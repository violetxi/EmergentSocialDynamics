"""
Trainer class and training loop are here
"""
import os
# set to false to suppress warnings about acotr and qvalue networks needs to be 
# updated manually as we do that in the training loop
os.environ['RL_WARNINGS']='False'
import wandb
import argparse
import datetime
import numpy as np
from defaults import DEFAULT_ARGS
from typeguard import typechecked
from typing import List, Optional
from copy import deepcopy
from tqdm import tqdm

import torch
from tensordict import TensorDict
from torchrl.modules.tensordict_module.common import SafeModule

from social_rl.agents.base_agent import BaseAgent
from social_rl.utils.reward_standardizer import RewardStandardizer
from social_rl.utils.utils import (
    load_config_from_path,
    ensure_dir,
)
torch.autograd.set_detect_anomaly(True)



@typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a MARL model')
    parser.add_argument(
        '--config_path', type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--seed', type=int, 
        default=DEFAULT_ARGS['seed'],
        help='Random seed'
    )
    parser.add_argument(
        '--log_dir', type=str, 
        default=DEFAULT_ARGS['log_dir'],
        help='Directory to save logs'
    )
    parser.add_argument(
        '--batch_size', type=int, 
        default=DEFAULT_ARGS['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '--num_episodes', type=int, 
        default=DEFAULT_ARGS['num_episodes'],
        help='Total number of episodes to train on'
    )
    parser.add_argument(
        '--max_episode_len', type=int,
        default=DEFAULT_ARGS['max_episode_len'],
        help='Max number of steps per episode'
    )
    parser.add_argument(
        '--warm_up_steps', type=int,
        default=DEFAULT_ARGS['warm_up_steps'],
        help='Number of steps to warm up the replay buffer'
    )
    parser.add_argument(
        '--project_name', type=str,
        default=DEFAULT_ARGS['project_name'],
        help='Name of the project for wandb'
    )
    args = parser.parse_args()
    return args



@typechecked
class Trainer:
    """Trainer class contains a reward standardizer to regularize rewards such 
    that it has mean=0, std=1.
    Args:
        args (argparse.Namespace): arguments
    """
    def __init__(
        self, 
        args: argparse.Namespace
    ) -> None:
        self.reward_standardizer = RewardStandardizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args        
        self._init_log_dir()
        self.config = load_config_from_path(args.config_path, args)        
        self._init_env(args.seed)
        self._init_agents()
        self._init_buffer()
        self._init_wandb()


    def _init_save_postfix(self) -> str:
        postfix = ""
        for arg, default_value in DEFAULT_ARGS.items():
            current_value = getattr(args, arg)
            if current_value != default_value:
                postfix += f'-{arg}_{current_value}'
        return postfix
    

    def _init_log_dir(self) -> None:
        """Initialize log directory to save buffer, checkpoints, and logs
        By default, log_dir is ./logs/ and the name of the log directory is
        the name of the config file without the extension
        """
        self.postfix = self._init_save_postfix()      
        self.args.log_dir = os.path.join(
            self.args.log_dir,       
            os.path.basename(self.args.config_path).split('.')[0]
            )        
        self.args.log_dir += self.postfix
        ensure_dir(self.args.log_dir)
        
        self.checkpoint_dir = os.path.join(
            self.args.log_dir,            
            'checkpoints'
            )
        ensure_dir(self.checkpoint_dir)
        

    def _init_wandb(self) -> None:
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{os.path.basename(self.args.config_path).split('.')[0]}"
        run_name += self.postfix   
        run_name += f":{cur_time}"
        wandb.init(
            project=self.args.project_name, 
            config=args.__dict__,
            name=run_name
            )


    def _get_test_env_seed(self) -> int:
        """
        Initialize test environment seed to ensure no overlap with trianing env
        """
        env_config = self.config.env_config
        low = self.args.seed + self.args.num_episodes + 1
        high = low + self.args.seed + self.args.max_episode_len
        return np.random.randint(low, high)
        

    def _init_env(
        self, 
        seed: int
        ) -> None:
        env_config = self.config.env_config
        self.env_name = env_config.env_name
        env_class = env_config.env_class        
        # Do not use batch_size for environment
        kwargs = {
            "device": self.device,
        }
        self.env = env_class(seed, env_config, kwargs)
        test_env_seed = self._get_test_env_seed()
        self.test_env = env_class(test_env_seed, env_config, kwargs)


    def _init_agent(
        self, 
        agent_idx: int, 
        agent_id: str
    ) -> BaseAgent:
        """Initialize each agent's world model, actor, value and qvalue networks
        args:
            agent_idx: index of agent in env
            agent_id: id of agent in env
        """ 
        actor_config = self.config.agent_config.actor_config
        actor_net_module = actor_config.net_module(actor_config.net_kwargs)
        module = actor_config.dist_wrapper(actor_net_module)        
        actor_module = SafeModule(
            module, 
            in_keys=actor_config.in_keys, 
            out_keys=actor_config.intermediate_keys
        )
        actor = actor_config.wrapper_class(
            module=actor_module, 
            in_keys=actor_config.intermediate_keys,
            out_keys=actor_config.out_keys,
            spec=actor_config.action_spec,
            distribution_class=actor_config.dist_class  
        )
        actor = actor.to(self.device)

        qvalue_config = self.config.agent_config.qvalue_config
        qvalue_module = qvalue_config.net_module(qvalue_config.net_kwargs)
        # outkeys defaults to state_action_value with obs as inkey
        qvalue = qvalue_config.wrapper_class(
            qvalue_module, 
            in_keys=qvalue_config.in_keys,
            out_keys=qvalue_config.out_keys
        )
        qvalue = qvalue.to(self.device)
        
        wm_config = self.config.agent_config.wm_config                
        wm_module = wm_config.wm_module_cls(agent_idx, wm_config)        
        world_model = wm_config.wrapper_class(
            wm_module, 
            wm_config.in_keys, 
            wm_config.out_keys
        )
        world_model = world_model.to(self.device)

        replay_buffer_config = self.config.agent_config.replay_buffer_config        
        replay_buffer = replay_buffer_config.buffer_class(**replay_buffer_config.buffer_kwargs)

        if self.config.agent_config.value_config is not None:
            value_config = self.config.agent_config.value_config
            value_module = value_config.net_module(value_config.net_kwargs)
            # outkeys defaults to state_value with obs as inkey     
            value = value_config.wrapper_class(
                value_module, 
                in_keys=value_config.in_keys,
                out_keys=value_config.out_keys
            )
            value = value.to(self.device)
        else:
            value = None

        if hasattr(self.config.agent_config, 'intr_reward_weight'):
            intr_reward_weight = self.config.agent_config.intr_reward_weight
            agent = self.config.agent_config.agent_class(
                agent_idx=agent_idx,
                agent_id=agent_id,
                config=self.config.agent_config,
                actor=actor,
                qvalue=qvalue,
                world_model=world_model,
                replay_buffer=replay_buffer,
                intr_reward_weight=intr_reward_weight,
                value=value
            )
            self._step_intr_reward = {}
        else:
            agent = self.config.agent_config.agent_class(
                agent_idx=agent_idx, 
                agent_id=agent_id,
                config=self.config.agent_config,
                actor=actor,
                value=value, 
                qvalue=qvalue, 
                world_model=world_model, 
                replay_buffer=replay_buffer        
            )

        return agent


    def _init_agents(self) -> None:
        agent_config = self.config.agent_config
        agent_ids = self.env._env.agents    # get agent ids from env
        assert len(agent_ids) == agent_config.num_agents, \
            f"Number of agents in env ({len(agent_ids)}) does not match number of agents in config ({agent_config.num_agents})"
        
        self.agents = {}        
        for agent_idx in range(agent_config.num_agents):
            agent_id = agent_ids[agent_idx]            
            agent = self._init_agent(agent_idx, agent_id)
            self.agents[agent_id] = agent        
        print(f"Finished initializing {agent_config.num_agents} agents")


    def _init_buffer(
        self
    ) -> None:
        """Initialize replay buffer for each agent for warm-up steps
        """
        tensordict = self.env.reset()
        print(f"Starting warm-up steps for {self.args.warm_up_steps} steps")

        for i in tqdm(range(self.args.warm_up_steps)):
            with torch.no_grad():
                tensordict = tensordict.to(self.device)
                tensordict = self._step_episode(tensordict)

            for _, agent in self.agents.items():                
                if i > 0:
                    # do not log the first step because actions are taken randomly
                    agent.replay_buffer.add(tensordict.clone())                   


    def _step_episode(
        self, 
        tensordict: TensorDict
    ) -> TensorDict:        
        actions = {}
        for agent_id, agent in self.agents.items():
            # Switch to eval mode for environment interaction
            agent.set_eval()            
            action = agent.act(tensordict.clone())
            if len(action.shape) == 2:
                action = torch.argmax(action, dim=1)[0]
            actions[agent_id] = action

            if hasattr(agent, 'intr_reward'):
                self.reward_standardizer.update(agent.intr_reward)
                intr_reward = self.reward_standardizer.standardize(agent.intr_reward)
                self._step_intr_reward[agent_id] = intr_reward

        tensordict["action"] = deepcopy(actions)
        tensordict = self.env.step(tensordict.detach().cpu())        
        tensordict["prev_action"] = deepcopy(actions)
        
        # combine intrinsic reward with extrinsic reward (the intrinsic reward is already scaled)
        if hasattr(self, '_step_intr_reward') and self._step_intr_reward:
            tensordict["intr_reward"] = deepcopy(self._step_intr_reward)
            self._step_intr_reward = {}
            for agent_id, intr_reward in tensordict["intr_reward"].items():
                extr_reward = tensordict.get(("next", "reward", agent_id))
                tensordict.set(("extr_reward", agent_id), extr_reward)
                tensordict.set(("next", "reward", agent_id), intr_reward + extr_reward)         

        return tensordict


    def convert_wm_to_actor_tensordict(
        self, 
        tensordict: TensorDict, 
        agent_id: str,
        required_keys: List[str] = ["observation", "done", "action", "latent", "next"]
    ) -> TensorDict:
        """Convert world model's tensordict to actor's tensordict, i.e., 
        make sure each key only has tensor value for current agent
        """
        tensordict_out = TensorDict({}, batch_size=tensordict.batch_size)
        for key in required_keys:
            if isinstance(tensordict[key], TensorDict):
                if key == "next":                    
                    tensordict = self.convert_wm_to_actor_tensordict(
                        tensordict[key], 
                        agent_id, 
                        required_keys=[
                            "observation", "done", "reward"
                        ]
                    )
                    tensordict_out[key] = tensordict
                else:
                    if key == "action":
                        # convert action to one-hot vector in one line 
                        action = tensordict.get((key, agent_id))
                        action = torch.nn.functional.one_hot(
                            action, 
                            num_classes=self.env.action_spec[agent_id].space.n
                        )
                        tensordict_out[key] = action
                    else:
                        tensordict_out[key] = tensordict.get((key, agent_id))            

            elif isinstance(tensordict[key], torch.Tensor):
                tensordict_out[key] = tensordict.get(key)
            else:
                raise NotImplementedError(f"Type of {key} is {type(tensordict[key])} which is not supported")
            
        return tensordict_out        


    def train_episode(
        self,
        episode: int,
        tensordict: TensorDict,
        tensordict_test: Optional[TensorDict] = None
        ) -> None:
        """Train agents for one episode, in a parallelized environment env.step() takes all 
        agents' actions as input and returns the next obs, reward, done, info for each agent
        """
        for t in tqdm(range(self.args.max_episode_len)):
            self.step += 1
            with torch.no_grad():
                # Disable gradient computation
                tensordict = tensordict.to(self.device)          
                tensordict = self._step_episode(tensordict)
                # evaluating in test environment
                if tensordict_test is not None:
                    tensordict_test = tensordict_test.to(self.device)
                    tensordict_test = self._step_episode(tensordict_test)

            for agent_id, agent in self.agents.items():
                # keep adding new experience
                agent.replay_buffer.add(tensordict.clone())
                # log reward (intrinsic and extrinsic)            
                if hasattr(agent, 'intr_reward'):                    
                    intr_reward = tensordict.get(("intr_reward", agent_id)).item()
                    wandb.log(
                        {f"{agent_id}_intr_reward": intr_reward},
                        step=self.step
                        )
                    extr_reward = tensordict.get(("extr_reward", agent_id)).item()
                    wandb.log(
                        {f"{agent_id}_reward": extr_reward},
                        step=self.step
                        )
                else:
                    wandb.log(
                        {f"{agent_id}_reward": tensordict.get(('next', 'reward', agent_id)).item()},
                        step=self.step
                        )
                
                # log test reward
                if tensordict_test is not None:           
                    if hasattr(agent, 'intr_reward'):
                        intr_reward_test = tensordict_test.get(("intr_reward", agent_id)).item()
                        wandb.log(
                            {f"{agent_id}_intr_reward_test": intr_reward_test},
                            step=self.step
                            )
                        extr_reward_test = tensordict_test.get(("extr_reward", agent_id)).item()
                        wandb.log(
                            {f"{agent_id}_reward_test": extr_reward_test},
                            step=self.step
                            )                        
                    else:
                        wandb.log(
                            {f"{agent_id}_reward_test": tensordict_test.get(('next', 'reward', agent_id)).item()},
                            step=self.step
                            )
            
            if tensordict['done'].all():
                return
        
            if t > 0:
                # update wm and actor
                for agent_id, agent in self.agents.items():
                    # swtich to train mode for learning
                    agent.set_train()  
                    tensordict_batch = agent.replay_buffer.sample().to(self.device)                
                    wm_loss_dict, tensordict_wm = agent.update_wm_grads(tensordict_batch)
                    tensordict_actor = self.convert_wm_to_actor_tensordict(tensordict_wm, agent_id)
                    actor_loss_dict = agent.update_actor_grads(tensordict_actor)
                    agent.step_optimizer()
                    # log wm_dict and actor_dict
                    for key, value in wm_loss_dict.items():
                        wandb.log(
                            {f"{agent_id}_wm_{key}": value},
                            step=self.step
                            )
                    for key, value in actor_loss_dict.items():
                        wandb.log(
                            {f"{agent_id}_{key}": value},
                            step=self.step
                        )
        
        for agent_id, agent in self.agents.items():
            model_save_path = f"{self.checkpoint_dir}/{agent_id}_ep{episode}_model_weights.pth"
            agent.save_model_weights(model_save_path)
            

    def train(self) -> None:
        self.step = 0
        for episode in tqdm(range(self.args.num_episodes)):            
            tensordict = self.env.reset(seed=episode)            
            test_env_seed = self._get_test_env_seed()
            tensordict_eval = self.test_env.reset(seed=test_env_seed)
            self.train_episode(episode, tensordict, tensordict_eval)



if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()