"""
Evaluation class to evaluate trained agents given an environment.
Evaluation function is used to evaluate in bulk
"""
import os
os.environ['RL_WARNINGS']='False'
import argparse
from defaults import DEFAULT_ARGS
from typeguard import typechecked
from typing import List, Dict
from tqdm import tqdm
import numpy as np

import torch
from tensordict import TensorDict
from torchrl.envs.common import _EnvWrapper
from torchrl.modules.tensordict_module.common import SafeModule

from social_rl.agents.base_agent import BaseAgent
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
        '--model_folder', type=str,
        required=True,  
        help='Path to folder containing model checkpoints'
    )
    parser.add_argument(
        '--result_folder', type=str,
        default='./results',
        help='Path to folder to save results'
    )
    parser.add_argument(
        '--num_episodes', type=int,
        default=100,
        help='Number of episodes to evaluate for'
    )
    parser.add_argument(
        '--eval_ckpt_type', type=str,
        default='last',
        help='Type of checkpoint to evaluate, can be last or every n'
    )    
    args = parser.parse_args()
    return args



@typechecked
class RunEvaluation:
    """This class is used to run evaluation on a trained model        
    """
    def __init__(
            self, 
            args: argparse.Namespace
            ) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_args = self._recreate_train_args() 
        self.config = load_config_from_path(args.config_path, self.train_args)
        self._init_env()
        self._init_agents()
            

    def _recreate_train_args(self) -> argparse.Namespace:
        train_args = argparse.Namespace(**DEFAULT_ARGS)
        if '-' in self.args.model_folder:
            updated_train_args = self.args.model_folder.split('-')        
        return train_args


    def _init_env(self) -> None:
        """
        Initialize environment that has a different seed from training
        """
        seed = self.train_args.seed + np.random.randint(1000, 10000)
        env_config = self.config.env_config
        self.env_name = env_config.env_name
        env_class = env_config.env_class        
        # Do not use batch_size for environment
        kwargs = {
            "device": self.device,
        }
        self.env = env_class(seed, env_config, kwargs)
        print(f"Finished initializing {self.env_name} environment with seed {seed}")


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


    def _choose_ckpt(self) -> Dict[str, str]:
        """Choose checkpoint to evaluate
        """
        ckpt_files = [
            f for f in os.listdir(self.args.model_folder) if f.endswith('.pth') and 'agent' in f
            ]
        eps = [int(f.split('_')[2][2:]) for f in ckpt_files]
        if self.args.eval_ckpt_type == 'last':
            ckpt_ep = f"ep{np.max(eps)}"
        ckpt_files = [f for f in ckpt_files if ckpt_ep in f]
        ckpt_f_dict = {'_'.join(f.split('_')[:2]): f for f in ckpt_files}
        return ckpt_f_dict
    

    def _init_agents(self) -> None:
        agent_config = self.config.agent_config
        agent_ids = self.env._env.agents
        assert len(agent_ids) == agent_config.num_agents, \
            f"Number of agents in env ({len(agent_ids)}) does not match number of agents in config ({agent_config.num_agents})"

        self.agents = {}        
        ckpt_path_dict = self._choose_ckpt()
        for agent_idx in range(agent_config.num_agents):
            agent_id = agent_ids[agent_idx]            
            agent = self._init_agent(agent_idx, agent_id)
            model_path = os.path.join(self.args.model_folder, ckpt_path_dict[agent_id])                      
            agent.load_model_weights(model_path)
            agent.set_eval()
            self.agents[agent_id] = agent       
        print(f"Finished initializing and loading weights for {agent_config.num_agents} agents")



@typechecked
class Evaluator:
    """
    Class to evaluate trained agents given an environment.
    Args:
        agents: list of agents to evaluate
        env: environment to evaluate agents on        
    """
    def __init__(
            self, 
            agents: List[BaseAgent], 
            env: _EnvWrapper,
            eval_episodes: int,
            ) -> None:
        self.agents = agents
        self.env = env
        self.eval_episodes = eval_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

    
    def get_actions(
            self, 
            tensordict: TensorDict
            ) -> TensorDict:
        """Get actions from agents
        Args:
            tensordict: input to agents
        Returns:
            actions: actions from agents
        """
        actions = {}  
        for agent_id, agent in self.agents.items(): 
            action = agent.act(tensordict.clone())
            if len(action.shape) == 2:
                action = torch.argmax(action, dim=1)[0]
            actions[agent_id] = action
        tensordict["actions"] = actions
        tensordict_out = self.env.step(tensordict.detach().cpu())
        return tensordict_out


    def evaluate(
            self, 
            num_episodes: int
            ) -> None:
        """Evaluate agents on environment for num_episodes
        Args:
            num_episodes: number of episodes to evaluate for
        """
        self.env.reset()
        for agent in self.agents:
            agent.set_eval()
        episode_rewards = []             
        for episode in range(num_episodes):
            episode_reward = {}            
            tensordict = self.env.reset()
            done = False
            while not done:
                tensordict = self.get_actions(tensordict)                
                done = tensordict.get(("next", "done"))
                for agent_id, reward in tensordict.get(("next", "rewards")).items():
                    if agent_id not in episode_reward:
                        episode_reward[agent] = [reward.item()]
                    else:
                        episode_reward[agent].append(reward.item())                        
        return episode_rewards




if __name__ == '__main__':
    args = parse_args()
    evaluation_run = RunEvaluation(args)