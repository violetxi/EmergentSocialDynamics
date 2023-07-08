"""
Trainer class and training loop are here
"""
import argparse
from defaults import DEFAULT_ARGS
from typeguard import typechecked
from typing import List
from copy import deepcopy

import torch
from tensordict import TensorDict
from torchrl.modules.tensordict_module.common import SafeModule

from social_rl.agents.base_agent import BaseAgent
from social_rl.utils.utils import (
    load_config_from_path,
    ensure_dir,
)



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
        '--epochs', type=int,
        default=DEFAULT_ARGS['epochs'],
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--num_episodes', type=int, 
        default=DEFAULT_ARGS['num_episodes'],
        help='Total number of episodes to train on'
    )
    parser.add_argument(
        '--episode_length', type=int,
        default=DEFAULT_ARGS['episode_length'],
        help='Max number of steps per episode'
    )
    parser.add_argument(
        '--warm_up_steps', type=int,
        default=DEFAULT_ARGS['warm_up_steps'],
        help='Number of steps to warm up the replay buffer'
    )
    parser.add_argument(
        '--val_freq', type=int, 
        default=DEFAULT_ARGS['val_freq'],
        help='Validation frequency'
    )
    args = parser.parse_args()
    return args



@typechecked
class Trainer:
    def __init__(
        self, 
        args: argparse.Namespace
    ) -> None:
        self.args = args
        ensure_dir(args.log_dir)
        self.config = load_config_from_path(args.config_path, args)
        self._init_env(args.seed)
        self._init_agents()
        self._init_buffer()


    def _init_env(
        self, 
        seed: int
    ) -> None:
        env_config = self.config.env_config
        self.env_name = env_config.env_name
        env_class = env_config.env_class
        # @TODO: change device after servers are back online
        # Do not use batch_size for environment
        kwargs = {
            "device": "cpu"
        }
        self.env = env_class(seed, env_config, kwargs)
        # self.env = env_class(seed, env_config)
        print(f"Finished initializing {self.env_name} environment")


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
        #action_spec = self.env.action_spec[agent_id]
        actor = actor_config.wrapper_class(
            module=actor_module, 
            in_keys=actor_config.intermediate_keys,
            out_keys=actor_config.out_keys,
            spec=actor_config.action_spec,
            distribution_class=actor_config.dist_class  
        )

        value_config = self.config.agent_config.value_config
        value_module = value_config.net_module(value_config.net_kwargs)
        # outkeys defaults to state_value with obs as inkey     
        value = value_config.wrapper_class(
            value_module, 
            in_keys=value_config.in_keys,
            out_keys=value_config.out_keys
        ) 

        qvalue_config = self.config.agent_config.qvalue_config
        qvalue_module = qvalue_config.net_module(qvalue_config.net_kwargs)
        # outkeys defaults to state_action_value with obs as inkey
        qvalue = value_config.wrapper_class(
            qvalue_module, 
            in_keys=qvalue_config.in_keys,
            out_keys=value_config.out_keys
        )
        
        wm_config = self.config.agent_config.wm_config                
        wm_module = wm_config.wm_module_cls(agent_idx, wm_config)        
        world_model = wm_config.wrapper_class(
            wm_module, 
            wm_config.in_keys, 
            wm_config.out_keys
        )

        replay_buffer_config = self.config.agent_config.replay_buffer_config        
        replay_buffer_wm = replay_buffer_config.buffer_class(**replay_buffer_config.buffer_kwargs)
        replay_buffer_actor = replay_buffer_config.buffer_class(**replay_buffer_config.buffer_kwargs)
        agent = self.config.agent_config.agent_class(
            agent_idx=agent_idx, 
            agent_id=agent_id,
            config=self.config.agent_config,
            actor=actor,
            value=value, 
            qvalue=qvalue, 
            world_model=world_model, 
            replay_buffer_wm=replay_buffer_wm, 
            replay_buffer_actor=replay_buffer_actor
        )        
        return agent


    def _init_agents(
        self
    ) -> None:
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

        for i in range(self.args.warm_up_steps):
            print(f"Warm-up step {i}")
            tensordict = self._step_episode(tensordict)

            for agent_id, agent in self.agents.items():
                if i > 0:
                    agent.replay_buffer_wm.add(tensordict.clone())
                    #agent.replay_buffer_actor.add(tensordict.clone())        


    def _step_episode(
        self, 
        tensordict: TensorDict
    ) -> TensorDict:        
        actions = {}           
        for agent_id, agent in self.agents.items():
            action = agent.act(tensordict.clone())
            if len(action.shape) == 2:
                action = torch.argmax(action, dim=1)[0]
            actions[agent_id] = action
        tensordict["action"] = deepcopy(actions)
        tensordict = self.env.step(tensordict)
        tensordict["prev_action"] = deepcopy(actions)
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
        tensordict: TensorDict
    ) -> None:
        """Train agents for one episode, in a parallelized environment env.step() takes all 
        agents' actions as input and returns the next obs, reward, done, info for each agent
        """
        for t in range(self.args.episode_length):                       
            tensordict = self._step_episode(tensordict)    
            for _, agent in self.agents.items():
                # keep adding new experience
                agent.replay_buffer_wm.add(tensordict.clone())
                #agent.replay_buffer_actor.add(tensordict.clone())
            
            if tensordict['done'].all():
                return
        
            for agent_id, agent in self.agents.items():
                tensordict = agent.replay_buffer_wm.sample()
                wm_loss_dict, tensordict_wm = agent.update_wm(tensordict)
                tensordict_actor = self.convert_wm_to_actor_tensordict(tensordict_wm, agent_id)
                actor_loss_dict = agent.update_actor(tensordict_actor)
                # log wm_dict and actor_dict


    def train(self) -> None:
        for episode in range(self.args.num_episodes): 
            tensordict = self.env.reset()           
            self.train_episode(tensordict)




if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()   