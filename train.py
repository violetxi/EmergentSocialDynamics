import os
import yaml
import datetime
import argparse
import numpy as np
from copy import deepcopy
import torch

from tianshou.data import VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from social_rl.model.core import CNN
from social_rl.tianshou_elign.data import Collector
from social_rl.tianshou_elign.env import VectorEnv
from social_rl.envs.social_dilemma.pettingzoo_env import parallel_env
from social_rl.policy.multi_agent_policy_manager import MultiAgentPolicyManager

from default_args import DefaultGlobalArgs


def get_args():
    global_config = DefaultGlobalArgs()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # add default arguments    
    for field, default in global_config.__annotations__.items():
        parser.add_argument(f'--{field}', default=getattr(global_config, field), type=default)    
    args = parser.parse_known_args()[0]
    return args


def get_env(env_config):
    # load ssd with wrapped as pettingzoo's parallel environment     
    return parallel_env(**env_config)


class TrainRunner:
    def __init__(
            self, 
            args: argparse.Namespace
            ) -> None:
        self.args = args
        self._load_config()
        self._setup_env()
        self.set_seed()
        self._setup_agents()
        self._setup_collectors()

    def set_seed(self) -> None:
        # @TODO: allow automatically generating seeds for N test and M train envs
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.device == 'cuda':
            torch.cuda.manual_seed(seed)
        #self.train_envs.seed()
        #self.test_envs.seed(seed)

    def _load_config(self) -> None:
        with open(self.args.config, 'r') as stream:
            configs = yaml.safe_load(stream)
        self.env_config = configs['Environment']        
        self.net_config = configs['Net']
        self.policy_config = configs['PPOPolicy']

    def _setup_env(self) -> None:        
        # this is just a dummpy for setting up other things later
        env = get_env(self.env_config)
        self.action_space = env._env.action_space
        self.env_agents = env.possible_agents        
        self.train_envs = VectorEnv([lambda : deepcopy(env) for i in range(self.args.train_num)])
        self.test_envs = VectorEnv([lambda : deepcopy(env) for i in range(self.args.test_num)])
    
    def _setup_single_agent(self, agent_id) -> PPOPolicy:
        net = CNN(self.net_config)
        device = self.args.device
        action_shape = self.action_space.n
        actor = Actor(net, action_shape, device=device).to(device)
        critic = Critic(net, device=device).to(device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=self.args.lr)        
        dist = torch.distributions.Categorical
        
        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=self.args.gamma,
            **self.policy_config
        )
        return policy

    def _setup_agents(self) -> None:
        all_agents = {}
        for agent in self.env_agents:
            all_agents[agent] = self._setup_single_agent(agent)
        self.policy = MultiAgentPolicyManager(
            list(all_agents.values()), 
            self.env_agents,
            self.action_space
            )

    def _setup_collectors(self) -> None:
        train_buffer = VectorReplayBuffer(
            self.args.buffer_size * self.args.train_num,
            buffer_num=self.args.train_num,
        )
        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            train_buffer,
            exploration_noise=True,
            #preprocess_fn=preprocess_fn
            )
        self.test_collector = Collector(
            self.policy, 
            self.test_envs, 
            exploration_noise=True,
            #preprocess_fn=preprocess_fn
            )

    def _get_save_path(self):
        default_args = vars(DefaultGlobalArgs())
        curr_args = vars(self.args)
        args_diffs = []
        for k, v in curr_args.items():
            # only 'config' will not be included in the DefaultGlobalArgs
            if k == 'config':
                args_diffs.insert(
                    0, f'{v.split("/")[-1].split(".")[0]}'
                    )
            else:
                if v != default_args[k]:
                    args_diffs.append(f'{k}_{curr_args[k]}')
        return '-'.join(args_diffs)

    """ Defining call back functions """
    def save_best_fn(self, policy):
        for agent_id in self.env_agents:
            model_save_path = os.path.join(
                self.log_path, f"best_policy-{agent_id}.pth")
            torch.save(policy.policies[agent_id].state_dict(), model_save_path)

    def stop_fn(self, mean_rewards):
        return mean_rewards >= args.reward_threshold

    def reward_metric(self, rews):
        # rews: (n_ep, n_agent)        
        return rews
    

    def run(self) -> None:
        args = self.args
        log_name = self._get_save_path()
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb_run_name = f"{log_name}-{cur_time}"
        # log        
        self.log_path = os.path.join(args.logdir, log_name)
        os.makedirs(self.log_path, exist_ok=True)        
        # logger = WandbLogger(name=wandb_run_name)
        # logger.load(SummaryWriter(self.log_path))
        # run trainer
        train_result = onpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=self.save_best_fn,
            reward_metric=self.reward_metric,    # used in Collector
            step_per_collect=args.step_per_collect,
            stop_fn=self.stop_fn,
            #logger=logger
            )
        # run testing
        self.policy.eval()
        eval_result = self.test_collector.collect(n_episode=2)            
        print(f"\n========== Test Result==========\n{eval_result}")     


if __name__ == "__main__":
    args = get_args()
    train_runner = TrainRunner(args)
    train_runner.run()