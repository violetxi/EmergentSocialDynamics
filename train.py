import os
import datetime
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env import DummyVectorEnv, PettingZooEnv
from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo_env import parallel_env
import yaml
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
    env = parallel_env(**env_config)
    env = parallel_to_aec(env)
    # wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)
    return env


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.output_dim = config['output_dim']
        self.encoder = nn.Sequential(
            nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], stride=config['stride']),
            nn.ReLU(),            
            nn.Flatten(),
            nn.Linear(config['flatten_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim']),
            nn.ReLU(),
        )
        self.dist_mean = nn.Linear(config['output_dim'], config['output_dim'])
        self.dist_std = nn.Linear(config['output_dim'], config['output_dim'])
        self.preprocessing = Compose([
            ToPILImage(),
            Grayscale(),
            ToTensor(),                         
        ])

    def forward(self, obs, state=None, info={}):        
        transformed_obs = torch.stack([self.preprocessing(ob) for ob in obs.obs.curr_obs])
        embds = self.encoder(transformed_obs.to("cuda"))
        logits = embds
        return logits, state
    

class DummyDist(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        action = torch.Tensor([7]).long().to("cuda")
        # print(f"Dummy action {action}")
        return action


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
        self.env = get_env(self.env_config)
        # @TODO: in visualization script, remember to set 
        # self.env.env.env.env.collect_frames = True
        # @TODO: allow multiple train and test envs for parallel training
        self.train_envs = DummyVectorEnv([lambda : self.env])
        self.test_envs = DummyVectorEnv([lambda : self.env])

    #def _setup_single_agent(self) -> PPOPolicy:
    def _setup_single_agent(self, agent_id) -> PPOPolicy:
        net = CNN(self.net_config)
        action_shape = self.env.action_space.n
        device = self.args.device
        actor = Actor(net, action_shape, device=device).to(device)
        critic = Critic(net, device=device).to(device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=self.args.lr)
        # @TODO: remove this later just using this to figure out 
        # if reward is distributed correctly per agent
        if agent_id == 'agent-0':
            dist = DummyDist
        else:
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
        for agent in self.env.agents:
            #all_agents[agent] = self._setup_single_agent()
            all_agents[agent] = self._setup_single_agent(agent)
        self.policy = MultiAgentPolicyManager(list(all_agents.values()), self.env)

    def _setup_collectors(self) -> None:
        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            VectorReplayBuffer(self.args.buffer_size, len(self.train_envs)),
            exploration_noise=True,
            )
        self.test_collector = Collector(
            self.policy, 
            self.test_envs, 
            exploration_noise=True,
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
        for agent_id in self.env.agents:
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