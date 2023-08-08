"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
import os
import argparse
import  numpy as np

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
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env import DummyVectorEnv, PettingZooEnv
from pettingzoo.utils.conversions import parallel_to_aec

from pettingzoo_env import parallel_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args


def get_env(env_name, env_kwargs):
    # Step 1: Load ssd environment
    env = parallel_env(env_kwargs, render_mode="rgb_array")
    env = parallel_to_aec(env)
    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)
    return env


class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.output_dim = output_dim    # actor wrapper expects this attribute
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1),
            nn.ReLU(),            
            nn.Flatten(),
            nn.Linear(1014, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )
        self.dist_mean = nn.Linear(output_dim, output_dim)
        self.dist_std = nn.Linear(output_dim, output_dim)
        self.preprocessing = self.tranforms = Compose([
            ToPILImage(),
            Grayscale(),
            ToTensor(),                         
        ])

    def forward(self, obs, state=None, info={}):        
        transformed_obs = torch.stack(
            [self.preprocessing(ob) for ob in obs.obs.curr_obs])
        embds = self.encoder(transformed_obs.to("cuda"))
        logits = embds
        return logits, state



if __name__ == "__main__":
    # setup env
    env_name = "harvest"
    env_args = dict(
        env=env_name,
        num_agents=1,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0
    )    
    env = get_env(env_name, env_args)
    train_envs = DummyVectorEnv([lambda : env])
    test_envs = DummyVectorEnv([lambda : env])
    # seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    # model
    net = CNN(64)
    action_shape = 8
    device = "cuda"
    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)
    dist = torch.distributions.Categorical
    # args
    args=get_args()
    dqn_agent = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )
    all_agents = [dqn_agent]
    policy = MultiAgentPolicyManager(all_agents, env)
    agents = env.agents

    # Step 3: Define policies for each agent    
    # policies = MultiAgentPolicyManager([RandomPolicy()], env)

    # # Step 4: Convert the env to vector format
    # env = DummyVectorEnv([lambda: env])

    # # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    # collector = Collector(policies, env)

    # # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    # result = collector.collect(n_episode=2)
    # print(f"\n==========Random Result==========\n{result}")

#     # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "harvest", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "harvest", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.8

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)

    def reward_metric(rews):   
        return rews[:, 0]

    # log
    log_path = os.path.join(args.logdir, 'Harvest', 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ======== Step 5: Run the trainer =========
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    # ======== Step 6: Test the trained policy ========
    policy.eval()
    eval_collector = Collector(policy, test_envs)
    eval_result = test_collector.collect(n_episode=1)            
    print(f"\n==========DQN test Result==========\n{eval_result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")
    breakpoint()
