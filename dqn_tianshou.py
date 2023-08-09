"""This is a minimal example to show how to use Tianshou with a PettingZoo environment


Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
import os
import argparse

import torch
from torch import nn
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, DQNPolicy
from tianshou.trainer import offpolicy_trainer
from pettingzoo.utils.conversions import parallel_to_aec

from pettingzoo_env import parallel_env


def get_env(env_name, env_kwargs):
    # Step 1: Load ssd environment
    env = parallel_env(env_kwargs, render_mode="rgb_array")
    env = parallel_to_aec(env)
    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)
    return env


class CNN(nn.Module):
    def __init__(self, action_shape):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1),
            nn.ReLU(),            
            nn.Flatten(),
            nn.Linear(1014, 32),
            nn.ReLU(),
            nn.Linear(32, action_shape),                        
        )
        self.preprocessing = self.tranforms = Compose([
            ToPILImage(),
            Grayscale(),
            ToTensor(),                         
        ])

    def forward(self, obs, state=None, info={}):
        transformed_obs = torch.stack(
            [self.preprocessing(ob) for ob in obs.curr_obs])
        logits = self.model(transformed_obs.to("cuda"))
        return logits, state



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--estimation-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    return parser.parse_args()




if __name__ == "__main__":    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    net = CNN(env.action_space.n).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    dqn_agent = DQNPolicy(
        net, 
        optim, 
        discount_factor=args.discount_factor, 
        estimation_step=args.estimation_step, 
        target_update_freq=args.target_update_freq
        )
    all_agents = [dqn_agent]
    policy = MultiAgentPolicyManager(all_agents, env)
    agents = env.agents

    # Step 3: Define policies for each agent
    #policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)
    policies = MultiAgentPolicyManager([RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    #collector = Collector(policies, env)
    collector = Collector(policy, env, exploration_noise=True)


    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=2)
    print(f"\n==========Random Result==========\n{result}")

#     # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(1_000_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=1000)  # batch size * training_num
    output_dir = f"discount_factor{args.discount_factor}_estimation_step{args.estimation_step}_target_update_freq{args.target_update_freq}"

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_foldler = os.path.join("log", "harvest", "dqn", output_dir)
        model_save_path = os.path.join(model_save_foldler, "best_policy.pth")
        os.makedirs(model_save_foldler, exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 3000

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1000,
        step_per_epoch=5000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=2048,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    eval_collector = Collector(policy, test_envs)
    test_result = test_collector.collect(n_episode=2)
    # return result, policy.policies[agents[1]]
    print(f"\n==========DQN test Result==========\n{test_result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")
