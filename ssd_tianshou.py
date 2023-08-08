"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
import os

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
        logits = self.model(transformed_obs)
        return logits, state



if __name__ == "__main__":    
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
    net = CNN(env.action_space.n)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    dqn_agent = DQNPolicy(
        net, 
        optim, 
        discount_factor=0.9, 
        estimation_step=3, 
        target_update_freq=320
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
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=2)
    print(f"\n==========Random Result==========\n{result}")

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

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
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
    #print(f"\n==========DQN train Result==========\n{result}")
    print(f"\n==========DQN test Result==========\n{test_result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")


"""This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

# import argparse
# import os
# from copy import deepcopy
# from typing import Optional, Tuple

# import gymnasium
# import numpy as np
# import torch
# from tianshou.data import Collector, VectorReplayBuffer
# from tianshou.env import DummyVectorEnv
# from tianshou.env.pettingzoo_env import PettingZooEnv
# from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
# from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger
# from tianshou.utils.net.common import Net
# from torch.utils.tensorboard import SummaryWriter

# from pettingzoo.classic import tictactoe_v3


# def get_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=1626)
#     parser.add_argument("--eps-test", type=float, default=0.05)
#     parser.add_argument("--eps-train", type=float, default=0.1)
#     parser.add_argument("--buffer-size", type=int, default=20000)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument(
#         "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
#     )
#     parser.add_argument("--n-step", type=int, default=3)
#     parser.add_argument("--target-update-freq", type=int, default=320)
#     parser.add_argument("--epoch", type=int, default=50)
#     parser.add_argument("--step-per-epoch", type=int, default=1000)
#     parser.add_argument("--step-per-collect", type=int, default=10)
#     parser.add_argument("--update-per-step", type=float, default=0.1)
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument(
#         "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
#     )
#     parser.add_argument("--training-num", type=int, default=10)
#     parser.add_argument("--test-num", type=int, default=10)
#     parser.add_argument("--logdir", type=str, default="log")
#     parser.add_argument("--render", type=float, default=0.1)
#     parser.add_argument(
#         "--win-rate",
#         type=float,
#         default=0.6,
#         help="the expected winning rate: Optimal policy can get 0.7",
#     )
#     parser.add_argument(
#         "--watch",
#         default=False,
#         action="store_true",
#         help="no training, " "watch the play of pre-trained models",
#     )
#     parser.add_argument(
#         "--agent-id",
#         type=int,
#         default=2,
#         help="the learned agent plays as the"
#         " agent_id-th player. Choices are 1 and 2.",
#     )
#     parser.add_argument(
#         "--resume-path",
#         type=str,
#         default="",
#         help="the path of agent pth file " "for resuming from a pre-trained agent",
#     )
#     parser.add_argument(
#         "--opponent-path",
#         type=str,
#         default="",
#         help="the path of opponent agent pth file "
#         "for resuming from a pre-trained agent",
#     )
#     parser.add_argument(
#         "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
#     )
#     return parser


# def get_args() -> argparse.Namespace:
#     parser = get_parser()
#     return parser.parse_known_args()[0]


# def get_agents(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
#     optim: Optional[torch.optim.Optimizer] = None,
# ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
#     env = get_env()
#     observation_space = (
#         env.observation_space["observation"]
#         if isinstance(env.observation_space, gymnasium.spaces.Dict)
#         else env.observation_space
#     )
#     args.state_shape = observation_space.shape or observation_space.n
#     args.action_shape = env.action_space.shape or env.action_space.n
#     if agent_learn is None:
#         # model
#         net = Net(
#             args.state_shape,
#             args.action_shape,
#             hidden_sizes=args.hidden_sizes,
#             device=args.device,
#         ).to(args.device)
#         if optim is None:
#             optim = torch.optim.Adam(net.parameters(), lr=args.lr)
#         agent_learn = DQNPolicy(
#             net,
#             optim,
#             args.gamma,
#             args.n_step,
#             target_update_freq=args.target_update_freq,
#         )
#         if args.resume_path:
#             agent_learn.load_state_dict(torch.load(args.resume_path))

#     if agent_opponent is None:
#         if args.opponent_path:
#             agent_opponent = deepcopy(agent_learn)
#             agent_opponent.load_state_dict(torch.load(args.opponent_path))
#         else:
#             agent_opponent = RandomPolicy()

#     if args.agent_id == 1:
#         agents = [agent_learn, agent_opponent]
#     else:
#         agents = [agent_opponent, agent_learn]
#     policy = MultiAgentPolicyManager(agents, env)
#     return policy, optim, env.agents


# def get_env(render_mode=None):
#     return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


# def train_agent(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
#     optim: Optional[torch.optim.Optimizer] = None,
# ) -> Tuple[dict, BasePolicy]:
#     # ======== environment setup =========
#     train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
#     test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
#     # seed
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     train_envs.seed(args.seed)
#     test_envs.seed(args.seed)

#     # ======== agent setup =========
#     policy, optim, agents = get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
#     )

#     # ======== collector setup =========
#     train_collector = Collector(
#         policy,
#         train_envs,
#         VectorReplayBuffer(args.buffer_size, len(train_envs)),
#         exploration_noise=True,
#     )
#     test_collector = Collector(policy, test_envs, exploration_noise=True)
#     # policy.set_eps(1)
#     train_collector.collect(n_step=args.batch_size * args.training_num)

#     # ======== tensorboard logging setup =========
#     log_path = os.path.join(args.logdir, "tic_tac_toe", "dqn")
#     writer = SummaryWriter(log_path)
#     writer.add_text("args", str(args))
#     logger = TensorboardLogger(writer)

#     # ======== callback functions used during training =========
#     def save_best_fn(policy):
#         if hasattr(args, "model_save_path"):
#             model_save_path = args.model_save_path
#         else:
#             model_save_path = os.path.join(
#                 args.logdir, "tic_tac_toe", "dqn", "policy.pth"
#             )
#         torch.save(
#             policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
#         )

#     def stop_fn(mean_rewards):
#         return mean_rewards >= args.win_rate

#     def train_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

#     def test_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

#     def reward_metric(rews):
#         return rews[:, args.agent_id - 1]

#     # trainer
#     result = offpolicy_trainer(
#         policy,
#         train_collector,
#         test_collector,
#         args.epoch,
#         args.step_per_epoch,
#         args.step_per_collect,
#         args.test_num,
#         args.batch_size,
#         train_fn=train_fn,
#         test_fn=test_fn,
#         stop_fn=stop_fn,
#         save_best_fn=save_best_fn,
#         update_per_step=args.update_per_step,
#         logger=logger,
#         test_in_train=False,
#         reward_metric=reward_metric,
#     )

#     return result, policy.policies[agents[args.agent_id - 1]]


# # ======== a test function that tests a pre-trained agent ======
# def watch(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
# ) -> None:
#     env = DummyVectorEnv([lambda: get_env(render_mode="rgb_array")])
#     policy, optim, agents = get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent
#     )
#     policy.eval()
#     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#     collector = Collector(policy, env, exploration_noise=True)
#     result = collector.collect(n_episode=1, render=args.render)
#     rews, lens = result["rews"], result["lens"]
#     print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")


# if __name__ == "__main__":
#     # train the agent and watch its performance in a match!
#     args = get_args()
#     result, agent = train_agent(args)
#     watch(args, agent)