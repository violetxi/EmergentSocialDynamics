"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
import os
import sys
import argparse
import  numpy as np
import cv2 

import torch
from torch import nn
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)

from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env import PettingZooEnv, DummyVectorEnv
from pettingzoo.utils.conversions import parallel_to_aec

from pettingzoo_env import parallel_env
#sys.path.append('/ccn2/u/ziyxiang/EmergentSocialDynamics')
from tianshou_srl.data.collector import Collector


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True, type=str, help='path to the checkpoint file')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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


def get_env(env_kwargs):
    # Step 1: Load ssd environment
    env = parallel_env(
        env_kwargs, 
        render_mode="rgb_array", 
        collect_frames=True)
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
        if len(obs.obs.curr_obs.shape) == 3:            
            transformed_obs = self.preprocessing(obs.obs.curr_obs)
            transformed_obs = transformed_obs.unsqueeze(0)
        else:
            transformed_obs = torch.stack(
                [self.preprocessing(ob) for ob in obs.obs.curr_obs])
        embds = self.encoder(transformed_obs.to("cuda"))
        logits = embds
        return logits, state


def save_video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h, c = frames[0].shape
    out = cv2.VideoWriter(filename, fourcc, 30.0, (w * 10, h * 10))
    for frame in frames:
        frame = cv2.resize(frame.astype(np.uint8), (w * 10, h * 10), interpolation=cv2.INTER_AREA)
        # We write every frame to the output video file. We first ensure the frame is in the correct format
        out.write(frame)    


if __name__ == "__main__":
    # setup env
    env_name = 'harvest'
    env_args = dict(
        env=env_name,
        num_agents=1,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0
    )    
    env = get_env(env_args)
    agents = env.agents       
    # seed
    seed = 0    
    torch.manual_seed(seed)
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
    ppo_agent = PPOPolicy(
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
    all_agents = [ppo_agent]
    policy = MultiAgentPolicyManager(all_agents, env)    
    policy.policies['agent-0'].load_state_dict(torch.load(args.ckpt_path))
    policy.eval()
    # prepare collector and get results
    env = DummyVectorEnv([lambda: env]) 
    eval_collector = Collector(policy, env, exploration_noise=True) 
    eval_result = eval_collector.collect(n_episode=2, collect_frames=True)
    # save video
    video_dir = os.path.join(os.path.dirname(args.ckpt_path), 'videos')
    os.makedirs(video_dir, exist_ok=True)
    model_str = args.ckpt_path.split('/')[-3]
    params_str = args.ckpt_path.split('/')[-2]    
    frames = eval_result['frames']    
    for ep_n in range(len(frames)):
        ep_frames = frames[ep_n]
        video_path = os.path.join(video_dir, f'{model_str}_{params_str}_{ep_n}.mp4')
        save_video(frames, video_path)    
    # print results
    print(f"Reward for {len(frames)} episodes for {len(agents)} agents")
    print(f"Mean total reward across episode: {eval_result['rews']}")
    print(f"std: {eval_result['rews_std']}")
    

# Step 3: Define policies for each agent    
    # policies = MultiAgentPolicyManager([RandomPolicy()], env)

    # # Step 4: Convert the env to vector format
    # env = DummyVectorEnv([lambda: env])

    # # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    # collector = Collector(policies, env)

    # # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    # result = collector.collect(n_episode=2)
    # print(f"\n==========Random Result==========\n{result}")