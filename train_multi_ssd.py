from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import json
import os
import torch
import pprint
import argparse
import wandb
import numpy as np

from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager

from tianshou_elign.data import Collector, ReplayBuffer
from tianshou_elign.env import (
    VectorEnv,
    BaseRewardLogger
)
from tianshou_elign.policy import (
    SACDMultiPolicy,    
    SACDMultiCCPolicy,    
)
from tianshou_elign.trainer import offpolicy_trainer
from pettingzoo_env import parallel_env


# this section is temporary for testing code that is indeed working..
# @TODO: move to a separate file later
from torch import nn
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)

config = {
    'in_channels': 1,
    'out_channels': 6,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'bias': True,
    'flatten_dim': 1014,
    'num_layers': 2,
    'hidden_dim': 64,
    'output_dim': 64
}


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
        logits = self.encoder(obs.observation.curr_obs.to("cuda"))
        return logits, state


def get_args():
    parser = argparse.ArgumentParser()

    # State arguments.
    parser.add_argument('--task', type=str, default='simple_spread_in')
    parser.add_argument('--save-video', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--video-file', type=str, default='videos/simple.mp4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Training arguments
    parser.add_argument('--buffer-size', type=int, default=2000000)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--training-num', type=int, default=2)
    parser.add_argument('--test-num', type=int, default=1)

    # Model arguments
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--centralized', action='store_true', default=False)

    # SAC special
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=0)
    parser.add_argument('--n-step', type=int, default=1)

    # Task specific
    parser.add_argument('--num-good-agents', type=int, default=0)
    parser.add_argument('--num-adversaries', type=int, default=0)
    parser.add_argument('--obs-radius', type=float, default=float('inf'))
    parser.add_argument('--amb-init', type=int, default=0)
    parser.add_argument('--rew-shape', action='store_true', default=False)

    # Enable wandb logging or not
    parser.add_argument('--wandb-enabled', action='store_true', default=False)

    # Enable grads logging or not
    parser.add_argument('--grads-logging', action='store_true', default=False)

    # Specify the intrinsic reward type or no intrinsic reward
    # options include ['no', 'elign_self', 'elign_team', 'elign_adv', 'elign_both', 'curio_self', 'curio_team']
    parser.add_argument('--intr-rew', type=str, default='no')

    # add noise to world model or not
    parser.add_argument('--wm-noise-level', type=float, default=0.0)

    args = parser.parse_known_args()[0]
    return args


def preprocess_fn(batch):
    """Preprocess observation in image format
    """
    transform = Compose([ToPILImage(), Grayscale(), ToTensor(),])
    for agent_id in batch.obs.keys():
        ob = batch.obs.get(agent_id).observation.curr_obs
        processed_ob = torch.stack([transform(ob_i) for ob_i in ob])
        batch.obs[agent_id].observation.curr_obs = processed_ob
    return batch

def train_multi_sacd(args=get_args()):
    #wandb_dir = '/scr/zixianma/multiagent/' if torch.cuda.is_available() else 'log/'
    wandb_dir = 'log/'
    if args.wandb_enabled:
        wandb.init(dir=wandb_dir, sync_tensorboard=True)
        run_name = args.logdir[args.logdir.rfind('/') + 1:]
        wandb.run.name = run_name
        wandb.config.update(args)
    torch.set_num_threads(4)  # 1 for poor CPU

    base_env_kwargs = {
        'env': 'harvest',
        'num_agents': 4,
        'use_collective_reward': False,
        'inequity_averse_reward': False,
        'alpha': 0.0,
        'beta': 0.0
    }
    env_args = {
        'base_env_kwargs': base_env_kwargs,
        'max_cycles': 1000,
        'render_mode': 'rgb_array',
        'collect_frames': False,
    }
    train_envs = VectorEnv(
        [lambda: parallel_env(**env_args) for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: parallel_env(**env_args) for _ in range(args.test_num)])    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed) 
    # agents 
    num_agents = base_env_kwargs['num_agents']
    net = CNN(config)
    action_shape = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actors = [Actor(net, action_shape, device=device).to(device) 
              for _ in range(num_agents)]
    critic1s = [Critic(net, device=device).to(device) 
               for _ in range(num_agents)]
    critic2s = [Critic(net, device=device).to(device)
                for _ in range(num_agents)]    
    # Optimizers.
    actor_optims = [Adam(actor.parameters(), lr=args.actor_lr)
                    for actor in actors]
    critic1_optims = [Adam(critic1.parameters(), lr=args.critic_lr)
                      for critic1 in critic1s]
    critic2_optims = [Adam(critic2.parameters(), lr=args.critic_lr)
                      for critic2 in critic2s]    
    # Policy
    dist = torch.distributions.Categorical
    agent_ids = train_envs.envs[0].possible_agents
    if args.centralized:
        if args.intr_rew == 'no':
            policy = SACDMultiCCPolicy(
                actors, actor_optims,
                critic1s, critic1_optims,
                critic2s, critic2_optims,
                dist, args.tau, args.gamma, args.alpha,
                reward_normalization=args.rew_norm,
                ignore_done=args.ignore_done,
                estimation_step=args.n_step,
                grads_logging=args.grads_logging)
    else:
        if args.intr_rew == 'no':
            policy = SACDMultiPolicy(
                agent_ids,
                actors, actor_optims,
                critic1s, critic1_optims,
                critic2s, critic2_optims,
                dist, args.tau, args.gamma, args.alpha,
                #reward_normalization=args.rew_norm,
                ignore_done=args.ignore_done,
                estimation_step=args.n_step,
                grads_logging=args.grads_logging)

   # Load existing models if checkpoint is specified.
    if args.checkpoint:
        policy.load(args.checkpoint)

    # Setup the logger.
    reward_logger = BaseRewardLogger


    # collector    
    train_collector = Collector(
        policy, 
        train_envs, 
        ReplayBuffer(args.buffer_size),
        num_agents=num_agents,
        reward_logger=reward_logger,
        preprocess_fn=preprocess_fn
        )
    test_collector = Collector(
        policy, test_envs,
        num_agents=num_agents,
        reward_logger=reward_logger,
        preprocess_fn=preprocess_fn
        )

    # log
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    writer = SummaryWriter(args.logdir)

    def save_fn(): return policy.save(args.logdir) if args.save_models else None
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, save_fn=save_fn, writer=writer)
    breakpoint()
    train_collector.close()
    test_collector.close()

    if args.save_models:
        policy.save(args.logdir, type='final')

    if args.save_video:
        pprint.pprint(result)
        # Let's watch its performance!
        env = make_multiagent_env(
            args.task, benchmark=args.benchmark, optional=task_params)
        # Change max steps for a longer visualization
        env.world.max_steps = 1000
        collector = Collector(policy, env, num_agents=num_agents,
                              reward_logger=reward_logger,
                              benchmark_logger=vis_benchmark_logger)
        result = collector.collect(
            n_episode=1, render=args.render, render_mode='rgb_array')
        from tianshou.env import create_video
        create_video(result['frames'], args.video_file)
        print('Final reward: ', result["rew"])
        collector.close()


if __name__ == '__main__':
    train_multi_sacd()
