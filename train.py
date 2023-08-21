import os
import cv2
import glob
import yaml
import pickle
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict, Union, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
# to supress tensorboard pkg_resources deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")
from tianshou.data import VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.logger.wandb import WandbLogger

from social_rl.model.core import CNN
from social_rl.tianshou_elign.data import Collector
from social_rl.tianshou_elign.env import VectorEnv
from social_rl.envs.social_dilemma.pettingzoo_env import parallel_env
from social_rl.policy.multi_agent_policy_manager import MultiAgentPolicyManager
from social_rl.util.utils import ensure_dir

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
        self.train_step_agent_rews = None
        self.eval_step_agent_rews = None
        self._load_config()
        self._setup_env()
        self.set_seed()
        self._setup_agents()
        self._setup_collectors()
        self.log_name = self._get_save_path()
        self.log_path = os.path.join(args.logdir, self.log_name)        
        os.makedirs(self.log_path, exist_ok=True)

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
        self.config = configs
        self.env_config = configs['Environment']        
        self.net_config = configs['Net']
        self.policy_config = configs['PPOPolicy']

    def _setup_env(self) -> None:        
        # this is just a dummpy for setting up other things later
        env = get_env(self.env_config)
        self.agent_ids = env.possible_agents
        self.action_space = env._env.action_space
        # these will be used for evaluation and plotting
        self.env_agents = env.possible_agents
        self.agent_colors = env.get_agent_colors()
        # trainer will use this to collect data
        self.train_envs = VectorEnv([lambda : deepcopy(env) for i in range(self.args.train_env_num)])
        self.test_envs = VectorEnv([lambda : deepcopy(env) for i in range(self.args.test_env_num)])        
    
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
            self.args.buffer_size * self.args.train_env_num,
            buffer_num=self.args.train_env_num,
        )        
        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            train_buffer,
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
            elif k in ['ckpt_dir', 'eval_only']:
                pass
            else:
                if v != default_args[k]:
                    args_diffs.append(f'{k}_{curr_args[k]}')
        return '-'.join(args_diffs)

    """ Defining call back functions for onpolicy_trainer """
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
    
    def train(self) -> None:
        args = self.args
        # logger setup     
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb_run_name = f"{self.log_name}-{cur_time}"        
        # logger = WandbLogger(
        #     save_interval=args.save_interval,
        #     project=args.project_name,
        #     name=wandb_run_name,
        #     config=self.config,
        #     )
        # writer = SummaryWriter(self.log_path)
        # writer.add_text("args", str(args))
        # logger.load(writer)        
        # run trainer
        train_result = onpolicy_trainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_env_num,   # number of episodes tested during training
            batch_size=args.batch_size,            
            step_per_collect=args.step_per_collect,
            episode_per_collect=args.episode_per_collect,
            save_best_fn=self.save_best_fn,
            reward_metric=self.reward_metric,    # used in Collector                        
            stop_fn=self.stop_fn,
            #logger=logger
            )
        # save training result
        train_result_save_path = os.path.join(self.log_path, 'train_result.pkl')
        with open(train_result_save_path, 'wb') as f:
            pickle.dump(train_result, f)        
        print(f"\n========== Test Result during training==========\n{train_result}")

    def eval(self) -> None:
        args = self.args
        if args.eval_only:
            ckpts = glob.glob(f'{args.ckpt_dir}/*.pth')
            agent_ckpts = {ckpt.split('-')[-1].split('.')[0]: ckpt for ckpt in ckpts}            
            for agent_id, agent_policy in self.policy.policies.items():
                agent_policy.load_state_dict(torch.load(agent_ckpts[agent_id]))
        # run testing
        self.policy.eval()
        print(f"\n========== Eval after training ==========")        
        eval_result = self.test_collector.collect(
            n_episode=args.eval_eps,
            render_mode='rgb_array'
            )
        #step_agent_reward = np.array(eval_result.pop('step_agent_rews'))
        step_agent_reward = eval_result.pop('step_agent_rews')
        breakpoint()
        frames = eval_result.pop('frames')
        self.save_results(step_agent_reward, frames)
        #print(f"\n========== Eval after training ==========\n{eval_result}")

    def _convert_save_data(
            self, 
            data: np.ndarray
            ) -> List[Dict[str, np.ndarray]]:
        """Convert data from np.ndarray (n_episode, n_step, n_agent) to 
        [{agent_id: [ep1_rew, ...]}, {agent_id: [ep2_rew, ...]}, ...]        
        """
        output = []
        for i in range(data.shape[0]):
            output.append({
                agent_id: data[i, :, j] 
                for j, agent_id in enumerate(self.env_agents)
                })
        return output

    def save_results(
            self, 
            data: np.ndarray,
            episode_frames: Optional[List[List[np.ndarray]]] = None
            ) -> None:        
        args = self.args
        ensure_dir(args.logdir)
        task_name = self.log_path.split('/')[1].split('_')[0]        
        # use config to identify model types
        # if args.ckpt_dir is not None:
        #     model_name = args.ckpt_dir.split('/')[-1]
        # else:
        model_name = os.path.basename(args.config).split('.')[0]        
        # save eval step-wise agent reward data, in case of reruns with duplicated 
        # model and hypere-parameter settings, replace the data
        result_path = os.path.join(args.logdir, f"{task_name}.pkl")
        data = self._convert_save_data(data)
        if os.path.exists(result_path):
            print(f"{result_path} alreadying exist, appending data..")
            with open(result_path, 'rb') as f:
                existing_data = pickle.load(f)
                existing_models = [k for data in existing_data for k in data.keys()]
                model_idx = existing_models.index(model_name) \
                    if model_name in existing_models else None
                # if same model&hyperparam is tested in the past, replace
                if model_idx is not None:
                    print("Model already exist, replacing..")
                    existing_data[model_idx] = {model_name: data}
                else:
                    print("Model not exist, appending..")
                    existing_data.append({model_name: data})
        else:
            existing_data = [{model_name: data}]
        with open(result_path, 'wb') as f:
            pickle.dump(existing_data, f)
        print(f"data saved to {result_path}..")
        # create videos
        video_folder = os.path.join(self.log_path, "videos", model_name)
        ensure_dir(video_folder)
        for i, run_frames in enumerate(episode_frames):
            video_path = os.path.join(video_folder, f"{model_name}-ep_{i}.mp4")            
            self.save_video(run_frames, data[i], video_path)

    # write a function to save list of frames to video
    def save_video(
            self, 
            frames: List[np.ndarray], 
            rewards: Dict[str, List],
            filename: str
            ) -> None:
        """Save frames to video
        Args:
            frames: list of frames
            filename: filename to save video to
        """
        print(f"Saving video to {filename}..")
        #height, width, layers = frames[0].shape
        height = 500
        width = 300
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 30, (width*2, height))
        max_steps = len(frames)
        for i, frame in enumerate(frames): 
            reward_img = self.render_reward_curve(rewards, i, max_steps)
            reward_img = cv2.resize(reward_img, (width, height), interpolation=cv2.INTER_AREA)
            reward_img.astype('uint8')
            frame = frame.astype('uint8')
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            combined_frame = cv2.hconcat([frame, reward_img])
            video.write(combined_frame)
        cv2.destroyAllWindows()
        video.release()

    def render_reward_curve(self, rewards_dict, cur_steps, max_steps):        
        """Render reward curve as an image"""
        fig, ax = plt.subplots()
        for agent_id, rewards in rewards_dict.items():
            ax.plot(
                rewards[:cur_steps], 
                label=f"Agent {agent_id}", 
                color=self.agent_colors[agent_id],
                alpha=0.5)

        ax.set_title("Reward at Each Step for Each Agent")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        ax.set_xlim([0, max_steps])  # Set x-axis limits
        ax.legend()  # Add a legend

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

    def run(self) -> None:
        if not self.args.eval_only:
            self.train()                
            self.eval()
        else:
            assert self.args.ckpt_dir is not None, \
                "ckpt_dir must be specified for eval_only"
            self.eval()


if __name__ == "__main__":
    args = get_args()
    train_runner = TrainRunner(args)
    train_runner.run()