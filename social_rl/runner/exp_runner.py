import os
import cv2
import glob
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from importlib import import_module
from typing import List, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)
# to supress tensorboard pkg_resources deprecated warnings
from tianshou.data import VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from social_rl.tianshou_elign.data.collector import Collector
from social_rl.tianshou_elign.env.vecenv import VectorEnv
from social_rl.tianshou_elign.trainer.onpolicy import onpolicy_trainer
from social_rl.envs.social_dilemma.pettingzoo_env import parallel_env
from social_rl.policy.multi_agent_policy_manager import MultiAgentPolicyManager
from social_rl.policy.mappo import MAPPOPolicy
from social_rl.policy.multi_agent_policy_sharing_parameters_manager import \
    MultiAgentPolicySharingParametersManager
from social_rl.utils.loggers.wandb_logger import WandbLogger
from social_rl.utils.utils import ensure_dir

from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import (
    DictConfig,
    OmegaConf
    )


def get_env(env_config):
    # load ssd with wrapped as pettingzoo's parallel environment     
    return parallel_env(**env_config)


class TrainRunner:
    def __init__(
            self, 
            args: argparse.Namespace,
            ) -> None:        
        self.args = args
        self.train_step_agent_rews = None
        self.eval_step_agent_rews = None        
        self._load_config()        
        self._setup_env()        
        self.set_seed()
        self._setup_agents()        
        self._setup_collectors()        

    def set_seed(self) -> None:
        seed = self.args.exp_run.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.exp_run.device == 'cuda':
            torch.cuda.manual_seed(seed)            

    def _load_config(self) -> None:
        hydra_cfg = HydraConfig.get()
        # path to save and load checkpoints
        self.output_dir = os.path.join(get_original_cwd(), hydra_cfg.run.dir)
        self.log_name = self.output_dir.split('/')[-3]
        # configuration for environment and agents
        self.config = OmegaConf.to_container(self.args, resolve=True)
        self.env_config = self.config['environment']        
        self.net_config = self.config['model']['net']        
        self.policy_config = self.config['model']['PPOPolicy']
        if 'IMPolicy' in self.config['model']:            
            self.impolicy_config = self.config['model']['IMPolicy']
            self.im_policy = True
            if 'world_model' in self.config['model']['IMPolicy']:                
                self.model_based = True
            else:                
                self.model_based = False
        else:
            self.model_based = False
            self.im_policy = False

    def _create_seed_env(
            self,
            seeds: List[int],
            ) -> VectorEnv:
        """For environment to be reproducible across runs, need to seed them 
        during initialization."""
        vec_envs = []
        for seed in seeds:
            self.env_config['base_env_kwargs']['seed'] = seed
            env_config = deepcopy(self.env_config)
            #vec_envs.append(lambda: get_env(env_config))
            vec_envs.append(lambda config=env_config: get_env(config))
        return VectorEnv(vec_envs)

    def _setup_env(self) -> None:        
        # this is just a dummpy for setting up other things later
        self.env_name = self.env_config['base_env_kwargs']['env']       
        seed = self.args.exp_run.seed
        train_env_seeds = list(range(seed, seed + self.args.exp_run.train_env_num))        
        test_env_seeds = list(range(
            seed + self.args.exp_run.train_env_num, 
            seed + self.args.exp_run.train_env_num + self.args.exp_run.test_env_num
            ))
        # trainer will use this to collect data
        self.train_envs = self._create_seed_env(train_env_seeds)
        print("Setting up testing environments...")        
        self.test_envs = self._create_seed_env(test_env_seeds)
        # this will be used getting information about the environment
        env = get_env(self.env_config)
        self.agent_ids = env.possible_agents
        self.action_space = env._env.action_space
        # these will be used for evaluation and plotting
        self.env_agents = env.possible_agents
        self.agent_colors = env.get_agent_colors()        
    
    def _setup_single_agent(self, agent_id) -> PPOPolicy:
        net_module = import_module(
            self.net_config['module_path']
            )
        feature_net_cls = getattr(
            net_module,
            self.net_config['class_name']
            )
        net = feature_net_cls(self.net_config)
        device = self.args.exp_run.device
        action_shape = self.action_space.n
        actor = Actor(net, action_shape, device=device)
        critic = Critic(net, device=device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(
            actor_critic.parameters(), 
            lr=self.args.agent.optim.ppo_lr
            )
        dist = torch.distributions.Categorical
        # setup policy using intrinsic motivation
        if self.im_policy:
            # base policy optimization
            ppo = PPOPolicy(
                    actor=actor,
                    critic=critic,
                    optim=optim,
                    dist_fn=dist,
                    **self.policy_config
                )            
            # world model based IM policy
            if self.model_based:
                # from pixel to feature space
                feature_net_module = import_module(
                    self.impolicy_config['world_model']['args']['feature_net']['module_path']
                    )
                feature_net_cls = getattr(
                    feature_net_module, 
                    self.impolicy_config['world_model']['args']['feature_net']['class_name']
                    )
                feature_net_config = self.impolicy_config['world_model']['args']['feature_net']['config']
                feature_net = feature_net_cls(feature_net_config).to(device)
                wm_module = import_module(
                    self.impolicy_config['world_model']['module_path']
                    )
                wm_cls = getattr(
                    wm_module,
                    self.impolicy_config['world_model']['class_name']
                )
                im_model = wm_cls(
                    feature_net=feature_net,
                    **self.impolicy_config['world_model']['kwargs'],
                    device=device,
                    )
                im_model = im_model
                im_wm_optim = torch.optim.Adam(
                    im_model.parameters(),
                    lr=self.args.agent.optim.wm_lr
                    )
                policy_module = import_module(
                    self.impolicy_config['module_path']
                    )
                policy_cls = getattr(
                    policy_module,
                    self.impolicy_config['class_name']
                    )
                policy = policy_cls(
                    policy=ppo,
                    model=im_model,
                    optim=im_wm_optim,
                    **self.impolicy_config['args']
                    ).cuda()
            else:                
                # module free IM policy
                policy_module = import_module(
                    self.impolicy_config['module_path']
                    )
                policy_cls = getattr(
                    policy_module,
                    self.impolicy_config['class_name']
                    )
                policy = policy_cls(
                    policy=ppo,
                    **self.impolicy_config['args']
                    ).cuda()
        else:
            # model free non-IM policy        
            if self.config['model']['name'] == 'mappo':
                policy = MAPPOPolicy(
                    actor=actor,
                    critic=critic,
                    optim=optim,
                    dist_fn=dist,
                    **self.policy_config
                ).cuda()
            else:
                # individual PPO             
                policy = PPOPolicy(
                    actor=actor,
                    critic=critic,
                    optim=optim,
                    dist_fn=dist,
                    **self.policy_config
                ).cuda()

        return policy

    def _setup_agents(self) -> None:
        all_agents = {}
        if self.config['model']['name'] == 'mappo':
            # in MAPPO, all agents share the same policy
            agent_policy = self._setup_single_agent(self.env_agents[0])
            for agent_id in self.env_agents:
                all_agents[agent_id] = agent_policy
            self.policy = MultiAgentPolicySharingParametersManager(
                list(all_agents.values()),
                self.env_agents,
                self.action_space
                )
        else:
            for agent_id in self.env_agents:
                all_agents[agent_id] = self._setup_single_agent(agent_id)
            self.policy = MultiAgentPolicyManager(
                list(all_agents.values()), 
                self.env_agents,
                self.action_space
                )
        
    def preprocess_fn(self, obs):
        """Preprocess observation in image format to tensor format
        used in collector
        :param obs: observation in image format (num_envs, )
        """
        transform = Compose([ToPILImage(), Grayscale(), ToTensor(),])        
        for i, env_ob in enumerate(obs):
            for agent_id, agent_ob in env_ob.items():
                ob = agent_ob['observation']['curr_obs']                
                if len(ob.shape) == 3:
                    processed_ob = transform(ob).unsqueeze(0)
                else:
                    processed_ob = torch.stack([transform(ob_i) for ob_i in ob])
                agent_ob['observation']['curr_obs'] = processed_ob.pin_memory()
                                        
                cent_obs = agent_ob['observation']['cent_obs']
                if len(ob.shape) == 3:
                    processed_cent_obs = transform(cent_obs).unsqueeze(0)
                else:
                    processed_cent_obs = torch.stack([transform(cent_obs) for cent_obs in cent_obs])
                agent_ob['observation']['cent_obs'] = processed_cent_obs.pin_memory()
        return obs

    def _setup_collectors(self) -> None:
        train_buffer = VectorReplayBuffer(
            self.args.exp_run.buffer_size * self.args.exp_run.train_env_num,
            buffer_num=self.args.exp_run.train_env_num,            
        )        
        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            train_buffer,
            preprocess_fn=self.preprocess_fn,
            exploration_noise=True,
            )
        self.test_collector = Collector(
            self.policy, 
            self.test_envs, 
            preprocess_fn=self.preprocess_fn,
            exploration_noise=True,
            )

    """ Defining call back functions for onpolicy_trainer """
    def save_best_fn(self, policy):
        for agent_id in self.env_agents:
            model_save_path = os.path.join(
                self.output_dir, f"best_policy-{agent_id}.pth")
            torch.save(policy.policies[agent_id].state_dict(), model_save_path)

    def stop_fn(self, mean_rewards):
        return mean_rewards >= self.args.trainer.stop_fn.reward_threshold

    def reward_metric(self, rews):
        # rews: (n_ep, n_agent)
        return rews
    
    def train(self) -> None:
        args = self.args
        # logger setup
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        wandb_run_name = f"{self.log_name}-{cur_time}"        
        logger = WandbLogger(
            update_interval=args.wandb.update_interval,
            save_interval=args.wandb.save_interval,
            project=args.wandb.project_name,
            name=wandb_run_name,
            config=self.config,
            )
        writer = SummaryWriter(self.output_dir)
        writer.add_text("args", str(args))
        logger.load(writer)
        # run trainer
        train_result = onpolicy_trainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=args.trainer.max_epoch,
            step_per_epoch=args.trainer.step_per_epoch,
            repeat_per_collect=args.trainer.repeat_per_collect,
            episode_per_test=args.trainer.test_eps,   # number of episodes tested during training
            batch_size=args.trainer.batch_size,            
            step_per_collect=args.trainer.step_per_collect,
            episode_per_collect=args.trainer.episode_per_collect,
            save_best_fn=self.save_best_fn,
            reward_metric=self.reward_metric,    # used in Collector                        
            stop_fn=self.stop_fn,
            logger=logger
            )
        # save training result
        train_result_save_path = os.path.join(self.output_dir, 'train_result.pkl')
        with open(train_result_save_path, 'wb') as f:
            pickle.dump(train_result, f)        
        print(f"\n========== Test Result during training==========\n{train_result}")

    def eval(
            self,
            ckpt_dir: Optional[str],
            ) -> None:
        args = self.args
        assert args.exp_run.result_dir is not None, \
            "Please specify result_dir in config file or command line"        
        if args.exp_run.eval_only:                     
            ckpts = glob.glob(f'{ckpt_dir}/*.pth')
            agent_ckpts = {ckpt.split('-')[-1].split('.')[0]: ckpt for ckpt in ckpts}            
            for agent_id, agent_policy in self.policy.policies.items():
                agent_policy.load_state_dict(torch.load(agent_ckpts[agent_id]))
        # run testing
        self.policy.eval()        
        eval_result = self.test_collector.collect(
            n_episode=args.trainer.test_eps,
            render_mode='rgb_array'
            )
        step_agent_reward = np.array(eval_result.pop('step_agent_rews'))
        # save agent behavior details for cleanup only    
        if self.env_name == 'cleanup':
            output_dir = os.path.join(
                args.exp_run.result_dir, 'agent_info', self.env_name, args.model.name
                )
            ensure_dir(output_dir)
            agent_infos = eval_result.pop('agent_info')
            for idx, agent_info in enumerate(agent_infos):                
                filename = os.path.join(
                    output_dir, 
                    f"{args.model.name}-episode_{idx + 1}.csv"
                    )                
                df = pd.DataFrame.from_dict(agent_info[0], orient='index')
                df.to_csv(filename)

        frames = eval_result.pop('frames')
        self.save_results(step_agent_reward, frames, ckpt_dir)
        print(f"\n========== Eval after training ==========\n{eval_result}")

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
            episode_frames: Optional[List[List[np.ndarray]]] = None,
            ckpt_dir: Optional[str] = None
            ) -> None:        
        args = self.args
        ensure_dir(args.exp_run.result_dir)
        # load train config, and create result file path
        train_config = OmegaConf.load(
            os.path.join(ckpt_dir, '.hydra', 'config.yaml')
            )
        task_name = train_config.environment.base_env_kwargs.env
        num_agents = train_config.environment.base_env_kwargs.num_agents
        result_path = os.path.join(
            args.exp_run.result_dir, f"{task_name}_{num_agents}agents.pkl"
            )
        # get model name and hyperparam
        model_name = args.model.name    
        print(model_name)
        data = self._convert_save_data(data)    
        # save single model result
        single_model_result_path = os.path.join(
            args.exp_run.result_dir, f"{task_name}_{num_agents}_{model_name}.pkl")
        with open(single_model_result_path, 'wb') as f:
            pickle.dump(data, f)
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
        # save frames for video generation        
        # for i, run_frames in enumerate(episode_frames):            
        #     video_folder = os.path.join(
        #         args.exp_run.result_dir, 
        #         "frames", 
        #         f"{model_name}_ep{i}"
        #         )
        #     ensure_dir(video_folder)
        #     self.save_behavior_vis(run_frames, data[i], video_folder)

    # Combine frames and reward curve horizontally
    def save_behavior_vis(
            self, 
            frames: List[np.ndarray], 
            rewards: Dict[str, List],
            video_folder: str
            ) -> None:
        """Save frames to video
        Args:
            frames: list of frames
            rewards: dict of rewards
            video_folder: folder to save video
        """        
        height = 500
        width = 300
        max_steps = len(frames)
        for i, frame in enumerate(frames):
            reward_img = self.render_reward_curve(rewards, i, max_steps)
            reward_img = cv2.resize(reward_img, (width, height), interpolation=cv2.INTER_AREA)
            reward_img.astype('uint8')
            frame = frame.astype('uint8')
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            combined_frame = cv2.hconcat([frame, reward_img])
            frame_path = os.path.join(video_folder, f"frame_{i}.png")
            cv2.imwrite(frame_path, combined_frame)

    def render_reward_curve(self, rewards_dict, cur_steps, max_steps):        
        """Render reward curve as an image"""
        fig, ax = plt.subplots()
        for agent_id, rewards in rewards_dict.items():
            ax.plot(
                np.cumsum(rewards[:cur_steps]), 
                label=f"Agent {agent_id}", 
                color=self.agent_colors[agent_id],
                alpha=0.5)

        ax.set_title("Cumulative Reward for Each Agent")
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
        if not self.args.exp_run.eval_only:
            self.train()
        else:                        
            assert self.args.exp_run.ckpt_dir is not None, \
                "ckpt_dir must be provided for eval_only"   
            self.eval(self.args.exp_run.ckpt_dir)
