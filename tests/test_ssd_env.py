import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from social_rl.envs.social_dilemma.env_creator import get_env_creator
from social_rl.envs.social_dilemma.pettingzoo_env import parallel_env


def test_env_render(env_name):    
    env_args = dict(
        env=env_name,
        num_agents=5,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0
    )
    env = parallel_env(env_args, render_mode="rgb_array")
    env.reset(seed=0)
    n_steps = 300
    # save frames
    frame_folder = os.path.join("tests", env_name)
    os.makedirs(frame_folder, exist_ok=True)
    dpi_val = 300
    for step in range(n_steps):
        actions = [
            env._env.action_space.sample() for agent_id in env.possible_agents
            ]      
        obs, rewards, done, truncations, info = env.step(actions)        
        # frame = env.render()
        # if step == 0:
        #     print(f"environment {env_name} frame shape: {frame.shape}")
        # h, w, _ = frame.shape
        # h = h * 30
        # w = w * 30
        # fig_size_width = w / dpi_val
        # fig_size_height = h / dpi_val
        # frame_resize = cv2.resize(frame.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)        
        # plt.figure(figsize=(fig_size_width, fig_size_height), dpi=dpi_val)
        # plt.imshow(frame_resize)
        # plt.axis('off')  # Hide the axis
        # filename = os.path.join(frame_folder, f'{step}.pdf')
        # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        # plt.close()


if __name__ == "__main__":
    test_env_render("harvest")
    test_env_render("cleanup")
