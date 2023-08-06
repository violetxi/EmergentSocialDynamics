import cv2
import numpy as np
import envs.env_creator as env_creator
from envs.pettingzoo_env import ssd_parallel_env


def test_env_render(env_name):    
    creator = env_creator.get_env_creator(
        env=env_name,
        num_agents=2,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0
    )
    raw_env = creator(None)
    env = ssd_parallel_env(raw_env, max_cycles=1000)
    env.reset()
    n_steps = 300
    frames = []
    for step in range(n_steps):
        actions = {}
        for agent_id in env.agents:            
            actions[agent_id] = env.action_spaces[agent_id].sample()
        obs, rewards, done, info = env.step(actions)
        frames.append(env.render(mode="rgb_array"))
        #print(f"Step: {step} | Obs: {obs} | Rewards: {rewards} | Done: {done} | Info: {info}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h, c = frames[0].shape
    out = cv2.VideoWriter(f'{env_name}.mp4', fourcc, 30.0, (w * 10, h * 10))
    for frame in frames:
        frame = cv2.resize(frame.astype(np.uint8), (w * 10, h * 10), interpolation=cv2.INTER_AREA)
        # We write every frame to the output video file. We first ensure the frame is in the correct format
        out.write(frame)

    # Release the VideoWriter
    out.release()



if __name__ == "__main__":
    test_env_render("harvest")
    test_env_render("cleanup")
    #test_env_ray("harvest")
    #test_env_ray("cleanup")
