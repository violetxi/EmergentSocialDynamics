"""Uses Stable-Baselines3 to train agents in the Knights-Archers-Zombies environment using SuperSuit vector envs.

This environment requires using SuperSuit's Black Death wrapper, to handle agent death.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

import envs.env_creator as env_creator
from envs.harvest import HarvestEnv
from envs.cleanup import CleanupEnv
from envs.pettingzoo_env import ssd_parallel_env



def train(env, steps: int = 10_000, seed: int | None = 0):
    env.reset()
    print(f"Starting training on {env.ssd_env.__class__.__name__}.")
    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env, num_games: int = 100, render_mode: str | None = None):    
    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env_name = "harvest"
    creator = env_creator.get_env_creator(
        env=env_name,
        num_agents=1,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0
    )
    raw_env = creator(None)
    env = ssd_parallel_env(raw_env, max_cycles=1000)
    train(env, steps=10000, seed=0)
    eval(env, num_games=10, render_mode=None)
    # raw_env = creator(None)
    # env_fn = knights_archers_zombies_v10

    # # Set vector_state to false in order to use visual observations (significantly longer training time)
    # env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # # Train a model (takes ~5 minutes on a laptop CPU)
    # train(env_fn, steps=81_920, seed=0, **env_kwargs)

    # # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    # eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # # Watch 2 games (takes ~10 seconds on a laptop CPU)
    # eval(env_fn, num_games=2, render_mode=None, **env_kwargs)
