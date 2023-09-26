from social_rl.envs.social_dilemma.cleanup import CleanupEnv
from social_rl.envs.social_dilemma.harvest import HarvestEnv


def get_env_creator(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    seed=0
):
    if env == "harvest":
        return HarvestEnv(
            num_agents=num_agents,
            return_agent_actions=True,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )

    elif env == "cleanup":    
        return CleanupEnv(
            num_agents=num_agents,
            return_agent_actions=True,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )

    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")

