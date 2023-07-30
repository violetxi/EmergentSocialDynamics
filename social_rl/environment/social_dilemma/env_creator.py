from social_rl.environment.social_dilemma.cleanup import CleanupEnv
from social_rl.environment.social_dilemma.harvest import HarvestEnv



def get_env_creator(
    env,
    num_agents,    
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0
):
    if env == "harvest":
        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )

    elif env == "cleanup":
        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )
    else:
        raise ValueError(f"env must be one of harvest, cleanup, not {env}")

    return env_creator
