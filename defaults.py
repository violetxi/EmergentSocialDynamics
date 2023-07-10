"""Save some values that are the same for training, validation and evaluation.
Don't change these values. If you want to change them, do it through args..
"""
DEFAULT_ARGS = {    
    'seed': 0,
    'log_dir': './logs/',
    'batch_size': 2048,
    'num_episodes': int(1e4),
    'max_episode_len': int(1e3),
    'warm_up_steps': int(1e4),
    'val_freq': int(1e4),
    # wandb info
    'project_name': 'emergent-social-dynamics',
}