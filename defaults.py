"""Save some values that are the same for training, validation and evaluation.
"""
DEFAULT_ARGS = {    
    'seed': 0,
    'log_dir': './logs/',
    'batch_size': 8,
    'num_episodes': int(1e4),
    'episode_length': int(1e3),
    'warm_up_steps': int(1e4),
    'val_freq': 100,
    # wandb info
    'project_name': 'emergent-social-dynamics',
}

