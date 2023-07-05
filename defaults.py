"""Save some values that are the same for training, validation and evaluation.
"""
DEFAULT_ARGS = {    
    'seed': 0,
    'log_dir': './logs/',
    'batch_size': 8,
    'epochs': int(1e5),
    'num_episodes': int(1e4),
    'episode_length': int(1e4),
    'warm_up_steps': int(1e1),
    'val_freq': 100,
}

