"""Save some values that are the same for training, validation and evaluation.
"""
DEFAULT_ARGS = {    
    'seed': 0,
    'log_dir': './logs/',
    'batch_size': 512,
    'epochs': 1e5,
    'num_episodes': 1e4,
    'episode_length': 1e4,
    'warm_up_steps': 1e5,
    'val_freq': 100,
}

