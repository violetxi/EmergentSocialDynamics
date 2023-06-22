import os


def load_config_from_path(path: str) -> dict:
    """
    Load config from path, assuming config is a python file with class Config,  
    and all parameters are defined as class attributes
    """
    base_lib_path = 'social_rl.config.'
    config_name = os.path.basename(path).split('.')[0]
    config_path = base_lib_path + config_name
    config = __import__(config_path, fromlist=['*'])
    return config.Config()
    