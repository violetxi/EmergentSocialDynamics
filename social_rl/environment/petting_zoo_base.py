from torchrl.env.common import _EnvWrapper




class PettingZooBase(_EnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)