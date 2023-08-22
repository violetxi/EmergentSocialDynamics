import torch
from torch import nn
from torchvision.transforms import (    
    Compose, 
    ToPILImage,
    Grayscale, 
    ToTensor    
)
from tianshou.data import to_torch


class CNN(nn.Module):
    def __init__(self, config):
        """ CNN tested to work with model free agents (PPO) """
        super(CNN, self).__init__()
        self.output_dim = config['output_dim']
        self.encoder = nn.Sequential(
            nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], stride=config['stride']),
            nn.ReLU(),            
            nn.Flatten(),
            nn.Linear(config['flatten_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim']),
            nn.ReLU(),
        )
        self.dist_mean = nn.Linear(config['output_dim'], config['output_dim'])
        self.dist_std = nn.Linear(config['output_dim'], config['output_dim'])
        self.preprocessing = Compose([
            ToPILImage(),
            Grayscale(),
            ToTensor(),
        ])

    def preprocess_fn(self, obs):
            """Preprocess observation in image format
            """
            transform = Compose([ToPILImage(), Grayscale(), ToTensor(),])            
            ob = obs.observation.curr_obs
            processed_ob = torch.stack([transform(ob_i.permute(2, 0, 1)) for ob_i in ob])
            return processed_ob

    def forward(self, obs, state=None, info={}):
        obs = to_torch(obs, dtype=torch.float32)
        processed_obs = self.preprocess_fn(obs)
        logits = self.encoder(processed_obs.to("cuda"))
        return logits, state


class CNNICM(CNN):
    def __init__(self, config):
        super().__init__(config)
    
    # override forward method such that there is no state returned
    def forward(self, obs, state=None, info={}):
        obs = to_torch(obs, dtype=torch.float32)
        processed_obs = self.preprocess_fn(obs)
        logits = self.encoder(processed_obs.to("cuda"))
        return logits