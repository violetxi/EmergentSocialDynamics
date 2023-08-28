import torch
from torch import nn
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

    def forward(self, obs, state=None, info={}):        
        process_obs = obs.observation.curr_obs.cuda()        
        if len(process_obs.shape) == 5:
            # stacked inputs assuming images are grayscaled
            bs, ts, c, h, w = process_obs.shape
            process_obs = process_obs.reshape(bs, ts, h, w)
            logits = self.encoder(process_obs).reshape(bs, -1)
        else:            
            logits = self.encoder(process_obs)
        return logits, state


class CNNICM(CNN):
    def __init__(self, config):
        super().__init__(config)

    # override forward method such that there is no state returned
    def forward(self, obs, state=None, info={}):
        process_obs = obs.observation.curr_obs.cuda()        
        if len(process_obs.shape) == 5:
            # stacked inputs assuming images are grayscaled
            bs, ts, c, h, w = process_obs.shape
            process_obs = process_obs.reshape(bs, ts, h, w)
            logits = self.encoder(process_obs).reshape(bs, -1)
        else:            
            logits = self.encoder(process_obs)
        return logits
    
    