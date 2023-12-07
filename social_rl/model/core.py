import torch
from torch import nn


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
    

class ConvGRU(nn.Module):
    def __init__(self, config):
        """ CNN tested to work with model free agents (PPO) """
        super(ConvGRU, self).__init__()
        self.output_dim = config['output_dim']
        self.conv = nn.Sequential(
            nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], stride=config['stride']),
            nn.ReLU(),            
        )
        self.conv2gru = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['flatten_dim'], config['cnn_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['cnn_hidden_dim'], config['cnn_output_dim']),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=config['rnn_input_size'],
            hidden_size=config['rnn_hidden_size'],
            num_layers=config['rnn_num_layers'],
            batch_first=config['rnn_batch_first'],
        )
        # get logits from GRU output
        self.fc = nn.Linear(config['rnn_hidden_size'], config['output_dim'])
        self.dist_mean = nn.Linear(config['output_dim'], config['output_dim'])
        self.dist_std = nn.Linear(config['output_dim'], config['output_dim'])
        # to keep track of its hidden state
        self.state = None

    def encode(self, x):
        x = self.conv(x)        
        x = self.conv2gru(x)                        
        return x

    def forward(self, obs, state=None):
        # state is the initial hidden state
        obs = obs.observation.curr_obs.cuda(non_blocking=True)    
        # stacked inputs assuming images are grayscaled        
        bs, ts, c, h, w = obs.shape        
        processed_obs = []
        for i in range(ts):
            processed_obs.append(self.encode(obs[:, i, :, :, :]))        
        processed_obs = torch.stack(processed_obs).permute(1, 0, 2)        
        # during batch training, uses reset state every time
        if bs > 50:
            if state is None:
                state = torch.zeros((
                    self.gru.num_layers, bs, self.gru.hidden_size
                    )).cuda()
            out, state = self.gru(processed_obs, state)
        else:
            # during data collection, use state with past history
            if self.state is None:                
                self.state = torch.zeros((
                    self.gru.num_layers, bs, self.gru.hidden_size
                    )).cuda()
            out, self.state = self.gru(processed_obs, self.state)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits, self.state
    

class ConvGRUICM(ConvGRU):
    def __init__(self, config):
        super(ConvGRUICM, self).__init__(config)        

    # override forward method such that there is no state returned    
    def forward(self, obs, state=None, info={}):
        # state is the initial hidden state        
        obs = obs.observation.curr_obs.cuda(non_blocking=True)        
        # stacked inputs assuming images are grayscaled        
        bs, ts, c, h, w = obs.shape
        processed_obs = []
        for i in range(ts):                   
            processed_obs.append(self.encode(obs[:, i, :, :, :]))
        processed_obs = torch.stack(processed_obs).permute(1, 0, 2)
        # during batch training, uses reset state every time
        if bs > 50:
            if state is None:
                state = torch.zeros((
                    self.gru.num_layers, bs, self.gru.hidden_size
                    )).cuda()
            out, state = self.gru(processed_obs, state)
        else:
            # during data collection, use state with past history
            if self.state is None:                
                self.state = torch.zeros((
                    self.gru.num_layers, bs, self.gru.hidden_size
                    )).cuda()
            out, self.state = self.gru(processed_obs, self.state)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits



class ConvGRUMAPPO(nn.Module):
    def __init__(self, config):
        """ CNN tested to work with model free agents (PPO) """
        super(ConvGRUMAPPO, self).__init__()
        self.output_dim = config['output_dim']
        self.conv = nn.Sequential(
            nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], stride=config['stride']),
            nn.ReLU(),            
        )
        self.conv2gru = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['flatten_dim'], config['cnn_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['cnn_hidden_dim'], config['cnn_output_dim']),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=config['rnn_input_size'],
            hidden_size=config['rnn_hidden_size'],
            num_layers=config['rnn_num_layers'],
            batch_first=config['rnn_batch_first'],
        )
        # get logits from GRU output
        self.fc = nn.Linear(config['rnn_hidden_size'], config['output_dim'])
        self.dist_mean = nn.Linear(config['output_dim'], config['output_dim'])
        self.dist_std = nn.Linear(config['output_dim'], config['output_dim'])
        # to keep track of its hidden state
        self.state = None

    def encode(self, x):
        x = self.conv(x)        
        x = self.conv2gru(x)                        
        return x

    def forward(self, obs, state=None):        
        # state is the initial hidden state
        obs = obs.observation.curr_obs.cuda(non_blocking=True)             
        bs, n_agents, ts, c, h, w = obs.shape
        obs = obs.reshape(n_agents * bs, ts, c, h, w)
        processed_obs = []
        for i in range(ts):
            processed_obs.append(self.encode(obs[:, i, :, :, :]))        
        # (bs * n_agents, ts, rnn_input_size)
        processed_obs = torch.stack(processed_obs).permute(1, 0, 2)
        # (n_agents, bs, ts, rnn_input_size)
        processed_obs = processed_obs.reshape(bs, n_agents, ts, -1)             
        logits_out = []

        if bs > 50:
            # during batch training, uses reset state every time
            if state is None:
                state = [
                    torch.zeros((self.gru.num_layers, bs, self.gru.hidden_size)).cuda() for _ in range(n_agents)                    
                ]
            for i in range(n_agents):
                out, state[i] = self.gru(processed_obs[:, i, :, :], state[i])
                last_out = out[:, -1, :]
                logits = self.fc(last_out)
                logits_out.append(logits)            
        else:
            # during data collection, use state with past history            
            if self.state is None:           
                self.state = [
                    torch.zeros((self.gru.num_layers, bs, self.gru.hidden_size)).cuda() for _ in range(n_agents)
                    ]
            for i in range(n_agents):
                out, self.state[i] = self.gru(processed_obs[:, i, :, :], self.state[i])
                last_out = out[:, -1, :]
                logits = self.fc(last_out)
                logits_out.append(logits)        
        logits_out = torch.stack(logits_out).reshape(n_agents * bs, self.output_dim, 1)                              
        return logits_out, torch.stack(self.state) if self.state else state
    


class ConvGRUMAPPOCritic(nn.Module):
    def __init__(self, config):
        """ CNN tested to work with model free agents (PPO) """
        super(ConvGRUMAPPOCritic, self).__init__()
        self.output_dim = config['output_dim']
        self.conv = nn.Sequential(
            nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], stride=config['stride']),
            nn.ReLU(),            
        )
        self.conv2gru = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['flatten_dim'], config['cnn_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['cnn_hidden_dim'], config['cnn_output_dim']),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=config['rnn_input_size'],
            hidden_size=config['rnn_hidden_size'],
            num_layers=config['rnn_num_layers'],
            batch_first=config['rnn_batch_first'],
        )
        # get logits from GRU output
        self.fc = nn.Linear(config['rnn_hidden_size'], config['output_dim'])
        self.dist_mean = nn.Linear(config['output_dim'], config['output_dim'])
        self.dist_std = nn.Linear(config['output_dim'], config['output_dim'])
        # to keep track of its hidden state
        self.state = None

    def encode(self, x):
        x = self.conv(x)        
        x = self.conv2gru(x)                        
        return x

    def forward(self, obs, state=None):        
        # state is the initial hidden state
        obs = obs.observation.curr_obs.cuda(non_blocking=True)                 
        bs, ts, c, h, w = obs.shape

        processed_obs = []
        for i in range(ts):
            processed_obs.append(self.encode(obs[:, i, :, :, :]))        
        # (bs, ts, rnn_input_size)
        processed_obs = torch.stack(processed_obs).permute(1, 0, 2)        

        if bs > 50:
            # during batch training, uses reset state every time
            if state is None:
                state = torch.zeros((self.gru.num_layers, bs, self.gru.hidden_size)).cuda()            

            out, state = self.gru(processed_obs, state)
            last_out = out[:, -1, :]
            logits = self.fc(last_out)                
        else:
            # during data collection, use state with past history            
            if self.state is None:           
                self.state = torch.zeros((self.gru.num_layers, bs, self.gru.hidden_size)).cuda()
            
            out, self.state = self.gru(processed_obs, self.state)
            last_out = out[:, -1, :]
            logits = self.fc(last_out)                    
        return logits, self.state