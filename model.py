import torch
import torch.nn as nn


class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed, config):
        
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        # TODO Build model generator per configuration.  
        self.config = config 
        self.fc1_units = self.config['fc1_units']
        self.fc2_units = self.config['fc2_units']


        # Create sequential container with two hidden layers 
        self.model = nn.Sequential(nn.Linear(state_size, self.fc1_units),
                                   nn.ReLU(),
                                   nn.Linear(self.fc1_units, self.fc2_units),
                                   nn.ReLU(),
                                   nn.Linear(self.fc2_units, action_size))

    def forward(self, state):
        
        return self.model.forward(state)

