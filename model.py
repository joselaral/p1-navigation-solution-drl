import torch
import torch.nn as nn


class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Create sequential container with two hidden layers 
        self.model = nn.Sequential(nn.Linear(state_size, fc1_units),
                                   nn.Linear(fc1_units, fc2_units),
                                   nn.Linear(fc2_units, action_size))

    def forward(self, state):
        
        return self.model.forward(state)

