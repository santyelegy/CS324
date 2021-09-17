from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.W_hx=nn.Linear(input_dim,hidden_dim)
        self.W_hh=nn.Linear(hidden_dim,hidden_dim)
        self.W_ph=nn.Linear(hidden_dim,output_dim)
        self.seq_length=seq_length

    def forward(self, x):
        # Implementation here ...
        h=0
        for i in range(0,self.seq_length):
            h=F.tanh(self.W_hx(x)+self.W_hh(h))
        out=self.W_ph(h)
        out=F.softmax(out)
        return out
        
    # add more methods here if needed
