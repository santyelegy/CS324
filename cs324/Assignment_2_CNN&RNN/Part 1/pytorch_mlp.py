from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__() 
        self.layer=nn.Sequential(nn.Linear(n_inputs,n_hidden[0]))
        for i in range(1,len(n_hidden)):
            self.layer.add_module('relu',nn.ReLU(True))
            self.layer.add_module('linear',nn.Linear(n_hidden[i-1],n_hidden[i]))
        self.layer.add_module('softmax',nn.Softmax())

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out=self.layer(x)
        #out=out.argmax(1)
        return out
