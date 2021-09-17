from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.layer_list=[Linear(n_inputs,n_hidden[0])]
        for i in range(1,len(n_hidden)):
            self.layer_list.append(ReLU())
            self.layer_list.append(Linear(n_hidden[i-1],n_hidden[i]))

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for a in self.layer_list:
            x=a.forward(x)
        return x

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for a in self.layer_list[::-1]:
            dout = a.backward(dout)
        return dout

    def update(self,learning_rate):
        for a in self.layer_list:
            a.update(learning_rate)
        
