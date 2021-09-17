from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from torch import nn

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.count=0
    self.stack = nn.Sequential()
    self.add_convLayer(3,1,1,n_channels,64)
    self.stack.add_module("pool2d-1",nn.MaxPool2d(3,stride=2,padding=1))
    self.add_convLayer(3,1,1,64,128)
    self.stack.add_module("pool2d-2",nn.MaxPool2d(3,2,1))
    self.add_convLayer(3,1,1,128,256)
    self.add_convLayer(3,1,1,256,256)
    self.stack.add_module("pool2d-3",nn.MaxPool2d(3,2,1))
    self.add_convLayer(3,1,1,256,512)
    self.add_convLayer(3,1,1,512,512)
    self.stack.add_module("pool2d-4",nn.MaxPool2d(3,2,1))
    self.add_convLayer(3,1,1,512,512)
    self.add_convLayer(3,1,1,512,512)
    self.stack.add_module("pool2d-5",nn.MaxPool2d(3,2,1))
    self.linear=nn.Linear(512,n_classes)


  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    x = self.stack(x)
    x = x.view(x.size(0), -1)
    #print(x.size())
    out=self.linear(x)
    return out

  def add_convLayer(self,k,s,p,in_channels,out_channels):
    self.stack.add_module("conv2d-{0}".format(self.count),nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=s,padding=p,kernel_size=k))
    self.stack.add_module("batch norm-{0}".format(self.count),nn.BatchNorm2d(out_channels))
    self.stack.add_module("relu-{0}".format(self.count),nn.ReLU())
    self.count+=1
