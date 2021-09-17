from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import torch
from torch import optim
import os
from pytorch_mlp import MLP
import sklearn
from sklearn.datasets import make_moons
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 10
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    acc=(predictions.argmax(1) == targets).type(torch.float).sum().item()/len(targets)
    return acc

def train(train_set,model,learning_rate):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    optimizer= optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    loss=torch.nn.CrossEntropyLoss()
    print_training=True
    model.train()
    # YOUR TRAINING CODE GOES HERE
    avgacc=0
    avgloss=0
    length=len(train_set["data"])
    for i in range(length):
        sample=train_set["data"][i]
        label=torch.from_numpy(train_set["label"][i]).long()
        out=model(torch.from_numpy(sample.astype(np.float32)))
        #print("out size",torch.Tensor.size(out))
        #print("label size",torch.Tensor.size(label))
        loss_val=loss(out,label)
        avgacc+=accuracy(out,label)
        loss_val.backward()
        optimizer.step()
        avgloss+=loss_val
    if print_training:
        print("finish epoch, average train acc ",avgacc/length," average train loss ",avgloss/length)
    return avgacc/length

def eval(test_set,model):
    print("start evaling...")
    model.eval()
    acc=0
    length=len(test_set["data"])
    for i in range(length):
        result=model(torch.from_numpy(test_set["data"][i].astype(np.float32)))
        acc+=accuracy(result,torch.from_numpy(test_set["label"][i]).long())
    print("test acc ",acc/length) 
    return acc/length


def main(args):
    """
    Main function
    """
    hidden_units=args.dnn_hidden_units.split(',')
    hidden_units = list(map(int,hidden_units))
    #add an output layer to this model
    hidden_units.append(2)
    epoches=args.max_steps
    eval_freq=args.eval_freq
    model=MLP(n_inputs=2,n_hidden=hidden_units,n_classes=2)
    train_set,test_set=make_dataset(10)
    train_acc_list=[]
    eval_acc_list=[]
    for i in range(epoches):
        start = time.time()
        train_acc=train(train_set,model,args.learning_rate)
        train_acc_list.append(train_acc)
        end = time.time()
        print("epoch ",i ," using time:",end-start)
        if i%eval_freq==0:
            eval_acc=eval(test_set,model)
            eval_acc_list.append(eval_acc)

def make_dataset(batch_size):
    x,y=make_moons(n_samples=10000, shuffle=True, noise=None, random_state=None)
    train_set={"data":[],"label":[]}
    test_set={"data":[],"label":[]}
    count=0
    for i in range(int(len(x)/10)):
        data_list=[]
        label_list=[]
        for index in range(i*10,i*10+10):
            data_list.append(x[index])
            label_list.append(y[index])
        count+=1
        if count%10<8:
            train_set["data"].append(make_batch(data_list))
            train_set["label"].append(to_numpy(label_list))
        else:
            test_set["data"].append(make_batch(data_list))
            test_set["label"].append(to_numpy(label_list))
    return train_set,test_set

def make_batch(data_list):
    #data size: 2
    #reshape each data to 1,2
    #batch shape batch_size,2
    #data shape
    data_len=np.shape(data_list[0])[0]
    batch_data=np.zeros((len(data_list),data_len))
    for i in range(len(data_list)):
        batch_data[i]=data_list[i]
    return batch_data


def to_numpy(label_list):
    #shape batch_size
    batch_label=np.zeros(len(label_list))
    for index in range(len(label_list)):
        batch_label[index]=label_list[index]
    return batch_label

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args)