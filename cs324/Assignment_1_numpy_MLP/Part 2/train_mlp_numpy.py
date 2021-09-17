from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import SoftMax_CrossEntropy
import sklearn
from sklearn.datasets import make_moons
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
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
    batch_size,len=np.shape(predictions)
    total_acc=0
    for i in range(batch_size):
        a=np.argmax(predictions[i])
        b=np.argmax(targets[i])
        if a==b:
            accuracy=1
        else:
            accuracy=0
        total_acc+=accuracy
    return total_acc/batch_size

def train(train_set,model,SGD=False):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    loss_cal=SoftMax_CrossEntropy()
    avgacc=0
    avgloss=0
    length=len(train_set["data"])
    for i in range(length):
        sample=train_set["data"][i]
        label=train_set["label"][i]
        out=model.forward(sample)
        avgacc+=accuracy(out,label)
        avgloss+=loss_cal.forward(out,label)
        model.backward(loss_cal.backward())
        model.update(LEARNING_RATE_DEFAULT)
    print("finish epoch, average train acc ",avgacc/length," average train loss ",avgloss/length)

def eval(test_set,model):
    acc=0
    length=len(test_set["data"])
    for i in range(length):
        result=model.forward(test_set["data"][i])
        acc+=accuracy(result,test_set["label"][i])
    print("test acc ",acc/length) 


def main():
    """
    Main function
    """
    model=MLP(n_inputs=2,n_hidden=[20,20,2],n_classes=2)
    train_set,test_set=make_dataset(10)
    for i in range(10):
        print("epoch ",i ," begin:")
        train(train_set,model)
        eval(test_set,model)

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
            train_set["label"].append(one_hot(label_list))
        else:
            test_set["data"].append(make_batch(data_list))
            test_set["label"].append(one_hot(label_list))
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


def one_hot(label_list):
    #shape batch_size,2
    batch_label=np.zeros((len(label_list),2))
    for index in range(len(label_list)):
        if label_list[index]>0:
            batch_label[index]=np.array([0,1])
        else:
            batch_label[index]=np.array([1,0])
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
    main()