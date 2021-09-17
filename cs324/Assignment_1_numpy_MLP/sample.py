import numpy as np
import sklearn
from sklearn.datasets import make_circles  
from sklearn.datasets import make_moons  
import matplotlib.pyplot as plt    

loc1=1
loc2=10
scale1=1
scale2=1

## return two list with tuple (x,l) x is variable and l is the label 
# (for label, 1 means from sample 1 and -1 means form sample 2)
def make_samples():
    train_set1=np.random.normal(loc1,scale1,(80,2))
    #print("this is set 1: ",train_set1)
    train_set2=np.random.normal(loc2,scale2,(80,2))
    test_set1=np.random.normal(loc1,scale1,(20,2))
    test_set2=np.random.normal(loc2,scale2,(20,2))
    train_mix=[]
    test_mix=[]
    for i in train_set1:
        train_mix.append((i,1))
    for i in test_set1:
        test_mix.append((i,1))
    for i in train_set2:
        train_mix.append((i,-1))
    for i in test_set2:
        test_mix.append((i,-1))
    return train_mix, test_mix

def make_moon_samples():  
    fig=plt.figure(1)  
    plt.subplot(122)  
    x1,y1=make_moons(n_samples=1000,noise=0.05)  
    plt.title('make_moons function example')  
    plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1)  
    plt.show() 

def seperate(dataset):
    inputs=np.zeros([len(dataset),2])
    labels=np.zeros(len(dataset))
    for i in range(len(dataset)):
        x,y=dataset[i]
        inputs[i]=x
        labels[i]=y
    return inputs,labels



