import numpy as np
from modules import *
from right_models import *
from train_mlp_numpy import *
def softmax_test():
    arr=np.random.randn(2).reshape(1,2)
    label=np.array([1,0]).reshape(1,2)
    my=SoftMax_CrossEntropy()
    true=CrossEntropyLoss()
    print("-----SoftMax-CrossEntropy Layer-----")
    print("forward test:")
    print(my.forward(arr,label))
    print(true(arr,label,requires_acc=False))
    print("backward test")
    print((my.backward()==true.gradient()).all())

def relu_test():
    arr=np.random.randn(20).reshape(1,20)
    eta=np.random.randn(20).reshape(1,20)
    my=ReLU()
    true=True_Relu()
    print("-----ReLU Layer-----")
    print("forward test:")
    print((my.forward(arr)==true.forward(arr)).all())
    print("backward test:")
    print((my.backward(eta)==true.backward(eta)).all())



def linear_test():
    my=Linear(20,2)
    ture=True_Linear([20,2])
    ture.W=my.weight
    data=np.random.randn(20).reshape([1,20])
    print("-----Linear Layer-----")
    print("forward result test:")
    print((my.forward(data)==ture.forward(data)).all())

    print("backward result test:")
    eta=np.random.randn(2).reshape([1,2])
    print((my.backward(eta)==ture.backward(eta)).all())

    print("bias grand test:")
    print((my.grads["bias"]==ture.b_grad).all())

    print("weight grand test:")
    print((my.grads["weight"]==ture.W_grad).all())

    print("weight test: ")
    print((my.weight==ture.W).all())


def bp_test():
    my=Linear(20,2)
    data=np.random.randn(20).reshape([1,20])
    eta=np.random.randn(2).reshape([1,2])
    print("-----check if bp work-----")
    print("-----before bp------")
    print(my.weight)
    print(my.bias)
    my.forward(data)
    my.backward(eta)
    my.update(0.01)
    print("-----after bp-----")
    print(my.weight)
    print(my.bias)



def weight_init_test():
    shape=(20,2)
    weight_true = np.random.randn(*shape) * (2 / shape[0]**0.5)
    weight_false=np.random.normal(0,0.0001,shape)
    print("true size:")
    print(np.shape(weight_true))
    print(weight_true)
    print("false size:")
    print(np.shape(weight_false))
    print(weight_false)

def dataset_test():
    test,eval=make_dataset(10)
    print(test["data"][0])
    print(np.shape(test["data"][0]))
    print(test["label"][0])
    print(np.shape(test["label"][0]))
if __name__ == '__main__':
    #bp_test()
    #linear_test()
    #relu_test()
    #softmax_test()
    #weight_init_test()
    dataset_test()