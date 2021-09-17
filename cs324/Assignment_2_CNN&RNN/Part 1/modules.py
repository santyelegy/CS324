import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        #input dim n,output dim m  n=d^(l-1) m=d^l
        #weight n*m
        #bias batch*m
        shape=(in_features,out_features)
        self.weight = np.random.randn(*shape) * (2 / shape[0]**0.5)
        self.bias=np.zeros((1,out_features))
        self.input=0
        self.grads={}
        
    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        #x: batch*n
        #print(np.shape(x),np.shape(self.weight),np.shape(self.bias))
        out=np.dot(x,self.weight)+self.bias
        self.input=x
        
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        dx=np.dot(dout,self.weight.T)
        #update weight and bias
        delta=np.dot(self.input.T,dout)
        self.grads['weight']=delta
        self.grads['bias']=dout
        return dx

    def update(self,learning_rate):
        self.weight=self.weight-learning_rate*self.grads['weight']
        self.bias=self.bias-learning_rate*self.grads['bias']


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        #x: 1*m
        self.input=x
        out= 1 * (x > 0) * x
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx=(self.input>0)*dout
        return dx

    def update(self,learning_rate):
        return

class SoftMax_CrossEntropy(object):
    def exp_normalize(self,x):
        b = x.max(axis=-1,keepdims=True)
        y = np.exp(x - b)
        return y / y.sum(axis=-1,keepdims=True)
    def forward(self,x,y):
        softmax=self.exp_normalize(x)
        self.softmax=softmax
        self.input=y
        loss=np.einsum('ij,ij->', y, np.log(softmax), optimize=True) / y.shape[0]
        return -loss

    def backward(self):
        return self.softmax-self.input
