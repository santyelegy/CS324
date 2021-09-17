import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs=n_inputs
        self.max_epochs=max_epochs
        self.learning_rate=learning_rate
        self.w=np.random.rand(n_inputs).T
        self.b=0
        
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label=np.dot(input,self.w)+self.b
        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for i in range(0,len(training_inputs)):
            output=self.forward(training_inputs[i])
            if output*labels[i]>0:
                pass
            else:
                #do BP algorithm
                #print("differ:",self.learning_rate*output)
                self.b=self.b+self.learning_rate*labels[i]
                self.w=self.w+self.learning_rate*labels[i]*training_inputs[i]

from sample import make_samples,seperate

def eval(test_set,model):
    size=len(test_set)
    acc=0
    for a in test_set:
        i,l=a
        label=model.forward(i)
        if label*l>0:
            acc+=1
    print("accuracy: ",acc/size)

train_set,test_set=make_samples()
training_inputs,labels=seperate(train_set)
#print(training_inputs,labels)
model=Perceptron(2)
for i in range(0,5):
    model.train(training_inputs,labels)
    eval(test_set,model)
