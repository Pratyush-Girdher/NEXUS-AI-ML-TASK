import numpy as np


class FullyConnected:

    def __init__(self,input_size=1352,output_size=10):

        self.weights=np.random.randn(
            input_size,
            output_size
        )*0.01

        self.bias=np.zeros(output_size)


    def forward(self,input):

        batch=input.shape[0]

        self.input=input.reshape(batch,-1)

        return np.dot(
            self.input,
            self.weights
        )+self.bias


    def backward(self,grad,lr=0.2):

        batch=self.input.shape[0]

        dW=np.dot(
            self.input.T,
            grad
        )/batch

        db=np.mean(grad,axis=0)

        self.weights-=lr*dW
        self.bias-=lr*db



class Softmax:

    def forward(self,input):

        exp=np.exp(
            input-
            np.max(input,axis=1,keepdims=True)
        )

        self.output=exp/np.sum(
            exp,
            axis=1,
            keepdims=True
        )

        return self.output


    def backward(self,labels):

        batch=labels.shape[0]

        grad=self.output.copy()

        grad[
            np.arange(batch),
            labels
        ]-=1

        return grad/batch