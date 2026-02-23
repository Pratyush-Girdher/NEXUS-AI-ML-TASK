import pickle

from dataset import load_mnist
from activation import ReLU
from pooling import MaxPool
from fc_layer import Softmax
from loss import accuracy


model=pickle.load(
open("models/trained_model.pkl","rb")
)

conv=model["conv"]
fc=model["fc"]

relu=ReLU()
pool=MaxPool()
softmax=Softmax()


X,y=load_mnist()

X=X[60000:60100]
y=y[60000:60100]


conv_out=conv.forward(X)

relu_out=relu.forward(conv_out)

pool_out=pool.forward(relu_out)

fc_out=fc.forward(pool_out)

probs=softmax.forward(fc_out)

print("Test Accuracy:",accuracy(probs,y))