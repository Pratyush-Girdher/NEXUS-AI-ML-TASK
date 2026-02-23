from src.dataset import load_mnist
from src.conv_layer import ConvLayer
from src.activation import ReLU
from src.pooling import MaxPool
from src.fc_layer import FullyConnected, Softmax


X,y = load_mnist()

conv = ConvLayer()

relu = ReLU()

pool = MaxPool()

fc = FullyConnected()

softmax = Softmax()


batch = X[:5]


conv_out = conv.forward(batch)

relu_out = relu.forward(conv_out)

pool_out = pool.forward(relu_out)

fc_out = fc.forward(pool_out)

probabilities = softmax.forward(fc_out)


print("Final Output Shape:", probabilities.shape)