import pickle
import os

from dataset import load_mnist
from conv_layer import ConvLayer
from activation import ReLU
from pooling import MaxPool
from fc_layer import FullyConnected, Softmax
from loss import CrossEntropyLoss, accuracy


print("Loading MNIST Dataset...")

X, y = load_mnist()

print("Dataset Loaded")


# SMALL DATASET (FAST TRAINING)
X = X[:200]
y = y[:200]


# Layers
conv = ConvLayer()

relu = ReLU()

pool = MaxPool()

fc = FullyConnected()

softmax = Softmax()

loss_fn = CrossEntropyLoss()


epochs = 3
batch_size = 5


for epoch in range(epochs):

    print("\n====================")
    print("Epoch :", epoch + 1)

    total_loss = 0
    total_acc = 0

    batch_count = 0


    for i in range(0, len(X), batch_size):

        print("Processing Batch :", i)

        batch = X[i:i+batch_size]
        labels = y[i:i+batch_size]


        # Forward Pass

        conv_out = conv.forward(batch)

        relu_out = relu.forward(conv_out)

        pool_out = pool.forward(relu_out)

        fc_out = fc.forward(pool_out)

        probs = softmax.forward(fc_out)


        # Loss + Accuracy

        loss = loss_fn.forward(probs, labels)

        acc = accuracy(probs, labels)


        total_loss += loss

        total_acc += acc

        batch_count += 1


        # Backprop (FC Only)

        grad = softmax.backward(labels)

        fc.backward(
            grad,
            lr=0.2
        )


    print("\nEpoch Results")

    print(
        "Loss :",
        total_loss / batch_count
    )

    print(
        "Accuracy :",
        total_acc / batch_count
    )


# AUTO CREATE MODELS FOLDER
os.makedirs("models", exist_ok=True)


model = {

    "conv": conv,
    "fc": fc

}


pickle.dump(

    model,

    open(
        "models/trained_model.pkl",
        "wb"
    )

)

print("\nModel Saved Successfully ✅")