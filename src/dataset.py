import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():

    print("Loading MNIST Dataset...")

    X,y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False
    )

    X = X.astype(np.float32)/255.0

    y = y.astype(int)

    X = X.reshape(-1,28,28,1)

    return X,y