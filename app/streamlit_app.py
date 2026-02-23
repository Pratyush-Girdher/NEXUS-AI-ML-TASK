import sys
import os

root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(root)

sys.path.append(
    os.path.join(root,"src")
)


import streamlit as st
import numpy as np
import pickle
from PIL import Image


from src.activation import ReLU
from src.pooling import MaxPool
from src.fc_layer import Softmax


# Load Model

model = pickle.load(

    open(
        "models/trained_model.pkl",
        "rb"
    )

)

conv = model["conv"]

fc = model["fc"]

relu = ReLU()

pool = MaxPool()

softmax = Softmax()


st.title("CNN Digit Classifier (From Scratch)")


st.write(
"Upload a handwritten digit image (0-9)"
)


uploaded = st.file_uploader(

    "Upload Image",

    type=["png","jpg","jpeg"]

)


if uploaded:

    img = Image.open(uploaded).convert("L")

    img = img.resize((28,28))


    arr = np.array(img)/255.0

    arr = arr.reshape(1,28,28,1)


    conv_out = conv.forward(arr)

    relu_out = relu.forward(conv_out)

    pool_out = pool.forward(relu_out)

    fc_out = fc.forward(pool_out)

    probs = softmax.forward(fc_out)


    prediction = np.argmax(probs)


    st.image(img,width=150)

    st.success(

        f"Predicted Digit : {prediction}"

    )