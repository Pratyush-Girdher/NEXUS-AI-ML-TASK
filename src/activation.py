import numpy as np

class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0,input)