import numpy as np


class MaxPool:

    def __init__(self, size=2):

        self.size = size


    def forward(self, input):

        self.input = input

        batch_size, H, W, C = input.shape

        S = self.size

        out_h = H // S
        out_w = W // S

        output = np.zeros(
            (batch_size,
             out_h,
             out_w,
             C)
        )

        for b in range(batch_size):

            for c in range(C):

                for i in range(out_h):

                    for j in range(out_w):

                        region = input[
                            b,
                            i*S:(i+1)*S,
                            j*S:(j+1)*S,
                            c
                        ]

                        output[b,i,j,c] = np.max(region)

        return output