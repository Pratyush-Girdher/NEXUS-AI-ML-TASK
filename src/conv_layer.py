import numpy as np

class ConvLayer:
    def __init__(self, num_filters=8, filter_size=3):
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size,
            1
        )*0.1
    def forward(self, input):
        self.input = input
        batch_size, H,W,C = input.shape
        F = self.filter_size

        out_h = H-F+1
        out_w = W-F +1

        output = np.zeros(
            (batch_size,
             out_h,
             out_w,
             self.num_filters)
        )
        for b in range(batch_size):

            image = input[b]

            for f in range(self.num_filters):

                kernel = self.filters[f]

                for i in range(out_h):

                    for j in range(out_w):

                        region = image[
                            i:i+F,
                            j:j+F,
                            :
                        ]

                        output[b,i,j,f] = np.sum(
                            region * kernel
                        )

        return output
