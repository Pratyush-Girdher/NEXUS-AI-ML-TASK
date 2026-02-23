import numpy as np


class CrossEntropyLoss:

    def forward(self, predictions, labels):

        batch_size = predictions.shape[0]

        # avoid log(0)
        epsilon = 1e-9

        correct_probs = predictions[
            np.arange(batch_size),
            labels
        ]

        loss = -np.log(
            correct_probs + epsilon
        )

        return np.mean(loss)

def accuracy(pred,labels):

    p=np.argmax(pred,axis=1)

    return np.mean(p==labels)