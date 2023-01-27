import numpy as np


def probs2labels(predictions, is_multiclass):
    if is_multiclass:
        return np.argmax(predictions, axis=-1)
    else:
        return np.where(predictions > .5, 1, 0)
