import numpy as np


def _keras_model_prob2labels(predictions: np.array, is_multiclass: bool = False) -> np.array:
    if is_multiclass:
        output = []
        for prediction in predictions:
            values = []
            for val in prediction:
                values.append(round(val))
            output.append(np.float64(values))
    else:
        output = np.zeros(predictions.shape)
        for i, prediction in enumerate(predictions):
            class_prediction = np.argmax(prediction)
            output[i][class_prediction] = 1
    return output
