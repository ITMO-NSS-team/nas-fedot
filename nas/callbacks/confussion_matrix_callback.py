import io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from fedot.core.data.data import InputData
from nas.metrics.confusion_matrix import plot_confusion_matrix


def plot2image(figure):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(figure)
    buffer.seek(0)

    img = tf.image.decode_png(buffer.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)

    return img


class ConfusionMatrixPlotter(tf.keras.callbacks.Callback):
    def __init__(self, data: InputData, normalize: bool = False, color_map=plt.cm.YlGn, title=None, save_dir=None):
        self.val = data.features
        self.target = data.target
        self.title = title
        self.normalize = normalize
        self.color_map = color_map
        self.data = data
        self.figure = plt.figure(figsize=(8, 8))
        self.save_dir = save_dir
        plt.title(self.title)

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.val)
        predicted = np.argmax(predicted, axis=1)
        conf_matrix = confusion_matrix(self.target, predicted)
        conf_matrix = plot_confusion_matrix(conf_matrix, self.data.supplementary_data, False, cmap=self.color_map)
        conf_matrix = plot2image(conf_matrix)

        file_writer = tf.summary.create_file_writer(self.save_dir + '/confusion_matrix')
        with file_writer.as_default():
            tf.summary.image('Conf_Matrix', conf_matrix, step=epoch)
