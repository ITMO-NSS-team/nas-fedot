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
    def __init__(self, data: InputData, normalize: bool = False, cmap=plt.cm.Blues, title=None, dir = None):
        self.val = data.features
        self.target = data.target
        self.title = title
        self.normalize = normalize
        self.cmap = cmap
        self.classes_num = data.num_classes
        self.classes = np.unique(data.target)
        self.figure = plt.figure(figsize=(8, 8))
        self.dir = dir
        plt.title(self.title)

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.val)
        labels = np.argmax(predicted, axis=1)
        conf_matrix = confusion_matrix(self.target, labels)
        conf_matrix = plot_confusion_matrix(conf_matrix, self.classes, False)
        conf_matrix = plot2image(conf_matrix)

        file_writer = tf.summary.create_file_writer(self.dir + '/cm')
        with file_writer.as_default():
            tf.summary.image('Conf_Matrix', conf_matrix, step=epoch)

