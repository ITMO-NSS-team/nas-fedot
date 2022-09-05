from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from nas.operations.evaluation.callbacks.utils import plot2image
from nas.operations.evaluation.metrics import plot_confusion_matrix


class ConfusionMatrixPlotter(tf.keras.callbacks.Callback):
    def __init__(self, data_generator, normalize: bool = False, color_map=plt.cm.YlGn, title=None, save_dir=None,
                 true_labels: List = None):
        self.data_generator = data_generator
        self.title = title
        self.color_map = color_map
        self.normalize = normalize
        self.save_dir = save_dir
        self._true_labels = true_labels

    @property
    def figure(self):
        figure = plt.figure(figsize=(8, 8))
        plt.title(self.title)
        return figure

    @property
    def true_labels(self):
        labels = np.arange(1, len(np.unique(self.data_generator.data_generator.targets)))
        labels = self._true_labels if self._true_labels else labels
        return labels

    # def __init__(self, dataset: InputData, normalize: bool = False, color_map=plt.cm.YlGn, title=None, save_dir=None):
    #     self.val = dataset.x
    #     self.target = dataset.target
    #     self.title = title
    #     self.normalize = normalize
    #     self.color_map = color_map
    #     self.dataset = dataset
    #     self.figure = plt.figure(figsize=(8, 8))
    #     self.save_dir = save_dir
    #     plt.title(self.title)

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.data_generator)
        predicted = np.argmax(predicted, axis=1)
        targets = np.argmax(self.data_generator.data_generator.targets, axis=1)
        conf_matrix = confusion_matrix(targets, predicted)
        true_labels = self.true_labels
        conf_matrix = plot_confusion_matrix(conf_matrix, true_labels, False,
                                            cmap=self.color_map)
        conf_matrix = plot2image(conf_matrix)

        file_writer = tf.summary.create_file_writer(self.save_dir + '/confusion_matrix')
        with file_writer.as_default():
            tf.summary.image('Conf_Matrix', conf_matrix, step=epoch)
