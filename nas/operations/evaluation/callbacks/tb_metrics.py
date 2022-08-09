from typing import List
from abc import abstractmethod

import psutil
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import tensorflow as tf
import sklearn.metrics as m

from nas.operations.evaluation.callbacks.utils import plot2image


class CustomCallbacksNas:
    @staticmethod
    @abstractmethod
    def apply(reference, predicted, **kwargs):
        return NotImplementedError()


class RAMProfiler(CustomCallbacksNas):
    @staticmethod
    def apply(reference, predicted, **kwargs):
        return psutil.virtual_memory().percent


class GPUProfiler(CustomCallbacksNas):
    @staticmethod
    def apply(reference, predicted, **kwargs):
        memory_usage = tf.config.experimental.get_memory_info('GPU:0')
        return memory_usage['current'] / (1024 ** 3)


class F1ScoreCallback(CustomCallbacksNas):
    @staticmethod
    def apply(reference, predicted, **kwargs):
        if len(np.unique(reference)) == 2:
            additional_params = {'average': 'weighted'}
        else:
            additional_params = {'average': 'micro'}
        return m.f1_score(y_true=reference, y_pred=predicted, **additional_params)


# TODO fix
class GraphPlotter(tf.keras.callbacks.Callback):
    def __init__(self, graph, log_path):
        self.graph = graph
        self.writer = tf.summary.create_file_writer(str(Path(log_path, 'graph')))

    def on_train_end(self, logs=None):
        graph = plot2image(self.graph.show())

        with self.writer.as_default():
            tf.summary.image('Graph', graph, step=0)

        plt.close()


class NASCallbackTF(tf.keras.callbacks.Callback):
    def __init__(self, data, callbacks_list: List, log_path, mode='on_epoch'):
        self.data = data
        self.callback_list = callbacks_list
        self.mode = mode
        self.writer = tf.summary.create_file_writer(str(Path(log_path, 'custom_callbacks')))

    @staticmethod
    def _apply_func(target, predicted, func):
        return func.apply(reference=target, predicted=predicted)

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.data.data_generator)
        predicted = np.argmax(predicted, axis=1)
        for func in self.callback_list:
            metric = self._apply_func(self.data.target, predicted, func=func)
            with self.writer.as_default():
                tf.summary.scalar(func.__name__, data=metric, step=epoch)
