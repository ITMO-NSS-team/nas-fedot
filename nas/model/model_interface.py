from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Type, List, TYPE_CHECKING, Union, Optional

import tensorflow as tf
from fedot.core.data.data import InputData

from nas.data.dataset.builder import BaseNNDataset

if TYPE_CHECKING:
    from nas.graph.cnn_graph import NasGraph


class BaseModelInterface(ABC):
    """
    Class Interface for model handling
    """

    def __init__(self, model_class, data_transformer: Type[BaseNNDataset], **additional_model_params):
        self.data_transformer = data_transformer
        self._model_class = model_class
        self.model = None
        self.additional_model_params = additional_model_params

    @staticmethod
    @abstractmethod
    def prepare_data(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: Union[str, os.PathLike, pathlib.Path]):
        raise NotImplementedError


# TODO docstrings
class ModelTF(BaseModelInterface):
    def __init__(self, model_class: Type[tf.keras.Model], data_transformer, **additional_model_params):
        super().__init__(model_class, data_transformer, **additional_model_params)

    @staticmethod
    def prepare_data(data_transformer, data: InputData, mode: str, batch_size: int):
        data_generator = data_transformer.build(data, mode=mode, batch_size=batch_size)
        return data_generator

    def compile_model(self, graph: NasGraph, output_shape: int = 1, eagerly_flag: bool = None,
                      lr: Optional[float] = None, optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                      metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                      loss: Optional[str, tf.keras.losses.Loss] = None):
        learning_rate = lr if lr else self.additional_model_params.get('lr', 1e-3)
        metrics = metrics if metrics else self.additional_model_params.get('metrics', [tf.keras.metrics.Accuracy()])
        loss = loss if loss else self.additional_model_params.get('loss')
        optimizer = optimizer if optimizer \
            else self.additional_model_params.get('optimizer', tf.keras.optimizers.Adam)(learning_rate)
        if not loss:
            loss = tf.keras.losses.BinaryCrossentropy() if output_shape == 1 \
                else tf.keras.losses.CategoricalCrossentropy()

        self.model = self._model_class(graph, output_shape)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=eagerly_flag)
        return self

    def fit(self, train_data: InputData, val_data: InputData, epochs, batch_size, callbacks: List = None, verbose=None):
        # TODO add verbose
        train_dataset = self.prepare_data(self.data_transformer, train_data, 'train', batch_size)
        val_dataset = self.prepare_data(self.data_transformer, val_data, 'val', batch_size)
        self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size,
                       callbacks=callbacks)
        return self

    def predict(self, test_data: InputData, batch_size: int = 1, callbacks: List = None, verbose=None, **kwargs):
        test_dataset = self.prepare_data(self.data_transformer, test_data, 'test', batch_size)
        predictions = self.model.predict(test_dataset, batch_size=batch_size, callbacks=callbacks)
        return predictions

    def save(self, save_path: Union[str, os.PathLike, pathlib.Path]):
        pass
