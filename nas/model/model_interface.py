from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Type, List

import tensorflow
from fedot.core.data.data import InputData

from nas.data.dataset.builder import BaseNNDataset
from nas.graph.cnn_graph import NasGraph


class BaseModelInterface(ABC):
    """
    Class Interface for model handling
    """

    def __init__(self, model_class, data_transformer: Type[BaseNNDataset]):
        self.data_transformer = data_transformer
        self._model_class = model_class
        self.model = None

    @staticmethod
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
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


class ModelTF(BaseModelInterface):
    def __init__(self, model_class: Type[tensorflow.keras.Model], data_transformer):
        super().__init__(model_class, data_transformer)

    @staticmethod
    def prepare_data(*args, **kwargs):
        pass

    def compile_model(self, graph: NasGraph, metrics: List, optimizer, loss: Union[str, tensorflow.keras.losses.Loss],
                      output_shape: int = 1, eagerly_flag: bool = True):
        self.model = self._model_class(graph)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=eagerly_flag)
        return self

    def fit(self, train_data: InputData, val_data: InputData, epochs, batch_size):
        train_generator = self.prepare_data(train_data)
        val_generator = self.prepare_data(val_data)
        pass

    def predict(self, *args, **kwargs):
        pass
