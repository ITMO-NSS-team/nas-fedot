from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, List, Optional

from nas.graph.BaseGraph import NasGraph

if TYPE_CHECKING:
    pass


class BaseModelInterface(ABC):
    """
    Class Interface for model handling
    """

    def __init__(self, model, device, loss_func=None, optimizer=None, **kwargs):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_function = loss_func
        self.callbacks = None

        self.additional_model_params = kwargs

    def set_callbacks(self, callbacks_lst: List):
        self.callbacks = callbacks_lst

    def set_fit_params(self, optimizer, loss_func):
        self.loss_function = loss_func
        self.optimizer = optimizer

    @staticmethod
    # @abstractmethod
    def prepare_data(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def save(self, save_path: Union[str, os.PathLike, pathlib.Path]):
        raise NotImplementedError


class NeuralSearchModel(BaseModelInterface):
    """
    Class to handle all required logic to evaluate graph as neural network regardless of  framework.
    """

    def __init__(self, model, device: str = 'cpu', **kwargs):
        super().__init__(model, device=device, **kwargs)

    def compile_model(self, graph: NasGraph, input_shape, output_shape, **kwargs):
        """
        This method builds model from given graph.
        """
        self.model = self.model()  # TODO: implement defined argument passing instead of kwargs
        self.model.init_model(graph=graph, input_shape=input_shape, out_shape=output_shape, **kwargs)
        self.set_computation_device()

    def set_computation_device(self):
        self.model.set_device(self.device)

    def fit_model(self, train_data, val_data: Optional = None, epochs: int = 1, **kwargs):
        self.model.fit(train_data,
                       val_data=val_data,
                       optmizer=self.optimizer,
                       loss=self.loss_function,
                       epochs=epochs,
                       device=self.device,
                       **kwargs)

    def predict(self, test_data, **kwargs):
        return self.model.predict(test_data, self.device)
# TODO docstrings
# class ModelTF(BaseModelInterface):
#     def __init__(self, model_class: Type[tf.keras.Model], data_transformer, **additional_model_params):
#         super().__init__(model_class, data_transformer, **additional_model_params)
#
#     @staticmethod
#     def prepare_data(data_transformer, data: InputData, mode: str, batch_size: int):
#         data_generator = data_transformer.build(data, mode=mode, batch_size=batch_size)
#         return data_generator
#
#     def compile_model(self, graph: NasGraph, output_shape: int = 1, eagerly_flag: bool = None,
#                       lr: Optional[float] = None, optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
#                       metrics: Optional[List[tf.keras.metrics.Metric]] = None,
#                       loss: Optional[str, tf.keras.losses.Loss] = None):
#         learning_rate = lr if lr else self.additional_model_params.get('lr', 1e-3)
#         metrics = metrics if metrics else self.additional_model_params.get('metrics', [tf.keras.metrics.Accuracy()])
#         loss = loss if loss else self.additional_model_params.get('loss')
#         optimizer = optimizer if optimizer \
#             else self.additional_model_params.get('optimizer', tf.keras.optimizers.Adam)(learning_rate)
#         if not loss:
#             loss = tf.keras.losses.BinaryCrossentropy() if output_shape == 1 \
#                 else tf.keras.losses.CategoricalCrossentropy()
#
#         self.model = self._model_class(graph, output_shape)
#         self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=eagerly_flag)
#         return self
#
#     def fit(self, train_data: InputData, val_data: InputData, epochs, batch_size, callbacks: List = None, verbose=None):
#         # TODO add verbose
#         train_dataset = self.prepare_data(self.data_transformer, train_data, 'train', batch_size)
#         val_dataset = self.prepare_data(self.data_transformer, val_data, 'val', batch_size)
#         self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size,
#                        callbacks=callbacks)
#         return self
#
#     def predict(self, test_data: InputData, batch_size: int = 1, callbacks: List = None, verbose=None, **kwargs):
#         test_dataset = self.prepare_data(self.data_transformer, test_data, 'test', batch_size)
#         predictions = self.model.predict(test_dataset, batch_size=batch_size, callbacks=callbacks)
#         return predictions
#
#     def save(self, save_path: Union[str, os.PathLike, pathlib.Path]):
#         pass
