from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING, Union, Optional

import numpy as np
import torch.nn
import tqdm
from torch.utils.data import Dataset

from nas.data.dataset.builder import BaseNNDataset
from nas.graph.BaseGraph import NasGraph

if TYPE_CHECKING:
    pass


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


class TorchModel(BaseModelInterface):
    def __init__(self, model_class: torch.nn.Module, graph: NasGraph, input_shape: int, out_shape: int):
        super().__init__(model_class)
        self._device = None
        self._writer = None
        self._model = model_class().init_model(input_shape, out_shape, graph)

    def __call__(self, data: Optional[torch.Tensor, np.ndarray], **kwargs):
        data = torch.Tensor(data)
        self.model.eval()
        data = data.to(self._device)
        with torch.no_grad():
            return self.model(data)

    def fit(self, train_data: Dataset, batch_size: int, train_parameters, opt_epochs: int = None,
            val_data: Optional[Dataset] = None):
        epochs = opt_epochs if opt_epochs is not None else train_parameters.epochs
        callbacks = train_parameters.callbacks
        scheduler = train_parameters.scheduler
        optimizer = train_parameters.optimizer
        metrics = train_parameters.metrics
        loss = train_parameters.loss_func

        train_loop = tqdm.trange(epochs, position=0)
        for epoch in train_loop:
            train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}')
            train_logs = self._one_epoch_train(train_data, loss, optimizer, scheduler, metrics)
            val_logs = {} if val_data is None else self._one_epochs_val(val_data, loss, metrics)
            train_loop.set_postfix(learning_rate=optimizer.param_groups[0]['lr'],
                                   **train_logs, **val_logs)

    def predict(self, *args, **kwargs):
        pass

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
