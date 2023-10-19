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
        """
        Initialize the Model class with a specified architecture and training parameters.
        Args
            :param model (torch.nn.Module or function): The model class which will be trained
            :param device (): Device for calculations ('cuda' / 'cpu')
            :param loss_func(): Loss Function used in this task
            :param optimizer(): Optimizer used during backpropagation
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_function = loss_func
        self.callbacks = None
        self.additional_model_params = kwargs

    def set_callbacks(self, callbacks_lst: List):
        """
        Set Callbacks that should run after each epoch

        Parameters
        ----------
        callbacks_lst : list
             A list containing callback objects like EarlyStoppingCallback etc..
        """
        self.callbacks = callbacks_lst

    def set_fit_params(self, optimizer, loss_func):
        """
        Set fitting params such as loss funciton & optimizers. In case if they were not defined at initialization time.
        """
        self.loss_function = loss_func
        self.optimizer = optimizer

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        """
        This methods compiles the model based on passed arguments/parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, *args, **kwargs):
        """
        This method fits the compiled model using provided data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        This method is responsible for prediction process.
        """
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
