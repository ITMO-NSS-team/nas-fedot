from typing import Type, Union, Callable, Optional, List, Tuple

from torch.nn import Module
from torch.optim import Optimizer, AdamW

from nas.graph.base_graph import NasGraph
from nas.model.model_interface import BaseModelInterface


class ModelConstructor:
    """
    Class that creates a new instance of Trainer session class with given train parameters.
    """

    def __init__(self,
                 trainer: Type[BaseModelInterface],
                 model_class,
                 loss_function: Union[Callable, Module],
                 optimizer: Type[Optimizer],
                 callbacks_lst: Optional[Union[List, Tuple]] = None,
                 device: str = 'cpu',
                 **additional_model_parameters):
        self.trainer = trainer
        self._model_class = model_class
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._callbacks = callbacks_lst
        self._device = device
        self._additional_model_parameters = additional_model_parameters

    def build(self, input_shape: Union[Tuple[int], List[int]], output_shape: int, graph: NasGraph):
        """
        Creates new Trainer class instance for train/evaluation session and initializes model from given graph object.

        :param input_shape (int) - The shape of an individual sample in HWC format.
        :param output_shape (int) - Number of output channels.
        :param graph (NASGraph object) -
        :return class with compiled keras or pytorch model
        """
        trainer = self.trainer(model=self._model_class, **self._additional_model_parameters)
        trainer.compile_model(graph, input_shape, output_shape)
        trainer.set_computation_device(self._device)
        trainer.set_loss(self._loss_function)
        trainer.set_optimizer(self._optimizer)
        trainer.set_callbacks(self._callbacks)
        return trainer
