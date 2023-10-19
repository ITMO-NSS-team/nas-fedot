from typing import Type

from nas.graph.BaseGraph import NasGraph
from nas.model.model_interface import BaseModelInterface


class ModelConstructor:
    """
    Class that creates a new instance of a model from given parameters.
    """

    def __init__(self, trainer: Type[BaseModelInterface], model_class, **additional_model_parameters):
        self.trainer = trainer(model_class, **additional_model_parameters)

    def build(self, input_shape: int, output_shape: int, graph: NasGraph):
        """
        Builds the actual model with all its layers based on graph structure and compiles it for training/evaluation.

        :param input_shape (int) - The shape of an individual sample's input vector in this network
        :param output_shape (int) - The number of classes to predict per sample
        :param graph (NASGraph object) -
        :return class with compiled keras or pytorch model
        """
        self.trainer.compile_model(graph=graph, input_shape=input_shape, output_shape=output_shape)
        return self.trainer
