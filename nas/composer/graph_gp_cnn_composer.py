import random
from dataclasses import dataclass
from functools import partial
from typing import (
    Tuple,
    List
)

import numpy as np
import pandas as pd

from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiser
from nas.layer import LayerTypesIdsEnum, activation_types
from nas.nas_node import NNNodeGenerator

random.seed(1)
np.random.seed(1)


@dataclass
class GPNNComposerRequirements(GPComposerRequirements):
    conv_kernel_size: Tuple[int, int] = (3, 3)
    conv_strides: Tuple[int, int] = (1, 1)
    pool_size: Tuple[int, int] = (2, 2)
    pool_strides: Tuple[int, int] = (2, 2)
    min_num_of_neurons: int = 50
    max_num_of_neurons: int = 200
    min_filters: int = 64
    max_filters: int = 128
    channels_num: int = 3
    max_drop_size: int = 0.5
    image_size: List[int] = None
    conv_types: List[LayerTypesIdsEnum] = None
    cnn_secondary: List[LayerTypesIdsEnum] = None
    pool_types: List[LayerTypesIdsEnum] = None
    train_epochs_num: int = 5
    batch_size: int = 72
    num_of_classes: int = 10
    activation_types = activation_types
    max_num_of_conv_layers = 4
    min_num_of_conv_layers = 2

    def __post_init__(self):
        if not self.cnn_secondary:
            self.cnn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
        if not self.conv_types:
            self.conv_types = [LayerTypesIdsEnum.conv2d]
        if not self.pool_types:
            self.pool_types = [LayerTypesIdsEnum.maxpool2d, LayerTypesIdsEnum.averagepool2d]
        if not self.primary:
            self.primary = [LayerTypesIdsEnum.dense]
        if not self.secondary:
            self.secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
        if self.max_drop_size > 1:
            self.max_drop_size = 1
        if not all([side_size > 3 for side_size in self.image_size]):
            raise ValueError(f'Specified image size is unacceptable')
        self.conv_kernel_size, self.conv_strides = permissible_kernel_parameters_correct(self.image_size,
                                                                                         self.conv_kernel_size,
                                                                                         self.conv_strides, False)
        self.pool_size, self.pool_strides = permissible_kernel_parameters_correct(self.image_size,
                                                                                  self.pool_size,
                                                                                  self.pool_strides, True)
        if self.min_num_of_neurons < 1:
            raise ValueError(f'min_num_of_neurons value is unacceptable')
        if self.max_num_of_neurons < 1:
            raise ValueError(f'max_num_of_neurons value is unacceptable')
        if self.max_drop_size > 1:
            raise ValueError(f'max_drop_size value is unacceptable')
        if self.channels_num > 3 or self.channels_num < 1:
            raise ValueError(f'channels_num value must be anywhere from 1 to 3')
        if self.train_epochs_num < 1:
            raise ValueError(f'epochs number less than 1')
        if self.batch_size < 1:
            raise ValueError(f'batch size less than 1')
        if self.min_filters < 2:
            raise ValueError(f'min_filters value is unacceptable')
        if self.max_filters < 2:
            raise ValueError(f'max_filters value is unacceptable')

    @property
    def filters(self):
        filters = [self.min_filters]
        i = self.min_filters
        while i < self.max_filters:
            i = i * 2
            filters.append(i)
        return filters


from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.convert import graph_structure_as_nx_graph

from nas.graph_keras_eval import create_nn_model, keras_model_fit, keras_model_predict
from nas.graph_cnn_gp_operators import *


# from graph_icebergs_classification import CustomGraphModel


class CustomGraphModel(OptGraph):
    def __init__(self, nodes=None, cnn_nodes=None, fitted_model=None):
        super().__init__(nodes)
        self.cnn_nodes = cnn_nodes if not cnn_nodes is None else []
        self.model = fitted_model

    def __repr__(self):
        return f"{self.depth}:{self.length}:{len(self.cnn_nodes)}"

    def evaluate(self, data: pd.DataFrame):
        nodes = data.columns.to_list()
        _, labels = graph_structure_as_nx_graph(self)
        return len(nodes)

    def __eq__(self, other) -> bool:
        return self is other

    def add_cnn_node(self, new_node: OptNode):
        """
        Append new node to graph list
        """
        self.cnn_nodes.append(new_node)

    def update_cnn_node(self, old_node: OptNode, new_node: OptNode):
        index = self.cnn_nodes.index(old_node)
        self.cnn_nodes[index] = new_node

    def replace_cnn_nodes(self, new_nodes):
        self.cnn_nodes = new_nodes

    def fit(self, input_data: InputData, verbose=False, input_shape: tuple = None,
            min_filters: int = None, max_filters: int = None, classes: int = 3, batch_size=24, epochs=15):
        if not self.model:
            self.model = create_nn_model(self, input_shape, classes)
        train_predicted = keras_model_fit(self.model, input_data, verbose=True, batch_size=batch_size, epochs=epochs)
        # TODO mb need to add target in output
        return train_predicted

    def predict(self, input_data: InputData, output_mode: str = 'default'):
        evaluation_result = keras_model_predict(self.model, input_data)
        return evaluation_result


class GPNNGraphOptimiser(GPGraphOptimiser):
    def __init__(self, initial_graph, requirements, graph_generation_params,
                 metrics, parameters, log, archive_type=None):
        self.metrics = metrics

        super().__init__(initial_graph=initial_graph, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         metrics=metrics,
                         parameters=parameters,
                         log=log, archive_type=archive_type)

        self.parameters.graph_generation_function = random_cnn_graph
        self.graph_generation_function = partial(self.parameters.graph_generation_function,
                                                 graph_class=CustomGraphModel,
                                                 requirements=self.requirements,
                                                 primary_node_func=NNNodeGenerator.primary_node,
                                                 secondary_node_func=NNNodeGenerator.secondary_node)

        if initial_graph and type(initial_graph) != list:
            self.population = [initial_graph] * requirements.pop_size
        else:
            self.population = initial_graph or self._make_population(self.requirements.pop_size)
            # print(self.population)

    def _make_population(self, pop_size: int):
        return [self.graph_generation_function() for _ in range(pop_size)]

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, input_shape, min_filters, max_filters, classes, batch_size, epochs,
                         graph) -> float:

        graph.fit(train_data, True, input_shape, min_filters, max_filters, classes, batch_size, epochs)
        return metric_function(graph, test_data)

    def compose(self, data):
        train_data, test_data = train_test_data_setup(data, 0.8)
        # train_data, test_data = data, data
        composer_requirements = self.requirements
        input_shape = [size for size in composer_requirements.image_size]
        input_shape.append(composer_requirements.channels_num)
        input_shape = tuple(input_shape)

        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            self.metrics, train_data, test_data, input_shape,
                                            composer_requirements.min_filters, composer_requirements.max_filters,
                                            composer_requirements.num_of_classes, composer_requirements.batch_size,
                                            composer_requirements.train_epochs_num)
        self.optimise(metric_function_for_nodes)
        return self.best_individual.graph
