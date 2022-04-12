import random
from dataclasses import dataclass
from functools import partial
from copy import deepcopy
from typing import (
    Tuple,
    List,
    Any,
    Optional
)
from uuid import uuid4

import numpy as np
import pandas as pd

from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser
from nas.layer import LayerTypesIdsEnum, activation_types, LayerParams
from nas.graph_cnn_gp_operators import random_cnn_graph

from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.convert import graph_structure_as_nx_graph

from nas.graph_keras_eval import create_nn_model, keras_model_fit, keras_model_predict
from nas.graph_cnn_gp_operators import *

random.seed(1)
np.random.seed(1)


@dataclass
class GPNNComposerRequirements(PipelineComposerRequirements):
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
            self.cnn_secondary = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value]
        if not self.conv_types:
            self.conv_types = [LayerTypesIdsEnum.conv2d.value]
        if not self.pool_types:
            self.pool_types = [LayerTypesIdsEnum.maxpool2d.value, LayerTypesIdsEnum.averagepool2d.value]
        if not self.primary:
            self.primary = [LayerTypesIdsEnum.dense.value]
        if not self.secondary:
            self.secondary = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value]
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


class CustomGraphAdapter(DirectAdapter):
    def __init__(self, base_graph_class=None, base_node_class=None, log=None):
        super().__init__(base_graph_class=base_graph_class, base_node_class=base_node_class, log=log)
        self.base_graph_params = {}

    def adapt(self, adaptee: Any) -> OptGraph:
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph
        for node in opt_graph.nodes:
            self.base_graph_params[node.distance_to_primary_level] = node.content['params']
            node.__class__ = OptNode
        return opt_graph

    def restore(self, opt_graph: OptGraph):
        obj = deepcopy(opt_graph)
        obj.__class__ = self.base_graph_class
        for node in obj.nodes:
            node.__class__ = self.base_node_class
        return obj


class NNGraph(OptGraph):
    def __init__(self, nodes=None, fitted_model=None):
        super().__init__(nodes)
        self.model = fitted_model

    def __repr__(self):
        return f"{self.depth}:{self.length}:{self.cnn_depth}"

    def __eq__(self, other) -> bool:
        return self is other

    @property
    def cnn_depth(self):
        depth = [node for node in self.nodes if 'conv' in node.content]
        return len(depth)

    def fit(self, input_data: InputData, verbose=False, input_shape: tuple = None,
            min_filters: int = None, max_filters: int = None, classes: int = 3, batch_size=24, epochs=15):
        if not self.model:
            self.model = create_nn_model(self, input_shape, classes)
        train_predicted = keras_model_fit(self.model, input_data, verbose=verbose, batch_size=batch_size, epochs=epochs)
        # TODO mb need to add target in output
        return train_predicted

    def predict(self, input_data: InputData, output_mode: str = 'default', is_multiclass: bool = False) -> OutputData:
        evaluation_result = keras_model_predict(self.model, input_data, output_mode, is_multiclass=is_multiclass)
        return evaluation_result


class NNNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List['NNNode']]):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        if 'params' in content:
            self.content = content

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()


class GPNNGraphOptimiser(EvoGraphOptimiser):
    def __init__(self, initial_graph, requirements, graph_generation_params,
                 metrics, parameters, log):
        self.metrics = metrics

        super().__init__(initial_graph=initial_graph, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         metrics=metrics,
                         parameters=parameters,
                         log=log)

        self.parameters.graph_generation_function = random_cnn_graph
        self.graph_generation_function = partial(self.parameters.graph_generation_function,
                                                 graph_class=NNGraph,
                                                 requirements=self.requirements,
                                                 node_func=NNNode)

        if initial_graph and type(initial_graph) != list:
            self.population = [initial_graph] * requirements.pop_size
        else:
            self.population = initial_graph or self._make_population(self.requirements.pop_size)

    def _make_population(self, pop_size: int):
        initial_graphs = [self.graph_generation_function() for _ in range(pop_size)]
        ind_graphs = []
        for g in initial_graphs:
            new_ind = Individual(deepcopy(self.graph_generation_params.adapter.adapt(g)))
            ind_graphs.append(new_ind)
        return ind_graphs

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, input_shape, min_filters, max_filters, classes, batch_size, epochs,
                         graph) -> float:
        graph.fit(train_data, True, input_shape, min_filters, max_filters, classes, batch_size, epochs)
        return [metric_function(graph, test_data)]

    def compose(self, data):
        train_data, test_data = train_test_data_setup(data, 0.8)
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
