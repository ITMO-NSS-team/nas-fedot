import random
import os
from functools import partial
from copy import deepcopy
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser

from nas.composer.cnn_graph import NNGraph
from nas.composer.cnn_graph_node import NNNode
from nas.composer.cnn_graph_operator import random_conv_graph_generation

random.seed(1)
np.random.seed(1)


class GPNNGraphOptimiser(EvoGraphOptimiser):
    def __init__(self, initial_graph, requirements, graph_generation_params,
                 metrics, parameters, log):
        self.metrics = metrics

        super().__init__(initial_graph=initial_graph, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         metrics=metrics,
                         parameters=parameters,
                         log=log)

        self.parameters.graph_generation_function = random_conv_graph_generation
        self.graph_generation_function = partial(self.parameters.graph_generation_function,
                                                 graph_class=NNGraph,
                                                 requirements=self.requirements,
                                                 node_func=NNNode)
        self.population = initial_graph or self._make_population(self.requirements.pop_size)

    def save(self, save_folder: str = None, history: bool = True, image: bool = True):
        print(f'Saving files into {os.path.abspath(save_folder)}')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if not isinstance(self.best_individual.graph, NNGraph):
            graph = self.graph_generation_params.adapter.restore(self.best_individual.graph)
        else:
            graph = self.best_individual.graph
        graph.save(path=save_folder)
        if history:
            self.history.save(json_file_path=f'{save_folder}_opt_history.json')
        if image:
            graph.show(path=f'{save_folder}_optimized_graph.png')

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
        graph.fit(train_data, True, input_shape, train_data.num_classes, batch_size, epochs)
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
