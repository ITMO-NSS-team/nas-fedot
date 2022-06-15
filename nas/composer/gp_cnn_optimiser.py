import random
import os
from typing import Callable, Union, List, Optional
from functools import partial
from copy import deepcopy
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser

from nas.composer.cnn_graph import CNNGraph
from nas.composer.cnn_graph_node import CNNNode
from nas.cnn_builder import NASDirector

random.seed(1)
np.random.seed(1)


class GPNNGraphOptimiser(EvoGraphOptimiser):
    def __init__(self, initial_graph: Optional[List[str]], requirements, graph_generation_params,
                 graph_builder, metrics, parameters, log):
        super().__init__(initial_graph=initial_graph, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         metrics=metrics,
                         parameters=parameters,
                         log=log)
        self.metrics = metrics
        self.graph_builder = graph_builder
        self.director = NASDirector()
        self.initial_graph = [self._define_builder(initial_graph)] if initial_graph else None
        self.population = self.initial_graph if initial_graph else self._make_population(self.requirements.pop_size)

    def save(self, save_folder: str = None, history: bool = True, image: bool = True):
        print(f'Saving files into {os.path.abspath(save_folder)}')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if not isinstance(self.best_individual.graph, CNNGraph):
            graph = self.graph_generation_params.adapter.restore(self.best_individual.graph)
        else:
            graph = self.best_individual.graph
        graph.save(path=save_folder)
        if history:
            self.history.save(json_file_path=f'{save_folder}_opt_history.json')
        if image:
            graph.show(path=f'{save_folder}_optimized_graph.png')

    def _define_builder(self, initial_graph=None):
        self.director.set_builder(self.graph_builder(nodes_list=initial_graph, requirements=self.requirements))
        return self.director.create_nas_graph()

    def _make_population(self, pop_size: int):
        initial_graphs = [self._define_builder() for _ in range(pop_size)]
        ind_graphs = []
        for g in initial_graphs:
            new_ind = Individual(deepcopy(self.graph_generation_params.adapter.adapt(g)))
            ind_graphs.append(new_ind)
        return ind_graphs

    def metric_for_nodes(self, metric_function, train_data: InputData, test_data: InputData, requirements,
                         graph) -> List[float]:
        graph.fit(train_data, True, requirements=requirements)
        out = [metric_function(graph, test_data)]
        return out

    def compose(self, data):
        train_data, test_data = train_test_data_setup(data, 0.8)
        self.history.clean_results()
        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            self.metrics, train_data, test_data,
                                            self.requirements)
        self.optimise(metric_function_for_nodes)
        return self.best_individual.graph
