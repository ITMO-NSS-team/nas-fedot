import os
import gc
import pathlib

from tensorflow.keras.backend import clear_session

from typing import List, Optional
from functools import partial
from copy import deepcopy
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser

from nas.data.split_data import generator_train_test_split
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.composer.cnn.cnn_builder import NASDirector
from nas.utils.utils import seed_all
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

seed_all(1)


class GPNNGraphOptimiser(EvoGraphOptimiser):
    def __init__(self, initial_graph: Optional[List[str]], requirements, graph_generation_params, graph_builder,
                 metrics, parameters, log, verbose=0, save_path=None):
        super().__init__(initial_graph=initial_graph, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         metrics=metrics,
                         parameters=parameters,
                         log=log)
        self._save_path = save_path
        self.verbose = verbose
        self.metrics = metrics
        self.graph_builder = graph_builder
        self.director = NASDirector()
        self.initial_graph = [self._define_builder(initial_graph)] if initial_graph else None
        self.population = self.initial_graph if initial_graph else self._make_population(self.requirements.pop_size)

    @property
    def save_path(self):
        if not self._save_path.exists():
            pathlib.Path(self._save_path).mkdir(parents=True)
        return self._save_path

    def save(self, history: bool = True, image: bool = True):
        print(f'Saving files into {self.save_path.resolve()}')
        if not isinstance(self.best_individual.graph, CNNGraph):
            graph = self.graph_generation_params.adapter.restore(self.best_individual.graph)
        else:
            graph = self.best_individual.graph
        graph.save(path=self.save_path)
        if history:
            self.history.save()
        if image:
            graph.show(path=self.save_path / 'opt_history.json')

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
                         verbose, graph) -> List[float]:
        graph.fit(train_data, requirements=requirements, verbose=verbose, results_path=self.save_path)
        out = [metric_function(graph, test_data)]
        clear_session()
        gc.collect()
        del graph.model
        return out

    def compose(self, train_data):
        self.history.clean_results()
        train_data, test_data = generator_train_test_split(train_data, .7, True)
        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            self.metrics, train_data, test_data,
                                            self.requirements, self.verbose)
        self.optimise(metric_function_for_nodes)
        # TODO
        return self.graph_generation_params.adapter.restore(self.best_individual.graph)
