import gc
import multiprocessing
from typing import List

import numpy as np
from fedot.core.optimisers.gp_comp.evaluation import SimpleDispatcher
from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.optimizer import GraphGenerationParams
from tensorflow.keras.backend import clear_session

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.cnn.cnn_graph import NNGraph
from nas.utils.utils import seed_all

seed_all(1)


class NNGraphOptimiser(EvoGraphOptimizer):
    def __init__(self, initial_graphs: List[NNGraph], requirements: NNComposerRequirements,
                 graph_generation_params: GraphGenerationParams, graph_optimizer_params: GPGraphOptimizerParameters,
                 objective, verbose=0):
        super().__init__(initial_graphs=initial_graphs, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         objective=objective, graph_optimizer_params=graph_optimizer_params)
        self.eval_dispatcher = SimpleDispatcher(adapter=graph_generation_params.adapter)
        self.save_path = None
        self.verbose = verbose
        self.metrics = objective

    def save(self, history: bool = True, image: bool = True):
        print(f'Saving files into {self.save_path.resolve()}')
        if not isinstance(self.generations.best_individuals[0].graph, NNGraph):
            graph = self.graph_generation_params.adapter.restore(self.generations.best_individuals[0].graph)
        else:
            graph = self.generations.best_individuals[0].graph
        graph.save(path=self.save_path)
        if image:
            graph.show(path=self.save_path / 'opt_graph.png')

    def _calculate_objective_function(self, graph, objective_function, data_producer, requirements,
                                      verbose) -> List[float]:
        fitness_history = []

        for fold_id, (train_data, test_data) in enumerate(data_producer()):
            graph.fit(train_data, requirements=requirements, verbose=verbose, results_path=self.save_path)
            ind_fitness = objective_function(graph, test_data)

            fitness_history.append(ind_fitness)

            clear_session()
            gc.collect()
            graph.model = None

        fitness = float(np.mean(fitness_history))

        return [fitness]

    def with_save_path(self, save_path):
        self.save_path = save_path
