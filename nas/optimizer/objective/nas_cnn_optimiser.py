from typing import List

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams

from nas.composer.requirements import NNComposerRequirements
from nas.graph.base_graph import NasGraph
from nas.utils.utils import seed_all

seed_all(1)


class NNGraphOptimiser(EvoGraphOptimizer):
    def __init__(self, initial_graphs: List[NasGraph], requirements: NNComposerRequirements,
                 graph_generation_params: GraphGenerationParams, graph_optimizer_params: GPAlgorithmParameters,
                 objective: Objective, **kwargs):
        super().__init__(initial_graphs=initial_graphs, requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         objective=objective, graph_optimizer_params=graph_optimizer_params)
        self.save_path = kwargs.get('save_path')

    def save(self, history: bool = True):
        self.log.message(f'Saving files into {self.save_path.resolve()}')
        if not isinstance(self.generations.best_individuals[0].graph, NasGraph):
            graph = self.graph_generation_params.adapter.restore(self.generations.best_individuals[0].graph)
        else:
            graph = self.generations.best_individuals[0].graph
        if history:
            self.history.save(self.save_path)
        graph.save(path=self.save_path)
