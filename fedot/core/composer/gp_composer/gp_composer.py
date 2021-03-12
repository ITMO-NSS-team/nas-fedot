from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Optional,
)

from fedot.core.chain_validation import validate
from fedot.core.composer.chain import Chain, SharedChain
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.composer.node import NodeGenerator
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum, crossover_by_type
from fedot.core.composer.optimisers.gp_operators import random_ml_chain
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, MutationStrengthEnum, mutation_by_type
from fedot.core.composer.optimisers.selection import SelectionTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.composer.write_history import write_composer_history_to_csv
from fedot.core.models.data import InputData
from fedot.core.models.data import train_test_data_setup


@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 50
    num_of_generations: Optional[int] = 50
    crossover_prob: Optional[float] = None
    mutation_prob: Optional[float] = None
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean


class GPComposer(Composer):
    def __init__(self):
        super().__init__()
        self.shared_cache = {}

    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable], optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False) -> Chain:

        train_data, test_data = train_test_data_setup(data, 0.8)
        self.shared_cache.clear()

        if not optimiser_parameters:
            self.optimiser_parameters = GPChainOptimiserParameters(chain_generation_function=random_ml_chain,
                                                                   crossover_types=[CrossoverTypesEnum.subtree,
                                                                                    CrossoverTypesEnum.onepoint],
                                                                   crossover_types_dict=crossover_by_type,
                                                                   mutation_types=[MutationTypesEnum.simple,
                                                                                   MutationTypesEnum.local_growth,
                                                                                   MutationTypesEnum.reduce],
                                                                   mutation_types_dict=mutation_by_type,
                                                                   selection_types=[SelectionTypesEnum.tournament])
        else:
            self.optimiser_parameters = optimiser_parameters

        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data, True)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=NodeGenerator.primary_node,
                                     secondary_node_func=NodeGenerator.secondary_node, chain_class=Chain,
                                     parameters=self.optimiser_parameters)

        best_chain, self.history = optimiser.optimise(metric_function_for_nodes)
        historical_fitness = [chain.fitness for chain in self.history]

        if is_visualise:
            ComposerVisualiser.visualise_history(self.history, historical_fitness)

        write_composer_history_to_csv(historical_fitness=historical_fitness, historical_chains=self.history,
                                      pop_size=composer_requirements.pop_size)

        print('GP composition finished')
        return best_chain

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, is_chain_shared: bool,
                         chain: Chain) -> float:
        validate(chain)
        if is_chain_shared:
            chain = SharedChain(base_chain=chain, shared_cache=self.shared_cache)
        chain.fit(input_data=train_data)
        return metric_function(chain, test_data)
