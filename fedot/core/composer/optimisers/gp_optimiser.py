import math
from dataclasses import dataclass
from functools import partial
from typing import (
    List,
    Callable,
    Any,
    Optional
)

import numpy as np
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum, crossover
from fedot.core.composer.optimisers.heredity import heredity, GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, mutation
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.composer.optimisers.selection import SelectionTypesEnum, selection
from fedot.core.composer.timer import CompositionTimer


@dataclass
class GPChainOptimiserParameters:
    chain_generation_function: Callable
    selection_types: List[SelectionTypesEnum]
    crossover_types: List[CrossoverTypesEnum]
    mutation_types: List[MutationTypesEnum]
    crossover_types_dict: dict
    mutation_types_dict: dict
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none
    genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.steady_state


class GPChainOptimiser:
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable,
                 chain_class: Callable, parameters: Optional[GPChainOptimiserParameters] = None):
        self.requirements = requirements
        self.primary_node_func = primary_node_func
        self.secondary_node_func = secondary_node_func
        self.history = []
        self.chain_class = chain_class
        self.parameters = parameters

        self.chain_generation_function = partial(self.parameters.chain_generation_function, chain_class=chain_class,
                                                 requirements=self.requirements,
                                                 primary_node_func=self.primary_node_func,
                                                 secondary_node_func=self.secondary_node_func)

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            raise AttributeError(f'Object chain_class has no required attributes for gp_optimizer')

        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def optimise(self, objective_function, offspring_rate=0.5):

        if self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
            num_of_new_individuals = math.ceil(self.requirements.pop_size * offspring_rate)
        else:
            num_of_new_individuals = self.requirements.pop_size - 1

        with CompositionTimer() as t:

            self.history = []

            for ind in self.population:
                ind.fitness = objective_function(ind)

            self._add_to_history(self.population)

            for generation_num in range(self.requirements.num_of_generations - 1):
                print(f'Generation num: {generation_num}')

                individuals_to_select = regularized_population(reg_type=self.parameters.regularization_type,
                                                               population=self.population,
                                                               objective_function=objective_function,
                                                               chain_class=self.chain_class)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_new_individuals * 2)

                new_population = []

                for ind_num, parent_num in zip(range(num_of_new_individuals), range(0, len(selected_individuals), 2)):
                    new_population.append(
                        self.reproduce(selected_individuals[parent_num], selected_individuals[parent_num + 1]))

                    new_population[ind_num].fitness = objective_function(new_population[ind_num])

                self.population = heredity(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                           self.population,
                                           new_population, self.requirements.pop_size - 1)

                self.population.append(self.best_individual)

                self._add_to_history(self.population)

                print('spent time:', t.minutes_from_start)
                print(f'Best metric is {self.best_individual.fitness}')

                if t.is_max_time_reached(self.requirements.max_lead_time, generation_num):
                    break

        return self.best_individual, self.history

    @property
    def best_individual(self) -> Any:
        best_ind = min(self.population, key=lambda ind: ind.fitness)
        equivalents = self.simpler_equivalents_of_best_ind(best_ind)

        if equivalents:
            best_candidate_id = min(equivalents, key=equivalents.get)
            best_ind = self.population[best_candidate_id]
        return best_ind

    def simpler_equivalents_of_best_ind(self, best_ind: Any) -> dict:
        sort_inds = np.argsort([ind.fitness for ind in self.population])[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = best_ind.fitness == self.population[i].fitness
            has_less_num_of_models_than_best = len(self.population[i].nodes) < len(best_ind.nodes)
            if is_fitness_equals_to_best and has_less_num_of_models_than_best:
                simpler_equivalents[i] = len(self.population[i].nodes)
        return simpler_equivalents

    def reproduce(self, selected_individual_first, selected_individual_second) -> Any:
        new_ind = crossover(self.parameters.crossover_types,
                            selected_individual_first,
                            selected_individual_second,
                            self.requirements, self.parameters.crossover_types_dict)

        new_ind = mutation(types=self.parameters.mutation_types,
                           chain_class=self.chain_class,
                           chain=new_ind,
                           requirements=self.requirements,
                           secondary_node_func=self.secondary_node_func,
                           primary_node_func=self.primary_node_func, types_dict=self.parameters.mutation_types_dict)
        return new_ind

    def _make_population(self, pop_size: int) -> List[Any]:
        return [self.chain_generation_function() for _ in range(pop_size)]

    def _add_to_history(self, individuals: List[Any]):
        [self.history.append(ind) for ind in individuals]
