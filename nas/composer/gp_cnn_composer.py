from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Optional,
    Tuple,
    List
)

from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.optimisers.selection import SelectionTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.composer.write_history import write_composer_history_to_csv
from fedot.core.models.data import InputData
from fedot.core.models.data import train_test_data_setup
from nas.cnn_crossover import CrossoverTypesEnum
from nas.cnn_crossover import crossover_by_type
from nas.cnn_gp_operators import permissible_kernel_parameters_correct, random_cnn_chain
from nas.cnn_mutation import MutationTypesEnum
from nas.cnn_mutation import mutation_by_type
from nas.layer import LayerTypesIdsEnum, activation_types
from nas.nas_chain import NASChain
from nas.nas_node import NNNodeGenerator


@dataclass
class GPNNComposerRequirements(GPComposerRequirements):
    conv_kernel_size: Tuple[int, int] = (3, 3)
    conv_strides: Tuple[int, int] = (1, 1)
    pool_size: Tuple[int, int] = (2, 2)
    pool_strides: Tuple[int, int] = (2, 2)
    min_num_of_neurons: int = 50
    max_num_of_neurons: int = 200
    min_filters = 64
    max_filters = 128
    channels_num = 3
    max_drop_size: int = 0.5
    image_size: List[int] = None
    conv_types: List[LayerTypesIdsEnum] = None
    cnn_secondary: List[LayerTypesIdsEnum] = None
    pool_types: List[LayerTypesIdsEnum] = None
    train_epochs_num: int = 10
    batch_size: int = 24
    num_of_classes = 3
    activation_types = activation_types
    max_num_of_conv_layers = 5
    min_num_of_conv_layers = 3

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


class GPNNComposer(Composer):
    def __init__(self):
        super().__init__()

    def compose_chain(self, data: InputData, initial_chain: Optional[NASChain],
                      composer_requirements: Optional[GPNNComposerRequirements],
                      metrics: Optional[Callable], optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False) -> NASChain:
        train_data, test_data = train_test_data_setup(data, 0.8)

        input_shape = [size for size in composer_requirements.image_size]
        input_shape.append(composer_requirements.channels_num)
        input_shape = tuple(input_shape)

        if not optimiser_parameters:
            self.optimiser_parameters = GPChainOptimiserParameters(chain_generation_function=random_cnn_chain,
                                                                   crossover_types=[CrossoverTypesEnum.subtree],
                                                                   crossover_types_dict=crossover_by_type,
                                                                   mutation_types=[MutationTypesEnum.simple],
                                                                   mutation_types_dict=mutation_by_type,
                                                                   selection_types=[SelectionTypesEnum.tournament])
        else:
            self.optimiser_parameters = optimiser_parameters
        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data, input_shape,
                                            composer_requirements.min_filters, composer_requirements.max_filters,
                                            composer_requirements.num_of_classes, composer_requirements.batch_size,
                                            composer_requirements.train_epochs_num)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=NNNodeGenerator.primary_node,
                                     secondary_node_func=NNNodeGenerator.secondary_node, chain_class=NASChain,
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
                         test_data: InputData, input_shape, min_filters, max_filters, classes, batch_size, epochs,
                         chain: NASChain) -> float:
        chain.fit(train_data, True, input_shape, min_filters, max_filters, classes, batch_size, epochs)
        return metric_function(chain, test_data)