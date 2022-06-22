import os
import datetime
from typing import List, Union, Optional

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes

from nas.utils.utils import set_root, seed_all
from nas.utils.var import project_root, default_nodes_params
from nas.composer.nas_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.nas_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn.cnn_adapters import CustomGraphAdapter
from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.data.load_images import DataLoader
from nas.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.mutations.cnn_val_rules import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.metrics.metrics import calculate_validation_metric
from nas.composer.cnn.cnn_builder import CNNBuilder

set_root(project_root)
seed_all(942212)


# TODO extend initial approximation with desirable nodes params. Add ability to load graph as initial approximation
def run_test(train_path, test_path: Optional[str] = None, verbose: Union[str, int] = 'auto', epochs: int = 1,
             save_path: str = None, image_size: int = None, max_cnn_depth: int = None, max_nn_depth: int = None,
             batch_size: int = None, opt_epochs: int = 5, initial_graph_struct: List[str] = None, default_params=None,
             samples_limit: int = None, timeout: datetime.timedelta = None, has_skip_connections: bool = False,
             pop_size: int = 5, num_of_generations: int = 10, split_ratio: float = 0.8):
    """
    Run example with custom dataset and params
    :param train_path: Path to dataset
    :param test_path: path to test dataset if data was split before
    :param verbose: verbose value: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = progress bar, 2 = one line per epoch
    :param epochs: number of train epochs
    :param save_path: save path to optimized graph, history and graph struct
    :param image_size: dataset's image size. if None then size of first image in dataset will be picked as image size
    :param max_cnn_depth: max possible depth of convolutional part of the graph
    :param max_nn_depth: max possible depth of dense part of the graph
    :param batch_size: number of samples per gradient update. if None will be set to 16
    :param opt_epochs: number epochs for fitness estimate
    :param initial_graph_struct: graph's initial approximation
    :param default_params: default parameters for initial graph approximation for each node type.
        If None, then parameters will be random generated based on requirements
    :param samples_limit: sample limit per class
    :param timeout: runtime restrictions
    :param has_skip_connections: parameter for initial graph. If True them graph with skip connections will be generated
    :param pop_size: population size for evolution
    :param num_of_generations: number of generations
    :param split_ratio: train/test data ratio
    """
    train_data = DataLoader.from_directory(dir_path=train_path, image_size=image_size, samples_limit=samples_limit)
    channel_num = train_data.features[0].shape[-1]
    image_size = train_data.features[0].shape[0] if not image_size else image_size
    input_shape = [image_size, image_size, channel_num] if image_size else train_data.features[0].shape
    if not test_path:
        train_data, test_data = train_test_data_setup(train_data, split_ratio, True)
    else:
        test_data = DataLoader.from_directory(dir_path=test_path, image_size=image_size, samples_limit=samples_limit)

    rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, graph_has_several_starts,
             graph_has_wrong_structure, flatten_check]
    mutations = [cnn_simple_mutation, single_drop_mutation, single_add_mutation,
                 single_change_mutation, single_edge_mutation]
    metric_func = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    # TODO fix verbose for evolution
    # TODO unit tests + get results with ResNet34
    requirements = GPNNComposerRequirements(input_shape=input_shape, pop_size=pop_size,
                                            num_of_generations=num_of_generations, max_num_of_conv_layers=max_cnn_depth,
                                            max_nn_depth=max_nn_depth, primary=['conv2d'], secondary=['dense'],
                                            batch_size=batch_size, timeout=timeout, epochs=opt_epochs,
                                            has_skip_connection=has_skip_connections,
                                            default_parameters=default_params)
    optimiser_params = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                  mutation_types=mutations,
                                                  crossover_types=[CrossoverTypesEnum.subtree],
                                                  regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CNNGraph, base_node_class=CNNNode), rules_for_constraint=rules)
    optimiser = GPNNGraphOptimiser(initial_graph=initial_graph_struct, requirements=requirements,
                                   graph_generation_params=graph_generation_params, graph_builder=CNNBuilder,
                                   metrics=metric_func, parameters=optimiser_params, verbose=verbose,
                                   log=default_log(logger_name='Custom-run', verbose_level=4))

    print(f'================ Starting optimisation process with following params: population size: {pop_size}; '
          f'number of generations: {num_of_generations}; number of train epochs: {opt_epochs}; '
          f'image size: {input_shape}; batch size: {batch_size} ================')

    optimized_network = optimiser.compose(data=train_data)
    if save_path:
        print('save best graph structure...')
        optimiser.save(save_folder=save_path, history=True, image=True)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_network)
    optimized_network.fit(input_data=train_data, requirements=requirements, train_epochs=epochs, verbose=verbose)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, test_data)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = os.path.join(project_root, 'models', 'custom_example_model.json')
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    optimized_network.model.save_weights(os.path.join(project_root, 'models', 'custom_example_model.h5'))

    print("Done!")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dir_root = os.path.join(project_root, 'datasets', 'Blood-Cell-Classification', 'train')
    test_root = os.path.join(project_root, 'datasets', 'Blood-Cell-Classification', 'test')
    save_path = os.path.join(project_root, 'Blood-Cell-Cls')
    initial_graph_nodes = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dense', 'dense']
    default_parameters = default_nodes_params
    run_test(dir_root, test_root, verbose=1, epochs=20, save_path=save_path, image_size=90, max_cnn_depth=4,
             max_nn_depth=3, batch_size=16, opt_epochs=5, initial_graph_struct=None, default_params=None, samples_limit=20,
             has_skip_connections=True, pop_size=1, num_of_generations=1)
