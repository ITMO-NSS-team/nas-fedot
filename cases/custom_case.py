import os
import datetime
from typing import List, Union

from fedot.core.log import default_log
from nas.utils.utils import set_root
from nas.utils.var import PROJECT_ROOT, VERBOSE_VAL
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from nas.composer.gp_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.gp_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn_adapters import CustomGraphAdapter
from fedot.core.optimisers.optimizer import GraphGenerationParams
from nas.composer.cnn_graph_operator import generate_initial_graph
from nas.composer.cnn_graph_node import NNNode
from nas.composer.cnn_graph import NNGraph
from nas.data.load_images import DataLoader
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from nas.graph_cnn_mutations import cnn_simple_mutation, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.composer.metrics import calculate_validation_metric
from nas.composer.cnn_graph_operator import random_conv_graph_generation

set_root(PROJECT_ROOT)


# TODO extend initial approximation with desirable nodes params. Add ability to load graph as initial approximation
def run_test(path, verbose: Union[str, int] = 'auto', epochs: int = 1, save_path: str = None, image_size: int = None,
             max_cnn_depth: int = None, max_nn_depth: int = None, batch_size: int = None, opt_epochs: int = 5,
             initial_graph_struct: List[str] = None, samples_limit: int = None, timeout: datetime.timedelta = None):
    """
    Run example with custom dataset and params.
    :param path: Path to dataset
    :param verbose: verbose value: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = progress bar, 2 = one line per epoch.
    :param epochs: number of train epochs
    :param save_path: save path to optimized graph, history and graph struct
    :param image_size: dataset's image size. if None then size of first image in dataset will be picked as image size
    :param max_cnn_depth: max possible depth of convolutional part of the graph
    :param max_nn_depth: max possible depth of dense part of the graph
    :param batch_size: number of samples per gradient update. if None will be set to 16
    :param opt_epochs:
    :param initial_graph_struct: graph's initial approximation
    :param samples_limit: sample limit per class
    :param timeout: runtime restrictions
    """
    dataset = DataLoader.from_directory(dir_path=path, image_size=image_size, samples_limit=samples_limit)
    image_size = dataset.features[0].shape[0] if image_size is None else image_size
    train_data, test_data = train_test_data_setup(dataset, 0.8, True)
    rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, graph_has_several_starts,
             graph_has_wrong_structure]
    mutations = [cnn_simple_mutation, single_drop_mutation, single_edge_mutation, single_add_mutation,
                 single_change_mutation]
    metric_func = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    # TODO add grayscale support
    # TODO fix verbose for evolution
    requirements = GPNNComposerRequirements(input_shape=[image_size, image_size, 3], pop_size=5, num_of_generations=10,
                                            max_num_of_conv_layers=max_cnn_depth, max_nn_depth=max_nn_depth,
                                            primary=None, secondary=None, batch_size=batch_size, timeout=timeout,
                                            epochs=opt_epochs)
    optimiser_params = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                  mutation_types=[cnn_simple_mutation, single_edge_mutation],
                                                  crossover_types=[CrossoverTypesEnum.subtree],
                                                  regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=NNGraph, base_node_class=NNNode), rules_for_constraint=rules)
    if not initial_graph_struct:
        initial_graph = None
    else:
        initial_graph = [generate_initial_graph(NNGraph, NNNode, initial_graph_struct, requirements)]
    optimiser = GPNNGraphOptimiser(initial_graph=initial_graph, graph_generation_params=graph_generation_params,
                                   graph_generation_function=random_conv_graph_generation,
                                   metrics=metric_func, parameters=optimiser_params, requirements=requirements,
                                   log=default_log(logger_name='Custom-run', verbose_level=VERBOSE_VAL[verbose]))
    optimized_network = optimiser.compose(data=train_data)
    if save_path:
        print('save best graph structure...')
        optimiser.save(save_folder=save_path, history=True, image=True)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_network)
    optimized_network.fit(input_data=train_data, requirements=requirements, epochs=epochs, verbose=verbose)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, test_data)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = './models/custom_example_model.json'
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    optimized_network.model.save_weights('./models/custom_example_model.h5')

    print("Done!")


if __name__ == '__main__':
    dir_root = os.path.join(PROJECT_ROOT, 'datasets', 'Satellite-Image-Classification')
    save_path = os.path.join(PROJECT_ROOT, 'Satellite')
    initial_graph_nodes = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
                           'dense', 'dense']
    run_test(dir_root, verbose=1, epochs=20, save_path=save_path, initial_graph_struct=None,
             image_size=128, max_cnn_depth=8, max_nn_depth=3, batch_size=2, opt_epochs=10)
