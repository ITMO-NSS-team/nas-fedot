import os
import datetime
from typing import List, Union

from nas.utils.var import project_root, verbose_values
from nas.utils.utils import set_tf_compat, set_root

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

from nas.data.load_images import from_directory
from nas.composer.cnn.cnn_adapters import CustomGraphAdapter
from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.composer.nas_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.nas_cnn_composer import GPNNComposerRequirements
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_change_mutation, \
    single_drop_mutation, single_add_mutation
from nas.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.mutations.cnn_val_rules import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.metrics.metrics import calculate_validation_metric
from nas.composer.cnn.cnn_graph_operator import generate_initial_graph
from nas.composer.cnn.cnn_builder import CNNBuilder

root = project_root
set_root(root)


def start_test_example(path: str, epochs:  int = 1, verbose: Union[int, str] = 'auto',
                       initial_graph_struct: List[str] = None, per_class_limit: int = None,
                       timeout: datetime.timedelta = None):
    # size = 120
    # num_of_classes = 3
    timeout = datetime.timedelta(hours=20) if not timeout else timeout
    dataset_to_compose, dataset_to_validate, size = from_directory(path, per_class_limit=per_class_limit)
    num_of_classes = dataset_to_compose.num_classes
    cnn_node_types = ['conv2d']
    pool_types = ['max_pool2d', 'average_pool2d']
    nn_node_types = ['dense', 'serial_connection']
    secondary_node_types = ['serial_connection', 'dropout']
    rules = [has_no_self_cycled_nodes, has_no_cycle, flatten_check, has_no_flatten_skip,
             graph_has_several_starts, graph_has_wrong_structure]
    mutations = [cnn_simple_mutation, single_drop_mutation, single_edge_mutation, single_add_mutation,
                 single_change_mutation]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    skip_connection_ids = [0, 4, 8]
    skip_connections_len = 4
    requirements = GPNNComposerRequirements(
        conv_kernel_size=[3, 3], conv_strides=[1, 1], pool_size=[2, 2], min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=128, image_size=[size, size],
        conv_types=cnn_node_types, pool_types=pool_types, cnn_secondary=secondary_node_types,
        primary=nn_node_types, secondary=secondary_node_types, min_arity=2, max_arity=3,
        max_nn_depth=6, pop_size=10, num_of_generations=10, crossover_prob=0.8, mutation_prob=0.5,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    optimiser_params = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=mutations,
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CNNGraph, base_node_class=CNNNode),
        rules_for_constraint=rules
    )
    if not initial_graph_struct:
        initial_graph = None
    else:
        initial_graph = [generate_initial_graph(CNNGraph, CNNNode, initial_graph_struct, None, True,
                                                skip_connection_ids, skip_connections_len)]
    optimiser = GPNNGraphOptimiser(initial_graph=initial_graph, requirements=requirements,
                                   graph_generation_params=graph_generation_params, graph_builder=CNNBuilder,
                                   metrics=metric_function, parameters=optimiser_params,
                                   log=default_log(logger_name='Bayesian', verbose_level=verbose_values[verbose]))

    optimized_network = optimiser.compose(train_data=dataset_to_compose)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_network)
    save_path = os.path.join(root, 'test_graph')
    optimiser.save(save_folder=save_path, history=True, image=True)
    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3),
                          epochs=epochs, classes=num_of_classes, verbose=True)

    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    set_tf_compat()
    file_path = os.path.join(project_root, 'datasets', 'Satellite-Image-Classification')
    initial_graph_nodes = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
                           'dense', 'dense']
    start_test_example(path=file_path, epochs=20, initial_graph_struct=initial_graph_nodes, verbose='auto')
