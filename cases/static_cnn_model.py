import os
import datetime

from typing import List
from nas.patches.utils import project_root

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

from nas.layer import LayerParams, LayerTypesIdsEnum
from nas.patches.load_images import from_images
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphAdapter, CustomGraphNode
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements, GPNNGraphOptimiser
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from nas.graph_cnn_mutations import cnn_simple_mutation
from nas.composer.metrics import calculate_validation_metric

root = project_root()


def create_graph(graph: CustomGraphModel, node_type: str, params: List[LayerParams], parent=None):
    parent = None if not parent else [parent]
    if node_type.startswith('conv'):
        new_node = CustomGraphNode(nodes_from=parent, content={'name': params[0].layer_type,
                                                               'params': params[0], 'conv': True})
    elif node_type.startswith('drop'):
        new_node = CustomGraphNode(nodes_from=parent,
                                   content={'name': LayerTypesIdsEnum.dropout.value,
                                            'params': LayerParams(layer_type=LayerTypesIdsEnum.dropout.value,
                                                                  drop=0.2)})
    else:
        new_node = CustomGraphNode(nodes_from=parent, content={'name': params[1].layer_type,
                                                               'params': params[1]})
    graph.add_node(new_node)
    return new_node


def train_opt_graph(file_path: str, init_graph: CustomGraphModel = None, timeout: datetime.timedelta = None):

    if not timeout:
        timeout = datetime.timedelta(hours=20)

    size = 120
    num_of_classes = 3
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes)

    cnn_node_types = [LayerTypesIdsEnum.conv2d.value]
    conv_strides = (1, 1)
    conv_kernel_size = (3, 3)
    pool_types = [LayerTypesIdsEnum.maxpool2d.value, LayerTypesIdsEnum.averagepool2d.value]
    pool_size = (2, 2)
    pool_strides = (2, 2)
    nn_node_types = [LayerTypesIdsEnum.dense.value, LayerTypesIdsEnum.serial_connection.value]
    secondary_node_types = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value]
    activation = 'relu'
    num_of_filters = 16
    rules = [has_no_self_cycled_nodes, has_no_cycle]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    conv_layer_params = LayerParams(layer_type=cnn_node_types[0], activation=activation, kernel_size=conv_kernel_size,
                                    conv_strides=conv_strides, num_of_filters=num_of_filters, pool_size=pool_size,
                                    pool_strides=pool_strides, pool_type=pool_types[0])
    nn_layer_params = LayerParams(activation='relu', layer_type=nn_node_types[0], neurons=121)
    params = [conv_layer_params, nn_layer_params]
    nodes_list = ['conv_1', 'drop_1', 'conv_2', 'drop_2', 'nn_node_1', 'drop_nn_1', 'nn_node_2', 'drop_nn_2',
                  'nn_node_3']
    parent_node = None
    for ind, type in enumerate(nodes_list):
        parent_node = create_graph(graph=init_graph, node_type=type, params=params, parent=parent_node)

    requirements = GPNNComposerRequirements(
        conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=128, image_size=[size, size],
        conv_types=cnn_node_types, pool_types=pool_types, cnn_secondary=secondary_node_types,
        primary=nn_node_types, secondary=secondary_node_types, min_arity=2, max_arity=3,
        max_depth=6, pop_size=10, num_of_generations=10, crossover_prob=0.8, mutation_prob=0.5,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    optimiser_params = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=[cnn_simple_mutation],
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules
    )

    optimiser = GPNNGraphOptimiser(
        initial_graph=[init_graph], requirements=requirements, graph_generation_params=graph_generation_params,
        metrics=metric_function, parameters=optimiser_params,
        log=default_log(logger_name='Bayesian', verbose_level=0))

    optimized_graph = optimiser.compose(data=dataset_to_compose)
    optimized_graph.show(path='../test_result.png')

    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3),
                          epochs=20, classes=num_of_classes, verbose=False)

    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    file_path = os.path.join(root, 'Generated_dataset')
    initial_graph = CustomGraphModel()
    train_opt_graph(file_path=file_path, init_graph=initial_graph)

