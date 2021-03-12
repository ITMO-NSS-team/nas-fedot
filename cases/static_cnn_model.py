import os
import datetime
import random

from nas.patches.utils import project_root, set_tf_compat

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

from nas.layer import LayerTypesIdsEnum
from nas.patches.load_images import from_images
from nas.composer.graph_gp_cnn_composer import NNGraph, CustomGraphAdapter, NNNode
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements, GPNNGraphOptimiser
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation
from nas.graph_cnn_mutations import cnn_simple_mutation, has_no_flatten_skip
from nas.composer.metrics import calculate_validation_metric

root = project_root()


def add_skip_connections(graph: NNGraph):
    res_blocks_num = random.randint(0, graph.cnn_depth - 1)
    for _ in range(res_blocks_num):
        is_residual = random.randint(0, 1)
        for conv_id in range(graph.cnn_depth - 1):
            conv_node = graph.nodes[conv_id]
            if is_residual:
                res_depth = random.randint(conv_id, graph.cnn_depth - 1)
                if res_depth == 1:
                    continue
                if len(graph.nodes[res_depth].nodes_from) > 1:
                    continue
                graph.nodes[res_depth].nodes_from.append(conv_node)
    return graph


def create_graph(graph: NNGraph, node_type: str, params: dict, parent=None, residual_parent=None,
                 prev_conv: bool = False):
    parent = None if not parent else [parent]
    is_residual = 0
    drop = node_type.startswith('drop')
    is_skip = not drop and residual_parent is None
    is_end = not drop and residual_parent is not None

    if node_type.startswith('conv'):
        new_node = NNNode(nodes_from=parent, content={'name': params[0]['layer_type'],
                                                      'params': params[0], 'conv': True})
        prev_conv = True
    elif node_type.startswith('drop'):
        new_node = NNNode(nodes_from=parent,
                          content={'name': LayerTypesIdsEnum.dropout.value,
                                   'params': {'layer_type': LayerTypesIdsEnum.dropout.value,
                                              'drop': 0.2}})
    else:
        new_node = NNNode(nodes_from=parent, content={'name': params[1]['layer_type'],
                                                      'params': params[1]})
    if is_residual and is_end:
        new_node.nodes_from.append(residual_parent)
        residual_parent.content['skip_connection_to'] = new_node
        new_node.content['skip_connection_from'] = residual_parent
        residual_parent = None
    if is_residual and is_skip:
        residual_parent = new_node
    if prev_conv and not node_type.startswith('conv'):
        new_node.content['conv'] = True
        prev_conv = False

    graph.add_node(new_node)
    return new_node, residual_parent, prev_conv


def start_example_with_init_graph(file_path: str, timeout: datetime.timedelta = None):
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
    conv_layer_params = {'layer_type': cnn_node_types[0], 'activation': activation, 'kernel_size': conv_kernel_size,
                         'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
                         'pool_strides': pool_strides, 'pool_type': pool_types[0]}
    nn_layer_params = {'activation': 'relu', 'layer_type': nn_node_types[0], 'neurons': 121}
    params = [conv_layer_params, nn_layer_params]
    nodes_list = ['conv_1', 'drop_1', 'conv_2', 'drop_2', 'conv_3', 'drop_3', 'conv_4', 'drop_4',
                  'nn_node_1', 'drop_nn_1', 'nn_node_2', 'drop_nn_2', 'nn_node_3']
    initial_graph = NNGraph()

    parent_node = None
    residual_parent = None
    prev_conv = False
    for ind, type in enumerate(nodes_list):
        parent_node, residual_parent, prev_conv = create_graph(graph=initial_graph, node_type=type, params=params,
                                                               parent=parent_node, residual_parent=residual_parent,
                                                               prev_conv=prev_conv)
    initial_graph = add_skip_connections(initial_graph)
    initial_graph.show()
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
        adapter=CustomGraphAdapter(base_graph_class=NNGraph, base_node_class=NNNode),
        rules_for_constraint=rules
    )

    optimiser = GPNNGraphOptimiser(
        initial_graph=[initial_graph], requirements=requirements, graph_generation_params=graph_generation_params,
        metrics=metric_function, parameters=optimiser_params,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_graph = optimiser.compose(data=dataset_to_compose)
    optimized_graph.show(path='../test_result.png')

    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3),
                          epochs=20, classes=num_of_classes, verbose=True)

    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    file_path = os.path.join(root, 'Generated_dataset')
    set_tf_compat()
    start_example_with_init_graph(file_path=file_path)
