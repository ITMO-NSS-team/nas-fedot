import random
import datetime
from functools import partial

import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score as roc_auc, log_loss, accuracy_score

from fedot.core.data.data import InputData
from fedot.core.optimisers.graph import OptGraph, OptNode
from nas.layer import LayerParams, LayerTypesIdsEnum
from nas.patches.load_images import from_images
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphAdapter, CustomGraphNode
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from nas.graph_cnn_mutations import cnn_simple_mutation


random.seed(1)
np.random.seed(1)


def custom_metric(graph: CustomGraphModel, data):
    # graph.show()
    existing_variable_num = -graph.depth - graph.evaluate(data)
    return [existing_variable_num]


def custom_mutation(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            if random_node.nodes_from is not None and len(random_node.nodes_from) == 0:
                random_node.nodes_from = None
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph


def calculate_validation_metric(graph: CustomGraphModel, dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # the execution of the obtained composite models
    predicted = graph.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo", average="macro")
    y_values_pred = [[0, 0, 0] for _ in range(predicted.idx.size)]
    for i, predict in enumerate(predicted.predict):
        y_class_pred = np.argmax(predict)
        y_values_pred[i][y_class_pred] = 1

    y_pred = np.array([predict for predict in predicted.predict])
    y_values_pred = np.array(y_values_pred)
    log_loss_value = log_loss(y_true=dataset_to_validate.target,
                              y_pred=y_pred)
    accuracy_score_value = accuracy_score(dataset_to_validate.target,
                                          y_values_pred)

    return roc_auc_value, log_loss_value, accuracy_score_value


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def train_opt_graph(file_path: str, graph: CustomGraphModel = None, timeout: datetime.timedelta = None):

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
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    conv_layer_params = LayerParams(layer_type=cnn_node_types[0], activation=activation, kernel_size=conv_kernel_size,
                                    conv_strides=conv_strides, num_of_filters=num_of_filters, pool_size=pool_size,
                                    pool_strides=pool_strides, pool_type=pool_types[0])
    nn_layer_params = LayerParams(activation='relu', layer_type=nn_node_types[0], neurons=121)

    conv_node_1 = CustomGraphNode(nodes_from=None, layer_params=conv_layer_params,
                                  content={'name': conv_layer_params.layer_type})
    drop_1 = CustomGraphNode(nodes_from=[conv_node_1],
                             content={'name': secondary_node_types[1]},
                             layer_params=LayerParams(layer_type=secondary_node_types[1], drop=0.2))
    conv_node_2 = CustomGraphNode(nodes_from=[drop_1],
                                  content={'name': conv_layer_params.layer_type},
                                  layer_params=conv_layer_params)
    drop_2 = CustomGraphNode(nodes_from=[conv_node_2],
                             content={'name': secondary_node_types[1]},
                             layer_params=LayerParams(layer_type=secondary_node_types[1], drop=0.2))
    conv_node_3 = CustomGraphNode(nodes_from=[drop_2],
                                  content={'name': conv_layer_params.layer_type},
                                  layer_params=LayerParams(layer_type=LayerTypesIdsEnum.flatten))
    nn_node_1 = CustomGraphNode(nodes_from=[drop_2],
                                content={'name': nn_node_types[0]},
                                layer_params=nn_layer_params)
    drop_3 = CustomGraphNode(nodes_from=[nn_node_1],
                             content={'name': secondary_node_types[1]},
                             layer_params=LayerParams(layer_type=secondary_node_types[1], drop=0.2))
    nn_node_2 = CustomGraphNode(nodes_from=[drop_3],
                                content={'name': nn_node_types[0]},
                                layer_params=nn_layer_params)

    nodes_list = [conv_node_1, drop_1, conv_node_2, drop_2, nn_node_1, drop_3, nn_node_2]
    if graph:
        for node in nodes_list:
            graph.add_node(node)

    requirements = GPNNComposerRequirements(
        conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=128, image_size=[size, size],
        conv_types=cnn_node_types, pool_types=pool_types, cnn_secondary=secondary_node_types,
        primary=cnn_node_types, secondary=secondary_node_types, min_arity=2, max_arity=3,
        max_depth=6, pop_size=10, num_of_generations=10, crossover_prob=0.8, mutation_prob=0.5,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    optimiser_params = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=[cnn_simple_mutation],
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules
    )

    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_params,
        requirements=requirements, initial_graph=[graph],
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_graph = optimiser.optimise(partial(custom_metric, data=dataset_to_compose))
    optimized_graph.show(path='test_result.png')

    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3), epochs=20, classes=num_of_classes)

    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    file_path = 'Generated_dataset'
    initial_graph = CustomGraphModel(cnn_depth=4)
    train_opt_graph(file_path=file_path, graph=initial_graph)

