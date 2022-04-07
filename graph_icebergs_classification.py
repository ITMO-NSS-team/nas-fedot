import os
import random
import sys
import datetime

import numpy as np
import tensorflow as tf

from typing import Tuple
from sklearn.metrics import roc_auc_score as roc_auc, log_loss, accuracy_score

from nas.composer.graph_gp_cnn_composer import GPNNGraphOptimiser, GPNNComposerRequirements
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphNode, CustomGraphAdapter
from nas.layer import LayerTypesIdsEnum
from nas.cnn_data import from_json

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from nas.graph_cnn_mutations import cnn_simple_mutation
from fedot.core.optimisers.graph import OptGraph


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)


random.seed(2)
np.random.seed(2)


def calculate_validation_metric(graph: CustomGraphModel, dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # the execution of the obtained composite models
    predicted = graph.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    y_pred = [np.float64(predict[0]) for predict in predicted.predict]
    log_loss_value = log_loss(y_true=dataset_to_validate.target,
                              y_pred=y_pred)
    y_pred = [round(predict[0]) for predict in predicted.predict]
    accuracy_score_value = accuracy_score(y_true=dataset_to_validate.target,
                                          y_pred=y_pred)

    return roc_auc_value, log_loss_value, accuracy_score_value


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


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


def run_custom_example(filepath: str, timeout: datetime.timedelta = None):
    num_of_classes = 2
    dataset_to_compose, dataset_to_validate = from_json(filepath)
    if not timeout:
        timeout = datetime.timedelta(hours=20)

    secondary = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value]
    conv_types = [LayerTypesIdsEnum.conv2d.value]
    pool_types = [LayerTypesIdsEnum.maxpool2d.value, LayerTypesIdsEnum.averagepool2d.value]
    nn_primary = [LayerTypesIdsEnum.dense.value]
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[cnn_simple_mutation],
        crossover_types=[CrossoverTypesEnum.subtree],
        regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)
    requirements = GPNNComposerRequirements(
        conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=64, image_size=[75, 75],
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=secondary,
        primary=nn_primary, secondary=secondary, min_arity=2, max_arity=3,
        max_depth=6, pop_size=5, num_of_generations=5,
        crossover_prob=0, mutation_prob=0,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    optimiser = GPNNGraphOptimiser(
        initial_graph=None,
        requirements=requirements,
        graph_generation_params=graph_generation_params,
        metrics=metric_function,
        parameters=optimiser_parameters,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_network = optimiser.compose(data=dataset_to_compose)
    optimized_network.show(path='iceberg_result.png')

    print('Best model structure:')
    for node in optimized_network.nodes:
        print(node)

    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_network)
    optimized_network.fit(input_data=dataset_to_compose, input_shape=(75, 75, 3), epochs=20, classes=num_of_classes)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = 'model_ice.json'
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    # saving the weights of the model
    optimized_network.model.save_weights('model_ice.h5')

if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

    # a dataset that will be used as a train and test set during composition
    setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)

    file_path = 'IcebergsDataset/train.json'
    run_custom_example(file_path)
