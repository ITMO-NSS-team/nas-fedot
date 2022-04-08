import os
import random
import statistics
import sys
import datetime

import numpy as np
import tensorflow as tf

from typing import Tuple
from sklearn.metrics import roc_auc_score as roc_auc, log_loss, accuracy_score

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.data.data import InputData

from nas.layer import LayerTypesIdsEnum
from nas.patches.load_images import from_images
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphNode, CustomGraphAdapter
from nas.composer.graph_gp_cnn_composer import GPNNGraphOptimiser, GPNNComposerRequirements

from fedot.core.log import default_log
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from nas.graph_cnn_mutations import cnn_simple_mutation

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)

random.seed(2)
np.random.seed(2)


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def calculate_validation_metric_multiclass(graph: CustomGraphModel,
                                           dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # the execution of the obtained composite models
    predicted = graph.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    y_pred = []
    roc_auc_values = []
    for predict, true in zip(predicted.predict, dataset_to_validate.target):
        roc_auc_score = roc_auc(y_true=true, y_score=predict)
        roc_auc_values.append(roc_auc_score)
    roc_auc_value = statistics.mean(roc_auc_values)
    for predict in predicted.predict:
        values = []
        for val in predict:
            values.append(round(val))
        y_pred.append(np.float64(values))
    y_pred = np.array(y_pred)
    log_loss_value = log_loss(y_true=dataset_to_validate.target,
                              y_pred=y_pred)
    accuracy_score_value = accuracy_score(y_true=dataset_to_validate.target, y_pred=y_pred)

    return roc_auc_value, log_loss_value, accuracy_score_value


def calculate_validation_metric(graph: CustomGraphModel, dataset_to_validate: InputData) -> Tuple[float, float, float]:
    # the execution of the obtained composite models
    predicted = graph.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target, y_score=predicted.predict,
                            multi_class="ovo", average="macro")
    y_values_pred = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(predicted.idx.size)]
    for i, predict in enumerate(predicted.predict):
        y_class_pred = np.argmax(predict)
        y_values_pred[i][y_class_pred] = 1

    y_pred = np.array([predict for predict in predicted.predict])
    y_values_pred = np.array(y_values_pred)
    log_loss_value = log_loss(y_true=dataset_to_validate.target, y_pred=y_pred)
    accuracy_score_value = accuracy_score(dataset_to_validate.target, y_values_pred)
    return roc_auc_value, log_loss_value, accuracy_score_value


def run_patches_classification(file_path, timeout: datetime.timedelta = None):
    size = 120
    num_of_classes = 10
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes)

    if not timeout:
        timeout = datetime.timedelta(hours=20)

    secondary = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value]
    conv_types = [LayerTypesIdsEnum.conv2d.value]
    pool_types = [LayerTypesIdsEnum.maxpool2d.value, LayerTypesIdsEnum.averagepool2d.value]
    nn_primary = [LayerTypesIdsEnum.dense.value]
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=[cnn_simple_mutation],
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)
    requirements = GPNNComposerRequirements(
        conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=128, image_size=[size, size],
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=secondary,
        primary=nn_primary, secondary=secondary, min_arity=2, max_arity=3,
        max_depth=6, pop_size=10, num_of_generations=10, crossover_prob=0.8, mutation_prob=0.5,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    optimiser = GPNNGraphOptimiser(
        initial_graph=None, requirements=requirements, graph_generation_params=graph_generation_params,
        metrics=metric_function, parameters=optimiser_parameters,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_network = optimiser.compose(data=dataset_to_compose)
    optimized_network.show(path='graph_10_cls_result.png')
    print('Best model structure:')
    for node in optimized_network.nodes:
        print(node)

    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3),
                          epochs=20, classes=num_of_classes)
    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = 'model_10cls.json'
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    # saving the weights of the model
    optimized_network.model.save_weights('model_10cls.h5')
    return optimized_network


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    file_path = '10cls_Generated_dataset.pickle'

    # a dataset that will be used as a train and test set during composition
    setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)

    run_patches_classification(file_path=file_path)
