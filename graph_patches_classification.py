import copy
import datetime
import os
import random
import statistics
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score as roc_auc, log_loss, accuracy_score

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from nas.composer.graph_gp_cnn_composer import CustomGraphModel, CustomGraphAdapter
from nas.composer.graph_gp_cnn_composer import GPNNGraphOptimiser, GPNNComposerRequirements
from nas.graph_cnn_crossover import cnn_subtree_crossover
from nas.graph_cnn_mutation import cnn_simple_mutation
from nas.graph_nas_node import NNNode
from nas.layer import LayerTypesIdsEnum
from nas.patches.load_images import from_images

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)
random.seed(2)
np.random.seed(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)


# class CustomGraphNode(NNNode):
#     def __init__(self, content: dict, nodes_from, layer_params):
#         super().__init__(content, nodes_from, layer_params)
#
#     def __repr__(self):
#         # return f"Node_repr_{self.layer_params.layer_type.name}"
#         return NNNode.__str__(self)


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def calculate_validation_metric_multiclass(graph: CustomGraphModel, dataset_to_validate: InputData) -> Tuple[
    float, float, float]:
    # the execution of the obtained composite models
    predicted = graph.predict(dataset_to_validate)
    y_pred = []
    roc_auc_values = []
    for predict, true in zip(predicted.predict, dataset_to_validate.target):
        roc_auc_score = roc_auc(y_true=true,
                                y_score=predict)
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
    accuracy_score_value = accuracy_score(y_true=dataset_to_validate.target,
                                          y_pred=y_pred)

    return roc_auc_value, log_loss_value, accuracy_score_value


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


def run_patches_classification(file_path, timeout: datetime.timedelta = None):
    size = 120
    num_of_classes = 3
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes, img_size=size)

    if not timeout:
        timeout = datetime.timedelta(hours=20)

    cnn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
    conv_types = [LayerTypesIdsEnum.conv2d]
    pool_types = [LayerTypesIdsEnum.maxpool2d, LayerTypesIdsEnum.averagepool2d]
    nn_primary = [LayerTypesIdsEnum.dense]
    nn_secondary = [LayerTypesIdsEnum.serial_connection, LayerTypesIdsEnum.dropout]
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    metric_function = [MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)]

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        # mutation_types=[MutationTypesEnum.simple],
        mutation_types=[cnn_simple_mutation],
        # crossover_types=[CrossoverTypesEnum.cnn_subtree],
        crossover_types=[cnn_subtree_crossover],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CustomGraphModel, base_node_class=NNNode),
        rules_for_constraint=rules)

    requirements = GPNNComposerRequirements(
        conv_kernel_size_range=(1, 4), conv_strides_range=(1, 2), pool_size_range=(2, 4),
        min_num_of_neurons=16, max_num_of_neurons=128, min_filters=16, max_filters=128,
        min_arity=1, max_arity=3, max_depth=4, min_num_of_conv_layers=1, max_num_of_conv_layers=4, max_params=1000000,
        pop_size=6, num_of_generations=10, crossover_prob=0.7, mutation_prob=0.4, train_epochs_num=5,
        num_of_classes=num_of_classes, timeout=timeout, image_size=[size, size], conv_types=conv_types,
        pool_types=pool_types, cnn_secondary=cnn_secondary, primary=nn_primary, secondary=nn_secondary)
    optimiser = GPNNGraphOptimiser(
        initial_graph=None, requirements=requirements, graph_generation_params=graph_generation_params,
        metrics=metric_function, parameters=optimiser_parameters,
        log=default_log(logger_name='Bayesian', verbose_level=1))

    optimized_network = optimiser.compose(data=dataset_to_compose)
    # optimized_network = optimiser.optimise(partial(custom_metric, data=data))
    net_show = copy.deepcopy(optimized_network)
    net_show.nodes = net_show.cnn_nodes + net_show.nodes
    net_show.show(path='result.png')

    print('Best model structure:')
    for node in optimized_network.cnn_nodes:
        val = node.layer_params.num_of_filters or \
              node.layer_params.drop or ''
        name = node.layer_params.layer_type.name or node
        print(f"{name}, {val}")
    for node in optimized_network.nodes:
        val = node.layer_params.neurons or ''
        name = node.layer_params.layer_type.name or node
        print(f"{name}, {val}")
    optimized_network_custom = CustomGraphModel(nodes=optimized_network.nodes, cnn_nodes=optimized_network.cnn_nodes,
                                                fitted_model=None)
    optimized_network_custom.fit(input_data=dataset_to_compose, input_shape=(size, size, 3), epochs=20,
                                 classes=num_of_classes)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network_custom, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = f'models_saved/{num_of_classes}cls_best_model.json'
    model_json = optimized_network_custom.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    # saving the weights of the model
    optimized_network_custom.model.save_weights(f'models_saved/{num_of_classes}cls'
                                                f'_{round(accuracy_score_on_valid_evo_composed, 3)}_acc_best.h5')


if __name__ == '__main__':
    # a dataset that will be used as a train and test set during composition
    dataset_path = 'Generated_dataset'
    run_patches_classification(file_path=dataset_path)
