import os
import random
import datetime

import numpy as np

from nas.patches.utils import project_root, set_tf_compat

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

from nas.patches.load_images import from_images
from nas.composer.graph_gp_cnn_composer import GPNNGraphOptimiser, GPNNComposerRequirements
from nas.layer import LayerTypesIdsEnum

from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from nas.graph_cnn_mutations import cnn_simple_mutation, has_no_flatten_skip
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation
from nas.composer.metrics import calculate_validation_metric

from nas.composer.graph_gp_cnn_composer import NNGraph, CustomGraphAdapter, NNNode

root = project_root()
random.seed(7)
np.random.seed(7)


def run_patches_classification(file_path, epochs: int = 20, timeout: datetime.timedelta = None):
    size = 120
    num_of_classes = 3
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes)

    if not timeout:
        timeout = datetime.timedelta(hours=20)

    secondary = [LayerTypesIdsEnum.serial_connection.value, LayerTypesIdsEnum.dropout.value,
                 LayerTypesIdsEnum.batch_normalization.value]
    conv_types = [LayerTypesIdsEnum.conv2d.value]
    pool_types = [LayerTypesIdsEnum.maxpool2d.value, LayerTypesIdsEnum.averagepool2d.value]
    nn_primary = [LayerTypesIdsEnum.dense.value]
    rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=[cnn_simple_mutation,
                                                                                 single_edge_mutation],
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=NNGraph, base_node_class=NNNode),
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
    print('save best graph structure...')
    optimized_network.show(path='../graph_patches_result.png')

    print('Best model structure:')
    for node in optimized_network.nodes:
        print(node)

    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_network)
    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3), epochs=epochs,
                          classes=num_of_classes, verbose=True)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate, is_multiclass=False)
    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed Accuracy is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = '../models/model_3cls.json'
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    # saving the weights of the model
    optimized_network.model.save_weights('../models/model_3cls.h5')
    return optimized_network


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    path = os.path.join(root, '_compressed_Generated_dataset.pickle')
    # A dataset that will be used as a train and test set during composition
    set_tf_compat()
    run_patches_classification(file_path=path, epochs=20)
