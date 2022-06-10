import os
import random
import datetime
from typing import List, Union

import numpy as np
from nas.utils.var import PROJECT_ROOT, VERBOSE_VAL
from nas.utils.utils import set_root, set_tf_compat

from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from nas.composer.cnn_adapters import CustomGraphAdapter
from nas.composer.cnn_graph_node import CNNNode
from nas.composer.cnn_graph import CNNGraph
from nas.composer.cnn_graph_operator import generate_initial_graph

from fedot.core.log import default_log
from nas.data.load_images import from_images
from nas.composer.gp_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.gp_cnn_composer import GPNNComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_change_mutation, \
    single_drop_mutation, single_add_mutation
from nas.graph_cnn_mutations import cnn_simple_mutation, has_no_flatten_skip, flatten_check, \
    graph_has_wrong_structure, graph_has_several_starts
from nas.composer.metrics import calculate_validation_metric

root = PROJECT_ROOT
set_root(root)
random.seed(17)
np.random.seed(17)


def run_patches_classification(file_path, epochs: int = 1, verbose: Union[int, str] = 'auto',
                               initial_graph_struct: List[str] = None, timeout: datetime.timedelta = None,
                               per_class_limit: int = None):
    size = 120
    num_of_classes = 10
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes,
                                                          per_class_limit=per_class_limit)
    timeout = datetime.timedelta(hours=20) if not timeout else timeout

    secondary = ['serial_connection', 'dropout']
    conv_types = ['conv2d']
    pool_types = ['max_pool2d', 'average_pool2d']
    nn_primary = ['dense']
    rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, flatten_check, graph_has_wrong_structure,
             graph_has_several_starts]
    mutations = [cnn_simple_mutation, single_drop_mutation, single_edge_mutation, single_add_mutation,
                 single_change_mutation]
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=mutations,
        crossover_types=[CrossoverTypesEnum.subtree], regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CNNGraph, base_node_class=CNNNode),
        rules_for_constraint=rules)
    requirements = GPNNComposerRequirements(
        conv_kernel_size=[3, 3], conv_strides=[1, 1], pool_size=[2, 2], min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=128, image_size=[size, size],
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=secondary,
        primary=nn_primary, secondary=secondary, min_arity=2, max_arity=3,
        max_nn_depth=6, pop_size=10, num_of_generations=10, crossover_prob=0.8, mutation_prob=0.5,
        train_epochs_num=5, num_of_classes=num_of_classes, timeout=timeout)
    if not initial_graph_struct:
        initial_graph = None
    else:
        initial_graph = [generate_initial_graph(CNNGraph, CNNNode, initial_graph_struct, False)]
    optimiser = GPNNGraphOptimiser(
        initial_graph=initial_graph, requirements=requirements, graph_generation_params=graph_generation_params,
        metrics=metric_function, parameters=optimiser_parameters,
        log=default_log(logger_name='Bayesian', verbose_level=VERBOSE_VAL[verbose]))

    optimized_network = optimiser.compose(data=dataset_to_compose)
    print('Best model structure:')
    for node in optimized_network.nodes:
        print(node)
    optimized_network = graph_generation_params.adapter.restore(optimized_network)
    print('Save best model structure...')
    save_path = os.path.join(root, 'graph_10cls')
    optimiser.save(save_path, history=True, image=True)
    optimized_network.fit(input_data=dataset_to_compose, input_shape=(size, size, 3), epochs=epochs,
                          classes=num_of_classes, verbose=True)
    # The quality assessment for the obtained composite models
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(optimized_network, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    json_file = '../models/model_10cls.json'
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    # Saving the weights of the model
    optimized_network.model.save_weights('../models/model_10cls.h5')
    return optimized_network


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    file_path = os.path.join(root, 'datasets', '10cls_Generated_dataset')
    # A dataset that will be used as a train and test set during composition
    set_tf_compat()
    run_patches_classification(file_path=file_path, epochs=1, per_class_limit=150, verbose='auto')
