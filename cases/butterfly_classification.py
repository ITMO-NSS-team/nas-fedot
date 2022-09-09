import datetime
import pathlib

import tensorflow as tf
from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimizer import GPGraphOptimizerParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task

import nas.composer.nn_composer_requirements as nas_requirements
import nas.data.load_images as loader
from nas.composer.nn_composer import NNComposer
from nas.data.data_generator import DataGenerator
from nas.data.data_generator import Preprocessor
from nas.data.setup_data import setup_data
from nas.graph.cnn.cnn_builder import CNNBuilder
from nas.graph.cnn.cnn_graph import NNGraph, NNNode
from nas.graph.nn_graph_builder import NNGraphBuilder
from nas.graph.node_factory import NNNodeFactory
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions
from nas.operations.validation_rules.cnn_val_rules import has_no_flatten_skip, flatten_count, \
    graph_has_several_starts, graph_has_wrong_structure, unique_node_types
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.utils.utils import set_root, project_root

set_root(project_root())


def build_butterfly_cls(save_path=None):
    set_root(project_root())
    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    dataset_path = pathlib.Path('../datasets/butterfly_cls/train')
    data = loader.NNData.data_from_folder(dataset_path, task)

    cv_folds = 3
    image_side_size = 20
    batch_size = 8
    epochs = 1
    optimization_epochs = 1

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    data_requirements = nas_requirements.DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = nas_requirements.ConvRequirements(input_shape=[image_side_size, image_side_size],
                                                          color_mode='RGB',
                                                          min_filters=32, max_filters=128,
                                                          kernel_size=[[3, 3], [1, 1], [5, 5], [7, 7]],
                                                          conv_strides=[[1, 1]],
                                                          pool_size=[[2, 2]], pool_strides=[[2, 2]],
                                                          pool_types=['max_pool2d', 'average_pool2d'])
    fc_requirements = nas_requirements.FullyConnectedRequirements(min_number_of_neurons=32,
                                                                  max_number_of_neurons=128)
    nn_requirements = nas_requirements.NNRequirements(conv_requirements=conv_requirements,
                                                      fc_requirements=fc_requirements,
                                                      primary=['conv2d'], secondary=['dense'],
                                                      epochs=epochs, batch_size=batch_size,
                                                      max_nn_depth=3, max_num_of_conv_layers=10,
                                                      has_skip_connection=True
                                                      )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                           optimizer_requirements=optimizer_requirements,
                                                           nn_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(hours=200),
                                                           pop_size=3,
                                                           num_of_generations=3)

    mutations = [MutationTypesEnum.single_add, MutationTypesEnum.single_drop, MutationTypesEnum.single_edge,
                 MutationTypesEnum.single_change, MutationTypesEnum.simple]

    validation_rules = [has_no_flatten_skip, flatten_count, graph_has_several_starts, graph_has_wrong_structure,
                        has_no_cycle, has_no_self_cycled_nodes, unique_node_types]

    optimizer_parameters = GPGraphOptimizerParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                      mutation_types=mutations,
                                                      crossover_types=[CrossoverTypesEnum.subtree],
                                                      regularization_type=RegularizationTypesEnum.none)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NNGraph, base_node_class=NNNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements, DefaultChangeAdvisor()))

    graph_generation_function = NNGraphBuilder()
    graph_generation_function.set_builder(CNNBuilder(requirements=requirements))

    builder = ComposerBuilder(task).with_composer(NNComposer).with_optimiser(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(objective_function).with_optimiser_params(optimizer_parameters). \
        with_initial_pipelines_generation_function(graph_generation_function.create_nas_graph). \
        with_graph_generation_param(graph_generation_parameters).with_history('../_results/debug/master_2')
    composer = builder.build()

    transformations = [tf.convert_to_tensor]
    data_preprocessor = Preprocessor()
    data_preprocessor.set_image_size((image_side_size, image_side_size)).set_features_transformations(transformations)
    composer.set_preprocessor(data_preprocessor)

    optimized_network = composer.compose_pipeline(train_data)

    train_data, val_data = train_test_data_setup(train_data, shuffle_flag=True)

    train_generator = setup_data(train_data, requirements.nn_requirements.batch_size, data_preprocessor, 'train',
                                 DataGenerator, True)
    val_generator = setup_data(val_data, requirements.nn_requirements.batch_size, data_preprocessor, 'train',
                               DataGenerator, True)

    optimized_network.fit(train_generator, val_generator, requirements=requirements, num_classes=train_data.num_classes,
                          verbose=1, optimization=False, shuffle=True)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test_data, data_preprocessor)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test_data, predicted_probabilities, predicted_labels)

    if save_path:
        composer.save(path=save_path)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    path = f'_results/debug/master_2/{datetime.datetime.now().date()}'
    print(tf.config.list_physical_devices('GPU'))
    build_butterfly_cls(path)
