import datetime
import os
import pathlib

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

import nas.data.nas_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from golem.core.adapter.adapter import DirectAdapter
from golem.core.optimisers.advisor import DefaultChangeAdvisor
from fedot.core.composer.composer_builder import ComposerBuilder
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.data.data_split import train_test_data_setup

from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task

import nas.composer.nn_composer_requirements as nas_requirements
from nas.composer.nn_composer import NasComposer
from nas.data import KerasDataset
from nas.data.dataset.builder import ImageDatasetBuilder
from nas.data.preprocessor import Preprocessor
from nas.graph.cnn_graph import NasNode
from nas.graph.graph_builder.resnet_builder import ResNetGenerator
from nas.graph.graph_builder.base_graph_builder import BaseGraphBuilder
from nas.graph.node.node_factory import NNNodeFactory
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions
from nas.operations.validation_rules.cnn_val_rules import *
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root
from nas.data.nas_data import InputDataNN

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

set_root(project_root())


def build_butterfly_cls(save_path=None):
    cv_folds = None
    image_side_size = 128
    batch_size = 64
    epochs = 10
    optimization_epochs = 1

    set_root(project_root())
    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    dataset_path = pathlib.Path(f'{project_root()}/../datasets/butterfly_cls/train')
    data = InputDataNN.data_from_folder(dataset_path, task)

    conv_layers_pool = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,
                        LayersPoolEnum.conv2d_7x7]

    mutations = [MutationTypesEnum.single_add, MutationTypesEnum.single_drop, MutationTypesEnum.single_edge,
                 MutationTypesEnum.single_change]

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    fc_requirements = nas_requirements.BaseLayerRequirements(min_number_of_neurons=32,
                                                             max_number_of_neurons=128)
    conv_requirements = nas_requirements.ConvRequirements(
        min_number_of_neurons=32, max_number_of_neurons=256,
        conv_strides=[[1, 1]],
        pool_size=[[2, 2]], pool_strides=[[2, 2]])
    model_requirements = nas_requirements.ModelRequirements(input_data_shape=[image_side_size, image_side_size],
                                                            color_mode='color',
                                                            num_of_classes=data.num_classes,
                                                            conv_requirements=conv_requirements,
                                                            fc_requirements=fc_requirements,
                                                            primary=conv_layers_pool,
                                                            secondary=[LayersPoolEnum.dense],
                                                            epochs=epochs, batch_size=batch_size,
                                                            max_nn_depth=1, max_num_of_conv_layers=36)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=model_requirements,
                                                           timeout=datetime.timedelta(minutes=5),
                                                           num_of_generations=3,
                                                           early_stopping_iterations=100,
                                                           early_stopping_timeout=float(datetime.timedelta(minutes=30).
                                                                                        total_seconds()),
                                                           n_jobs=1,
                                                           cv_folds=cv_folds)

    validation_rules = [ConvNetChecker.check_cnn, has_no_cycle, has_no_self_cycled_nodes, ]

    optimizer_parameters = GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                 mutation_types=mutations,
                                                 crossover_types=[CrossoverTypesEnum.subtree],
                                                 pop_size=10,
                                                 regularization_type=RegularizationTypesEnum.none)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(ResNetGenerator(model_requirements=requirements.model_requirements))

    builder = ComposerBuilder(task).with_composer(NasComposer).with_optimizer(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(objective_function).with_optimizer_params(optimizer_parameters). \
        with_initial_pipelines(graph_generation_function.build()). \
        with_graph_generation_param(graph_generation_parameters)

    data_preprocessor = Preprocessor()

    dataset_builder = ImageDatasetBuilder(dataset_cls=KerasDataset, image_size=(image_side_size, image_side_size),
                                          batch_size=requirements.model_requirements.batch_size,
                                          shuffle=True).set_data_preprocessor(data_preprocessor)

    composer = builder.build()
    composer.set_dataset_builder(dataset_builder)

    optimized_network = composer.compose_pipeline(train_data)

    train_data, val_data = train_test_data_setup(train_data, shuffle_flag=True)

    train_generator = dataset_builder.build(train_data, mode='train')
    val_generator = dataset_builder.build(val_data, mode='val')

    optimized_network.compile_model(model_requirements.input_shape, 'categorical_crossentropy',
                                    metrics=[tf.metrics.Accuracy()], optimizer=tf.keras.optimizers.Adam,
                                    n_classes=model_requirements.num_of_classes)

    optimized_network.fit(train_generator, val_generator, model_requirements.epochs, model_requirements.batch_size,
                          [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                                            mode='min'),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=3,
                                                                verbose=1,
                                                                min_delta=1e-4, mode='min')])

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test_data, dataset_builder)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test_data, predicted_probabilities, predicted_labels)

    if save_path:
        composer.save(path=save_path)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    path = f'_results/debug/master_2/{datetime.datetime.now().date()}'
    build_butterfly_cls(path)
