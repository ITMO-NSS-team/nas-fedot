import datetime
import os
import pathlib

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

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
import nas.data.load_images as loader
from nas.composer.nn_composer import NasComposer
from nas.data import KerasDataset
from nas.data.dataset import BaseNasDatasetBuilder
from nas.data.preprocessor import Preprocessor
from nas.data.setup_data import setup_data
from nas.graph.cnn.cnn_graph import NNNode
from nas.graph.cnn.resnet_builder import ResNetGenerator
from nas.graph.graph_builder import NNGraphBuilder
from nas.graph.node_factory import NNNodeFactory
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions
from nas.operations.validation_rules.cnn_val_rules import *
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

set_root(project_root())


def build_butterfly_cls(save_path=None):
    set_root(project_root())
    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    dataset_path = pathlib.Path('../datasets/butterfly_cls/train')
    data = loader.NasData.data_from_folder(dataset_path, task)

    cv_folds = None
    image_side_size = 64
    batch_size = 64
    epochs = 40
    optimization_epochs = 1
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
    nn_requirements = nas_requirements.ModelRequirements(input_data_shape=[image_side_size, image_side_size],
                                                         color_mode='color',
                                                         num_of_classes=data.num_classes,
                                                         conv_requirements=conv_requirements,
                                                         fc_requirements=fc_requirements,
                                                         primary=conv_layers_pool,
                                                         secondary=[LayersPoolEnum.dense],
                                                         epochs=epochs, batch_size=batch_size,
                                                         max_nn_depth=1, max_num_of_conv_layers=36)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(minutes=30),
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
        adapter=DirectAdapter(base_graph_class=NNGraph, base_node_class=NNNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    graph_generation_function = NNGraphBuilder()
    graph_generation_function.set_builder(ResNetGenerator(model_requirements=requirements.model_requirements))

    builder = ComposerBuilder(task).with_composer(NasComposer).with_optimizer(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(objective_function).with_optimizer_params(optimizer_parameters). \
        with_initial_pipelines(graph_generation_function.build()). \
        with_graph_generation_param(graph_generation_parameters)

    data_preprocessor = Preprocessor((image_side_size, image_side_size))

    data_transformer = BaseNasDatasetBuilder(dataset_cls=KerasDataset,
                                             batch_size=requirements.model_requirements.batch_size,
                                             shuffle=True).set_data_preprocessor(data_preprocessor)

    composer = builder.build()
    composer.set_data_transformer(data_transformer)

    optimized_network = composer.compose_pipeline(train_data)

    train_data, val_data = train_test_data_setup(train_data, shuffle_flag=True)

    train_generator = data_transformer.build(train_data, mode='train')
    val_generator = data_transformer.build(val_data, mode='val')
    # train_generator = setup_data(train_data, requirements.model_requirements.batch_size, data_preprocessor, 'train',
    #                              KerasDataset, True)
    # val_generator = setup_data(val_data, requirements.model_requirements.batch_size, data_preprocessor, 'train',
    #                            KerasDataset, True)

    optimized_network.model = ModelMaker(requirements.model_requirements.input_shape,
                                         optimized_network, converter.Struct, data.num_classes).build()
    optimized_network.fit(train_generator, val_generator, requirements=requirements, num_classes=train_data.num_classes,
                          verbose=1, optimization=False, shuffle=True)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test_data, data_transformer)
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
