import datetime
import os
import pathlib

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

from nas.model.model_interface import ModelTF
from nas.model.tensorflow.base_model import BaseNasTFModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

import nas.composer.requirements as nas_requirements
from nas.composer.nn_composer import NasComposer
from nas.data import KerasDataset
from nas.data.dataset.builder import ImageDatasetBuilder
from nas.data.preprocessor import Preprocessor
from nas.graph.BaseGraph import NasNode
from nas.graph.builder.resnet_builder import ResNetBuilder
from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.node.node_factory import NNNodeFactory
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions
from nas.operations.validation_rules.cnn_val_rules import *
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root
from nas.data.nas_data import InputDataNN

tf.config.experimental.set_memory_growth = True
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

set_root(project_root())


def build_butterfly_cls(save_path=None):
    cv_folds = None
    image_side_size = 24
    batch_size = 32
    epochs = 5
    optimization_epochs = 1
    num_of_generations = 2
    population_size = 2

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
                                                            max_nn_depth=1, max_num_of_conv_layers=40)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=model_requirements,
                                                           timeout=datetime.timedelta(minutes=60),
                                                           num_of_generations=num_of_generations,
                                                           early_stopping_iterations=100,
                                                           early_stopping_timeout=float(datetime.timedelta(minutes=30).
                                                                                        total_seconds()),
                                                           n_jobs=1,
                                                           cv_folds=cv_folds)

    data_preprocessor = Preprocessor()
    dataset_builder = ImageDatasetBuilder(dataset_cls=KerasDataset, image_size=(image_side_size, image_side_size),
                                          batch_size=requirements.model_requirements.batch_size,
                                          shuffle=True).set_data_preprocessor(data_preprocessor)

    # TODO may be add additional parameters to requirements class instead of passing them directly to model init method.
    model_interface = ModelTF(model_class=BaseNasTFModel, data_transformer=dataset_builder,
                              lr=1e-4, optimizer=tf.keras.optimizers.Adam,
                              metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')],
                              loss='categorical_crossentropy')

    validation_rules = [model_has_no_conv_layers, model_has_wrong_number_of_flatten_layers, model_has_several_starts,
                        has_no_cycle, has_no_self_cycled_nodes, check_dimensions]

    optimizer_parameters = GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                 mutation_types=mutations,
                                                 crossover_types=[CrossoverTypesEnum.subtree],
                                                 pop_size=population_size,
                                                 regularization_type=RegularizationTypesEnum.none)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(ResNetBuilder(model_requirements=requirements.model_requirements,
                                                        model_type='resnet_34'))

    builder = ComposerBuilder(task).with_composer(NasComposer).with_optimizer(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(objective_function).with_optimizer_params(optimizer_parameters). \
        with_initial_pipelines(graph_generation_function.build()). \
        with_graph_generation_param(graph_generation_parameters)

    composer = builder.build()

    new_train_data, new_test_data = train_test_data_setup(train_data, shuffle_flag=True)
    composer.set_model_interface(model_interface)
    optimized_network = composer.compose_pipeline(train_data)

    optimized_network.model_interface = model_interface
    optimized_network.model_interface.compile_model(optimized_network, train_data.num_classes)

    optimized_network.fit(new_train_data, new_test_data, epochs=epochs, batch_size=batch_size)
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
