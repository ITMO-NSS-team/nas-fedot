import datetime
import pathlib

import numpy as np
from albumentations.pytorch import ToTensorV2
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from golem.core.adapter.adapter import DirectAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.advisor import DefaultChangeAdvisor
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams
from sklearn.metrics import log_loss, roc_auc_score, f1_score

import albumentations as A
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

import nas.composer.requirements as nas_requirements
from nas.composer.nn_composer import NNComposer
from nas.data.dataset.builder import ImageDatasetBuilder
from nas.data.dataset.torch_dataset import TorchDataset
from nas.data.nas_data import InputDataNN
from nas.data.preprocessor import Preprocessor
from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.builder.resnet_builder import ResNetBuilder
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.node_factory import NNNodeFactory
from nas.model.constructor import ModelConstructor
from nas.operations.validation_rules.cnn_val_rules import *
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root

set_root(project_root())


def build_butterfly_cls(save_path=None):
    cv_folds = None
    image_side_size = 90
    batch_size = 16
    epochs = 20
    optimization_epochs = 1
    num_of_generations = 5
    population_size = 5

    set_root(project_root())
    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    dataset_path = pathlib.Path(project_root(), '../../datasets/butterfly')
    data = InputDataNN.data_from_folder(dataset_path, task)

    conv_layers_pool = [LayersPoolEnum.conv2d, LayersPoolEnum.pooling2d, LayersPoolEnum.adaptive_pool2d]

    mutations = [MutationTypesEnum.single_add, MutationTypesEnum.single_drop, MutationTypesEnum.single_edge,
                 MutationTypesEnum.single_change]

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    fc_requirements = nas_requirements.BaseLayerRequirements(min_number_of_neurons=32,
                                                             max_number_of_neurons=256)
    conv_requirements = nas_requirements.ConvRequirements(
        min_number_of_neurons=32, max_number_of_neurons=256,
        conv_strides=[1],
        pool_size=[2], pool_strides=[2])

    model_requirements = nas_requirements.ModelRequirements(input_data_shape=[image_side_size, image_side_size],
                                                            color_mode='color',
                                                            num_of_classes=data.num_classes,
                                                            conv_requirements=conv_requirements,
                                                            fc_requirements=fc_requirements,
                                                            primary=conv_layers_pool,
                                                            secondary=[LayersPoolEnum.linear],
                                                            epochs=epochs,
                                                            batch_size=batch_size,
                                                            max_nn_depth=1,
                                                            max_num_of_conv_layers=34)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=model_requirements,
                                                           timeout=datetime.timedelta(hours=100),
                                                           num_of_generations=num_of_generations,
                                                           early_stopping_iterations=100,
                                                           early_stopping_timeout=float(datetime.timedelta(minutes=30).
                                                                                        total_seconds()),
                                                           parallelization_mode='sequential',
                                                           n_jobs=1,
                                                           cv_folds=cv_folds)

    data_preprocessor = [A.RandomCrop(width=image_side_size, height=image_side_size),
                         A.HorizontalFlip(),
                         A.RandomBrightnessContrast(),
                         A.ToFloat(),
                         ToTensorV2()]
    dataset_builder = ImageDatasetBuilder(dataset_cls=TorchDataset, image_size=(image_side_size, image_side_size),
                                          shuffle=True).set_data_preprocessor(data_preprocessor)

    model_trainer = ModelConstructor(model_class=NASTorchModel, trainer=NeuralSearchModel, device='cuda:0',
                                     loss_function=CrossEntropyLoss(), optimizer=AdamW)

    validation_rules = [model_has_several_starts, model_has_no_conv_layers, model_has_wrong_number_of_flatten_layers,
                        model_has_several_roots,
                        has_no_cycle, has_no_self_cycled_nodes, skip_has_no_pools, model_has_dim_mismatch]

    optimizer_parameters = GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                 mutation_types=mutations,
                                                 crossover_types=[CrossoverTypesEnum.subtree],
                                                 pop_size=population_size,
                                                 regularization_type=RegularizationTypesEnum.none)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    builder = ResNetBuilder(model_requirements=requirements.model_requirements, model_type='resnet_18')
    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(builder)

    builder = ComposerBuilder(task).with_composer(NNComposer).with_optimizer(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(objective_function).with_optimizer_params(optimizer_parameters). \
        with_initial_pipelines(graph_generation_function.build()). \
        with_graph_generation_param(graph_generation_parameters)

    composer = builder.build()
    composer.set_trainer(model_trainer)
    composer.set_dataset_builder(dataset_builder)

    optimized_network = composer.compose_pipeline(train_data)

    if save_path:
        composer.save(path=save_path)

    trainer = model_trainer.build([image_side_size, image_side_size, 3], test_data.num_classes,
                                  optimized_network)

    train_data, val_data = train_test_data_setup(train_data, split_ratio=.7, shuffle_flag=False)
    train_dataset = DataLoader(dataset_builder.build(train_data), batch_size=requirements.model_requirements.batch_size,
                               shuffle=True)
    val_data = DataLoader(dataset_builder.build(val_data), batch_size=requirements.model_requirements.batch_size,
                          shuffle=True)
    test_dataset = DataLoader(dataset_builder.build(test_data), batch_size=requirements.model_requirements.batch_size,
                              shuffle=False)
    trainer.fit_model(train_dataset, val_data, epochs)
    predictions, targets = trainer.predict(test_dataset)
    history = composer.history

    loss = log_loss(targets, predictions)
    roc = roc_auc_score(targets, predictions, multi_class='ovo')
    f1 = f1_score(targets, np.argmax(predictions, axis=-1), average='weighted')

    print(f'Composed ROC AUC is {round(roc, 3)}')
    print(f'Composed LOG LOSS is {round(loss, 3)}')
    print(f'Composed F1 is {round(f1, 3)}')


if __name__ == '__main__':
    path = f'./_results/debug/master_2/{datetime.datetime.now().date()}'
    build_butterfly_cls(path)
