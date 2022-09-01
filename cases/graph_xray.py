import os
import datetime
import pathlib
from functools import partial
from sklearn.metrics import confusion_matrix

from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.data.data import DataTypesEnum
from fedot.core.optimisers.objective import Objective
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.adapters import DirectAdapter

from nas.data.dataloader import DataLoaderInputData, DataLoader, ImageDataset
from nas.data.split_data import SplitterGenerator
from nas.utils.utils import set_root, seed_all
from nas.utils.var import project_root
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.cnn.cnn_graph_node import NNNode
from nas.graph.cnn.cnn_graph import NNGraph
from nas.operations.evaluation.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.operations.evaluation.mutations import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions
from nas.operations.evaluation.metrics import plot_confusion_matrix
from nas.graph.cnn import CNNBuilder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

set_root(project_root)
seed_all(7482)


def run_nas(train, test, save, nn_requirements, epochs, validation_rules, mutations, objective_func, initial_graph,
            verbose, split_method, split_params):
    input_shape = train.supplementary_data.column_types['_image_size']

    metric_function = ClassificationMetricsEnum.logloss
    objective = Objective(metric_function)

    cnn_composer_parameters = NNComposerRequirements(input_shape=input_shape, **nn_requirements)

    optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                      mutation_types=mutations,
                                                      crossover_types=[CrossoverTypesEnum.subtree],
                                                      regularization_type=RegularizationTypesEnum.none)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NNGraph, base_node_class=NNNode),
        rules_for_constraint=validation_rules)

    optimizer = NNGraphOptimiser(initial_graph=initial_graph, requirements=cnn_composer_parameters,
                                 graph_generation_params=graph_generation_parameters, graph_builder=CNNBuilder,
                                 objective=objective, parameters=optimizer_parameters, verbose=verbose,
                                 save_path=save)

    print(
        f'\n\t Starting optimisation process with following params: '
        f'population size: {cnn_composer_parameters.pop_size}; '
        f'number of generations: {cnn_composer_parameters.num_of_generations}; '
        f'number of epochs: {cnn_composer_parameters.epochs}; '
        f'image size: {input_shape}; batch size: {cnn_composer_parameters.batch_size} \t\n')

    optimized_network = optimizer.optimise(train_data=train, split_method=split_method, split_params=split_params)
    optimized_network.fit(input_data=train, requirements=cnn_composer_parameters, train_epochs=epochs, verbose=verbose,
                          results_path=save)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test, predicted_probabilities, predicted_labels)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    if save:
        conf_matrix = confusion_matrix(test.target, predicted_labels.predict)
        plot_confusion_matrix(conf_matrix, test.supplementary_data.column_types['labels'], save=save)
        print('save best graph structure...')
        optimizer.save(history=True, image=True)
        json_file = os.path.join(save, 'model.json')
        model_json = optimized_network.model.to_json()
        with open(json_file, 'w') as f:
            f.write(model_json)
        _save_path = pathlib.Path(save, 'models', 'custom_example_model.h5')
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        optimized_network.model.save_weights(_save_path)


if __name__ == '__main__':
    data_root = '../datasets/butterfly_cls/train'
    folder_name = pathlib.Path(data_root).parts[2]
    save_path = pathlib.Path(f'../_results/{folder_name}/{datetime.datetime.now().date()}')
    task = Task(TaskTypesEnum.classification)

    img_size = 12
    batch_size = 64

    # TODO implement dataset augmentation func
    flip = partial(tf.image.random_flip_left_right, seed=1)
    saturation = partial(tf.image.random_saturation, lower=5, upper=10, seed=1)
    brightness = partial(tf.image.random_brightness, max_delta=.2, seed=1)
    contrast = partial(tf.image.random_contrast, lower=5, upper=10, seed=1)
    crop = partial(tf.image.random_crop, size=(img_size // 5, img_size // 5, 3), seed=1)
    resize = partial(tf.image.resize, size=(img_size, img_size))
    sup_data = SupplementaryData()
    sup_data.column_types = {'_image_size': [img_size, img_size, 3]}

    transformations = [resize]

    val_rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, graph_has_several_starts,
                 graph_has_wrong_structure, flatten_check]
    mutations_list = [cnn_simple_mutation, single_drop_mutation, single_add_mutation,
                      single_change_mutation, single_edge_mutation]
    metric = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    dataset = ImageDataset(data_root, batch_size, transformations)
    data_loader = DataLoader(dataset, True)
    true_labels = [f.parts[-1] for f in pathlib.Path(data_root).iterdir() if pathlib.Path(data_root).is_dir()]
    data = DataLoaderInputData.input_data_from_generator(data_loader, task, data_type=DataTypesEnum.image,
                                                         image_size=[img_size, img_size, 3], labels=true_labels)

    splitter = SplitterGenerator('holdout', train_size=.8, shuffle=True, random_state=42)
    for train, test in splitter.split(data):
        train_data, test_data = train, test

    conv_requirements = {'kernel_size': [3, 3], 'conv_strides': [1, 1], 'pool_size': [2, 2],
                         'pool_strides': [2, 2]}

    layer_requirements = {'min_num_of_neurons': 32, 'max_num_of_neurons': 256}

    requirements = {'pop_size': 10, 'num_of_generations': 15, 'max_num_of_conv_layers': 6,
                    'min_num_of_conv_layers': 4,
                    'max_nn_depth': 2, 'primary': ['conv2d'], 'secondary': ['dense'],
                    'batch_size': batch_size, 'epochs': 1, 'has_skip_connection': True,
                    'default_parameters': None, 'max_pipeline_fit_time': datetime.timedelta(hours=200)}
    requirements = requirements | conv_requirements | layer_requirements
    # TODO mb create dataclass for split params
    split_params = {'n_splits': 10, 'shuffle': True, 'random_state': 42}
    # sys.stdout = open(f'{folder_name}-{datetime.datetime.now().date()}-new', 'w')
    run_nas(train=train_data, test=test_data, save=save_path, nn_requirements=requirements,
            epochs=1, validation_rules=val_rules, mutations=mutations_list,
            objective_func=metric, initial_graph=None, verbose=1, split_method='k_fold', split_params=split_params)
    # sys.stdout.close()
