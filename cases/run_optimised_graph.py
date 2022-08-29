import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import pathlib
from functools import partial

import tensorflow as tf

from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.data.data import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes

from nas.data.dataloader import DataLoaderInputData, DataLoader, ImageDataset
from nas.data.split_data import generator_train_test_split
from nas.utils.utils import set_root, seed_all
from nas.utils.var import project_root
from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.nn_graph.cnn.cnn_graph import NNGraph
from nas.operations.evaluation.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.operations.evaluation.mutations import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.operations.evaluation.metrics.metrics import calculate_validation_metric, get_predictions

set_root(project_root)
seed_all(7482)


def run_nas(train, test, save, nn_requirements, epochs, verbose):
    input_shape = train.supplementary_data.column_types['_image_size']
    nn_requirements = NNComposerRequirements(input_shape=input_shape, **nn_requirements)
    save_path = save / 'optimized'
    save_path.mkdir(parents=True, exist_ok=True)

    optimized_network = NNGraph.load(save / 'model.json')
    optimized_network.fit(input_data=train, requirements=nn_requirements, train_epochs=epochs, verbose=verbose,
                          results_path=save_path)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test, predicted_probabilities, predicted_labels)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')


if __name__ == '__main__':
    data_root = '../datasets/butterfly_cls/train'
    folder_name = pathlib.Path(data_root).parts[2]
    save_path = pathlib.Path(f'../_results/{folder_name}/2022-07-28')
    task = Task(TaskTypesEnum.classification)

    img_size = 200
    batch_size = 8

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
    train_data, test_data = generator_train_test_split(data, .8, True)

    conv_requirements = {'kernel_size': [3, 3], 'conv_strides': [1, 1], 'pool_size': [2, 2],
                         'pool_strides': [2, 2]}

    layer_requirements = {'min_num_of_neurons': 32, 'max_num_of_neurons': 256}

    requirements = {'pop_size': 5, 'num_of_generations': 15, 'max_num_of_conv_layers': 50, 'min_num_of_conv_layers': 10,
                    'max_nn_depth': 2, 'primary': ['conv2d'], 'secondary': ['dense'],
                    'batch_size': batch_size, 'epochs': 5, 'has_skip_connection': True,
                    'default_parameters': None, 'max_pipeline_fit_time': datetime.timedelta(hours=200)}
    requirements = requirements | conv_requirements | layer_requirements
    sys.stdout = open(f'{folder_name}-{datetime.datetime.now().date()}-logs', 'w')
    run_nas(train=train_data, test=test_data, save=save_path, nn_requirements=requirements,
            epochs=30, batch_size=batch_size, validation_rules=val_rules, mutations=mutations_list,
            objective_func=metric, initial_graph=None, verbose=1)
    sys.stdout.close()
