import os
import datetime
from typing import Union
from sklearn.metrics import confusion_matrix

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes

from nas.utils.utils import set_root, seed_all
from nas.utils.var import project_root, default_nodes_params
from nas.composer.nas_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.nas_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn.cnn_adapters import CustomGraphAdapter
from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.data.load_images import ImageDataLoader
from nas.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.mutations.cnn_val_rules import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.metrics.metrics import calculate_validation_metric, get_predictions
from nas.metrics.confusion_matrix import plot_confusion_matrix
from nas.composer.cnn.cnn_builder import CNNBuilder
import nas.callbacks.tb_metrics as nas_callbacks

set_root(project_root)
seed_all(14322)


# TODO extend initial approximation with desirable nodes params. Add ability to load graph as initial approximation
def run_test(verbose: Union[str, int] = 'auto', epochs: int = 1, save_directory: str = None, image_size: int = None,
             max_cnn_depth: int = None, max_nn_depth: int = None, batch_size: int = None, opt_epochs: int = 5,
             timeout: datetime.timedelta = None, has_skip_connections: bool = False, pop_size: int = 5,
             num_of_generations: int = 10, split_ratio: float = 0.8, **kwargs):
    """
    Run example with custom dataset and params
    :param verbose: verbose value: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = progress bar, 2 = one line per epoch
    :param epochs: number of train epochs
    :param save_directory: save path to optimized graph, history and graph struct
    :param image_size: dataset's image size. if None then size of first image in dataset will be picked as image size
    :param max_cnn_depth: max possible depth of convolutional part of the graph
    :param max_nn_depth: max possible depth of dense part of the graph
    :param batch_size: number of samples per gradient update. if None will be set to 16
    :param opt_epochs: number epochs for fitness estimate
    :param timeout: runtime restrictions
    :param has_skip_connections: parameter for initial graph. If True them graph with skip connections will be generated
    :param pop_size: population size for evolution
    :param num_of_generations: number of generations
    :param split_ratio: train/test train_data ratio
    """

    train_path = kwargs.get('train_path', None)
    test_path = kwargs.get('test_path', None)
    images_path = kwargs.get('images_path', None)
    csv_path = kwargs.get('csv_path', None)
    img_id = kwargs.get('img_id', 'id')
    labels_id = kwargs.get('target', 'target')
    initial_graph_struct = kwargs.get('initial_graph_struct', None)
    default_params = kwargs.get('default_params', None)
    samples_limit = kwargs.get('samples_limit', None)

    if csv_path and images_path:
        train_data = ImageDataLoader.image_from_csv(img_path=images_path, labels_path=csv_path, img_id=img_id,
                                                    target=labels_id, image_size=image_size)
    elif train_path:
        train_data = ImageDataLoader.from_directory(dir_path=train_path, image_size=image_size,
                                                    samples_limit=samples_limit)
    else:
        raise ValueError('Wrong dataset type')

    channel_num = train_data.features[0].shape[-1]
    image_size = train_data.features[0].shape[0] if not image_size else image_size
    input_shape = [image_size, image_size, channel_num] if image_size else train_data.features[0].shape

    if not test_path:
        train_data, test_data = train_test_data_setup(train_data, split_ratio, True)
    else:
        test_data = ImageDataLoader.from_directory(dir_path=test_path, image_size=image_size,
                                                   samples_limit=samples_limit)

    validation_rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, graph_has_several_starts,
                        graph_has_wrong_structure, flatten_check]
    mutations = [cnn_simple_mutation, single_drop_mutation, single_add_mutation,
                 single_change_mutation, single_edge_mutation]
    metric_func = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)
    requirements = GPNNComposerRequirements(input_shape=input_shape, pop_size=pop_size,
                                            num_of_generations=num_of_generations, max_num_of_conv_layers=max_cnn_depth,
                                            max_nn_depth=max_nn_depth, primary=['conv2d'], secondary=['dense'],
                                            batch_size=batch_size, timeout=timeout, epochs=opt_epochs,
                                            has_skip_connection=has_skip_connections,
                                            default_parameters=default_params)
    optimiser_params = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                  mutation_types=mutations,
                                                  crossover_types=[CrossoverTypesEnum.subtree],
                                                  regularization_type=RegularizationTypesEnum.none)
    graph_generation_params = GraphGenerationParams(
        adapter=CustomGraphAdapter(base_graph_class=CNNGraph, base_node_class=CNNNode),
        rules_for_constraint=validation_rules)

    optimiser = GPNNGraphOptimiser(initial_graph=initial_graph_struct, requirements=requirements,
                                   graph_generation_params=graph_generation_params, graph_builder=CNNBuilder,
                                   metrics=metric_func, parameters=optimiser_params, verbose=verbose,
                                   log=default_log(logger_name='Custom-run', verbose_level=4))

    print(f'\n================ Starting optimisation process with following params: population size: {pop_size}; '
          f'number of generations: {num_of_generations}; number of train epochs: {opt_epochs}; '
          f'image size: {input_shape}; batch size: {batch_size} ================\n')

    optimized_network = optimiser.compose(train_data=train_data, _test_data=test_data)
    optimized_network.fit(input_data=train_data, requirements=requirements, train_epochs=epochs, verbose=verbose)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test_data)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test_data, predicted_probabilities, predicted_labels)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')
    conf_matrix = confusion_matrix(test_data.target, predicted_labels.predict)
    plot_confusion_matrix(conf_matrix, test_data.supplementary_data, save=save_directory)

    if save_directory:
        print('save best graph structure...')
        optimiser.save(save_folder=save_directory, history=True, image=True)
    json_file = os.path.join(project_root, 'models', 'custom_example_model.json')
    model_json = optimized_network.model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)
    optimized_network.model.save_weights(os.path.join(project_root, 'models', 'custom_example_model.h5'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dir_root = '/home/staeros/datasets/Blood-Cell-Classification/train'
    img_path = '/home/staeros/datasets/Traffic-Sign/'
    test_root = '/home/staeros/datasets/Blood-Cell-Classification/test'
    save_path = os.path.join(project_root, 'Blood-Cell-Cls')
    initial_graph_nodes = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dense', 'dense']
    default_parameters = default_nodes_params
    run_test(verbose=1, epochs=1, save_directory=save_path, image_size=24, max_cnn_depth=4,
             max_nn_depth=5, batch_size=16, opt_epochs=1, initial_graph_struct=None, default_params=None,
             has_skip_connections=True, pop_size=1, num_of_generations=1, train_path=dir_root, test_path=test_root)
