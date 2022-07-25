import os

from sklearn.metrics import confusion_matrix

from fedot.core.log import default_log
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.operators.mutation import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.adapters import DirectAdapter

from nas.utils.utils import set_root, seed_all
from nas.utils.var import project_root, default_nodes_params
from nas.composer.nas_cnn_optimiser import GPNNGraphOptimiser
from nas.composer.nas_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn.cnn_graph_node import CNNNode
from nas.composer.cnn.cnn_graph import CNNGraph
from nas.data.load_images import ImageDataLoader
from nas.mutations.nas_cnn_mutations import cnn_simple_mutation
from nas.mutations.cnn_val_rules import flatten_check, has_no_flatten_skip, graph_has_several_starts, \
    graph_has_wrong_structure
from nas.metrics.metrics import calculate_validation_metric, get_predictions
from nas.metrics.confusion_matrix import plot_confusion_matrix
from nas.composer.cnn.cnn_builder import CNNBuilder

set_root(project_root)
seed_all(14322)


def run_nas(train_data, test_data, val_split, save, nn_requirements, epochs, batch_size,
            validation_rules, mutations, objective_func, initial_graph, verbose):
    if not test_data:
        train_data, test_data = train_test_data_setup(train_data, val_split, True)

    input_shape = train_data.supplementary_data['image_size']
    nn_requirements = GPNNComposerRequirements(input_shape=input_shape, **nn_requirements)

    optimiser_params = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                  mutation_types=mutations,
                                                  crossover_types=[CrossoverTypesEnum.subtree],
                                                  regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CNNGraph, base_node_class=CNNNode),
        rules_for_constraint=validation_rules)

    optimiser = GPNNGraphOptimiser(initial_graph=initial_graph, requirements=nn_requirements,
                                   graph_generation_params=graph_generation_params, graph_builder=CNNBuilder,
                                   metrics=objective_func, parameters=optimiser_params, verbose=verbose,
                                   log=default_log(logger_name='10cls-run', verbose_level=4))

    print(f'\n\t Starting optimisation process with following params: population size: {nn_requirements.pop_size}; '
          f'number of generations: {nn_requirements.num_of_generations}; number of epochs: {nn_requirements.epochs}; '
          f'image size: {input_shape}; batch size: {batch_size} \t\n')

    optimized_network = optimiser.compose(train_data=train_data, test_data=test_data)
    optimized_network.fit(input_data=train_data, requirements=nn_requirements, train_epochs=epochs, verbose=verbose)

    predicted_labels, predicted_probabilities = get_predictions(optimized_network, test_data)
    roc_on_valid_evo_composed, log_loss_on_valid_evo_composed, accuracy_score_on_valid_evo_composed = \
        calculate_validation_metric(test_data, predicted_probabilities, predicted_labels)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed LOG LOSS is {round(log_loss_on_valid_evo_composed, 3)}')
    print(f'Composed ACCURACY is {round(accuracy_score_on_valid_evo_composed, 3)}')

    if save:
        conf_matrix = confusion_matrix(test_data.target, predicted_labels.predict)
        plot_confusion_matrix(conf_matrix, test_data.supplementary_data['labels'], save=save)
        print('save best graph structure...')
        optimiser.save(history=True, image=True)
        json_file = os.path.join(project_root, save, 'model.json')
        model_json = optimized_network.model.to_json()
        with open(json_file, 'w') as f:
            f.write(model_json)
        optimized_network.model.save_weights(os.path.join(project_root, 'models', 'custom_example_model.h5'))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data_root = '../datasets/10cls_Generated_dataset'
    labels_root = '../datasets/10cls_Generated_dataset/labels_10.json'
    save_path = './results/10-cls'
    task = Task(TaskTypesEnum.classification)
    train_data = ImageDataLoader.images_from_json(task, data_root, labels_root, None, None, 128, None)

    val_rules = [has_no_self_cycled_nodes, has_no_cycle, has_no_flatten_skip, graph_has_several_starts,
                 graph_has_wrong_structure, flatten_check]
    mutations_list = [cnn_simple_mutation, single_drop_mutation, single_add_mutation,
                      single_change_mutation, single_edge_mutation]
    metric = MetricsRepository().metric_by_id(ClassificationMetricsEnum.logloss)

    initial_graph_nodes = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dense', 'dense']
    requirements = {'pop_size': 5, 'num_of_generations': 10, 'max_num_of_conv_layers': 5,
                    'max_nn_depth': 2, 'primary': ['conv2d'], 'secondary': ['dense'],
                    'batch_size': 24, 'epochs': 5, 'has_skip_connection': True,
                    'default_parameters': default_nodes_params}

    run_nas(train_data=train_data, test_data=None, val_split=.7, nn_requirements=requirements, epochs=1, batch_size=10,
            save=save_path, validation_rules=val_rules, mutations=mutations_list, objective_func=metric,
            initial_graph=None, verbose=1)
