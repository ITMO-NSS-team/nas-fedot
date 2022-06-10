import datetime
import os
from nas.utils.var import TESTING_ROOT
from nas.composer.cnn_graph_operator import random_conv_graph_generation, generate_initial_graph
from nas.composer.cnn_graph_node import CNNNode
from nas.composer.cnn_graph import CNNGraph
from nas.composer.gp_cnn_composer import GPNNComposerRequirements

from nas.graph_cnn_mutations import has_no_flatten_skip, flatten_check, graph_has_several_starts
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes

NODES_LIST = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
              'dense', 'dense']


def get_requirements():
    timeout = datetime.timedelta(hours=20)
    secondary = ['serial_connection', 'dropout']
    conv_types = ['conv2d']
    pool_types = ['max_pool2d', 'average_pool2d']
    nn_primary = ['dense']
    requirements = GPNNComposerRequirements(
        conv_kernel_size=[3, 3], conv_strides=[1, 1], pool_size=[2, 2], min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=64, image_size=[75, 75],
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=secondary,
        primary=nn_primary, secondary=secondary, min_arity=2, max_arity=3,
        max_nn_depth=6, pop_size=20, num_of_generations=5,
        crossover_prob=0, mutation_prob=0,
        train_epochs_num=5, num_of_classes=3, timeout=timeout)
    return requirements


def test_static_graph_params_and_generation():
    loaded_static_graph = CNNGraph.load(os.path.join(TESTING_ROOT, 'static_graph.json'))
    generated_static_graph = generate_initial_graph(CNNGraph, CNNNode, NODES_LIST, None, True, [0, 2, 6], 3)
    assert loaded_static_graph.operator.is_graph_equal(generated_static_graph)


def test_static_graph_generation():
    graph = generate_initial_graph(CNNGraph, CNNNode, NODES_LIST, None, False)
    successful_generation = False
    if not successful_generation:
        for val in [has_no_flatten_skip, flatten_check, has_no_cycle, has_no_self_cycled_nodes,
                    graph_has_several_starts, has_no_self_cycled_nodes]:
            assert val(graph)


def test_graph_has_right_dtype():
    for _ in range(10):
        graph = random_conv_graph_generation(CNNGraph, CNNNode, GPNNComposerRequirements(image_size=[120, 120]))
        assert type(graph) == CNNGraph


def test_validation():
    rules_list = [has_no_flatten_skip, flatten_check, has_no_cycle, has_no_self_cycled_nodes,
                  graph_has_several_starts, has_no_self_cycled_nodes]
    for _ in range(10):
        graph = random_conv_graph_generation(CNNGraph, CNNNode, GPNNComposerRequirements(image_size=[120, 120]))
        for rule in rules_list:
            try:
                successful_generation = rule(graph)
            except ValueError:
                continue
            assert successful_generation
