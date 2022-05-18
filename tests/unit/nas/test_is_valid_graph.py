import datetime

from nas.graph_cnn_gp_operators import random_conv_graph_generation, generate_static_graph
from nas.composer.graph_gp_cnn_composer import NNGraph, NNNode
from nas.composer.graph_gp_cnn_composer import GPNNComposerRequirements

from nas.graph_cnn_mutations import has_no_flatten_skip, flatten_check
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.dag.graph_operator import GraphOperator

NODE_LIST = ['conv2d', 'dropout', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'dropout', 'flatten',
             'dense', 'dropout', 'dense', 'dropout', 'dense']


def load_data_and_req():
    timeout = datetime.timedelta(hours=20)
    secondary = ['serial_connection', 'dropout']
    conv_types = ['conv2d']
    pool_types = ['max_pool2d', 'average_pool2d']
    nn_primary = ['dense']
    requirements = GPNNComposerRequirements(
        conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), min_num_of_neurons=20,
        max_num_of_neurons=128, min_filters=16, max_filters=64, image_size=[75, 75],
        conv_types=conv_types, pool_types=pool_types, cnn_secondary=secondary,
        primary=nn_primary, secondary=secondary, min_arity=2, max_arity=3,
        max_nn_depth=6, pop_size=20, num_of_generations=5,
        crossover_prob=0, mutation_prob=0,
        train_epochs_num=5, num_of_classes=3, timeout=timeout)
    return requirements


def test_random_graph_generation():
    requirements = load_data_and_req()
    successful_generation = False
    for _ in range(100):
        graph = random_conv_graph_generation(NNGraph, NNNode, requirements)
        if not successful_generation:
            for val in [has_no_flatten_skip, flatten_check, has_no_cycle, has_no_self_cycled_nodes]:
                try:
                    successful_generation = val(graph)
                except ValueError:
                    continue
        else:
            break
    assert successful_generation
    assert type(graph) == NNGraph


def test_static_graph_generation():
    graph = generate_static_graph(NNGraph, NNNode, NODE_LIST)
    successful_generation = False
    if not successful_generation:
        for val in [has_no_flatten_skip, flatten_check, has_no_cycle, has_no_self_cycled_nodes]:
            try:
                successful_generation = val(graph)
            except ValueError:
                continue
    assert successful_generation
    assert type(graph) == NNGraph


def test_is_graph_valid():
    static_graph = generate_static_graph(NNGraph, NNNode, NODE_LIST)
