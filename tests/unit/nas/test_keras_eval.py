from nas.composer.gp_cnn_composer import GPNNComposerRequirements
from nas.composer.cnn_graph_node import NNNode
from nas.composer.cnn_graph import NNGraph
from nas.composer.cnn_graph_operator import random_conv_graph_generation
from nas.nn.graph_keras_eval import create_nn_model
from nas.utils.utils import project_root

root = project_root()
requirements = GPNNComposerRequirements(image_size=[120, 120], init_graph_with_skip_connections=True)
NODES_LIST = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
              'dense', 'dense']


def generate_graphs():
    graphs = []
    for _ in range(10):
        graphs.append(random_conv_graph_generation(NNGraph, NNNode, requirements))
    return graphs


def test_is_trainable_model():
    graphs = generate_graphs()
    success = True
    for graph in graphs:
        try:
            create_nn_model(graph, [120, 120, 3], 3)
        except ValueError:
            success = False
    assert success
