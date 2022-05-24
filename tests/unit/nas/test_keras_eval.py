import os
from nas.composer.graph_gp_cnn_composer import NNGraph, NNNode, GPNNComposerRequirements
from nas.graph_cnn_gp_operators import random_conv_graph_generation, generate_initial_graph
from nas.graph_keras_eval import create_nn_model
from keras.models import model_from_json
from nas.patches.utils import project_root

root = project_root()
requirements = GPNNComposerRequirements(image_size=[120, 120], init_graph_with_skip_connections=True)
nodes_list = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
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


def test_create_nn_model():
    graph = generate_initial_graph(NNGraph, NNNode, nodes_list, None, True, [0, 2, 6], 3)
    file_path = os.path.join(root, 'tests', 'unit', 'test_data', 'test_model.json')
    json_file = open(file_path, 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    generated_model = create_nn_model(graph, [120, 120, 3], 3)
    for layer_first, layer_second in zip(loaded_model.layers, generated_model.layers):
        successful_comparison = layer_first.get_config() == layer_second.get_config()
        assert successful_comparison
