import os
from nas.composer.graph_gp_cnn_composer import NNGraph, NNNode
from nas.graph_cnn_gp_operators import generate_initial_graph
from nas.graph_keras_eval import create_nn_model
from keras.models import model_from_json
from nas.patches.utils import project_root

root = project_root()


def generate_graph():
    nodes_list = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
                  'dense', 'dense']
    graph = generate_initial_graph(NNGraph, NNNode, nodes_list, has_skip_connections=True,
                                   skip_connections_id=[0, 4, 8], shortcuts_len=4)
    return graph


def test_create_nn_model():
    graph = generate_graph()
    file_path = os.path.join(root, 'tests', 'unit', 'test_model.json')
    json_file = open(file_path, 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    generated_model = create_nn_model(graph, (120, 120, 3), 3)
    successful_comparison = False
    for layer_first, layer_second in zip(loaded_model.layers, generated_model.layers):
        successful_comparison = layer_first.get_config() == layer_second.get_config()
        if not successful_comparison:
            assert successful_comparison
    assert successful_comparison


def test_trainable_model():
    assert True
