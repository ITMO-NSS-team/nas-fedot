from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.model.nn import create_nn_model
from nas.utils.utils import project_root
from nas.utils.var import default_nodes_params
from nas.graph.nn_graph.cnn import NNGraphBuilder, CNNBuilder

root = project_root()
requirements = NNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                      num_of_generations=1, max_num_of_conv_layers=4,
                                      max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                      batch_size=4, epochs=1,
                                      has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2,
                                      batch_norm_prob=-1, dropout_prob=-1,
                                      default_parameters=default_nodes_params)
NODES_LIST = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dense', 'dense']


def generate_graphs():
    director = NNGraphBuilder()
    director.set_builder(CNNBuilder(NODES_LIST, requirements=requirements))
    graphs = []
    for _ in range(10):
        graph = director.create_nas_graph()
        graphs.append(graph)
    return graphs


def test_is_trainable_model():
    graphs = generate_graphs()
    success = True
    for graph in graphs:
        try:
            create_nn_model(graph, [120, 120, 3], 4)
        except ValueError:
            success = False
    assert success
