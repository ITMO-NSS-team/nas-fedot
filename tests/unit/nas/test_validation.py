from nas.graph_cnn_mutations import has_no_flatten_skip, graph_has_wrong_structure, flatten_check
from nas.composer.graph_gp_cnn_composer import NNGraph, NNNode
from nas.graph_cnn_gp_operators import generate_static_graph


def generate_graph():
    nodes_list = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'flatten', 'dense', 'dropout', 'dense', 'dense']
    graph = generate_static_graph(NNGraph, NNNode, nodes_list)
    return graph


def test_validation():
    rules_list = [has_no_flatten_skip, graph_has_wrong_structure, flatten_check]
    graph = generate_graph()
    successful_generation = False
    for rule in rules_list:
        successful_generation = rule(graph)
        if not successful_generation:
            break
    assert successful_generation
