from tests.unit.nas.utility_functions import get_graph
from nas.operations.validation_rules.cnn_val_rules import *


def test_get_wrong_graph():
    graph = get_graph()
    node_to_add = graph.nodes[5]
    node_to_add.nodes_from = graph.nodes[3].nodes_from
    # graph.delete_node(node_to_delete)
    graph.update_node(graph.nodes[3], node_to_add)

    graph.save('/home/staeros/nas-fedot/tests/unit/test_data')
    assert True
