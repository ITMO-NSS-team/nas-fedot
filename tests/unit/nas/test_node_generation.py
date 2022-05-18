from fedot.core.dag.graph_operator import GraphOperator
from nas.composer.graph_gp_cnn_composer import NNNode
from nas.graph_cnn_gp_operators import create_dropout_node, create_secondary_node, create_conv2d_node, \
    create_primary_nn_node


# TODO add unit tests for checking nn_layers.py;
#  for checking is generated graph is valid(params and nodes) for both static and random graphs
#  (instead of compare nodes compare graphs);
#  for mutations(is new params valid);
#  for validation rules;
#  for load images func(pickle and folder cases);
#  Implement ability to turn on/off skip connections generation;
#  Rework skip connection generation
def test_dropout_node():
    generated_node = create_dropout_node(NNNode)
    print("!")
