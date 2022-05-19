import os
from nas.var import TESTING_ROOT
from nas.composer.graph_gp_cnn_composer import NNNode, NNGraph, GPNNComposerRequirements
from nas.graph_cnn_gp_operators import generate_initial_graph, random_conv_graph_generation

NODES_LIST = ['conv2d', 'conv2d', 'dropout', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense', 'dropout',
              'dense', 'dense']


# TODO add unit tests for checking nn_layers.py;
#  for checking is generated graph is valid(params and nodes) for both static and random graphs
#  (instead of compare nodes compare graphs) --- DONE;
#  for mutations(are new params valid);
#  for validation rules;
#  for load images func(pickle and folder cases);
#  Implement ability to turn on/off skip connections generation;
#  Rework skip connection generation
