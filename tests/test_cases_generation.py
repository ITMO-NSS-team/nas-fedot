import os

from nas.graph.cnn import CNNBuilder, NNGraphBuilder
from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.utils.var import default_nodes_params, tests_root
from nas.graph.cnn.cnn_graph import NNGraph

NODES_LIST = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense',
              'dense', 'dense']


requirements = NNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                      num_of_generations=1, max_num_of_conv_layers=4,
                                      max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                      batch_size=4, epochs=1,
                                      has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2,
                                      batch_norm_prob=-1, dropout_prob=-1,
                                      default_parameters=default_nodes_params)

director = NNGraphBuilder()
director.set_builder(CNNBuilder(NODES_LIST, requirements))
graph = NNGraph.load(os.path.join(tests_root, 'graph_with_flatten_skip.json'))
graph.show()
graph.save()
print('Done!')
