import os

from nas.composer.cnn.cnn_builder import CNNBuilder, NASDirector
from nas.composer.nas_cnn_composer import GPNNComposerRequirements
from nas.utils.var import default_nodes_params, tests_root
from nas.composer.cnn.cnn_graph import CNNGraph

NODES_LIST = ['conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'flatten', 'dense',
              'dense', 'dense']


requirements = GPNNComposerRequirements(input_shape=[120, 120, 3], pop_size=1,
                                        num_of_generations=1, max_num_of_conv_layers=4,
                                        max_nn_depth=3, primary=['conv2d'], secondary=['dense'],
                                        batch_size=4, epochs=1,
                                        has_skip_connection=True, skip_connections_id=[0, 2, 5], shortcuts_len=2,
                                        batch_norm_prob=-1, dropout_prob=-1,
                                        default_parameters=default_nodes_params)

director = NASDirector()
director.set_builder(CNNBuilder(NODES_LIST, requirements))
graph = CNNGraph.load(os.path.join(tests_root, 'graph_with_flatten_skip.json'))
graph.show()
graph.save(os.path.join(tests_root, 'static_graph'))
print('Done!')
