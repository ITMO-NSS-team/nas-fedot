import random
from typing import List, Optional

from nas.graph.cnn.cnn_graph import NNGraph, CNNNode
from nas.composer.nn_composer_requirements import NNComposerRequirements

# TODO mb need to move add dense layers from keras_eval and increase the number of nn layers in requirements


def get_layer_params(layer_type: str, requirements=None):
    layer_params = _get_random_layer_params(layer_type, requirements)
    return layer_params


def _get_conv2d_requirements(requirements: NNComposerRequirements):
    conv_node_type = random.choice(requirements.nn_requirements.primary)
    activation = random.choice(requirements.nn_requirements.activation_types).value
    kernel_size = random.choice(requirements.nn_requirements.conv_requirements.kernel_size)
    conv_strides = random.choice(requirements.nn_requirements.conv_requirements.conv_strides)
    num_of_filters = random.choice(requirements.nn_requirements.conv_requirements.filters)
    pool_size = random.choice(requirements.nn_requirements.conv_requirements.pool_size)
    pool_strides = random.choice(requirements.nn_requirements.conv_requirements.pool_strides)
    pool_type = random.choice(requirements.nn_requirements.conv_requirements.pool_types)
    return {'layer_type': conv_node_type, 'activation': activation, 'kernel_size': kernel_size,
            'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
            'pool_strides': pool_strides, 'pool_type': pool_type}


def _get_random_layer_params(layer_type: str, requirements: NNComposerRequirements):
    layer_params = {'n_jobs': 1}
    if layer_type == 'conv2d':
        layer_params = _get_conv2d_requirements(requirements) | layer_params
    elif layer_type == 'serial_connection':
        layer_params = {'layer_type': layer_type} | layer_params
    elif layer_type == 'dropout':
        drop_value = random.randint(1, (requirements.nn_requirements.max_drop_size * 10)) / 10
        layer_params = {'drop': drop_value} | layer_params
    elif layer_type == 'batch_normalization':
        momentum = random.uniform(0, 1)
        epsilon = random.uniform(0, 1)
        layer_params = {'momentum': momentum, 'epsilon': epsilon}
    elif layer_type == 'dense':
        activation = random.choice(requirements.nn_requirements.activation_types).value
        neurons = random.choice(requirements.nn_requirements.fc_requirements.neurons_num)
        layer_params = {'layer_type': layer_type, 'neurons': neurons, 'activation': activation} | layer_params
    return layer_params


def _generate_random_struct(requirements: NNComposerRequirements) -> List[str]:
    """ function for generate random graph structure if initial structure isn't specified"""

    conv_depth = random.randint(requirements.nn_requirements.min_num_of_conv_layers,
                                requirements.nn_requirements.max_num_of_conv_layers)
    nn_depth = random.randint(requirements.nn_requirements.min_nn_depth, requirements.nn_requirements.max_nn_depth)
    struct = ['conv2d']
    for i in range(1, conv_depth + nn_depth):
        if i < conv_depth:
            node = random.choice(requirements.nn_requirements.primary) if i != conv_depth - 1 else 'flatten'
        else:
            node = random.choice(requirements.nn_requirements.secondary)
        struct.append(node)
    return struct


def _add_skip_connections(graph: NNGraph, params):
    skip_connections_id = params[0]
    shortcut_len = params[1]
    for current_node in skip_connections_id:
        is_first_conv = current_node <= graph.cnn_depth
        is_second_conv = current_node + shortcut_len < graph.cnn_depth
        if is_first_conv == is_second_conv and (current_node + shortcut_len) < len(graph.nodes):
            graph.nodes[current_node + shortcut_len].nodes_from.append(graph.nodes[current_node])
        else:
            print('Wrong connection. Connection dropped.')


class CNNBuilder:

    def __init__(self, nodes_list: Optional[List[str]] = None,
                 requirements: NNComposerRequirements = None):
        self.nodes = list(nodes_list) if nodes_list else _generate_random_struct(requirements)
        self.requirements = requirements

    # TODO fix
    def _skip_connection_params(self):
        connections = set()
        skips_len = random.randint(2, len(self.nodes) // 2)
        max_number_of_skips = len(self.nodes) // 3
        for _ in range(max_number_of_skips):
            node_id = random.randint(0, len(self.nodes))
            connections.add(node_id)
        return connections, skips_len

    def _add_node(self, node_type, nodes_from):
        node_params = get_layer_params(node_type, self.requirements)
        node = CNNNode(content={'name': node_type, 'params': node_params}, nodes_from=nodes_from)
        if node_type == 'flatten':
            return node
        if random.random() > self.requirements.nn_requirements.batch_norm_prob:
            batch_norm_params = get_layer_params('batch_normalization', self.requirements)
            node.content['params'] = node.content['params'] | batch_norm_params
        if random.random() > self.requirements.nn_requirements.dropout_prob:
            dropout_params = get_layer_params('dropout', self.requirements)
            node.content['params'] = node.content['params'] | dropout_params
        return node

    def build_graph(self) -> NNGraph:
        graph = NNGraph()
        parent_node = None
        for node_name in self.nodes:
            node = self._add_node(node_name, parent_node)
            parent_node = [node]
            graph.add_node(node)
        if self.requirements.nn_requirements.has_skip_connection:
            _add_skip_connections(graph, self._skip_connection_params())
        return graph


if __name__ == '__main__':
    r = NNComposerRequirements(primary=None, secondary=None, input_shape=[120, 120, 3])
    graph = CNNBuilder(NNGraph, requirements=r).build_graph()
    print("DONE!")
