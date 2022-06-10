import random
from typing import List, Callable, Union, Optional

from nas.composer.cnn_graph import CNNGraph, CNNNode
from nas.composer.gp_cnn_composer import GPNNComposerRequirements

# TODO mb need to move add dense layers from keras_eval and increase the number of nn layers in requirements
from nas.utils.var import DEFAULT_NODES_PARAMS


def _get_layer_params(layer_type: str, requirements=None):
    if requirements is not None:
        layer_params = _get_random_layer_params(layer_type, requirements)
    else:
        layer_params = DEFAULT_NODES_PARAMS[layer_type]
    return layer_params


def _get_conv2d_requirements(requirements):
    conv_node_type = random.choice(requirements.primary)
    activation = random.choice(requirements.activation_types).value
    kernel_size = requirements.conv_kernel_size
    conv_strides = requirements.conv_strides
    num_of_filters = random.choice(requirements.filters)
    image_size = requirements.input_shape
    pool_size = requirements.pool_size
    pool_strides = requirements.pool_strides
    pool_type = random.choice(requirements.pool_types)
    requirements.input_shape = image_size
    return {'layer_type': conv_node_type, 'activation': activation, 'kernel_size': kernel_size,
            'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
            'pool_strides': pool_strides, 'pool_type': pool_type}


def _get_random_layer_params(layer_type: str, requirements):
    layer_params = None
    if layer_type == 'conv2d':
        layer_params = _get_conv2d_requirements(requirements)
    elif layer_type == 'serial_connection':
        layer_params = {'layer_type': layer_type}
    elif layer_type == 'dropout':
        drop_value = random.randint(1, (requirements.max_drop_size * 10)) / 10
        layer_params = {'drop': drop_value}
    elif layer_type == 'batch_normalization':
        momentum = random.uniform(0, 1)
        epsilon = random.uniform(0, 1)
        layer_params = {'momentum': momentum, 'epsilon': epsilon}
    elif layer_type == 'dense':
        activation = random.choice(requirements.activation_types).value
        neurons = random.randint(requirements.min_num_of_neurons, requirements.max_num_of_neurons)
        layer_params = {'layer_type': layer_type, 'neurons': neurons, 'activation': activation}
    return layer_params


def _generate_random_struct(requirements: GPNNComposerRequirements) -> List[str]:
    conv_depth = random.randint(requirements.min_num_of_conv_layers, requirements.max_num_of_conv_layers)
    nn_depth = random.randint(requirements.min_nn_depth, requirements.max_nn_depth)
    struct = ['conv2d']
    for i in range(1, conv_depth + nn_depth):
        if i < conv_depth:
            node = random.choice(requirements.primary + requirements.secondary) if i != conv_depth - 1 else 'flatten'
        else:
            node = random.choice(requirements.secondary)
        struct.append(node)
    return struct


class NASDirector:
    _builder = None

    def set_builder(self, builder):
        self._builder = builder

    def create_nas_graph(self):
        return self._builder.build_graph()


class CNNBuilder:

    def __init__(self, nodes_list: Optional[List[str]] = None,
                 requirements=None):
        self.nodes = list(nodes_list) if nodes_list else _generate_random_struct(requirements)
        self.requirements = requirements

    def _add_node(self, node_type, nodes_from):
        node_params = _get_layer_params(node_type, self.requirements)
        node = CNNNode(content={'name': node_type, 'params': node_params}, nodes_from=nodes_from)
        if node_type == 'flatten':
            return node
        if random.random() > self.requirements.batch_norm_prob:
            batch_norm_params = _get_layer_params('batch_normalization', self.requirements)
            node.content['params'] = node.content['params'] | batch_norm_params
        if random.random() > self.requirements.dropout_prob:
            dropout_params = _get_layer_params('dropout', self.requirements)
            node.content['params'] = node.content['params'] | dropout_params
        return node

    def build_graph(self) -> CNNGraph:
        graph = CNNGraph()
        parent_node = None
        for node_name in self.nodes:
            node = self._add_node(node_name, parent_node)
            parent_node = [node]
            graph.add_node(node)
        return graph


if __name__ == '__main__':
    r = GPNNComposerRequirements(primary=None, secondary=None, input_shape=[120, 120, 3])
    graph = CNNBuilder(CNNGraph, requirements=r).build_graph()
    print("DONE!")
