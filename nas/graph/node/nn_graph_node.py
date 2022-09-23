from typing import Optional, List, Tuple, Union, Callable

import numpy as np
from fedot.core.optimisers.graph import OptNode

from nas.graph.node.nn_node_params import GraphLayers


def get_node_params_by_type(node, requirements):
    return GraphLayers().layer_by_type(node, requirements)


def count_conv_layer_params(node, input_shape):
    kernel_size = node.content['params'].get('kernel_size')
    stride = node.content['params'].get('conv_strides')
    num_of_filters = node.content['params'].get('num_of_filters')
    params = (np.dot(*kernel_size)*input_shape + 1) * num_of_filters
    return params


def count_fc_layer_params(node, input_shape):
    out_shape = node.content['params'].get('neurons')
    return (input_shape * out_shape) + 1


class NNNode(OptNode):
    def __init__(self, content: dict, nodes_from: Optional[List] = None,
                 input_shape: Union[List[float], Tuple[float]] = None):
        super().__init__(content, nodes_from)
        self.nodes_from = nodes_from
        if 'params' in content:
            self.content = content
            self.content['name'] = self.content['name'].value

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()

    @property
    def input_shape(self):
        return None

    # TODO fix
    def get_number_of_trainable_params(self, input_shape) -> Callable:
        is_conv = 'conv' in self.content['name']
        if isinstance(input_shape, NNNode):
            number_of_filters = input_shape.content['params'].get('num_of_filters')
            number_of_neurons = input_shape.content['params'].get('neurons')
        else:
            number_of_filters = input_shape[-1]
            number_of_neurons = input_shape[0] * input_shape[1]
        params = 0
        if is_conv:
            params = count_conv_layer_params(self, number_of_filters)
        elif 'dense' in self.content['name']:
            if input_shape.content['name'] == 'flatten':
                parent = input_shape.nodes_from[0]
                number_of_filters = parent.content['params'].get('num_of_filters')
                number_of_neurons = number_of_filters * np.dot(*parent.content['params'].get('kernel_size'))
            else:
                number_of_filters = input_shape.content['params'].get('num_of_filters')
                number_of_neurons = input_shape.content['params'].get('neurons')
            params = count_fc_layer_params(self, number_of_neurons)
        else:
            number_of_filters = input_shape.content['params'].get('num_of_filters')
            number_of_neurons = input_shape.content['params'].get('neurons')
            params = number_of_filters * np.dot(*input_shape.content['params'].get('kernel_size'))
        return params
