from typing import Union, Dict

import torch.nn as nn
from golem.core.dag.graph_node import GraphNode

from nas.graph.node.nas_graph_node import NasNode


def conv2d(node: NasNode, **inputs_dict):
    """
    TODO
    """
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('out_shape')
    kernel_size = node.parameters.get('kernel_size')
    stride = node.parameters.get('stride', 1)
    padding = node.parameters.get('padding')
    return nn.Conv2d(input_dim, out_shape, kernel_size, stride, padding)


def linear(node: NasNode, **inputs_dict):
    """
    TODO
    """
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('neurons')
    return nn.Linear(input_dim, out_shape, **inputs_dict)


def dropout(node: NasNode, **kwargs):
    dropout_prob = node.parameters.get('drop')
    return nn.Dropout(p=dropout_prob)


def batch_norm(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    eps = node.parameters.get('epsilon')
    momentum = node.parameters.get('momentum')
    return nn.BatchNorm2d(input_dim, eps, momentum)


def pooling(node: NasNode, **inputs_dict):
    kernel_size = node.parameters.get('pool_size')
    stride = node.parameters.get('pool_stride')
    padding = node.parameters.get('padding', 0)
    pool_layer = nn.MaxPool2d if node.parameters['mode'] == 'max' else nn.AvgPool2d
    return pool_layer(kernel_size, stride, padding=padding)


def flatten(*args, **kwargs):
    return nn.Flatten


class TorchLayerFactory:
    @staticmethod
    def get_layer(node: Union[GraphNode, NasNode]) -> Dict:
        _layers = {'conv2d': conv2d,
                   'linear': linear,
                   'dropout': dropout,
                   'batch_norm': batch_norm,
                   'pooling2d': pooling,
                   'flatten': flatten}
        layer = {}
        layer_type = node.name
        layer_fun = _layers.get(layer_type)
        layer['weighted_layer'] = layer_fun  # TODO rename 'weighted layer' to more appropriate name
        if layer_fun is None:
            raise ValueError(f'Wrong layer type: {layer_type}')
        if 'momentum' in node.parameters:
            layer['normalization'] = _layers.get('batch_norm')
        return layer

    @staticmethod
    def get_activation(activation_name: str):
        activations = {'relu': nn.ReLU,
                       'elu': nn.ELU,
                       'selu': nn.SELU,
                       'softmax': nn.Softmax,
                       'sigmoid': nn.Sigmoid}
        activation = activations.get(activation_name)
        if activation is None:
            raise ValueError(f'Wrong activation function: {activation_name}')
        return activation

    @classmethod
    def tmp_layer_initialization(cls, input_shape: int, node: NasNode):
        name = 'conv2d' if 'conv2d' in node.name else node.name
        weighted_layer = cls.get_layer(name)(input_shape, node)
        normalization = cls.get_layer('batch_norm')(weighted_layer.out_features, node)
        activation = cls.get_activation(node.parameters.get('activation')())
        drop = cls.get_layer('dropout')(node)

        return {'weighted_layer': weighted_layer,
                'normalization': normalization,
                'activation': activation,
                'dropout': drop}


class NASLayer(nn.Module):
    def __init(self, node: NasNode, input_channels: int):
        super().__init()
        self._weighted_layer = TorchLayerFactory.get_layer(node.name)
        self._normalization = TorchLayerFactory.get_layer('batch_norm')
        self._activation = TorchLayerFactory.get_activation(node.parameters.get('activation'))
        self._dropout = TorchLayerFactory.get_layer('dropout')
