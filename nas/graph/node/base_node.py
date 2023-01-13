from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from functools import partial

from nas.composer.base_requirements import ModelRequirements
from nas.repository.layer_types_enum import LayersPoolEnum


@dataclass
class BaseNodeFactory:
    @staticmethod
    def batch_norm(requirements: ModelRequirements) -> dict:
        layer_params = dict()
        if random.uniform(0, 1) < requirements.batch_norm_prob:
            layer_params['momentum'] = 0.99
            layer_params['epsilon'] = 0.001
            layer_params['input_channels'] = None
            layer_params['out_channels'] = None
        return layer_params

    @staticmethod
    def _base_conv2d(requirements: ModelRequirements, **kwargs) -> dict:
        layer_params = BaseNodeFactory.batch_norm(requirements)
        layer_params['input_channels'] = None  # TODO
        layer_params['out_channels'] = random.choice(requirements.conv_requirements.out_channels)
        layer_params['activation'] = random.choice(requirements.activation_types).value
        layer_params['stride'] = random.choice(requirements.conv_requirements.conv_strides)
        return layer_params

    def conv2d(self, requirements: ModelRequirements, kernel_size: int) -> dict:
        layer_params = self._base_conv2d(requirements)
        layer_params['kernel_size'] = [kernel_size, kernel_size]
        return {'params': layer_params, 'name': f'conv2d_{kernel_size}x{kernel_size}'}

    @staticmethod
    def dense(requirements: ModelRequirements, **kwargs) -> dict:
        layer_params = BaseNodeFactory.batch_norm(requirements)
        layer_params['input_channels'] = None  # TODO
        layer_params['out_channels'] = random.choice(requirements.fc_requirements.out_features)
        return {'params': layer_params, 'name': 'dense'}

    @staticmethod
    def dropout(requirements: ModelRequirements) -> dict:
        layer_params = dict()
        layer_params['drop'] = random.randint(1, requirements.max_drop_size * 10) / 10
        layer_params['input_channels'] = None
        layer_params['out_channels'] = None
        return {'params': layer_params, 'name': 'dropout'}

    @staticmethod
    def flatten(*args, **kwargs) -> dict:
        layer_params = dict()
        layer_params['input_channels'] = None
        layer_params['out_channels'] = None
        return {'params': layer_params, 'name': 'flatten'}

    @staticmethod
    def pool(requirements: ModelRequirements, mode: str) -> dict:
        layer_params = dict()
        layer_params['input_channels'] = None
        layer_params['out_channels'] = None
        layer_params['pool_size'] = random.choice(requirements.conv_requirements.pool_size)
        layer_params['pool_strides'] = random.choice(requirements.conv_requirements.pool_strides)
        return {'params': layer_params, 'name': mode}


class BaseNodeTypes(Enum):
    conv2d_1x1 = partial(BaseNodeFactory().conv2d, kernel_size=1)
    conv2d_3x3 = partial(BaseNodeFactory().conv2d, kernel_size=3)
    conv2d_5x5 = partial(BaseNodeFactory().conv2d, kernel_size=5)
    conv2d_7x7 = partial(BaseNodeFactory().conv2d, kernel_size=7)
    dense = partial(BaseNodeFactory.dense)
    flatten = partial(BaseNodeFactory.flatten)
    dropout = partial(BaseNodeFactory.dropout)
    max_pool2d = partial(BaseNodeFactory.pool, mode='max_pool2d')
    avg_pool = partial(BaseNodeFactory.pool, mode='average_pool2d')


if __name__ == '__main__':
    from nas.composer.base_requirements import NNComposerRequirements
    from nas.graph.node.nn_graph_node import NNNode
    from nas.graph.cnn.cnn_graph import NNGraph

    default_requirements = NNComposerRequirements.default_params()
    nodes = [func.value(requirements=default_requirements.nn_requirements) for func in BaseNodeTypes]
    parents = None

    for i, node in enumerate(nodes):
        nodes[i] = NNNode(content=node, nodes_from=parents)
        parents = [nodes[i]]

    graph = NNGraph(nodes).set_input_channels(3)

    print('Done!')
