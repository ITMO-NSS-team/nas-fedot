import datetime
import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from nas.composer.nn_composer_requirements import ConvRequirements, BaseLayerRequirements, \
    ModelRequirements, NNComposerRequirements
from nas.graph.cnn.resnet_builder import ResNetGenerator
from nas.graph.node.nn_graph_node import NNNode, get_node_params_by_type
from nas.repository.layer_types_enum import ActivationTypesIdsEnum
from nas.repository.layer_types_enum import LayersPoolEnum


def add_shortcut_and_check(input_shape: List, output_shape: List) -> bool:
    """Adds conv 1x1 in shortcut with different strides and сompare"""

    stride = math.ceil(input_shape[0] / output_shape[0])
    layer_type = LayersPoolEnum.conv2d_1x1
    requirements = NNComposerRequirements()
    layer_params = get_node_params_by_type(layer_type, requirements.model_requirements)
    shortcut_node = NNNode(content={'name': layer_type.value, 'params': layer_params})
    shortcut_node.content['params']['conv_strides'] = [stride, stride]
    shortcut_node.content['params']['neurons'] = output_shape[-1]
    shape = get_shape(input_shape, shortcut_node)
    return shape


def get_shape(input_shape: List, node: NNNode) -> List:
    return ParamCounter().get_output_shape(node, input_shape)


def count_node_params(node: NNNode, input_shape: List) -> List:
    pass


@dataclass
class ParamCounter:
    @staticmethod
    def _conv(input_shape: Union[List, np.ndarray], node: NNNode) -> Union[np.ndarray, List]:
        layer_params = node.content['params']
        stride = layer_params['conv_strides'][0]
        output_array = [math.ceil(i / stride) for i in input_shape[:2]]
        channels_num = layer_params['neurons']
        output_array.append(channels_num)
        return output_array

    @staticmethod
    def _fully_connected(input_shape: Union[List, np.ndarray], node: NNNode) -> List:
        layer_params = node.content['params']
        output = layer_params['neurons']
        return [output]

    @staticmethod
    def _pooling(input_shape: List, node: NNNode) -> List:
        layer_params = node.content['params']
        pool_stride = layer_params['pool_strides']
        output = [math.ceil(i / pool_stride[0]) for i in input_shape[:2:]]
        output.append(input_shape[-1])
        return output

    @staticmethod
    def _flatten(input_shape: List[float], node: NNNode) -> List:
        output = math.prod(input_shape)
        return [output]

    @staticmethod
    def _get_type(name: str):
        if 'conv' in name:
            return 'conv'
        if 'dense' in name:
            return 'fully_connected'
        if 'flatten' in name:
            return 'flatten'
        if 'pool' in name:
            return 'pooling'

    def get_output_shape(self, node: NNNode, input_shape) -> Union[np.ndarray, List]:
        layer_types = {
            'conv': self._conv,
            'fully_connected': self._fully_connected,
            'flatten': self._flatten,
            'pooling': self._pooling
        }
        _node_type = self._get_type(node.content['name'])
        if _node_type in layer_types:
            return layer_types[_node_type](input_shape, node)
        else:
            raise ValueError


if __name__ == '__main__':
    cv_folds = 2
    image_side_size = 256
    batch_size = 8
    epochs = 1
    optimization_epochs = 1
    # conv_layers_pool = [LayersPoolEnum.conv2d_1x1, LayersPoolEnum.conv2d_3x3, LayersPoolEnum.conv2d_5x5,

    data_requirements = DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = ConvRequirements(input_shape=[image_side_size, image_side_size],
                                         cnn_secondary=[LayersPoolEnum.max_pool2d, LayersPoolEnum.average_poold2],
                                         color_mode='RGB',
                                         min_number_of_neurons=32, max_number_of_neurons=64,
                                         conv_strides=[[1, 1]],
                                         pool_size=[[2, 2]], pool_strides=[[2, 2]])
    fc_requirements = BaseLayerRequirements(min_number_of_neurons=32,
                                            max_number_of_neurons=64)
    nn_requirements = ModelRequirements(conv_requirements=conv_requirements,
                                        fc_requirements=fc_requirements,
                                        primary=[LayersPoolEnum.conv2d_3x3],
                                        secondary=[LayersPoolEnum.dense],
                                        epochs=epochs, batch_size=batch_size,
                                        max_nn_depth=1, max_num_of_conv_layers=10,
                                        has_skip_connection=True, activation_types=[ActivationTypesIdsEnum.relu]
                                        )
    optimizer_requirements = OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = NNComposerRequirements(data_requirements=data_requirements,
                                          optimizer_requirements=optimizer_requirements,
                                          model_requirements=nn_requirements,
                                          timeout=datetime.timedelta(hours=200),
                                          num_of_generations=1)
    graph = ResNetGenerator(requirements.model_requirements).build()
    # dimensions_check(graph, input_shape=[256, 256, 3])

    print('Done!')
