import datetime
from typing import Callable, List, Iterable

import tensorflow
from fedot.core.utils import DEFAULT_PARAMS_STUB

from nas.composer.nn_composer_requirements import NNComposerRequirements, OptimizerRequirements, DataRequirements, \
    ConvRequirements, FullyConnectedRequirements, NNRequirements
from nas.graph.cnn.cnn_graph import NNGraph
from nas.graph.cnn.resnet_builder import ResNetGenerator
from nas.graph.node.nn_graph_node import NNNode
from nas.nn import converter, ActivationTypesIdsEnum
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.default_parameters import default_nodes_params


def _get_layer_params(current_node: NNNode) -> dict:
    if current_node.content['params'] == DEFAULT_PARAMS_STUB:
        layer_params = default_nodes_params[current_node.content['name']]
    else:
        layer_params = current_node.content.get('params')
    return layer_params


def with_skip_connections(layer_func):
    def wrapper(*args, **kwargs):
        branch_manager: GraphBranchManager = kwargs.get('branch_manager')
        # node and it's layer representation
        current_node = kwargs.get('node')
        input_layer = layer_func(*args, **kwargs)

        # add to active branches new branches
        branch_manager.add_branch(current_node, input_layer)

        if len(current_node.nodes_from) > 1:
            # for cases where len(current_node.nodes_from) > 1 add skip connection
            # also add dimension equalizer for cases which have different dimensions
            # layer_to_add = branch_manager.get_last_layer(current_node)
            layers_to_add = branch_manager.get_last_layer(current_node.nodes_from)
            # dimensions check. add conv to equalize dimensions in shortcuts if different
            input_layer = tensorflow.keras.layers.Add([layers_to_add, input_layer])

        # update active branches
        branch_manager.update_branch(current_node, input_layer)
        return layer_func(*args, **kwargs)

    return wrapper


def with_activation(layer_func):
    def add_activation_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        activation_type = layer_params.get('activation')
        input_layer = layer_func(*args, **kwargs)
        if activation_type:
            activation = tensorflow.keras.layers.Activation(activation_type)
            input_layer = activation(input_layer)
        return input_layer

    return add_activation_to_layer


def with_batch_norm(layer_func):
    def add_batch_norm_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        momentum = layer_params.get('momentum')
        input_layer = layer_func(*args, **kwargs)
        if momentum:
            epsilon = layer_params.get('epsilon')
            batch_norm = tensorflow.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
            input_layer = batch_norm(input_layer)
        return input_layer

    return add_batch_norm_to_layer


def with_dropout(layer_func):
    def add_dropout_to_layer(*args, **kwargs):
        layer_params = _get_layer_params(kwargs.get('node'))
        drop = layer_params.get('drop')
        input_layer = layer_func(*args, **kwargs)
        if drop:
            dropout = tensorflow.keras.layers.Dropout(drop)
            input_layer = dropout(input_layer)
        return input_layer

    return add_dropout_to_layer


class GraphBranchManager:
    def __init__(self):
        self._streams = dict()

    @property
    def streams(self):
        return self._streams

    def add_branch(self, node: NNNode, layer):
        key = len(self._streams.keys())
        self._streams[key] = {'node': node, 'layer': layer}
        # self._streams[node] = layer
        return self

    def update_branch(self, current_node: NNNode, layer):
        # for cases where number of parents > 1 should be added skip connections
        number_of_parents = len(current_node.nodes_from)
        for i in range(len(self._streams.keys())):
            if i > number_of_parents or number_of_parents == 0:
                # exit loop for cases where connection id greater than number of parents
                # number of updated connections is equal to number of current node parents
                break
            if self._streams[i]['node'] == current_node.nodes_from[i]:
                self._streams[i] = {'node': current_node, 'layer': layer}

    def get_last_layer(self, parents: List[NNNode]) -> Iterable:
        """Returns all layers from active branches except main branch (0)"""
        list_to_return = {self._streams[i]['layer'] for i in self._streams.keys()
                          if self._streams[i]['node'] == parents[i % len(parents) and i != 0]}
        return list_to_return
        # for i in self._streams.keys():
        #     if self._streams[i]['node'] == parents[i % len(parents)]:
        #         return self._streams[i]['layer']


class KerasLayers:
    @with_batch_norm
    @with_skip_connections
    @with_activation
    @with_dropout
    def conv2d(self, node: NNNode, input: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)

        kernel_size = layer_params['kernel_size']
        strides = layer_params['conv_strides']
        filters = layer_params['neurons']

        conv2d_layer = tensorflow.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                      padding='same')
        return conv2d_layer(input)

    @with_batch_norm
    @with_skip_connections
    @with_activation
    @with_dropout
    def pool(self, node: NNNode, input, *args, **kwargs):
        layer_params = _get_layer_params(current_node=node)
        pool_size = layer_params.get('pool_size', [2, 2])
        pool_strides = layer_params.get('pool_strides')
        # hotfix
        if node.content['name'] == LayersPoolEnum.max_pool2d.value:
            pool_layer = tensorflow.keras.layers.MaxPooling2D(pool_size, pool_strides)(input)
        else:
            pool_layer = tensorflow.keras.layers.AveragePooling2D(pool_size, pool_strides)(input)
        return pool_layer

    @with_batch_norm
    @with_skip_connections
    @with_activation
    @with_dropout
    def dense(self, node: NNNode, input: tensorflow.Tensor, *args, **kwargs):
        layer_params = _get_layer_params(node)
        units = layer_params['neurons']

        dense_layer = tensorflow.keras.layers.Dense(units=units)

        return dense_layer(input)

    @with_batch_norm
    @with_skip_connections
    @with_activation
    @with_dropout
    def flatten(self, node: NNNode, input: tensorflow.Tensor, *args, **kwargs):
        return tensorflow.keras.layers.Flatten()(input)



    @staticmethod
    def _get_node_type(node: NNNode) -> str:
        node_type = node.content['name']
        if 'conv2d' in node_type:
            return 'conv2d'
        elif 'pool' in node_type:
            return 'pool'
        return node_type

    @classmethod
    def convert_by_node_type(cls, node: NNNode, input: tensorflow.Tensor, branch_manager: GraphBranchManager):
        layer_types = {
            'conv2d': cls.conv2d,
            'dense': cls.dense,
            'flatten': cls.flatten,
            'pool': cls.pool
        }
        node_type = cls._get_node_type(node)

        return layer_types[node_type](cls, node=node, input=input, branch_manager=branch_manager)


class ModelNas(tensorflow.keras.Model):
    def __init__(self, graph: NNGraph, converter: Callable, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.network_struct = converter(graph)
        self._branch_manager = GraphBranchManager()
        # self._classifier = tensorflow.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        x = tensorflow.keras.layers.Input(shape=inputs)
        for layer in self.network_struct:
            x = KerasLayers().convert_by_node_type(layer[0], input=x, branch_manager=self._branch_manager)
            print('HOLD')
        classifier = tensorflow.keras.layers.Dense(units=self.num_classes, activation='softmax')(x)
        return classifier


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
                                         min_filters=32, max_filters=64,
                                         conv_strides=[[1, 1]],
                                         pool_size=[[2, 2]], pool_strides=[[2, 2]])
    fc_requirements = FullyConnectedRequirements(min_number_of_neurons=32,
                                                 max_number_of_neurons=64)
    nn_requirements = NNRequirements(conv_requirements=conv_requirements,
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
                                          nn_requirements=nn_requirements,
                                          timeout=datetime.timedelta(hours=200),
                                          num_of_generations=1)
    graph = ResNetGenerator(requirements.nn_requirements).build()

    # print(tensorflow.config.list_physical_devices('GPU'))
    s = ModelNas(graph, converter.Struct, 75)

    s.call((224, 224, 3))

    print(1)
