import datetime
from typing import Callable

import tensorflow
from fedot.core.pipelines.convert import graph_structure_as_nx_graph

from nas.composer.nn_composer_requirements import NNComposerRequirements, OptimizerRequirements, DataRequirements, \
    ConvRequirements, FullyConnectedRequirements, NNRequirements
from nas.graph.cnn.cnn_graph import NNGraph
from nas.graph.cnn.resnet_builder import ResNetGenerator
from nas.graph.node.nn_graph_node import NNNode
from nas.model.layers.keras_layers import KerasLayers
from nas.nn import ActivationTypesIdsEnum
from nas.model import converter
from nas.model.branch_manager import GraphBranchManager
from nas.repository.layer_types_enum import LayersPoolEnum


class ModelNas(tensorflow.keras.Model):
    def __init__(self, graph: NNGraph, converter: Callable, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.network_struct = converter(graph)
        self._branch_manager = GraphBranchManager()
        # self._classifier = tensorflow.keras.layers.Dense(num_classes, activation='softmax')

    def _add_layer_recursive(self, inputs, branch_manager, node: NNNode = None):
        node = self.network_struct[0] if not node else node
        layer = KerasLayers().convert_by_node_type(node, input_layer=inputs, branch_manager=branch_manager)

        try:
            children_nodes = next(self.network_struct)
        except StopIteration:
            return

        # number_of_new_connections - is difference between sets of children_nodes parents.
        # i.e. child_1.nodes_from = [conv2, max_pool], child_2.nodes_from = [max_pool] -> conv2 - new connection
        number_of_new_connections = self._branch_manager.number_of_new_connections(children_nodes)

        if self._branch_manager.streams:
            self._branch_manager.update_keys(1)
            self._branch_manager.update_branch(node, layer)

        for _ in range(len(number_of_new_connections)):
            self._branch_manager._add_branch(node, layer)

        # self._branch_manager.update_branch(node, layer)

        for node in children_nodes:
            return self._add_layer_recursive(inputs=layer, branch_manager=branch_manager, node=node)

        return

    @staticmethod
    def _make_one_layer(input_layer, node: NNNode, branch_manager: GraphBranchManager, downsample: Callable):
        layer = KerasLayers().convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
        layer = KerasLayers.batch_norm(node=node, input_layer=layer)

        if len(node.nodes_from) > 1:
            # return
            parent_layer = branch_manager.get_parent_layer(node=node.nodes_from[1])['layer']
            if downsample and parent_layer.shape != layer.shape:
                parent_layer = downsample(parent_layer, layer.shape, node)
            layer = tensorflow.keras.layers.Add()([layer, parent_layer])

        layer = KerasLayers.activation(node=node, input_layer=layer)
        layer = KerasLayers.dropout(node=node, input_layer=layer)
        return layer

    def call(self, inputs, **kwargs):
        self.network_struct.reset()
        x = tensorflow.keras.layers.Input(shape=inputs)
        x = self._make_one_layer(x, self.network_struct.head, self._branch_manager, None)
        for node in self.network_struct:
            # for node in nodes:
                x = self._make_one_layer(input_layer=x, node=node, branch_manager=self._branch_manager,
                                         downsample=KerasLayers.downsample_block)  # create layer with batch_norm
                # _update active nn branches after each layer creation
                self._branch_manager.add_and_update(node, x, self.network_struct.get_children(node))

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
