from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Callable, Optional, List, TYPE_CHECKING

import tensorflow

from nas.model.layers.keras_layers import KerasLayers
from nas.model.layers.tf_layer_initializer import LayerInitializer
from nas.model.utils.branch_manager import GraphBranchManager
from nas.model.utils.converter import GraphStruct

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NasGraph
    from nas.graph.node.nn_graph_node import NNNode


class ModelBuilder:
    def __init__(self):
        self._builder = Optional[Callable] = None

    def set_builder(self, builder) -> ModelBuilder:
        self._builder = builder
        return self

    def build(self, graph: NasGraph, **kwargs):
        return self._builder(graph, kwargs)


class TFLayers(Enum):
    conv2s = partial(LayerInitializer.conv2d)


class NasKerasModel(tensorflow.keras.Model):
    def __init__(self, input_shape: List[int], graph: NasGraph,
                 graph_branch_manager: Optional[GraphBranchManager] = None,
                 n_classes: Optional[int] = None):
        super().__init__()
        self._graph_struct = GraphStruct(graph)
        self._branch_manager = graph_branch_manager
        self._model_structure = {node: LayerInitializer(node) for node in self._graph_struct}

    def call(self, inputs, training=None, mask=None):
        pass


class ModelMaker:
    def __init__(self, input_shape: List, graph: NasGraph, converter: Callable, num_classes: int = None):
        self.num_classes = num_classes
        self._graph_struct = converter(graph)
        self._branch_manager = GraphBranchManager()

        self._output_shape = 1 if num_classes == 2 else num_classes
        self._activation_func = 'sigmoid' if num_classes == 2 else 'softmax'
        self._loss_func = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

        self._input = tensorflow.keras.layers.Input(shape=input_shape)
        self._body = self._make_body
        self._classifier = tensorflow.keras.layers.Dense(self._output_shape, activation=self._activation_func)

    @staticmethod
    def _make_one_layer(input_layer, node: NNNode, branch_manager: GraphBranchManager, downsample: Optional[Callable]):
        layer = KerasLayers.convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
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

    def _make_body(self, inputs, **kwargs):
        # x = self._make_one_layer(inputs, self._graph_struct.head, self._branch_manager, None)
        x = inputs
        for node in self._graph_struct:
            x = self._make_one_layer(input_layer=x, node=node, branch_manager=self._branch_manager,
                                     downsample=KerasLayers.downsample_block)  # create layer with batch_norm
            # _update active deprecated branches after each layer creation
            self._branch_manager.add_and_update(node, x, self._graph_struct.get_children(node))
        return x

    def build(self):
        inputs = self._input
        body = self._body(inputs)
        del self._branch_manager
        self._branch_manager = None
        output = self._classifier(body)
        model = tensorflow.keras.Model(inputs=inputs, outputs=output, name='nas_model')

        return model
