from __future__ import annotations
import datetime
import pathlib
from typing import Callable, Optional, List, TYPE_CHECKING

import tensorflow
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from keras import optimizers

from nas.data import Preprocessor, setup_data, DataGenerator
from nas.graph.graph_builder import NNGraphBuilder
from nas.model.layers.keras_layers import KerasLayers
from nas.model.utils import converter
from nas.model.utils.branch_manager import GraphBranchManager
from nas.operations.evaluation.metrics.metrics import get_predictions, calculate_validation_metric
from nas.repository.layer_types_enum import LayersPoolEnum

if TYPE_CHECKING:
    from nas.graph.cnn.cnn_graph import NNGraph
    from nas.graph.node.nn_graph_node import NNNode


class ModelMaker:
    def __init__(self, input_shape: List, graph: NNGraph, converter: Callable, num_classes=None):
        super().__init__()
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
