from __future__ import annotations

from typing import Callable, Optional, List, TYPE_CHECKING, Union

import tensorflow
from golem.core.dag.graph_node import GraphNode

from nas.model.tensorflow.layer_initializer import LayerInitializer
from nas.model.tensorflow.layers import KerasLayers
from nas.model.utils.branch_manager import GraphBranchManager
from nas.model.utils.model_structure import _ModelStructure

if TYPE_CHECKING:
    from nas.graph.cnn_graph import NasGraph
    from nas.graph.node.nas_graph_node import NasNode


class ModelBuilder:
    def __init__(self):
        self._model_class = Optional[Callable] = None

    def set_model_class(self, model_class) -> ModelBuilder:
        self._model_class = model_class
        return self

    def build(self, graph: NasGraph, **kwargs):
        return self._model_class(graph, kwargs)


class KerasModelMaker:
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
    def _make_one_layer(input_layer, node: NasNode, branch_manager: GraphBranchManager, downsample: Optional[Callable]):
        layer = KerasLayers.convert_by_node_type(node=node, input_layer=input_layer, branch_manager=branch_manager)
        if node.content['params'].get('epsilon'):
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

from functools import partial


class BaseNasTFModel(tensorflow.keras.Model):
    def __init__(self, graph: NasGraph, n_classes: int = None):
        super().__init__()
        self.model_layers = None
        self.classifier = None
        # self.graph = graph
        self.initialize_layers(n_classes, graph)
        self.forward_pass = partial(self.make_model_forward_pass_recursive, graph=graph)
        self._inputs_dict = dict()

    def initialize_layers(self, n_classes: int, graph):
        output_shape = n_classes if n_classes > 2 else 1
        activation_function = 'softmax' if output_shape > 1 else 'sigmoid'
        self.model_layers = {hash(node): LayerInitializer().initialize_layer(node) for node in
                             graph.nodes}
        self.classifier = tensorflow.keras.layers.Dense(output_shape, activation=activation_function)

    def make_model_forward_pass_recursive(self, data_input, graph: NasGraph):
        visited_nodes = set()
        # save output of layers whom have more than 1 outputs in following format: hash(node): layer_output
        outputs_to_save = dict()

        def _make_one_layer(node: Union[NasNode, GraphNode]):

            # inputs: previous layer output (not shortcut)
            # get layer func
            layer_key = hash(node)
            node_layer = self.model_layers[layer_key]
            layer_inputs = None
            # if node is not in visited nodes, we simply calculate its output
            if node in visited_nodes:
                return outputs_to_save[layer_key]

            # store nodes in outputs_to_save if they have more than one successor or if it has more than 1 predecessor
            first_condition = len(graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_condition = layer_key not in outputs_to_save.keys()
            if first_condition and second_condition:
                outputs_to_save[layer_key] = None

            # to calculate output result we need to know layer_inputs. it could be obtained
            # by recursive calculating output of parent nodes or
            # by using inputs which are first layer inputs (original data).

            layer_inputs = [_make_one_layer(parent) for parent in node.nodes_from] if node.nodes_from else [data_input]

            # knowing layer inputs and layer func, calculate output of this layer
            # if node already in visited, then it has more than 1 child (it has several edges that led to itself)
            # hence its output already stored in outputs_to_save, and we could reuse its result as output.

            output = node_layer[0](layer_inputs[0])

            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output = tensorflow.keras.layers.Add()([output, shortcut])
                # Have to move BN, activation and dropout after add

            output = node_layer[1](output) if len(node_layer) > 1 else output
            output = LayerInitializer.activation(node)(output) if node.content['params'].get('activation') else output
            output = LayerInitializer.dropout(node)(output) if node.content['params'].get('drop') else output

            # at this step we have complete layer output which could be
            # stored to outputs dict to further skip connections assemble.
            if layer_key in outputs_to_save.keys():
                outputs_to_save[layer_key] = output

            # add node to visited
            visited_nodes.add(node)

            return output

        root_node = graph.root_node

        return _make_one_layer(root_node)

    def call(self, inputs, training=None, mask=None):
        inputs = tensorflow.cast(inputs, dtype='float32')
        inputs = self.forward_pass(inputs)
        output = self.classifier(inputs)
        return output
