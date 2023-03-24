from __future__ import annotations

from functools import partial
from typing import Callable, Optional, TYPE_CHECKING, Union

import tensorflow as tf
from golem.core.dag.graph_node import GraphNode

from nas.model.tensorflow.layer_initializer import LayerInitializer

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


class BaseNasTFModel(tf.keras.Model):
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
        self.classifier = tf.keras.layers.Dense(output_shape, activation=activation_function)

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
                output = tf.keras.layers.Add()([output, shortcut])
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
        inputs = tf.cast(inputs, dtype='float32')
        inputs = self.forward_pass(inputs)
        output = self.classifier(inputs)
        return output
