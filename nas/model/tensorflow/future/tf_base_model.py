from typing import Optional, Union, List, Tuple

import tensorflow as tf

from nas.composer.nn_composer_requirements import load_default_requirements
from nas.graph.cnn_graph import NasGraph
from nas.graph.graph_builder.base_graph_builder import BaseGraphBuilder
from nas.graph.graph_builder.resnet_builder import ResNetGenerator
from nas.model.tensorflow.future.tf_activations_enum import KerasActivations
from nas.model.tensorflow.future.tf_layer_initializer import LayerInitializer
from nas.model.utils.branch_manager import GraphBranchManager


def test_model():
    requirements = load_default_requirements()
    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(ResNetGenerator(model_requirements=requirements.model_requirements))
    model = NasModel(graph_generation_function.build(), [224, 224, 3], n_classes=75)
    print('Done')


class NasModel(tf.keras.Model):
    def __init__(self, graph: NasGraph, input_shape: Union[List, Tuple], n_classes: Optional[int] = None,
                 branch_manager: Optional[GraphBranchManager] = GraphBranchManager()):
        super().__init__()
        self.input_layer = tf.keras.layers.Input(input_shape)
        self.model_structure = [LayerInitializer.initialize_layer(node) for node in graph.graph_struct]
        self._graph_nodes = graph.graph_struct
        output_shape = 1 if not n_classes else n_classes
        self.classifier = tf.keras.layers.Dense(output_shape, activation='softmax')
        self.branch_manager = branch_manager

    def _one_block_output(self, node):
        pass

    def call(self, inputs, training=None, mask=None):
        inputs = self.input_layer(inputs)
        for node_id, layer in enumerate(self.model_structure):
            inputs = layer(inputs)
            node = self._graph_nodes[node_id]
            if node.content['params'].get('epsilon'):
                batch_norm = LayerInitializer.batch_norm(node)
            if len(node.nodes_from) > 1:
                parent_layer = self.branch_manager.get_parent_layer(node=node.nodes_from[1])['layer']
                inputs = tf.keras.layers.Add()([inputs, parent_layer])
            if node.content['params'].get('activation_func'):
                activation_func = tf.keras.layers.Activation(node.content['params'].get('activation_func'))
                inputs = activation_func(inputs)

            self.branch_manager.add_and_update(node, inputs, self._graph_nodes.get_children(node))

        output = self.classifier(inputs)
        return output



