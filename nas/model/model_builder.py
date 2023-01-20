from typing import List

import tensorflow as tf

from nas.graph.cnn.cnn_graph import NasGraph


def _make_skip_connection_block(*args, **kwargs):
    return


class NNBuilder:
    def __init__(self, graph: NasGraph, input_shape: List, classes: int):
        self.graph = graph
        self.nn_struct = graph.graph_struct
        self.input_shape = input_shape
        self.classes = classes

    def build(self):
        dict_representation = dict(self.graph)
        model = ModelNAS()
        for attr, value in dict_representation.items():
            layer = self._make_one_layer(attr, value)
            model.__setattr__(attr, layer)
        return model

    @property
    def skip_connection_blocks(self):
        return

    @property
    def skip_connection_destinations(self):
        return

    @_make_skip_connection_block
    def _make_one_layer(self, input_layer, node_name, parameters):
        pass

    @_make_skip_connection_block
    def _make_conv2d(self, *args, **kwargs):
        pass

    @_make_skip_connection_block
    def _make_dense(self, *args, **kwargs):
        pass


class ModelNAS(tf.kears.Model):
    def __init__(self):
        super().__init__()
