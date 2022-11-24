from typing import Any, List

from keras import layers, optimizers
from tensorflow import keras

import nas


def create_nn_model(graph: Any, input_shape: List, classes: int = 3):
    def _get_skip_connection_list(graph_structure):
        """Returns dictionary where key is node where skip connection is started and value is destination"""
        sc_layers = {}
        for destination_node in graph_structure._graph_struct:
            if len(destination_node.nodes_from) > 1:
                for source_node in destination_node.nodes_from[1:]:
                    sc_layers[source_node] = destination_node
        return sc_layers

    nn_structure = graph._graph_struct
    inputs = keras.Input(shape=input_shape, name='input_0')
    in_layer = inputs
    skip_connection_nodes_dict = _get_skip_connection_list(graph)
    skip_connection_destination_dict = {}
    for i, layer in enumerate(nn_structure):
        layer_type = layer.content['name']
        is_free_node = layer in graph.free_nodes
        if 'conv' in layer_type:
            in_layer = nas.nn.layers_keras.make_conv_layer(idx=i, input_layer=in_layer, current_node=layer,
                                                           is_free_node=False)
        elif layer_type == 'dense':
            in_layer = nas.nn.layers_keras.make_dense_layer(idx=i, input_layer=in_layer, current_node=layer)
        elif layer_type == 'flatten':
            flatten = layers.Flatten()
            in_layer = flatten(in_layer)

        in_layer = nas.nn.layers_keras.make_skip_connection_block(idx=i, input_layer=in_layer, current_node=layer,
                                                                  layers_dict=skip_connection_destination_dict)
        if 'pool' in layer_type:
            in_layer = nas.nn.layers_keras.make_pooling_layer(idx=i, input_layer=in_layer, current_node=layer,
                                                              is_free_node=False)

        if layer in skip_connection_nodes_dict:
            skip_connection_end_id = skip_connection_nodes_dict.pop(layer)
            if skip_connection_end_id not in skip_connection_destination_dict:
                skip_connection_destination_dict[skip_connection_end_id] = [in_layer]
            else:
                skip_connection_destination_dict[skip_connection_end_id].append(in_layer)

    # Output
    output_shape = 1 if classes == 2 else classes
    activation_func = 'sigmoid' if classes == 2 else 'softmax'
    loss_func = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    dense = layers.Dense(output_shape, activation=activation_func)
    outputs = dense(in_layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name='custom_model')
    model.compile(loss=loss_func, optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'], run_eagerly=True)

    return model


def build_nn_from_graph(graph, n_classes, requirements):
    input_shape = requirements.nn_requirements.conv_requirements.input_shape
    classes_num = n_classes

    graph.model = create_nn_model(graph, input_shape, classes_num)