import numpy as np

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from fedot.core.data.data import InputData, OutputData
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from nas import nn_layers


def _keras_model_prob2labels(predictions: np.array, is_multiclass: bool = False) -> np.array:
    if is_multiclass:
        output = []
        for prediction in predictions:
            values = []
            for val in prediction:
                values.append(round(val))
            output.append(np.float64(values))
    else:
        output = np.zeros(predictions.shape)
        for i, prediction in enumerate(predictions):
            class_prediction = np.argmax(prediction)
            output[i][class_prediction] = 1
    return output


def keras_model_fit(model, input_data: InputData, verbose: bool = True, batch_size: int = 24,
                    epochs: int = 10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('../models/mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    model.fit(input_data.features, input_data.target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=0.2,
              callbacks=[early_stopping, reduce_lr_loss, mcp_save])
    return keras_model_predict(model, input_data)


def keras_model_predict(model, input_data: InputData, output_mode: str = 'default',
                        is_multiclass: bool = False) -> OutputData:
    if output_mode == 'label':
        evaluation_result = model.predict_on_batch(input_data.features)
        evaluation_result = _keras_model_prob2labels(predictions=evaluation_result, is_multiclass=is_multiclass)
    elif output_mode == 'default':
        evaluation_result = model.predict_on_batch(input_data.features)
    else:
        raise ValueError('Wrong mode')
    return OutputData(idx=input_data.idx,
                      features=input_data.features,
                      predict=evaluation_result,
                      task=input_data.task, data_type=input_data.data_type)


def generate_structure(node: Any):
    if node.nodes_from:
        struct = []
        if len(node.nodes_from) == 1:
            struct.append(node)
            struct += generate_structure(node.nodes_from[0])
            return struct
        elif len(node.nodes_from) == 2:
            struct += generate_structure(node.nodes_from[0])
            struct.append(node)
            struct += generate_structure(node.nodes_from[1])
            return struct
        elif len(node.nodes_from) == 3:
            struct += generate_structure(node.nodes_from[0])
            struct.append(node)
            struct += generate_structure(node.nodes_from[1])
            struct.append(node)
            struct += generate_structure(node.nodes_from[2])
            return struct
    else:
        return [node]


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack, (model, training_config, weights)

    cls = Model
    cls.__reduce__ = __reduce__


def create_nn_model(graph: Any, input_shape: tuple, classes: int = 3):
    def _get_skip_connection_list(graph_structure):
        sc_layers = {}
        for node in graph_structure.nodes:
            if len(node.nodes_from) > 1:
                for n in node.nodes_from[1:]:
                    sc_layers[n] = node
        return sc_layers

    nn_structure = graph.nodes[::-1]
    make_keras_picklable()
    inputs = keras.Input(shape=input_shape)
    in_layer = inputs
    skip_connection_nodes_dict = _get_skip_connection_list(graph)
    skip_connection_destination_dict = {}
    for i, layer in enumerate(nn_structure):
        layer_type = layer.content['name']
        is_free_node = layer in graph.nodes_without_skip_connections
        if layer in skip_connection_nodes_dict:
            skip_connection_id = skip_connection_nodes_dict.pop(layer)
            if skip_connection_id not in skip_connection_destination_dict:
                skip_connection_destination_dict[skip_connection_id] = [in_layer]
            else:
                skip_connection_destination_dict[skip_connection_id].append(in_layer)
        in_layer = nn_layers.make_skip_connection_block(idx=i, input_layer=in_layer, current_node=layer,
                                                        layers_dict=skip_connection_destination_dict)
        if layer_type == 'conv2d':
            in_layer = nn_layers.make_conv_layer(idx=i, input_layer=in_layer, current_node=layer,
                                                 is_free_node=is_free_node)
        elif layer_type == 'dropout':
            in_layer = nn_layers.make_dropout_layer(idx=i, input_layer=in_layer, current_node=layer)
        elif layer_type == 'dense':
            in_layer = nn_layers.make_dense_layer(idx=i, input_layer=in_layer, current_node=layer)
        elif layer_type == 'flatten':
            flatten = layers.Flatten()
            in_layer = flatten(in_layer)
    # Output
    output_shape = 1 if classes == 2 else classes
    activation_func = 'sigmoid' if classes == 2 else 'softmax'
    loss_func = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    dense = layers.Dense(output_shape, activation=activation_func)
    outputs = dense(in_layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name='custom_model')
    model.compile(loss=loss_func, optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.summary()
    return model
