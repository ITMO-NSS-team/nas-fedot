import numpy as np

from typing import Any

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from nas.layer import LayerTypesIdsEnum
from fedot.core.data.data import InputData, OutputData
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


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
        evaluation_result = model.predict_proba(input_data.features)
        evaluation_result = _keras_model_prob2labels(predictions=evaluation_result, is_multiclass=is_multiclass)
    elif output_mode == 'default':
        evaluation_result = model.predict_proba(input_data.features)
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
    generated_struc = generate_structure(graph.root_node)
    nn_structure = generated_struc[::-1]
    make_keras_picklable()
    model = models.Sequential()
    cnn_nodes_count = 0
    for i, layer in enumerate(nn_structure):
        type = layer.content['params'].layer_type
        if 'conv' in layer.content and cnn_nodes_count is not None:
            cnn_nodes_count += 1
        if type == LayerTypesIdsEnum.conv2d.value:
            activation = layer.content['params'].activation
            kernel_size = layer.content['params'].kernel_size
            conv_strides = layer.content['params'].conv_strides
            filters_num = layer.content['params'].num_of_filters
            if i == 0:
                model.add(
                    layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation, input_shape=input_shape,
                                  strides=conv_strides))
            else:
                if not all([size == 1 for size in kernel_size]):
                    model.add(
                        layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation,
                                      strides=conv_strides))
            if layer.content['params'].pool_size:
                pool_size = layer.content['params'].pool_size
                pool_strides = layer.content['params'].pool_strides
                if layer.content['params'].pool_type == LayerTypesIdsEnum.maxpool2d.value:
                    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides))
                elif layer.content['params'].pool_type == LayerTypesIdsEnum.averagepool2d.value:
                    model.add(layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides))
        elif type == LayerTypesIdsEnum.dropout.value:
            drop = layer.content['params'].drop
            model.add(layers.Dropout(drop))
        elif type == LayerTypesIdsEnum.dense.value:
            activation = layer.content['params'].activation
            neurons_num = layer.content['params'].neurons
            model.add(layers.Dense(neurons_num, activation=activation))
        # adding Flatten layer after last layer from cnn part of the graph
        if cnn_nodes_count == graph.cnn_depth:
            model.add(layers.Flatten())
            cnn_nodes_count = None
    # Output
    output_shape = 1 if classes == 2 else classes
    activation_func = 'sigmoid' if classes == 2 else 'softmax'
    loss_func = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    model.add(layers.Dense(output_shape, activation=activation_func))

    model.compile(loss=loss_func, optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.summary()
    return model
