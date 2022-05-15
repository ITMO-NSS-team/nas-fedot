from typing import Any
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# from fedot_old.core.models.data import InputData, OutputData
from fedot.core.data.data import InputData, OutputData
from nas.layer import LayerTypesIdsEnum


def keras_model_fit(model, input_data: InputData, verbose: bool = True, batch_size: int = 24,
                    epochs: int = 10):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(f'{input_data.num_classes}_mdl_wts.hdf5', save_best_only=True, monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    model.summary()
    model.fit(input_data.features, input_data.target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=0.2,
              callbacks=[earlyStopping, reduce_lr_loss, mcp_save])
    return keras_model_predict(model, input_data)


def keras_model_predict(model, input_data: InputData):
    # evaluation_result = model.predict(input_data.features)
    evaluation_result = model.predict(input_data.features)
    # sum_rows = np.sum(evaluation_result,axis=1).tolist()
    # print(sum_rows)
    # print(np.sum(evaluation_result))
    # evaluation_result = log_loss(input_data.target, eval)
    return OutputData(idx=input_data.idx,
                      features=input_data.features,
                      predict=evaluation_result,
                      task=input_data.task, data_type=input_data.data_type)


def generate_structure(node: Any):
    if hasattr(node, 'nodes_from') and node.nodes_from:
        struct = []
        # while len(node.nodes_from != 0):
        #     struct.append(node)
        #     struct += generate_structure(node.nodes_from[0])

        if len(node.nodes_from) == 1:
            struct.append(node)
            struct += generate_structure(node.nodes_from[0])
            return struct
        elif len(node.nodes_from) == 2:
            struct.append(node)
            struct += generate_structure(node.nodes_from[0])
            struct.append(node)
            struct += generate_structure(node.nodes_from[1])
            return struct
        elif len(node.nodes_from) == 3:
            struct.append(node)
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
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def get_shape_dim(out_shape: tuple):
    out_shape_list = [el for el in out_shape if el]
    return np.prod(out_shape_list)


def create_nn_model(graph: Any, input_shape: tuple, classes: int = 3):
    generated_struc = generate_structure(graph.root_node)
    if any(generated_struc):
        nn_structure = graph.cnn_nodes + generated_struc
    else:
        nn_structure = graph
    # nn_structure = graph.nodes + generate_structure(graph.root_node)
    make_keras_picklable()
    model = models.Sequential()
    for i, layer in enumerate(nn_structure):
        type = layer.layer_params.layer_type
        if type == LayerTypesIdsEnum.conv2d:
            activation = layer.layer_params.activation.value
            kernel_size = layer.layer_params.kernel_size
            conv_strides = layer.layer_params.conv_strides
            filters_num = layer.layer_params.num_of_filters
            padding = layer.layer_params.padding
            batch_norm = layer.layer_params.batch_norm
            if i == 0:
                model.add(
                    layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation, input_shape=input_shape,
                                  strides=conv_strides, padding=padding))
            else:
                if not max(kernel_size) > max(model.layers[-1].output_shape[1:3]):
                    model.add(layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation,
                                            strides=conv_strides, padding=padding))
            if batch_norm:
                model.add(layers.BatchNormalization())
            if layer.layer_params.pool_size:
                pool_size = layer.layer_params.pool_size
                pool_strides = layer.layer_params.pool_strides
                if not max(pool_size) > max(model.layers[-1].output_shape[1:3]):
                    if layer.layer_params.pool_type == LayerTypesIdsEnum.maxpool2d:
                        model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides))
                    elif layer.layer_params.pool_type == LayerTypesIdsEnum.averagepool2d:
                        model.add(layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides))
        elif type == LayerTypesIdsEnum.dropout:
            drop = layer.layer_params.drop
            model.add(layers.Dropout(drop))
        elif type == LayerTypesIdsEnum.dense:
            activation = layer.layer_params.activation.value
            neurons_num = layer.layer_params.neurons
            batch_norm = layer.layer_params.batch_norm
            model.add(layers.Dense(neurons_num, activation=activation))
            if batch_norm:
                model.add(layers.BatchNormalization())
        # if i == len(graph.cnn_nodes) - 1:
        if i == len(graph.cnn_nodes) - 1:
            max_neurons_flatten = int(layer.layer_params.max_neurons_flatten or
                                      nn_structure[i - 1].layer_params.max_neurons_flatten or 1000)
            batch_norm = layer.layer_params.batch_norm
            while get_shape_dim(model.layers[-1].output_shape) > max_neurons_flatten:
                if model.layers[-1].output_shape[-1] >= 64:
                    print('too many neurons, added 1x1 convolution')
                    model.add(
                        layers.Conv2D(model.layers[-1].output_shape[-1] // 2, kernel_size=(1, 1), activation='relu',
                                      strides=(1, 1)))
                    if batch_norm:
                        model.add(layers.BatchNormalization())
                else:
                    print('too many neurons, added max pooling')
                    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(layers.Flatten())
            neurons_num = model.layers[-1].output_shape[1]
            model.add(layers.Dense(neurons_num, activation='relu'))
    # Output
    output_shape = 1 if classes == 2 else classes
    activation_func = 'sigmoid' if classes == 2 else 'softmax'
    loss_func = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    model.add(layers.Dense(output_shape, activation=activation_func))

    model.compile(loss=loss_func, optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.summary()
    return model
