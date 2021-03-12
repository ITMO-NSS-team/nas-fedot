from typing import Any

from keras import layers
from keras import models
from keras import optimizers

from fedot.core.models.data import InputData, OutputData
from nas.layer import LayerTypesIdsEnum
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import log_loss

def keras_model_fit(model, input_data: InputData, verbose: bool = True, batch_size: int = 24,
                    epochs: int = 10):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    model.fit(input_data.features, input_data.target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=0.2,
              callbacks=[earlyStopping, reduce_lr_loss, mcp_save])
    return keras_model_predict(model, input_data)

import numpy as np

def keras_model_predict(model, input_data: InputData):
    # evaluation_result = model.predict(input_data.features)
    evaluation_result = model.predict_proba(input_data.features)
    # sum_rows = np.sum(evaluation_result,axis=1).tolist()
    # print(sum_rows)
    # print(np.sum(evaluation_result))
    # evaluation_result = log_loss(input_data.target, eval)
    return OutputData(idx=input_data.idx,
                      features=input_data.features,
                      predict=evaluation_result,
                      task_type=input_data.task_type)


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
    else:
        return [node]


def create_nn_model(chain: Any, input_shape: tuple, classes: int = 3):
    nn_structure = chain.cnn_nodes + generate_structure(chain.root_node)
    model = models.Sequential()
    for i, layer in enumerate(nn_structure):
        type = layer.layer_params.layer_type
        if type == LayerTypesIdsEnum.conv2d:
            activation = layer.layer_params.activation.value
            kernel_size = layer.layer_params.kernel_size
            conv_strides = layer.layer_params.conv_strides
            filters_num = layer.layer_params.num_of_filters
            if i == 0:
                model.add(
                    layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation, input_shape=input_shape,
                                  strides=conv_strides))
            else:
                if not all([size == 1 for size in kernel_size]):
                    model.add(
                        layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation,
                                      strides=conv_strides))
                    # model.add(layers.Conv2D(filters_num, kernel_size=kernel_size, activation=activation,
                    #                   strides=conv_strides, data_format='channels_first'))
            if layer.layer_params.pool_size:
                pool_size = layer.layer_params.pool_size
                pool_strides = layer.layer_params.pool_strides
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
            model.add(layers.Dense(neurons_num, activation=activation))
        if i == len(chain.cnn_nodes) - 1:
            model.add(layers.Flatten())
            neurons_num = model.layers[len(model.layers) - 1].output_shape[1]
            model.add(layers.Dense(neurons_num, activation='relu'))
    # Output
    output_shape = 1 if classes == 2 else classes
    model.add(layers.Dense(output_shape, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.summary()
    return model
