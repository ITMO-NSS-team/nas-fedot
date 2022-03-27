from typing import Any

from keras import layers
from keras import models
from keras import optimizers

# from fedot_old.core.models.data import InputData, OutputData
from fedot.core.data.data import InputData, OutputData
from nas.layer import LayerTypesIdsEnum
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import log_loss

import pickle

from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def keras_model_fit(model, input_data: InputData, verbose: bool = True, batch_size: int = 24,
                    epochs: int = 10):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
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
    evaluation_result = model.predict_proba(input_data.features)
    # sum_rows = np.sum(evaluation_result,axis=1).tolist()
    # print(sum_rows)
    # print(np.sum(evaluation_result))
    # evaluation_result = log_loss(input_data.target, eval)
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
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def create_nn_model(chain: Any, input_shape: tuple, classes: int = 3):
    generated_struc = generate_structure(chain.root_node)
    nn_structure = chain.cnn_nodes + generated_struc
    # nn_structure = chain.nodes + generate_structure(chain.root_node)
    make_keras_picklable()
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
        # if i == len(chain.cnn_nodes) - 1:
        if i == len(chain.cnn_nodes) - 1:
            model.add(layers.Flatten())
            neurons_num = model.layers[len(model.layers) - 1].output_shape[1]
            model.add(layers.Dense(neurons_num, activation='relu'))
    # Output
    output_shape = 1 if classes == 2 else classes
    activation_func = 'sigmoid' if classes == 2 else 'softmax'
    loss_func = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    model.add(layers.Dense(output_shape, activation=activation_func))

    model.compile(loss=loss_func, optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.summary()
    return model
