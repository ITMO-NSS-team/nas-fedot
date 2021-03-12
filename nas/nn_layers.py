from typing import Any
from tensorflow.keras import layers
from nas.layer import LayerTypesIdsEnum


def make_dense_layer(idx: int, input_layer: Any, current_node: Any):
    activation = current_node.content['params']['activation']
    neurons_num = current_node.content['params']['neurons']
    dense_layer = layers.Dense(neurons_num, activation=activation, name=f'dense_layer_{idx}')
    dense_layer = dense_layer(input_layer)
    return dense_layer


def make_dropout_layer(idx: int, input_layer: Any, current_node: Any):
    drop = current_node.content['params']['drop']
    dropout = layers.Dropout(drop, name=f'dropout_layer_{idx}')
    dropout_layer = dropout(input_layer)
    return dropout_layer


def make_batch_norm_layer(idx: int, input_layer: Any, current_node: Any = None):
    momentum = current_node.content['params']['momentum']
    epsilon = current_node.content['params']['epsilon']
    batch_norm = layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=f'batch_norm_{idx}')
    batch_norm_layer = batch_norm(input_layer)
    return batch_norm_layer


def make_conv_layer(idx: int, input_layer: Any, current_node: Any = None, is_free_node=False):
    # Conv layer params
    activation = current_node.content['params']['activation']
    kernel_size = current_node.content['params']['kernel_size']
    conv_strides = current_node.content['params']['conv_strides']
    filters_num = current_node.content['params']['num_of_filters']
    conv_layer = layers.Conv2D(filters=filters_num, kernel_size=kernel_size, activation=activation,
                               strides=conv_strides, name=f'conv_layer_{idx}', padding='same')(input_layer)

    # Add pooling
    if is_free_node:
        if current_node.content['params']["pool_size"]:
            pool_size = current_node.content['params']["pool_size"]
            pool_strides = current_node.content['params']["pool_strides"]
            if current_node.content['params']["pool_type"] == LayerTypesIdsEnum.maxpool2d.value:
                pooling = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
            elif current_node.content['params']["pool_type"] == LayerTypesIdsEnum.averagepool2d.value:
                pooling = layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, padding='same')

            conv_layer = pooling(conv_layer)

    return conv_layer


def make_skip_connection_block(idx: int, input_layer: Any, current_node, layers_dict: dict):
    if current_node in layers_dict:
        tmp = layers_dict.pop(current_node)
        start_layer = tmp.pop(0)
        input_layer = layers.concatenate([start_layer, input_layer],
                                         axis=-1, name=f'residual_end_{idx}')
        input_layer = layers.Activation('relu')(input_layer)
        layers_dict[current_node] = tmp
    return input_layer
