from typing import Any
from tensorflow.keras import layers
from nas.layer import LayerTypesIdsEnum


def make_dense_layer(idx: int, input_layer: Any, current_node: Any):
    neurons_num = current_node.content['params']['neurons']
    activation = layers.Activation(current_node.content['params']['activation'])
    dense_layer = layers.Dense(neurons_num, name=f'dense_layer_{idx}')
    if current_node.content['batch_norm']:
        batch_norm_layer = _add_batch_norm(dense_layer, current_node)
        dense_layer = batch_norm_layer(dense_layer)
    dense_layer = activation(dense_layer)
    dense_layer = dense_layer(input_layer)
    return dense_layer


def make_dropout_layer(idx: int, input_layer: Any, current_node: Any):
    drop = current_node.content['params']['drop']
    dropout = layers.Dropout(drop, name=f'dropout_layer_{idx}')
    dropout_layer = dropout(input_layer)
    return dropout_layer


def make_conv_layer(idx: int, input_layer: Any, current_node: Any = None, is_free_node=False):
    # Conv layer params
    kernel_size = current_node.content['params']['kernel_size']
    conv_strides = current_node.content['params']['conv_strides']
    filters_num = current_node.content['params']['num_of_filters']
    activation = layers.Activation(current_node.content['params']['activation'])
    conv_layer = layers.Conv2D(filters=filters_num, kernel_size=kernel_size, strides=conv_strides,
                               name=f'conv_layer_{idx}', padding='same')(input_layer)
    if current_node.content['batch_norm']:
        conv_layer = _add_batch_norm(input_layer=conv_layer, current_node=current_node)
    conv_layer = activation(conv_layer)
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


def _add_batch_norm(input_layer: Any, current_node):
    batch_norm_layer = layers.BatchNormalization(momentum=current_node.content['params']['momentum'],
                                                 epsilon=current_node.content['params']['epsilon'])
    batch_norm_layer = batch_norm_layer(input_layer)
    return batch_norm_layer
