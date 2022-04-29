from typing import Any
from tensorflow.keras import layers


def make_dense_layer(idx: int, input_layer: Any, current_node: Any):
    """
    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node NNNode type
    """
    neurons_num = current_node.content['params']['neurons']
    activation = layers.Activation(current_node.content['params']['activation'])
    dense_layer = layers.Dense(units=neurons_num, name=f'dense_layer_{idx}')(input_layer)
    if 'momentum' in current_node.content['params']:
        dense_layer = _add_batch_norm(dense_layer, current_node)
    dense_layer = activation(dense_layer)
    return dense_layer


def make_dropout_layer(idx: int, input_layer: Any, current_node: Any):
    """
    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node NNNode type
    """
    drop = current_node.content['params']['drop']
    dropout = layers.Dropout(drop, name=f'dropout_layer_{idx}')
    dropout_layer = dropout(input_layer)
    return dropout_layer


def make_conv_layer(idx: int, input_layer: Any, current_node: Any = None, is_free_node=False):
    """
    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node NNNode type
    :param is_free_node: is node not belongs to any skip connection block
    """
    # Conv layer params
    kernel_size = current_node.content['params']['kernel_size']
    conv_strides = current_node.content['params']['conv_strides']
    filters_num = current_node.content['params']['num_of_filters']
    activation = layers.Activation(current_node.content['params']['activation'])
    conv_layer = layers.Conv2D(filters=filters_num, kernel_size=kernel_size, strides=conv_strides,
                               name=f'conv_layer_{idx}', padding='same')(input_layer)
    if 'momentum' in current_node.content['params']:
        conv_layer = _add_batch_norm(input_layer=conv_layer, current_node=current_node)
    conv_layer = activation(conv_layer)
    # Add pooling
    if is_free_node:
        if current_node.content['params']["pool_size"]:
            pool_size = current_node.content['params']["pool_size"]
            pool_strides = current_node.content['params']["pool_strides"]
            if current_node.content['params']["pool_type"] == 'max_pool2d':
                pooling = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
            elif current_node.content['params']["pool_type"] == 'average_pool2d':
                pooling = layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
            else:
                raise ValueError('Wrong pooling type!')
            conv_layer = pooling(conv_layer)

    return conv_layer


def make_skip_connection_block(idx: int, input_layer: Any, current_node, layers_dict: dict):
    """
    Method that makes skip connection if current node has any. Otherwise, returns current layer as result

    :param idx: layer index
    :param input_layer: input Keras layer
    :param current_node: current node
    :param layers_dict: dictionary with skip connection start/end pairs of nodes
    :return:
    """
    if current_node in layers_dict:
        tmp = layers_dict.pop(current_node)
        start_layer = tmp.pop(0)
        input_layer = layers.concatenate([start_layer, input_layer],
                                         axis=-1, name=f'residual_end_{idx}')
        input_layer = layers.Activation('relu')(input_layer)
        layers_dict[current_node] = tmp
    return input_layer


def _add_batch_norm(input_layer: Any, current_node: Any):
    """
    Method that adds batch normalization layer if current node has batch_norm parameters

    :param input_layer: input Keras layer
    :param current_node: current node
    :return:
    """
    batch_norm_layer = layers.BatchNormalization(momentum=current_node.content['params']['momentum'],
                                                 epsilon=current_node.content['params']['epsilon'])(input_layer)
    return batch_norm_layer
