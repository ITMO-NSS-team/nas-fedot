import random
from enum import Enum
from dataclasses import dataclass
from functools import partial

import tensorflow as tf


class FrameworkTypesEnum(Enum):
    keras = 'keras'
    torch = 'torch'


class LayersPoolEnum(Enum):
    conv2d = 'conv2d'
    dilation_conv2d = 'dilation_conv2d'
    flatten = 'flatten'
    dense = 'dense'
    dropout = 'dropout'
    batch_norm = 'batch_norm'


class GraphLayers:
    def __init__(self, layer_parameters, framework: FrameworkTypesEnum = FrameworkTypesEnum.keras):
        self.layer_parameters = layer_parameters

    def __call__(self, *args, **kwargs):
        layer_type = kwargs.get('node')
        return self.layer_by_type(layer_type)

    def _conv2d(self):
        """Returns dictionary with particular layer parameters as NNGraph"""
        layer_parameters = dict()

        layer_parameters['name'] = LayersPoolEnum.conv2d
        layer_parameters['activation'] = random.choice(self.layer_parameters.activation_types)
        layer_parameters['kernel_size'] = random.choice(self.layer_parameters.conv_requirements.kernel_size)
        layer_parameters['conv_strides'] = random.choice(self.layer_parameters.conv_requirements.conv_strides)
        layer_parameters['num_of_filters'] = random.choice(self.layer_parameters.conv_requirements.filters)
        return layer_parameters

    def _dilation_conv2d(self):
        layer_parameters = dict()

        layer_parameters['name'] = LayersPoolEnum.dilation_conv2d
        layer_parameters['activation'] = random.choice(self.layer_parameters.activation_types)
        layer_parameters['dilation_rate'] = random.choice(self.layer_parameters.conv_requirements.dilation_rate)
        layer_parameters['kernel_size'] = random.choice(self.layer_parameters.conv_requirements.kernel_size)
        layer_parameters['conv_strides'] = random.choice(self.layer_parameters.conv_requirements.conv_strides)
        layer_parameters['num_of_filters'] = random.choice(self.layer_parameters.conv_requirements.filters)
        return layer_parameters

    def _dense(self):
        layer_parameters = dict()
        layer_parameters['activation'] = random.choice(self.layer_parameters.activation_types)
        layer_parameters['neurons'] = random.choice(self.layer_parameters.fc_requirements.neurons_num)
        return layer_parameters

    def _dropout(self):
        layer_parameters = dict()

        layer_parameters['drop'] = random.randint(1, self.layer_parameters.max_drop_size * 10) / 10
        return layer_parameters

    def _flatten(self):
        return {'n_jobs': 1}

    def layer_by_type(self, layer_type: LayersPoolEnum,
                      framework_type: FrameworkTypesEnum = FrameworkTypesEnum.keras):
        layers = {
            LayersPoolEnum.conv2d: self._conv2d(),
            LayersPoolEnum.dilation_conv2d: self._dilation_conv2d(),
            LayersPoolEnum.flatten: self._flatten(),
            LayersPoolEnum.dense: self._dense(),
            LayersPoolEnum.dropout: self._dropout()
        }

        if layer_type in layers:
            return layers[layer_type]
        else:
            raise NotImplementedError


class KerasLayersEnum(Enum):
    conv2d = tf.keras.layers.Conv2D
    dense = tf.keras.layers.Dense
    dilation_conv = partial(tf.keras.layers.Conv2D, dilation_rate=(2, 2))
    flatten = tf.keras.layers.Flatten
    batch_normalization = tf.keras.layers.BatchNormalization
    dropout = tf.keras.layers.Dropout


if __name__ == '__main__':

    import nas.composer.nn_composer_requirements as nas_requirements
    import datetime

    cv_folds = 3
    image_side_size = 20
    batch_size = 8
    epochs = 1
    optimization_epochs = 1

    data_requirements = nas_requirements.DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = nas_requirements.ConvRequirements(input_shape=[image_side_size, image_side_size],
                                                          color_mode='RGB',
                                                          min_filters=32, max_filters=64,
                                                          kernel_size=[[3, 3], [1, 1], [5, 5], [7, 7]],
                                                          conv_strides=[[1, 1]],
                                                          pool_size=[[2, 2]], pool_strides=[[2, 2]],
                                                          pool_types=['max_pool2d', 'average_pool2d'])
    fc_requirements = nas_requirements.FullyConnectedRequirements(min_number_of_neurons=32,
                                                                  max_number_of_neurons=64)
    nn_requirements = nas_requirements.NNRequirements(conv_requirements=conv_requirements,
                                                      fc_requirements=fc_requirements,
                                                      primary=['conv2d'], secondary=['dense'],
                                                      epochs=epochs, batch_size=batch_size,
                                                      max_nn_depth=2, max_num_of_conv_layers=15,
                                                      has_skip_connection=True
                                                      )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                           optimizer_requirements=optimizer_requirements,
                                                           nn_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(hours=200),
                                                           pop_size=10,
                                                           num_of_generations=10)

    layers_repo = GraphLayers(requirements.nn_requirements)
    layers_repo(node=LayersPoolEnum.conv2d)

    print('Done!')
