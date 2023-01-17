import datetime

from nas.composer import nn_composer_requirements as nas_requirements
from nas.graph.cnn.cnn_builder import CNNGenerator
from nas.graph.graph_builder import NNGraphBuilder


def get_requirements():
    data_requirements = nas_requirements.DataRequirements(split_params={'cv_folds': 3})
    conv_requirements = nas_requirements.ConvRequirements(input_shape=[20, 20], color_mode='RGB',
                                                          min_filters_num=32, max_filters_num=128,
                                                          kernel_size=[3, 3], conv_strides=[1, 1],
                                                          pool_size=[2, 2], pool_strides=[2, 2],
                                                          pool_types=['max_pool2d', 'average_pool2d'])
    fc_requirements = nas_requirements.FullyConnectedRequirements(min_number_of_neurons=32,
                                                                  max_number_of_neurons=128)
    nn_requirements = nas_requirements.ModelRequirements(conv_requirements=conv_requirements,
                                                         fc_requirements=fc_requirements,
                                                         primary=['conv2d'], secondary=['dense'],
                                                         epochs=1, batch_size=8,
                                                         max_nn_depth=3, max_num_of_conv_layers=18,
                                                         has_skip_connection=True
                                                         )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=1)

    return nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                   optimizer_requirements=optimizer_requirements,
                                                   nn_requirements=nn_requirements,
                                                   timeout=datetime.timedelta(hours=1),
                                                   pop_size=1,
                                                   num_of_generations=1)


def get_graph():
    requirements = get_requirements()
    builder = NNGraphBuilder()
    cnn_builder = CNNGenerator(requirements=requirements)
    builder.set_builder(cnn_builder)

    return builder.build()
