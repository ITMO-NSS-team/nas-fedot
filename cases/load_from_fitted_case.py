import datetime
import os
import pathlib

import tensorflow as tf
from fedot.core.composer.advisor import DefaultChangeAdvisor
from golem.core.optimisers.adapters import DirectAdapter
from golem.core.optimisers.genetic.individual import Individual
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum, Mutation
from golem.core.optimisers.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams

import nas.composer.nn_composer_requirements as nas_requirements
from nas.graph.cnn_graph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.node_factory import NNNodeFactory
from nas.utils.utils import project_root, set_root

set_root(project_root())


def from_fitted():
    # path_to_model = pathlib.Path('../_results/debug/master/2022-09-05/fitted_model.h5')
    # # model = tf.keras.models.load_model(path_to_model)
    # model = tf.keras.applications.resnet.ResNet152()
    graph_path = os.path.join('/home/staeros/_results/resnet_comp/2/2022-10-04/graph.json')
    model_path = os.path.join('/home/staeros/_results/resnet_comp/2/2022-10-04/fitted_model.h5')
    model = tf.keras.models.load_model(model_path)
    graph = NasGraph.load(str(graph_path))
    graph.model = model
    history = OptHistory.load('/home/staeros/_results/resnet_comp/2/2022-10-04/history.json')
    history.show(per_time=False)
    # history = OptHistory.load('/home/staeros/_results/debug/master_2/2022-09-07/history.json')
    # history.show.fitness_line_interactive(per_time=False)

    cv_folds = 3
    image_side_size = 20
    batch_size = 8
    epochs = 1
    optimization_epochs = 1

    data_requirements = nas_requirements.DataRequirements(split_params={'cv_folds': cv_folds})
    conv_requirements = nas_requirements.ConvRequirements(input_shape=[image_side_size, image_side_size],
                                                          color_mode='RGB',
                                                          min_number_of_neurons=32, max_number_of_neurons=64,
                                                          conv_strides=[[1, 1]],
                                                          pool_size=[[2, 2]], pool_strides=[[2, 2]],
                                                          pool_types=['max_pool2d', 'average_pool2d'])
    fc_requirements = nas_requirements.BaseLayerRequirements(min_number_of_neurons=32,
                                                             max_number_of_neurons=64)
    nn_requirements = nas_requirements.ModelRequirements(conv_requirements=conv_requirements,
                                                         fc_requirements=fc_requirements,
                                                         primary=['conv2d'], secondary=['dense'],
                                                         epochs=epochs, batch_size=batch_size,
                                                         max_nn_depth=2, max_num_of_conv_layers=5,
                                                         has_skip_connection=True
                                                         )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                           optimizer_requirements=optimizer_requirements,
                                                           model_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(hours=200),
                                                           pop_size=10,
                                                           num_of_generations=10)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode),
        rules_for_constraint=[], node_factory=NNNodeFactory(requirements, DefaultChangeAdvisor()))

    path = pathlib.Path('/home/staeros/_results/broken_mutation/graph.json')
    graph = Individual(NasGraph.load(path))

    mutation = MutationTypesEnum.simple

    mutator = Mutation([mutation], requirements, graph_generation_parameters)

    mutator(graph)

    print('Done!')


if __name__ == '__main__':
    from_fitted()
