import random
from typing import List, Optional

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.graph.cnn.cnn_graph import NNGraph, NNNode
from nas.graph.grpah_generator import GraphGenerator
from nas.repository.layer_types_enum import LayersPoolEnum, GraphLayers


# def get_layer_params(layer_type: str, requirements=None):
#     layer_params = _get_random_layer_params(layer_type, requirements)
#     return layer_params
#
#
# def _get_conv2d_requirements(requirements: NNComposerRequirements):
#     conv_node_type = random.choice(requirements.nn_requirements.primary)
#     activation = random.choice(requirements.nn_requirements.activation_types).value
#     kernel_size = random.choice(requirements.nn_requirements.conv_requirements.kernel_size)
#     conv_strides = random.choice(requirements.nn_requirements.conv_requirements.conv_strides)
#     num_of_filters = random.choice(requirements.nn_requirements.conv_requirements.filters)
#     pool_size = random.choice(requirements.nn_requirements.conv_requirements.pool_size)
#     pool_strides = random.choice(requirements.nn_requirements.conv_requirements.pool_strides)
#     pool_type = random.choice(requirements.nn_requirements.conv_requirements.pool_types)
#     return {'layer_type': conv_node_type, 'activation': activation, 'kernel_size': kernel_size,
#             'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
#             'pool_strides': pool_strides, 'pool_type': pool_type}


# def get_layer_params(layer_type: KerasLayersEnum, requirements=None):
#     layer_params = _get_random_layer_params(layer_type, requirements)
#     return layer_params


def _get_conv2d_requirements(requirements: NNComposerRequirements):
    conv_node_type = random.choice(requirements.nn_requirements.primary)
    activation = random.choice(requirements.nn_requirements.activation_types).value
    kernel_size = random.choice(requirements.nn_requirements.conv_requirements.kernel_size)
    conv_strides = random.choice(requirements.nn_requirements.conv_requirements.conv_strides)
    num_of_filters = random.choice(requirements.nn_requirements.conv_requirements.filters)
    pool_size = random.choice(requirements.nn_requirements.conv_requirements.pool_size)
    pool_strides = random.choice(requirements.nn_requirements.conv_requirements.pool_strides)
    pool_type = random.choice(requirements.nn_requirements.conv_requirements.pool_types)
    return {'layer_type': conv_node_type, 'activation': activation, 'kernel_size': kernel_size,
            'conv_strides': conv_strides, 'num_of_filters': num_of_filters, 'pool_size': pool_size,
            'pool_strides': pool_strides, 'pool_type': pool_type}


def _get_random_layer_params(layer_type: str, requirements: NNComposerRequirements):
    layer_params = {'n_jobs': 1}
    if layer_type == 'conv2d':
        layer_params = _get_conv2d_requirements(requirements) | layer_params
    elif layer_type == 'serial_connection':
        layer_params = {'layer_type': layer_type} | layer_params
    elif layer_type == 'dropout':
        drop_value = random.randint(1, (requirements.nn_requirements.max_drop_size * 10)) / 10
        layer_params = {'drop': drop_value} | layer_params
    elif layer_type == 'batch_normalization':
        momentum = random.uniform(0, 1)
        epsilon = random.uniform(0, 1)
        layer_params = {'momentum': momentum, 'epsilon': epsilon}
    elif layer_type == 'dense':
        activation = random.choice(requirements.nn_requirements.activation_types).value
        neurons = random.choice(requirements.nn_requirements.fc_requirements.neurons_num)
        layer_params = {'layer_type': layer_type, 'neurons': neurons, 'activation': activation} | layer_params
    return layer_params


def _add_skip_connections(graph: NNGraph, params):
    skip_connections_id = params[0]
    shortcut_len = params[1]
    for current_node in skip_connections_id:
        if (current_node + shortcut_len) < len(graph.nodes):
            graph.nodes[current_node + shortcut_len].nodes_from.append(graph.nodes[current_node])
        else:
            print('Wrong connection. Connection dropped.')
        # is_first_conv = current_node <= graph.cnn_depth
        # is_second_conv = current_node + shortcut_len < graph.cnn_depth
        # if is_first_conv == is_second_conv and (current_node + shortcut_len) < len(graph.nodes):
        #     graph.nodes[current_node + shortcut_len].nodes_from.append(graph.nodes[current_node])
        # else:
        #     print('Wrong connection. Connection dropped.')


class CNNGenerator(GraphGenerator):

    def __init__(self, nodes_list: Optional[List[str]] = None,
                 requirements: NNComposerRequirements = None):
        self.nodes = nodes_list
        self.requirements = requirements

    # TODO fix
    @staticmethod
    def _get_skip_connection_params(graph):
        """Method for skip connection parameters generation"""
        connections = set()
        skips_len = random.randint(2, len(graph.nodes) // 2)
        max_number_of_skips = len(graph.nodes) // 3
        for _ in range(max_number_of_skips):
            node_id = random.randint(0, len(graph.nodes))
            connections.add(node_id)
        return connections, skips_len

    def _generate_random_struct(self) -> List[str]:
        """ Method for generate random graph structure if initial structure isn't specified"""

        conv_depth = random.randint(self.requirements.nn_requirements.min_num_of_conv_layers,
                                    self.requirements.nn_requirements.max_num_of_conv_layers)
        nn_depth = random.randint(self.requirements.nn_requirements.min_nn_depth,
                                  self.requirements.nn_requirements.max_nn_depth)
        struct = ['conv2d']
        for i in range(1, conv_depth + nn_depth):
            if i < conv_depth:
                node = random.choice(self.requirements.nn_requirements.primary) if i != conv_depth - 1 else 'flatten'
            else:
                node = random.choice(self.requirements.nn_requirements.secondary)
            struct.append(node)
        return struct

    def _add_node(self, node_type, nodes_from):
        node_params = get_layer_params(node_type, self.requirements)
        node = NNNode(content={'name': node_type, 'params': node_params}, nodes_from=nodes_from)
        if node_type == 'flatten':
            return node
        if random.random() > self.requirements.nn_requirements.batch_norm_prob:
            batch_norm_params = get_layer_params('batch_normalization', self.requirements)
            node.content['params'] = node.content['params'] | batch_norm_params
        if random.random() > self.requirements.nn_requirements.dropout_prob:
            dropout_params = get_layer_params('dropout', self.requirements)
            node.content['params'] = node.content['params'] | dropout_params
        return node

    def build_graph(self) -> NNGraph:
        graph = NNGraph()
        parent_node = None
        nodes = self.nodes or self._generate_random_struct()
        for node_name in nodes:
            node = self._add_node(node_name, parent_node)
            parent_node = [node]
            graph.add_node(node)
        if self.requirements.nn_requirements.has_skip_connection:
            _add_skip_connections(graph, self._get_skip_connection_params(graph))
        return graph


class ConvGraphMaker:
    def __init__(self, requirements, initial_struct: Optional[List] = None):
        self._initial_struct = initial_struct
        self._requirements = GraphLayers(layer_parameters=requirements)

    @property
    def initial_struct(self):
        return self._initial_struct

    @property
    def requirements(self):
        return self._requirements

    @staticmethod
    def _get_skip_connection_params(graph):
        """Method for skip connection parameters generation"""
        connections = set()
        skips_len = random.randint(2, len(graph.nodes) // 2)
        max_number_of_skips = len(graph.nodes) // 3
        for _ in range(max_number_of_skips):
            node_id = random.randint(0, len(graph.nodes))
            connections.add(node_id)
        return connections, skips_len

    def _generate_from_scratch(self):
        total_conv_nodes = random.randint(self.requirements.layer_parameters.min_num_of_conv_layers,
                                          self.requirements.layer_parameters.max_num_of_conv_layers)
        total_fc_nodes = random.randint(self.requirements.layer_parameters.min_nn_depth,
                                        self.requirements.layer_parameters.max_nn_depth)
        zero_node = random.choice([LayersPoolEnum.conv2d, LayersPoolEnum.dilation_conv2d])
        graph_nodes = [zero_node]
        for i in range(1, total_conv_nodes + total_fc_nodes):
            if i < total_conv_nodes:
                node = random.choice(self.requirements.layer_parameters.primary) \
                    if i != total_conv_nodes - 1 else LayersPoolEnum.flatten
            else:
                node = random.choice(self.requirements.layer_parameters.secondary)
            graph_nodes.append(node)
        return graph_nodes

    def _add_node(self, node_to_add, parent_node):
        node_params = self.requirements(node=node_to_add)
        node = NNNode(content={'name': node_to_add, 'params': node_params}, nodes_from=parent_node)
        # if node_to_add == KerasLayersEnum.flatten:
        #     return node
        # if random.random() > self.requirements.batch_norm_prob:
        #     batch_norm_params = get_layer_params(KerasLayersEnum.batch_normalization, self.requirements)
        #     node.content['params'] = node.content['params'] | batch_norm_params
        # if random.random() > self.requirements.dropout_prob:
        #     dropout_params = get_layer_params(KerasLayersEnum.dropout, self.requirements)
        #     node.content['params'] = node.content['params'] | dropout_params
        return node

    def build(self) -> NNGraph:
        graph = NNGraph()
        parent_node = None
        graph_nodes = self.initial_struct if self.initial_struct else self._generate_from_scratch()
        for node in graph_nodes:
            node = self._add_node(node, parent_node)
            parent_node = [node]
            graph.add_node(node)
        if self.requirements.layer_parameters.has_skip_connection:
            _add_skip_connections(graph, self._get_skip_connection_params(graph))
        return graph


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
                                                      primary=[LayersPoolEnum.conv2d, LayersPoolEnum.dilation_conv2d],
                                                      secondary=[LayersPoolEnum.dense],
                                                      epochs=epochs, batch_size=batch_size,
                                                      max_nn_depth=2, max_num_of_conv_layers=30,
                                                      has_skip_connection=True
                                                      )
    optimizer_requirements = nas_requirements.OptimizerRequirements(opt_epochs=optimization_epochs)

    requirements = nas_requirements.NNComposerRequirements(data_requirements=data_requirements,
                                                           optimizer_requirements=optimizer_requirements,
                                                           nn_requirements=nn_requirements,
                                                           timeout=datetime.timedelta(hours=200),
                                                           pop_size=10,
                                                           num_of_generations=10)

    maker = ConvGraphMaker(requirements=requirements.nn_requirements)
    maker.build()
    print('Done!')
