from random import choice

from fedot.core.optimisers.graph import OptGraph, OptNode
from nas.graph_nas_node import PrimaryNode, SecondaryNode, NNNodeGenerator
from nas.layer import LayerParams, LayerTypesIdsEnum
from nas.patches.load_images import from_images

image_size = [120, 120]


def train_opt_graph(file_path: str, graph: OptGraph = None):
    size = 120
    num_of_classes = 3
    dataset_to_compose, dataset_to_validate = from_images(file_path, num_classes=num_of_classes)

    conv_node_type = 'conv2d'
    activation = 'relu'
    kernel_size = (3, 3)
    conv_strides = (1, 1)
    num_of_filters = 16  # choice([16, 32, 64, 128])
    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_type = 'maxpool2d'

    layer_params = LayerParams(layer_type=conv_node_type, activation=activation, kernel_size=kernel_size,
                               conv_strides=conv_strides, num_of_filters=num_of_filters, pool_size=pool_size,
                               pool_strides=pool_strides, pool_type=pool_type)
    nn_layer_params = LayerParams(activation='relu', layer_type='dense', neurons=121)
    conv_node_1 = PrimaryNode(layer_params=layer_params)
    conv_node_2 = SecondaryNode(nodes_from=[conv_node_1], layer_params=layer_params)
    conv_node_3 = SecondaryNode(nodes_from=[conv_node_2], layer_params=LayerParams(layer_type='flatten'))
    nn_node_1 = SecondaryNode(nodes_from=[conv_node_3], layer_params=nn_layer_params)
    nn_node_2 = SecondaryNode(nodes_from=[nn_node_1], layer_params=nn_layer_params)

    nodes_list = [conv_node_1, conv_node_2, conv_node_3, nn_node_1, nn_node_2]
    if graph:
        for node in nodes_list:
            graph.add_node(node)

    graph.show(path='result.png')

    graph.fit()


if __name__ == '__main__':
    file_path = 'Generated_dataset'
    initial_graph = OptGraph()
    train_opt_graph(file_path=file_path, graph=initial_graph)

