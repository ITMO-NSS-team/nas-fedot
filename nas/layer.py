from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional


class ActivationTypesIdsEnum(Enum):
    softmax = 'softmax'
    elu = 'elu'
    selu = 'selu'
    softplus = 'softplus'
    relu = 'relu'
    softsign = 'softsign'
    tanh = 'tanh'
    hard_sigmoid = 'hard_sigmoid'
    sigmoid = 'sigmoid'
    linear = 'linear'

activation_types = [type_ for type_ in ActivationTypesIdsEnum]

class LayerTypesIdsEnum(Enum):
    conv2d = 'conv2d'
    flatten = 'flatten'
    dense = 'dense'
    dropout = 'dropout'
    maxpool2d = 'maxpool2d'
    averagepool2d = 'averagepool2d'
    serial_connection = 'serial_connection'
    description_layer = 'node_layer'

    @classmethod
    def description(cls, name):
        return cls.description_layer.value

@dataclass
class LayerParams:
    layer_type: LayerTypesIdsEnum
    pool_type: Optional[LayerTypesIdsEnum] = None
    neurons: int = None
    max_params: int = None
    activation: str = None
    drop: float = None
    pool_size: Tuple[int, int] = None
    kernel_size: Tuple[int, int] = None
    conv_strides: Tuple[int, int] = None
    pool_strides: Tuple[int, int] = None
    num_of_filters: int = None
    output_shape: List[float] = None
