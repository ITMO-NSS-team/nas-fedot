from ctypes import Union
from dataclasses import dataclass
from enum import Enum

import tensorflow as tf

from nas.repository.layer_types_enum import ActivationTypesIdsEnum


class KerasActivations:
    @staticmethod
    def relu():
        return tf.keras.activations.relu

    @classmethod
    def get_activations_func(cls, activation_func: Union[str, ActivationTypesIdsEnum]):
        activation_types = {}