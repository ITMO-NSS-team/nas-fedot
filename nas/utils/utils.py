import os
import sys

import tensorflow as tf

from pathlib import Path


def set_root(root: Path):
    os.chdir(root)
    sys.path.append(root)


def project_root() -> Path:
    """Returns FEDOT project root folder."""
    return Path(__file__).parent.parent.parent


def set_tf_compat():
    setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)
