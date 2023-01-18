from __future__ import annotations

from nas.composer.nn_composer_requirements import ConvRequirements


def test_set_shape_is_iterable():
    requirements = ConvRequirements()
    requirements.set_conv_params(stride=1)
    requirements.set_pooling_size(2)
    requirements.set_pooling_stride(3)
    assert hasattr(requirements.conv_strides, '__iter__')
    assert hasattr(requirements.pool_size, '__iter__')
    assert hasattr(requirements.pool_strides, '__iter__')


def test_set_params():
    requirements = ConvRequirements()
    requirements.set_conv_params(2).set_conv_params(3)
    requirements.set_pooling_size(1)
    requirements.set_pooling_size(2)
    requirements.set_pooling_size(3)
    requirements.set_pooling_stride(1)
    requirements.set_pooling_stride(2)
    requirements.set_pooling_stride(3)
    assert len(requirements.pool_size) == 3
    assert len(requirements.pool_strides) == 3
    assert len(requirements.conv_strides) == 3
