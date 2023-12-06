from __future__ import annotations

from nas.composer.requirements import BaseLayerRequirements


def test_neurons_num():
    requirements = BaseLayerRequirements(min_number_of_neurons=234, max_number_of_neurons=1456)
    powers_of_two = [2 ** n for n in range(12)]
    assert all(item in powers_of_two for item in requirements.neurons_num)


def test_neurons_range():
    try:
        BaseLayerRequirements(min_number_of_neurons=0, max_number_of_neurons=1)
    except ValueError:
        return True
    return False


def test_min_max_neurons():
    try:
        BaseLayerRequirements(min_number_of_neurons=200, max_number_of_neurons=100)
    except ValueError:
        return True
    return False


def test_base_probabilities():
    requirements = BaseLayerRequirements()
    assert requirements.batch_norm_prob
    assert requirements.dropout_prob
    assert requirements.max_dropout_val


def test_set_probs():
    requirements = BaseLayerRequirements()
    requirements.set_dropout_prob(0.2).set_batch_norm_prob(0.3).set_max_dropout_val(0.1)
    assert requirements.max_dropout_val == 0.1
    assert requirements.dropout_prob == 0.2
    assert requirements.batch_norm_prob == 0.3
