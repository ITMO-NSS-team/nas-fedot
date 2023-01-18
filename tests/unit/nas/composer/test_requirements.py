from __future__ import annotations

from nas.composer.nn_composer_requirements import BaseLayerRequirements


def test_fc_req():
    requirements = BaseLayerRequirements(min_number_of_neurons=234, max_number_of_neurons=1456)
    assert requirements.neurons_num in [2 ** n for n in range(11)]


def test_fc_req_the_least_neurons():
    try:
        BaseLayerRequirements(min_number_of_neurons=0, max_number_of_neurons=1)
    except ValueError:
        return True
    return False


def test_fc_req_min_max_neurons():
    try:
        BaseLayerRequirements(min_number_of_neurons=200, max_number_of_neurons=100)
    except ValueError:
        return True
    return False
