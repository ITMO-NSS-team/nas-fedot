from typing import Sequence, Union, Callable

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.graph import OptGraph

from nas.operations.validation_rules.cnn_val_rules import unique_node_types, flatten_count, has_no_flatten_skip, \
    graph_has_several_starts, graph_has_wrong_structure


# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class NNGraphVerifier:
    """ Class to verify graph using specified rules """
    def __init__(self,
                 rules: Sequence[VerifierRuleType] = ()):
        self._rules = rules
        self._log = default_log(self)

    def __call__(self, graph: Union[Graph, OptGraph]) -> bool:
        return self.verify(graph)

    def verify(self, graph: Union[Graph, OptGraph]) -> bool:
        # Check if all rules pass
        for rule in self._rules:
            try:
                if rule(graph) is False:
                    return False
            except ValueError as err:
                self._log.debug(f'Graph verification failed with error <{err}> '
                                f'for rule={rule} on graph={graph.root_node.descriptive_id}.')
                return False
        return True


def verifier_with_all_rules():
    rules = [unique_node_types, flatten_count,
             has_no_flatten_skip, graph_has_several_starts, graph_has_wrong_structure]
    return NNGraphVerifier(rules=rules)
