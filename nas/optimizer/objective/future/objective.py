from typing import Dict, Callable, Any, Optional

from golem.core.optimisers.objective import Objective


class NasObjective(Objective):
    """
    Implementation of GOLEM's Objective for NAS specific task.
    """

    def __init__(self,
                 quality_metrics: Dict[Any, Callable],
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        super().__init__(quality_metrics=quality_metrics,
                         complexity_metrics=complexity_metrics,
                         is_multi_objective=is_multi_objective)

    def __call__(self, trainer):
        pass
