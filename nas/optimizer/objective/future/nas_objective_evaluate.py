from fedot.core.optimisers.objective import DataSource
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective import ObjectiveEvaluate
from tqdm import tqdm

from nas.graph.BaseGraph import NasGraph


class ObjectiveEvaluate(ObjectiveEvaluate):
    """
    This class defines how Objective will be evaluated for neural network like graph structure.
    """
    def __init__(self,
                 objective,
                 data_producer: DataSource,
                 model_handler,
                 requirements,
                 verbose_level = None,
                 eval_n_jobs: int = 1,
                 **objective_kwargs):
        super().__init__(objective=objective, eval_n_jobs=eval_n_jobs, **objective_kwargs)
        self._data_producer = data_producer
        self._model_handler = model_handler


    def evaluate(self, graph: NasGraph) -> Fitness:
        # create pbar only if there is evaluation on folds
        fold_pbar = tqdm(self._data_producer) if folds else self._data_producer
        for fold_id, (train_data, test_data) in enumerate(fold_pbar):
            pass

