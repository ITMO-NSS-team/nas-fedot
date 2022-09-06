import gc
from typing import Optional, TypeVar

import numpy as np
from fedot.core.dag.graph import Graph
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.optimisers.fitness import Fitness
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import ObjectiveEvaluate
from fedot.core.optimisers.objective.objective import to_fitness
from fedot.core.repository.tasks import TaskTypesEnum, Task
from tensorflow.python.keras.backend import clear_session

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.data.data_generator import Preprocessor, DataGenerator
from nas.data.setup_data import setup_data
from nas.graph.cnn.cnn_graph import NNGraph

G = TypeVar('G', Graph, OptGraph)


class NNObjectiveEvaluate(ObjectiveEvaluate[G]):
    def __init__(self, objective, data_producer, preprocessor: Preprocessor, requirements: NNComposerRequirements,
                 pipeline_cache=None, preprocessing_cache=None, eval_n_jobs=1, **objective_kwargs):
        # Add cache
        super().__init__(objective, eval_n_jobs, **objective_kwargs)
        self._data_producer = data_producer
        self._requirements = requirements
        self._pipeline_cache = pipeline_cache
        self._preprocessing_cache = preprocessing_cache
        self._preprocessor = preprocessor
        self._log = default_log(self)

    def evaluate_objective(self, graph: NNGraph, data: InputData, fold_id: Optional[int] = None) -> None:
        shuffle = True if data.task != Task(TaskTypesEnum.ts_forecasting) else False
        data_to_train, data_to_validate = train_test_data_setup(data, shuffle_flag=True)

        train_generator = setup_data(data_to_train, self._requirements.nn_requirements.batch_size, self._preprocessor,
                                     'train', DataGenerator, shuffle)
        validation_generator = setup_data(data_to_validate, self._requirements.nn_requirements.batch_size,
                                          self._preprocessor, 'train', DataGenerator, shuffle)
        graph.fit(train_generator, validation_generator, self._requirements, data.num_classes, self._preprocessor,
                  shuffle=shuffle)

    def calculate_objective(self, graph: NNGraph, reference_data: InputData, fold_id: Optional[int] = None):
        # TODO
        test_generator = setup_data(reference_data, 1, self._preprocessor, 'test', DataGenerator, False)
        return self._objective(graph, reference_data=test_generator)

    def evaluate(self, graph: NNGraph) -> Fitness:
        graph.log = self._log
        graph_id = graph.root_node.descriptive_id

        self._log.debug(f'Fit for graph {graph_id} has started.')

        folds_metrics = []
        for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
            try:
                # TODO add cache support.
                self.evaluate_objective(graph, train_data, fold_id)
            except Exception as ex:
                self._log.warning(f'Continuing after graph fit error {ex}\nfor graph: {graph_id}')
                continue
            evaluated_fitness = self.calculate_objective(graph, reference_data=test_data)

            if evaluated_fitness.valid:
                folds_metrics.append(evaluated_fitness.values)
            else:
                self._log.warning(f'Continuing after objective evaluation error for graph: {graph_id}')
                continue

            clear_session()
            gc.collect()
            graph.model = None

        if folds_metrics:
            folds_metrics = tuple(np.mean(folds_metrics, axis=0))
            self._log.debug(f'Graph {graph_id} with evaluated metrics: {folds_metrics}')
        else:
            folds_metrics = None
        return to_fitness(folds_metrics, self._objective.is_multi_objective)
