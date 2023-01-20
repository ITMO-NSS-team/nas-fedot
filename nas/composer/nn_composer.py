from __future__ import annotations

import pathlib
from typing import Sequence, Tuple, Union, Optional

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.graph import OptGraph

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.data.dataset.builder import BaseNasDatasetBuilder
from nas.graph.cnn.cnn_graph import NasGraph
from nas.optimizer.objective.nn_objective_evaluate import NasObjectiveEvaluate


class NasComposer(Composer):
    def __init__(self, optimizer: EvoGraphOptimizer,
                 composer_requirements: NNComposerRequirements,
                 pipelines_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None):
        super().__init__(optimizer, composer_requirements)

        self.best_models = ()
        self._data_transformer: Optional[BaseNasDatasetBuilder] = None
        self.pipelines_cache = pipelines_cache
        self.preprocessing_cache = preprocessing_cache

    def _convert_opt_results_to_nn_graph(self, graphs: Sequence[OptGraph]) -> Tuple[NasGraph, Sequence[NasGraph]]:
        adapter = self.optimizer.graph_generation_params.adapter
        multi_objective = self.optimizer.objective.is_multi_objective
        best_graphs = [adapter.restore(graph) for graph in graphs]
        best_graph = best_graphs if multi_objective else best_graphs[0]
        return best_graph, best_graphs

    def set_data_transformer(self, transformer: BaseNasDatasetBuilder) -> NasComposer:
        self._data_transformer = transformer
        return self

    def set_callbacks(self, callbacks):
        raise NotImplementedError

    def compose_pipeline(self, data: Union[InputData, MultiModalData], optimization_verbose=None) -> NasGraph:
        """ Method for objective evaluation"""

        data.shuffle()
        if self.history:
            self.history.clean_results()

        data_producer = DataSourceSplitter(self.composer_requirements.cv_folds).build(data)

        objective_evaluator = NasObjectiveEvaluate(self.optimizer.objective, data_producer, self._data_transformer,
                                                   self.composer_requirements, self.pipelines_cache,
                                                   self.preprocessing_cache, optimization_verbose)
        objective_function = objective_evaluator.evaluate

        if self.composer_requirements.collect_intermediate_metric:
            self.optimizer.set_evaluation_callback(objective_evaluator.evaluate_intermediate_metrics)

        opt_result = self.optimizer.optimise(objective_function)
        best_model, self.best_models = self._convert_opt_results_to_nn_graph(opt_result)
        self.log.info('NAS composition has been finished')
        return best_model

    def save(self, path):
        path = pathlib.Path('..', path) if not isinstance(path, pathlib.Path) else path
        path.mkdir(parents=True, exist_ok=True)
        self.log.info(f'Saving results into {path.resolve()}')
        if self.best_models:
            graph = self.best_models[0]
            if not isinstance(graph, NasGraph):
                graph = self.graph_generation_params.adapter.restore(graph)
            graph.save(path)

        if self.history:
            self.history.save(path / 'history.json')
