import pathlib
from typing import Sequence, Tuple, Union, Optional

from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams

from nas.composer.nn_composer_requirements import NNComposerRequirements
from nas.data.data_generator import Preprocessor
from nas.graph.cnn.cnn_graph import NNGraph
from nas.optimizer.objective.nn_objective_evaluate import NNObjectiveEvaluate


class NNComposer(Composer):
    def __init__(self, optimizer: EvoGraphOptimizer,
                 composer_requirements: NNComposerRequirements, history: OptHistory = None,
                 pipelines_cache=None, preprocessing_cache=None,
                 graph_generation_params: Optional[GraphGenerationParams] = None):
        super().__init__(optimizer, composer_requirements)
        self.graph_generation_params = graph_generation_params

        self._preprocessor = Preprocessor()

        self.composer_requirements = composer_requirements
        self.pipelines_cache = pipelines_cache
        self.preprocessing_cache = preprocessing_cache

        self._full_history_dir = history
        self.best_models = ()

    def _convert_opt_results_to_nn_graph(self, graphs: Sequence[OptGraph]) -> Tuple[NNGraph, Sequence[NNGraph]]:
        adapter = self.optimizer.graph_generation_params.adapter
        multi_objective = self.optimizer.objective.is_multi_objective
        best_graphs = [adapter.restore(graph) for graph in graphs]
        best_graph = best_graphs if multi_objective else best_graphs[0]
        return best_graph, best_graphs

    def set_preprocessor(self, preprocessor):
        self._preprocessor = preprocessor
        return self

    def set_callbacks(self, callbacks):
        raise NotImplementedError

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> NNGraph:
        """ Method for objective evaluation"""

        data.shuffle()
        if self.history:
            self.history.clean_results()

        data_producer = DataSourceSplitter(self.composer_requirements.cv_folds).build(data)

        objective_evaluator = NNObjectiveEvaluate(self.optimizer.objective, data_producer, self._preprocessor,
                                                  self.composer_requirements, self.pipelines_cache,
                                                  self.preprocessing_cache)
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
            if not isinstance(graph, NNGraph):
                graph = self.graph_generation_params.adapter.restore(graph)
            graph.save(path)
            # graph.show(path / 'graph.png')

        if self.history:
            self.history.save(path / 'history.json')
