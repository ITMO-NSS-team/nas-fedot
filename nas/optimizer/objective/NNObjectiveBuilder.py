from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder


class NNObjectiveBuilder(DataObjectiveBuilder):
    def __init__(self, objective, max_pipeline_fit_time,
                 cv_folds, validation_blocks,
                 pipeline_cache, preprocessing_cache):
        super(NNObjectiveBuilder, self).__init__(objective, max_pipeline_fit_time, cv_folds, validation_blocks,
                                                 pipeline_cache, preprocessing_cache)

    def build(self, data, **kwargs):
        if not self.cv_folds:
            data_producer = self._build_kfolds_producer(data)
        else:
            data_producer = self._build_holdout_producer(data, **kwargs)

        # objective_evaluate =
