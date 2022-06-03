from fedot.core.optimisers.opt_history import OptHistory


def log_to_history(history: OptHistory, population, generations):
    """
    Default variant of callback that preserves optimisation history
    :param history: OptHistory for logging
    :param population: list of individuals obtained in last iteration
    :param generations: keeper of the best individuals from all iterations
    """
    history.add_to_history(population)
    history.add_to_archive_history(generations.best_individuals)
    if history.save_folder:
        history.save_current_results()
