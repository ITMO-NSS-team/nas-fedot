import json
import os
from collections.abc import Sequence
from copy import deepcopy

from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt
from typing import Tuple, List

from seaborn import histplot

from nas.graph.base_graph import NasGraph
from nas.graph.node.adapter import NasNodeOperatorAdapter
from nas.graph.node.nas_graph_node import NasNode
from nas.utils.utils import project_root

dataset_results_dir = {
    "MNIST": {
        # Less iterations:
        # "less-iter-kan": r"C:\dev\aim\nas_kan_results\_results\smaller-kans-mnist",
        # "less-iter-cnn": r"C:\dev\aim\nas_kan_results\_results\better-cnn-mnist",

        # More iterations:
        # "kan": r"C:\dev\aim\nas_kan_results\_results\kan-much-epochs-mnist",
        # "cnn": r"C:\dev\aim\nas_kan_results\_results\cnn-much-epochs-mnist",
        # "cnn-kan": r"C:\dev\aim\nas_kan_results\_results\cnn-kan-much-epochs-mnist",
        # "cnn-kan-smaller": r"C:\dev\aim\nas_kan_results\_results\mnist-kan-cnn-smaller",
        # "dense-kan": r"C:\dev\aim\nas_kan_results\_results\dense-kan-mnist"

        "cnn": r"C:\dev\aim\nas_kan_results\_results\ltr-cnn-mnist",
    },
    "FashionMNIST": {
        # Less iterations:
        # "less-iter-smaller-kan": r"C:\dev\aim\nas_kan_results\_results\smaller-kan-fashion",
        # "less-iter-kan": r"C:\dev\aim\nas_kan_results\_results\kan-fashion-mnist",
        # "less-iter-cnn": r"C:\dev\aim\nas_kan_results\_results\better-cnn-fashion",

        # More iterations:
        # "kan": r"C:\dev\aim\nas_kan_results\_results\kan-fashion-mnist-much-epochs",
        # "cnn": r"C:\dev\aim\nas_kan_results\_results\cnn-fashion-mnist-much-epochs",
        # "cnn-kan": r"C:\dev\aim\nas_kan_results\_results\kan-cnn-fashion-mnist-more-epochs",
        # "dense-kan": r"C:\dev\aim\nas_kan_results\_results\dense-kans-fashion",

        # LTR
        "cnn": r"C:\dev\aim\nas_kan_results\_results\ltr-cnn-fashion-mnist",
    },
    "EuroSAT": {
        # "kan": r"C:\dev\aim\nas_kan_results\_results\eurosat-kan",
        "kan": r"C:\dev\aim\nas_kan_results\_results\more-iters-kan-eurosat",
        "cnn-kan": r"C:\dev\aim\nas_kan_results\_results\cnn-kan-eurosat-many-epochs",
        "cnn": r"C:\dev\aim\nas_kan_results\_results\eurosat-cnn",
    },
}


def dir_dict_with_grid(root_dir, modification, dataset_names: List[Tuple[str, str]], algorithm_names,
                       only_existing_dirs=False):
    resultant_root = f"{root_dir}/{modification}"
    res = {}
    for (ds_presentation_name, ds_path_name) in dataset_names:
        ds_res = {}
        for algo_name in algorithm_names:
            result_dir = f"{resultant_root}-{algo_name}-{ds_path_name}"
            if not only_existing_dirs or os.path.isdir(result_dir):
                ds_res[algo_name] = result_dir
        res[ds_presentation_name] = ds_res
    return res


dataset_results_dir = dir_dict_with_grid(
    "C:/dev/aim/nas_kan_results/_results",
    "ltr",
    [
        ("MNIST", "mnist"), ("FashionMNIST", "fashion-mnist"),
        # ("EuroSAT", "eurosat")
    ],
    ["cnn", "dense-kan", "cnn-kan", "kan"],
    False
)


def deduplicate_by_lambda(items, key_func):
    seen = set()
    result = []
    for item in items:
        item_key = key_func(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


def get_individual_dump(history):
    res = []
    for gen in history.individuals:
        for ind in gen:
            res.append(ind)

    return res


def individuals_pool(history):
    return deduplicate_by_lambda(get_individual_dump(history), lambda x: x.uid)


def plot_opt_fitness_scatter(history_path):
    history = OptHistory.load(history_path)
    individuals = individuals_pool(history)

    plt.scatter([ind.fitness.values[0] for ind in individuals], [ind.fitness.values[1] for ind in individuals])


def visualise_pareto_front(front: Sequence[Individual | List[float]],
                           objectives_numbers: Tuple[int, int] = (0, 1),
                           objectives_names: Sequence[str] = ('ROC-AUC', 'Complexity'),
                           file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                           folder: str = f'../../tmp/pareto',
                           case_name: str = None,
                           minmax_x: List[float] = None,
                           minmax_y: List[float] = None,
                           color: str = 'red',
                           label: str = None
                           ):
    pareto_obj_first, pareto_obj_second = [], []
    for ind in front:
        fitness_list = ind.fitness.values if isinstance(ind, Individual) else ind
        fit_first = fitness_list[objectives_numbers[0]]
        pareto_obj_first.append(abs(fit_first))
        fit_second = fitness_list[objectives_numbers[1]]
        pareto_obj_second.append(abs(fit_second))

    # plt.xscale('log')
    plt.scatter(pareto_obj_first, pareto_obj_second, c=color)
    # plt.xscale('log')
    plt.plot(pareto_obj_first, pareto_obj_second, color=color, label=label)

    plt.title(f'Pareto frontier for {case_name}', fontsize=15)
    plt.xlabel(objectives_names[0], fontsize=15)
    plt.ylabel(objectives_names[1], fontsize=15)
    # plt.xscale('log')

    if minmax_x is not None:
        plt.xlim(minmax_x[0], minmax_x[1])
    if minmax_y is not None:
        plt.ylim(minmax_y[0], minmax_y[1])
    # fig.set_figwidth(8)
    # fig.set_figheight(8)
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')

        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()

    # plt.cla()
    # plt.clf()
    # plt.close('all')


def plot_opt_pareto_front(history, case_name: str, label: str):
    individuals = individuals_pool(history)

    # fig, ax = plt.subplots()

    visualise_pareto_front(
        history.final_choices.data, show=False, objectives_names=("Parameters", "LogLoss",), label=label, color=None,
        case_name=case_name,
        objectives_numbers=(1, 0)
    )


def plot_final_pareto_front(history, final_results, case_name: str, label: str, final_metric_name: str):
    individuals = individuals_pool(history)
    final_choices = history.final_choices.data

    front = []
    for ind in final_choices:
        # print(ind.fitness, ind.uid)
        # print(final_results[ind.uid])
        assert final_metric_name == "accuracy"
        complexity_metric = ind.fitness.values[1]

        final_metric = final_results[ind.uid][final_metric_name]  # For after-learning
        # final_metric = -ind.fitness.values[0]  # For in-optimizer score

        if final_metric < 0.6:
            continue

        front.append(
            (
                final_metric,
                complexity_metric
            )
        )

    # fig, ax = plt.subplots()

    visualise_pareto_front(
        front, show=False, objectives_names=("Parameters", final_metric_name), label=label, color=None,
        case_name=case_name,
        objectives_numbers=(1, 0)
    )


def plot_parameter_number_hist(history):
    param_values = []
    for ind in individuals_pool(history):
        param_values.append(ind.fitness.values[1])

    histplot(param_values, log_scale=True, bins=4)


def plot_all_pareto_fronts():
    for dataset in dataset_results_dir:
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            base_eval_path = dataset_results_dir[dataset][model]
            history_path = base_eval_path + "/history.json"
            history = OptHistory.load(history_path)
            final_results_json = json.load(open(base_eval_path + "/final_results.json"))

            # plot_opt_pareto_front(history, label=model, case_name=dataset)
            plot_final_pareto_front(history, final_results_json, label=model, case_name=dataset,
                                    final_metric_name="accuracy")
        plt.legend()
        plt.show()


def plot_parameter_numbers():
    for dataset in dataset_results_dir:
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            base_eval_path = dataset_results_dir[dataset][model]
            history_path = base_eval_path + "/history.json"
            history = OptHistory.load(history_path)

            plot_parameter_number_hist(history)
            plt.title(f"Parameter number distribution for {dataset} {model}")
            plt.show()


if __name__ == '__main__':
    plot_all_pareto_fronts()
