import json
import os
from collections.abc import Sequence
from copy import deepcopy
from itertools import groupby

import seaborn
from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

from seaborn import histplot
import seaborn as sns

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
                       only_existing_dirs=False, special_prefix_for_configs=None):
    """
    :param special_prefix_for_configs: indexed by [(dataset, model)], is List[prefix | None]
    Empty list designates ignoring this pair. None in the list means the :param modification: prefix.

    :returns: d[dataset][model]: List[path]; Configurations with empty lists are not present in the dict
    """
    if special_prefix_for_configs is None:
        special_prefix_for_configs = {}

    res = {}
    for (ds_presentation_name, ds_path_name) in dataset_names:
        ds_res = {}
        for algo_name in algorithm_names:
            config_res = []
            special_prefixes_trial = special_prefix_for_configs.get((ds_path_name, algo_name))
            if special_prefixes_trial is not None:
                if special_prefixes_trial == []:
                    continue
                input_prefixes = special_prefixes_trial
            else:
                input_prefixes = [modification]

            for prefix in input_prefixes:
                if prefix is None:
                    prefix = modification
                result_dir = f"{root_dir}/{prefix}-{algo_name}-{ds_path_name}"
                if not only_existing_dirs or os.path.isdir(result_dir):
                    config_res.append(result_dir)

            ds_res[algo_name] = config_res
        res[ds_presentation_name] = ds_res
    return res


result_root_dir = "C:/dev/aim/nas_kan_results/_results"

"""
See :dir_dict_with_grid: for the format
"""
dataset_results_dir = dir_dict_with_grid(
    result_root_dir,
    "ltr",
    [
        ("MNIST", "mnist"), ("FashionMNIST", "fashion-mnist"),
        ("EuroSAT", "eurosat")
    ],
    ["cnn", "dense-kan", "cnn-kan",
     "kan"
     ],
    only_existing_dirs=False,
    special_prefix_for_configs={
        # ("mnist", "dense-kan"): "ltr-re-eval",
        ("fashion-mnist", "kan"): "small",  # @ba51a28d977a6a518e2acc357e81289ec77bb1d3
        ("fashion-mnist", "cnn"): "small",  # @1f8815e20dc854c80e319f8d0935e8bee9d16544

        ("mnist", "cnn"): "moderate",  # @55d013d706306724fdc2be6754c1dd172b861589
        ("fashion-mnist", "cnn"): "moderate",  # @55d013d706306724fdc2be6754c1dd172b861589

        ("mnist", "cnn-kan"): "ltr-re-eval",
        # ("mnist", "cnn"): "ltr-re-eval",

        ("eurosat", "kan"): "ltr-re-eval"
    }
)


def transplant_path_dict(d, root_dir):
    res = {}
    for k, v in d.items():
        # res[os.path.join(root_dir, k)] = os.path.join(root_dir, v)
        res[k] = [os.path.join(root_dir, p) for p in v]
    return res


""" Doesn't exhibit new runs, new models. Is only treated as a container for additional stat evaluations """
extra_evals_folders = transplant_path_dict({
    "ltr-cnn-fashion-mnist": [
        "ci-ltr-cnn-fashion-mnist", "ci2-ltr-cnn-fashion-mnist"
    ],
    "ltr-dense-kan-fashion-mnist": [
        "ci-ltr-dense-kan-fashion-mnist", "ci2-ltr-dense-kan-fashion-mnist"
    ],
    "ltr-cnn-kan-fashion-mnist": ["ci-cnn-kan-fashion-mnist"],
    "ltr-kan-fashion-mnist": ["ci-kan-fashion-mnist"],

    "ltr-cnn-mnist": ["ci-ltr-cnn-mnist"],
    "ltr-cnn-kan-mnist": ["ci-ltr-cnn-kan-mnist"],
    "ltr-dense-kan-mnist": ["ltr-re-eval-dense-kan-mnist"],
    "ltr-kan-mnist": ["ci-ltr-kan-mnist"],
}, result_root_dir)


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


color_cycle_index = 0


def retrieve_next_color():
    global color_cycle_index
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    res = color_cycle[color_cycle_index]
    color_cycle_index += 1
    return res


def reset_next_color_index():
    global color_cycle_index
    color_cycle_index = 0


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
    front.sort(key=lambda t: t[1])
    pareto_obj_first, pareto_obj_second = [], []
    for ind in front:
        fitness_list = ind.fitness.values if isinstance(ind, Individual) else ind
        fit_first = fitness_list[objectives_numbers[0]]
        pareto_obj_first.append(abs(fit_first))
        fit_second = fitness_list[objectives_numbers[1]]
        pareto_obj_second.append(abs(fit_second))

    # Check if there are a lot of ys for any x
    n_has_duplicates = 0
    total_xs = 0
    for x, xys in groupby(list(zip(pareto_obj_first, pareto_obj_second)), lambda xy: xy[0]):
        total_xs += 1
        xys = list(xys)
        if len(xys) > 1:
            n_has_duplicates += 1
    print(f"{n_has_duplicates}/{total_xs}")
    has_duplicates = n_has_duplicates / total_xs >= 0.5

    if color is None:
        color = retrieve_next_color()

    if not has_duplicates:
        # plt.xscale('log')
        plt.scatter(pareto_obj_first, pareto_obj_second, c=color)
        # plt.xscale('log')
        plt.plot(pareto_obj_first, pareto_obj_second, color=color, label=label)
    else:
        seaborn.lineplot(x=pareto_obj_first, y=pareto_obj_second, label=label, color=color)

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
        reset_next_color_index()

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


def get_extra_final_results():
    # this_results_folder_name = os.path.basename(this_results_dir.rstrip("\\").rstrip("/"))
    # extra_final_result_files = extra_runs_folders.get(this_results_folder_name, [])
    res = {}
    for run_name, this_run_extra_evals_folders in extra_evals_folders.items():
        for f in this_run_extra_evals_folders:
            j = json.load(open(f + "/final_results.json"))
            for uid, graph_res in j.items():
                if uid not in res:
                    res[uid] = []
                if isinstance(graph_res, List):
                    res[uid].extend(graph_res)
                else:
                    res[uid].append(graph_res)
    return res


def plot_final_pareto_front(history, final_results, case_name: str, label: str, final_metric_name: str,
                            extra_results: Dict[str, List]):
    individuals = individuals_pool(history)
    final_choices = history.final_choices.data

    front = []
    model_averages = []
    for i, ind in enumerate(final_choices[::-1]):
        # print(ind.fitness, ind.uid)
        # print(final_results[ind.uid])
        assert final_metric_name == "accuracy"
        complexity_metric = ind.fitness.values[1]
        print(complexity_metric)

        # final_metric = -ind.fitness.values[0]  # For in-optimizer score

        this_extra_final_results = extra_results.get(ind.uid, [])
        this_results = final_results[ind.uid]
        if not isinstance(this_results, List):
            this_results = [this_results]
        this_results.extend(this_extra_final_results)

        print(len(this_results))
        # max_measurements = 10
        # if len(this_results) > max_measurements:
        #     this_results = this_results[:max_measurements]
        metric_sum = 0
        for this_run in this_results:
            final_metric = this_run[final_metric_name]  # For after-learning
            if final_metric < 0.6:
                continue

            metric_sum += final_metric
        model_averages.append(metric_sum / len(this_results))
        if len(model_averages) >= 2 and model_averages[-1] < model_averages[-2]:
            continue

        for this_run in this_results:
            final_metric = this_run[final_metric_name]
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
        objectives_numbers=(1, 0),
        file_name=f"{case_name}-pareto.png"
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
            extra_results = get_extra_final_results()
            plot_final_pareto_front(history, final_results_json, label=model, case_name=dataset,
                                    final_metric_name="accuracy", extra_results=extra_results)
        plt.legend()
        plt.show()
        reset_next_color_index()
        # exit()


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
            reset_next_color_index()


if __name__ == '__main__':
    plot_all_pareto_fronts()
    # plot_parameter_numbers()
