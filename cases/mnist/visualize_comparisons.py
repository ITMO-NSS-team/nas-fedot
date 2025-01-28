import json
import os
from collections.abc import Sequence
from copy import deepcopy
from itertools import groupby
from pathlib import Path

import seaborn
import tabulate
from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt, ticker, rcParams
from typing import Tuple, List, Dict

from numpy import average
from seaborn import histplot
import seaborn as sns

from cases.mnist.experiment_utils import parse_run_name
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
        # ("mnist", "dense-kan"): ["ltr-re-eval"],
        ("fashion-mnist", "kan"): [None, "small"],  # @ba51a28d977a6a518e2acc357e81289ec77bb1d3
        ("fashion-mnist", "cnn"): [None, "small"],  # @1f8815e20dc854c80e319f8d0935e8bee9d16544

        ("mnist", "cnn"): ["moderate"],  # @55d013d706306724fdc2be6754c1dd172b861589
        ("fashion-mnist", "cnn"): [None, "moderate"],  # @55d013d706306724fdc2be6754c1dd172b861589

        ("mnist", "cnn-kan"): ["ltr-re-eval"],
        # ("mnist", "cnn"): ["ltr-re-eval"],

        ("eurosat", "kan"): [
            "ltr-re-eval",
            "old-revision"
        ],
        ("eurosat", "cnn-kan"): [
            None,
            "old-revision"
        ],
        ("eurosat", "cnn"): [
            None,
            "old-revision"
        ]
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

    "old-revision-kan-eurosat": ["old-stats-extra-kan-eurosat", "old-stats-kan-eurosat"],
    "old-revision-cnn-kan-eurosat": ["old-stats-cnn-kan-eurosat"],
    "old-revision-cnn-eurosat": ["old-stats-cnn-eurosat"],
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


def visualise_pareto_front(front: Sequence[Individual | Tuple[float, float]],
                           objectives_numbers: Tuple[int, int] = (0, 1),
                           objectives_names: Sequence[str] = ('ROC-AUC', 'Complexity'),
                           file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                           folder: str = f'../../tmp/pareto',
                           case_name: str = None,
                           minmax_x: List[float] = None,
                           minmax_y: List[float] = None,
                           color: str = 'red',
                           label: str = None,
                           try_building_statintervals: bool = True,
                           plot_line_between_points: bool = False,
                           only_best_run: bool = False
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
    tmp_pareto_obj_first = []
    tmp_pareto_obj_second = []
    real_x = -1e10
    for x, xys in groupby(list(zip(pareto_obj_first, pareto_obj_second)), lambda xy: xy[0]):
        total_xs += 1
        if x - real_x < pareto_obj_first[-1] / 35:
            x = real_x
        else:
            real_x = x
        xys = list(xys)
        if len(xys) > 1:
            n_has_duplicates += 1

        if only_best_run:
            tmp_pareto_obj_first.append(x)
            tmp_pareto_obj_second.append(max([y for _, y in xys]))
        else:
            tmp_pareto_obj_first.extend([x] * len(xys))
            tmp_pareto_obj_second.extend([y for _, y in xys])
    pareto_obj_second = tmp_pareto_obj_second
    pareto_obj_first = tmp_pareto_obj_first

    print(f"{n_has_duplicates}/{total_xs} samples have duplicates")
    has_duplicates = n_has_duplicates / total_xs >= 0.5

    if color is None:
        color = retrieve_next_color()

    if not try_building_statintervals or not has_duplicates:
        # Didn't want to or failed to build stat-intervals
        # deduplicate_params = deduplicate_by_lambda(pareto_obj_first, lambda p: p)
        # dists_adj = [deduplicate_params[i + 1] - deduplicate_params[i] for i in range(len(deduplicate_params) - 1)]
        # target_width_anchor = min([d for d in dists_adj if d > deduplicate_params[-1] / 10])
        # anchored_width = min(dists_adj)
        sns.boxplot(x=pareto_obj_first, y=pareto_obj_second,
                    color=color,
                    native_scale=True,
                    # width=0.18 * target_width_anchor/anchored_width
                    fliersize=0
                    )
        # plt.scatter([], [], color=color, label=label, alpha=1)

        seaborn.lineplot(x=pareto_obj_first, y=pareto_obj_second, label=label, color=color)
        # label=None if plot_line_between_points else label,
        # plt.show()

        # plt.xscale('log')
        if plot_line_between_points:
            plt.plot(pareto_obj_first, pareto_obj_second, color=color, label=label)
    else:
        seaborn.lineplot(x=pareto_obj_first, y=pareto_obj_second, label=label, color=color)

    plt.title(f'Pareto frontier for {case_name}', fontsize=15)
    plt.xlabel(objectives_names[0], fontsize=15)
    plt.ylabel(objectives_names[1], fontsize=15)
    # plt.xscale('log')

    if minmax_x is not None:
        plt.xlim(*minmax_x)
    if minmax_y is not None:
        plt.ylim(*minmax_y)
    # fig.set_figwidth(8)
    # fig.set_figheight(8)
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')

        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight', dpi=600)
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


def plot_final_pareto_front(histories: List[OptHistory], final_results, case_name: str, label: str,
                            final_metric_name: str,
                            extra_results: Dict[str, List], remove_dominated: bool,
                            try_building_statintervals: bool,
                            plot_line_between_points: bool,
                            only_best_run: bool
                            ):
    """
    Called ones per trajectory to plot.
    The trajectory could be a scatter+line or stat-intervals if enough models have enough evaluation attempts.

    The trajectory could be obtained either from a single history or from the whole collection.
    The latter case is general, so the former one is considered as a partial case with list length = 1.
    """
    # individuals = individuals_pool(history)
    final_choices: List = sum([history.final_choices.data for history in histories], [])
    final_choices.sort(key=lambda ind: ind.fitness.values[1])

    front = []
    model_averages = []
    for i, ind in enumerate(final_choices):
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
        this_model_average = metric_sum / len(this_results)

        # Remove dominated by any of the previous models of the front
        if remove_dominated and len(model_averages) >= 2 and this_model_average < model_averages[-1]:
            continue
        model_averages.append(this_model_average)

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
        file_name=f"{case_name}-pareto.png",
        try_building_statintervals=try_building_statintervals,
        plot_line_between_points=plot_line_between_points,
        only_best_run=only_best_run
    )


def plot_parameter_number_hist(histories: List, relative=True):
    param_values = []
    for history in histories:
        for ind in individuals_pool(history):
            param_values.append(ind.fitness.values[1])

    histplot(param_values, log_scale=True, bins=4, stat="proportion" if relative else "count")


def format_parameter_count(n, decimals=1):
    """
    Format a number into a string with suffixes: k for thousands, M for millions, B for billions, etc.

    Args:
        n (int or float): The number to format.
        decimals (int): Number of decimal places.

    Returns:
        str: The formatted string.
    """
    suffixes = ['', 'k', 'M', 'B', 'T']
    magnitude = 0
    abs_n = abs(n)

    while abs_n >= 1000 and magnitude < len(suffixes) - 1:
        abs_n /= 1000.0
        magnitude += 1

    formatted = f"{n / (1000 ** magnitude):.{decimals}f}{suffixes[magnitude]}"

    # Remove trailing zeros and decimal point if not needed
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')

    return formatted


def table_of_best_models_for_dataset():
    """
    For each dataset, builds a separate LaTeX table with the best models of each type.
    The best model is determined by the highest accuracy of all its evaluation attempts.
    
    Use tabulate for the table building.
    """
    convkan_metrics = {
        "FashionMNIST": {"cnn": (157_030, 0.9014), "kan": (94_875, 0.8969), "cnn-kan": (94_950, 0.8884)},
        "MNIST": {"cnn": (157_030, 0.9912), "kan": (94_875, 0.9890), "cnn-kan": (94_950, 0.9875)},
    }
    extra_results = get_extra_final_results()
    for dataset in dataset_results_dir:
        dataset_table = []
        for arch in dataset_results_dir[dataset]:
            convkan_metric = convkan_metrics.get(dataset, {}).get(arch, None)

            print(f"Dataset: {dataset}, Model: {arch}")
            base_eval_paths = dataset_results_dir[dataset][arch]
            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), base_eval_paths))
            final_results = merge_dicts(list(
                map(lambda path: json.load(open(path + "/final_results.json")), base_eval_paths)))

            def best_model_by(first_better, filter):
                best_model_accuracy = None
                best_model_params = None
                for history in histories:
                    for ind in history.final_choices:
                        this_results = final_results[ind.uid]
                        if not isinstance(this_results, List):
                            this_results = [this_results]
                        this_results.extend(extra_results.get(ind.uid, []))

                        ind_metric_values = [this_run["accuracy"] for this_run in this_results]
                        this_model_aggregated = max(ind_metric_values)  # average(ind_metric_values)
                        this_model_params = ind.fitness.values[1]
                        if filter is not None and not filter(this_model_params, this_model_aggregated):
                            continue
                        if best_model_params is None or first_better((this_model_params, this_model_aggregated),
                                                                     (best_model_params, best_model_accuracy)):
                            best_model_accuracy = this_model_aggregated
                            best_model_params = this_model_params
                return best_model_params, best_model_accuracy

            best_accuracy_model = best_model_by(lambda new, old: new[1] > old[1], None)
            dataset_table.append((arch, best_accuracy_model[0], best_accuracy_model[1]))
            if convkan_metric is not None:
                smallest_model_winning_convkan = best_model_by(lambda new, old: new[0] < old[0],
                                                               lambda p, acc: acc > convkan_metric[1])
                if smallest_model_winning_convkan[
                    1] is not None and smallest_model_winning_convkan != best_accuracy_model:
                    dataset_table.append((arch, smallest_model_winning_convkan[0], smallest_model_winning_convkan[1]))

                # Add the ConvKAN metric to the table
                dataset_table.append((f"{arch} (ConvKAN)", convkan_metric[0], convkan_metric[1]))

        # Before, round the accuracy to 3 decimal places.
        dataset_table = [(arch, params, round(acc, 3)) for arch, params, acc in dataset_table]
        best_metric = max([acc for _, _, acc in dataset_table])
        # Highlight the best model's accuracy in bold
        dataset_table = [(arch, format_parameter_count(params), f"\\textbf{{{acc}}}" if acc == best_metric else acc) for
                         arch, params, acc in
                         dataset_table]
        print(dataset_table)

        table = tabulate.tabulate(dataset_table, headers=["Architecture", "#Parameters", "Accuracy"],
                                  tablefmt="latex_raw")
        print(table)


def merge_dicts(l):
    result = {}
    for d in l:
        result.update(d)
    return result


def plot_all_pareto_fronts(remove_domniated: bool, merge_runs: bool,
                           try_building_statintervals: bool = True,
                           plot_line_between_points: bool = False,
                           only_best_run: bool = False
                           ):
    for dataset in dataset_results_dir:
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}k' if x != 0 else 0))
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=14)

        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            base_eval_paths = dataset_results_dir[dataset][model]
            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), base_eval_paths))
            final_results = merge_dicts(list(
                map(lambda path: json.load(open(path + "/final_results.json")), base_eval_paths)))

            # plot_opt_pareto_front(history, label=model, case_name=dataset)
            extra_results = get_extra_final_results()

            if merge_runs:
                history_groups = [histories]
                group_model_names = [model]
            else:
                def listify(l):
                    return [[el] for el in l]

                history_groups = listify(histories)
                group_model_names = [f"{model} ({parse_run_name(Path(path).name)[0]})" for path in base_eval_paths]
            for group_model_name, group_histories in zip(group_model_names, history_groups):
                plot_final_pareto_front(group_histories, final_results, label=group_model_name, case_name=dataset,
                                        final_metric_name="accuracy", extra_results=extra_results,
                                        remove_dominated=remove_domniated,
                                        try_building_statintervals=try_building_statintervals,
                                        plot_line_between_points=plot_line_between_points,
                                        only_best_run=only_best_run
                                        )
        plt.legend()
        # plt.tight_layout()
        plt.show()
        reset_next_color_index()
        # exit()


def plot_parameter_numbers(merge_groups=False, relative=True):
    for dataset in dataset_results_dir:
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            if not merge_groups:
                for base_eval_path in dataset_results_dir[dataset][model]:
                    print(base_eval_path)
                    prefix, _, _ = parse_run_name(Path(base_eval_path).name)
                    print(f"Prefix: {prefix}")
                    history_path = base_eval_path + "/history.json"
                    history = OptHistory.load(history_path)

                    plot_parameter_number_hist([history], relative=relative)
                    plt.title(f"Parameter number distribution for {dataset} {model} ({prefix})")
                    plt.show()
                    reset_next_color_index()
            else:
                histories = list(
                    map(lambda path: OptHistory.load(path + "/history.json"), dataset_results_dir[dataset][model]))
                plot_parameter_number_hist(histories, relative=relative)
                plt.title(f"Parameter number distribution for {dataset} {model}")
                plt.show()
                reset_next_color_index()


if __name__ == '__main__':
    # Boxplot
    # plot_all_pareto_fronts(remove_domniated=True, merge_runs=True, try_building_statintervals=False,
    #                        plot_line_between_points=False, only_best_run=False)

    # Non-dominated stat-intervals
    # plot_all_pareto_fronts(remove_domniated=True, merge_runs=True, try_building_statintervals=True,
    #                        plot_line_between_points=True, only_best_run=False)

    # plot_parameter_numbers(merge_groups=True, relative=True)
    table_of_best_models_for_dataset()
