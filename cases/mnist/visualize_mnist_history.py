from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt

from nas.graph.base_graph import NasGraph
from nas.graph.node.adapter import NasNodeOperatorAdapter
from nas.graph.node.nas_graph_node import NasNode
from nas.utils.utils import project_root

# path = project_root() / "_results/debug/master_2/2024-08-09_23-30-56/history.json"
# path = r"D:\dev\aim\nas_kan_results\_results\smaller-kans-mnist\history.json"
path = r"D:\dev\aim\nas-fedot-kan\_results\debug\master_2\2024-08-14_17-28-44\history.json"

history = OptHistory.load(path)

print(history)

# Analyze if there is a graph in the history that has arity 2 at least for one node:
for generation in history.individuals:
    for ind in generation:
        graph = ind.graph
        for node in graph.nodes:
            if len(node.nodes_from) >= 2:
                print(f"Arity 2 node found in graph: {graph}, {node.uid}")
                break

# Check if some of the nodes are kan_linear:
from visualize_comparisons import get_individual_dump
linear_count = 0

for ind in get_individual_dump(history):
    graph = ind.graph
    for node in graph.nodes:
        if node.content["name"] == "kan_linear":
            print(f"KANLinear node found in graph: {graph}, {node.uid}")
            linear_count += 1
            break

        if node.content["name"] == "linear":
            print(f"Linear node found in graph: {graph}, {node.uid}")
            linear_count += 1
            break

print(f"Percentage of graphs containing linear layers: {linear_count / len(get_individual_dump(history))}")

# Plot hist of total parameter count or flops for all individuals in the history:
values = []
for generation in history.individuals:
    for ind in generation:
        values.append(ind.fitness.values[1])
        # values.append(DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode).restore(
        #     ind.graph).get_singleton_node_by_name('flatten').content["total_model_flops"])

from seaborn import histplot, distplot, displot

histplot(values, log_scale=True, bins=5)
plt.show()
# displot(values, log_scale=True, bins=5)
# plt.show()

# history.show(PlotTypesEnum.operations_animated_bar)
# history.show(PlotTypesEnum.operations_kde)
# history.show(PlotTypesEnum.fitness_line_interactive)

visualise_pareto(
    history.final_choices.data, show=True, objectives_names=("LogLoss", "Parameters")
)

print(history.get_leaderboard())
# 
# chosen = history.final_choices[0].graph
# PipelineVisualizer(chosen).visualise()
