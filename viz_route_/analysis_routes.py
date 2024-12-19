import os
import pickle
from queue import Queue
from sysconfig import parse_config_h
import numpy as np
import pandas as pd
from tqdm import tqdm
from retro_planner.viz_utils.route_tree import SynRoute, copy_route_tree
from value_function.dataset_collection_depth import maxDepth


def Route2DictTree(route):

    node_list = []
    names = []
    for i in range(len(route.mols)):
        name = route.mols[i]
        names.append(name)
        node = {
            'smiles': name,
            'child': [],
        }
        node_list.append(node)

    node_queue = Queue()
    node_queue.put((0, -1))  # target mol idx, and parent idx
    while not node_queue.empty():
        idx, parent_idx = node_queue.get()

        if parent_idx >= 0:
            # G.edge(names[parent_idx], names[idx], label='cost')
            node_list[parent_idx]['child'].append(node_list[idx])

        if route.children[idx] is not None:
            for c in route.children[idx]:
                node_queue.put((c, idx))
    return node_list[0]


def read_pkl_results(fname):
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    results = pd.DataFrame.from_dict(results)
    route_dict_list = []
    route_maxDepth = []
    for route in results['routes'].tolist():
        if route is not None:
            route_dict = Route2DictTree(route)
            route_dict_list.append(route_dict)
            route_maxDepth.append(maxDepth(route_dict))
        else:
            route_dict_list.append(None)
            route_maxDepth.append(np.inf)
    results['dict_routes'] = route_dict_list
    results['max_depth'] = route_maxDepth
    return results


def nan2inf(x):
    if pd.isna(x):
        return np.inf
    else:
        return x


if __name__ == '__main__':
    results_folder = '../retro_planner/results/'

    results_names_to_analysis = [
        # 'plan_graph_single_depth_value',
        'plan_graph_single_no_value',
        # 'plan_graph_single_depth_value_C2',
        'plan_graph_single_depth_value_split',
        # 'plan_graph_single_mcts_star_depth_value',
        'plan_graph_single_mcts_star_no_value',
        'plan_graph_single_mcts_star_depth_split_value'
    ]

    analysis_dict = {

    }
    results_list = []
    check_set = set()
    for fname in results_names_to_analysis:
        fpath = os.path.join(results_folder, fname + '.pkl')
        results = read_pkl_results(fpath)
        check_set.add(len(results))
        assert len(check_set) == 1
        results_list.append(results)
        analysis_dict[fname+'_max_depth'] = results['max_depth'].tolist()
        analysis_dict[fname+'_route_lens'] = [nan2inf(x)
                                              for x in results['route_lens'].tolist()]
    analysis_results = pd.DataFrame.from_dict(analysis_dict)
    analysis_results.to_csv(os.path.join(results_folder, '_'.join(
        results_names_to_analysis)+'.csv'), index=False)
