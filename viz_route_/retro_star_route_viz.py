import os
import pickle
import json
import pandas as pd
from tqdm import tqdm
from retro_planner.viz_utils.route_tree import SynRoute, copy_route_tree

if __name__ == '__main__':
    results_folder = '../retro_planner/results/'
    # dataset = 'retro_star_190'
    results_name = 'plan_graph_single_retro_star_no_value_multi_benchmark_paroutes_setn5_i750'

    # viz_route_save_folder = os.path.join(results_folder, dataset, results_name)
    viz_route_save_folder = os.path.join(results_folder, results_name)
    if not os.path.exists(viz_route_save_folder):
        os.makedirs(viz_route_save_folder)

    with open(os.path.join(results_folder, results_name + '.pkl'), 'rb') as f:
        read_data = pickle.load(f)
    read_data = pd.DataFrame.from_dict(read_data)
    best_routes_dict_list = []
    route_class = read_data['routes'].tolist()
    new_route_class = []
    for syn_class in tqdm(route_class):
        if syn_class is not None:
            best_routes_dict_list.append(syn_class.route_to_dict())
            new_syn_class = copy_route_tree(syn_class)
            new_route_class.append(new_syn_class)

        else:
            best_routes_dict_list.append({})
            new_route_class.append(None)
    with open(os.path.join(viz_route_save_folder, 'routes_dict.json'),'w') as f:
        json.dump(best_routes_dict_list, f)
    if 'all_succ_routes' in read_data.keys():
        all_succ_routes = read_data['all_succ_routes'].tolist()
        succ_routes_save_folder = os.path.join(viz_route_save_folder, 'succ_routes_json')
        if not os.path.exists(succ_routes_save_folder):
            os.makedirs(succ_routes_save_folder)
        for idx, routes_list in tqdm(enumerate(all_succ_routes), total=len(all_succ_routes)):
            if routes_list is not None:
                with open(os.path.join(succ_routes_save_folder, f'{idx}.json'), 'w') as f:
                    json.dump([[route.route_to_dict() for route in routes_list]],f)
            else:
                with open(os.path.join(succ_routes_save_folder, f'{idx}.json'), 'w') as f:
                    json.dump([[]],f)
    read_data['new_route_for_viz'] = new_route_class
    read_data.to_hdf(os.path.join(viz_route_save_folder, results_name + '.hdf5'), 'table')

    routes = read_data['new_route_for_viz'].tolist()
    graph_path = os.path.join(viz_route_save_folder, 'graph_routes')
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    for idx in tqdm(range(len(routes)), total=len(routes)):
        route = routes[idx]
        if route is not None:
            try:
                route_graph = route.viz_graph_route()
                route_graph.to_imagefile(graph_path + '/{}.png'.format(idx))
            except Exception as e:
                print(e)
                print('err {}'.format(idx))
    routes = read_data['routes'].tolist()

    smiles_routes_path = os.path.join(viz_route_save_folder, 'smiles_routes')
    if not os.path.exists(smiles_routes_path):
        os.makedirs(smiles_routes_path)
    for idx, route in tqdm(enumerate(routes), total=len(routes)):
        if route is not None:
            smiles_route = route.viz_route(smiles_routes_path + '/{}'.format(idx))