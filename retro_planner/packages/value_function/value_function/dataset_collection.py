from collections import defaultdict
import os
import pickle
from queue import Queue
from mlp_retrosyn.mlp_policies import preprocess
import numpy as np

import pandas as pd
from rxn_filter.utils import canonicalize_smiles
import torch
from retro_planner.common import args
from retro_planner.common.prepare_utils import prepare_single_step
from tqdm import tqdm


def read_askcos(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                data.extend(pickle.load(f))
            except:
                break
    return data


# ['fps', 'values', 'reaction_costs', 'target_values', 'reactant_fps', 'reactant_masks']
# costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))   0~np.log(1e-3)   0~7.0
'''
positive training example:
fps: fingerprint of the target molecule                             fps
values: ground truth cost of the target molecule                    values

negative training example:
r_values: sum of reactant values in a negative reaction sample      ==> model(reactant_fps)*reactant_masks
r_costs: negative reaction cost                                     reaction_costs
t_values: ground truth cost of the target molecule                  target_values
'''


def get_value_fn_dataset(tree: dict):
    data_dict = {
        'fps': [],
        'values': [],
        'reaction_costs': [],
        'target_values': [],
        'reactant_fps': [],
        'reactant_masks': [],
    }

    node_queue = Queue()
    node_queue.put((tree, None))
    while not node_queue.empty():
        node, parent = node_queue.get()

        if node['child']:
            reacts_list = []
            for c in node['child']:
                node_queue.put((c, node))
                reacts_list.append(c['smiles'])
            reacts_smiles = canonicalize_smiles('.'.join(reacts_list))
            prod = node['smiles']
            fp = preprocess(prod, fp_dim=2048)
            try:
                pred_results = one_step.run(prod, topk=10)
            except Exception as e:
                print(e)
                pred_results = None
            if pred_results is not None:
                pred_reactants = [canonicalize_smiles(
                    x) for x in pred_results['reactants']]
                pred_scores = pred_results['scores']
                costs = 0.0 - np.log(np.clip(np.array(pred_scores), 1e-3, 1.0))
                gt_value_list = costs[np.where(np.asanyarray(
                    pred_reactants) == reacts_smiles)[0]].tolist()
                if gt_value_list:
                    value = gt_value_list[0]
                    data_dict['fps'].append(np.packbits(
                        fp.reshape(1, -1).astype(int), axis=-1))
                    data_dict['values'].append(value)
                    for i in range(len(costs)):
                        pred_reactant_list = pred_reactants[i].split('.')
                        num_react = len(pred_reactant_list)
                        if num_react > 3:
                            continue
                        reactant_fps = [preprocess(x, fp_dim=2048).reshape(1, -1).astype(int)
                                        for x in pred_reactant_list]
                        reactant_fps = reactant_fps + \
                            [reactant_fps[-1]]*(3-num_react)
                        assert len(reactant_fps) == 3

                        reactant_fps = np.packbits(
                            np.concatenate(reactant_fps, axis=0), axis=-1)
                        reactant_masks = [1]*num_react + [0]*(3-num_react)
                        assert len(reactant_masks) == 3
                        reactant_masks = np.array(reactant_masks)

                        data_dict['reaction_costs'].append(costs[i])
                        data_dict['target_values'].append(value)
                        data_dict['reactant_fps'].append(reactant_fps)
                        data_dict['reactant_masks'].append(reactant_masks)

    return data_dict


if __name__ == '__main__':

    # 先把专利多步路径单独提取出来（只要全部叶子节点可买的路径可买数据集为retro_planner/building_block_dataset/zinc_stock_2021_10_3_canonical_smiles_total_10312151_add_8546.csv）
    merge_data_path = os.path.abspath('./merge_data_path')
    if not os.path.exists(merge_data_path):
        os.makedirs(merge_data_path)
    merge_patent_route_save_path = os.path.join(
        merge_data_path, 'patent_route.pkl')
    if not os.path.exists(merge_patent_route_save_path):
        merge_route_list = []
        patent_leaf_buyable_info_df = pd.read_csv(
            '../../muilti_step_datasets/askcos_share/leaf_buyable_info.csv')
        fname_list = patent_leaf_buyable_info_df['record_file_name'].tolist()
        record_idx = patent_leaf_buyable_info_df['this_record_idx'].tolist()
        is_all_leaf_buyable = patent_leaf_buyable_info_df['new_is_all_leaf_buyable'].tolist(
        )

        latent_fname = None
        latent_record = None
        for fname, rd_idx, buyable in tqdm(zip(fname_list, record_idx, is_all_leaf_buyable), total=len(record_idx)):
            if bool(buyable):
                if fname != latent_fname:
                    # 重新读取太慢了，fname不变就不读取
                    one_record = read_askcos(fname)
                    latent_fname = fname
                    latent_record = one_record
                    merge_route_list.append(
                        one_record[rd_idx]['Patent_number'])
                else:
                    assert fname == latent_fname
                    merge_route_list.append(
                        latent_record[rd_idx]['Patent_number'])
        torch.save(merge_route_list, merge_patent_route_save_path)
    else:
        merge_route_list = torch.load(merge_patent_route_save_path)

    # TODO

    one_step = prepare_single_step(
        args.mlp_templates,
        args.mlp_model_dump,
        device=0,
        use_filter=args.use_filter,
        filter_path=args.filter_path,
        keep_score=args.keep_score,
        one_step_model_type=args.use_graph_single,
        graph_model_dumb=args.graph_model_dumb,
        graph_dataset_root=args.graph_dataset_root)
    # x = 'CCCC(=O)OC'
    # one_step.run(x, topk=args.expansion_topk)

    # get_value_fn_dataset(merge_route_list[0]['tree'])
    data_dict_fname = os.path.join(
        merge_data_path, 'value_data_dic.pkl')
    if not os.path.exists(data_dict_fname):
        print(f'Starting Collection.')
        run_record_route_list = merge_route_list
        data_dict = defaultdict(list)
        for idx, route in tqdm(enumerate(run_record_route_list), total=len(run_record_route_list)):
            try:
                one_data_dict = get_value_fn_dataset(route['tree'])
                for k in one_data_dict:
                    data_dict[k] += one_data_dict[k]
                data_dict['idx_run'] += [idx]
            except Exception as e:
                print('Erro !!!!!!!!!!')
                print(idx)
                print(e)
            if idx % 100 == 0:
                torch.save(data_dict, data_dict_fname)
    else:
        print(f'Re collection From {data_dict_fname}.')

        data_dict = torch.load(data_dict_fname)
        idx_run = data_dict['idx_run']
        rerun_idx = idx_run[-1]
        run_record_route_list = merge_route_list[rerun_idx:]
        print(f'Starting Collection from index {rerun_idx}, there are {len(run_record_route_list)} left')
        for idx, route in tqdm(enumerate(run_record_route_list), total=len(run_record_route_list)):
            try:
                one_data_dict = get_value_fn_dataset(route['tree'])
                for k in one_data_dict:
                    data_dict[k] += one_data_dict[k]
                data_dict['idx_run'] += [rerun_idx + idx]
            except Exception as e:
                print('Erro !!!!!!!!!!')
                print(rerun_idx + idx)
                print(e)
            if idx % 100 == 0:
                torch.save(data_dict, data_dict_fname)
