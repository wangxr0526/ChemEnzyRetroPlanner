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

'''
ref1: https://github.com/binghong-ml/retro_star/issues/10
ref2: https://github.com/binghong-ml/retro_star/issues/9
按照Retro*的作者所说的制作数据集的方法没效果,甚至起了反效果？？？不知道为啥/(ㄒoㄒ)/~~，自己搞一份吧，脱离他的思路~~    
'''


'''
新思路：按照子树的最大深度来设计价值，
'''


def maxDepth(root):
    stack = Queue()
    if root:
        stack.put((root, 0))
    max_depth = 0
    while not stack.empty():
        tree_node, cur_depth = stack.get()
        if tree_node:
            max_depth = max(max_depth, cur_depth)
            for c in tree_node['child']:
                stack.put((c, cur_depth+1))
    return max_depth


def read_askcos(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                data.extend(pickle.load(f))
            except:
                break
    return data


def get_maxDepth_dataset(tree: dict):
    data_dict = {
        'target_smiles': [],
        'target_fps': [],
        'target_maxdepth': [],
    }

    node_queue = Queue()
    node_queue.put((tree, None))
    while not node_queue.empty():
        node, parent = node_queue.get()
        for c in node['child']:
            node_queue.put((c, node))
        maxdepth = maxDepth(node)
        prod = node['smiles']
        fp = preprocess(prod, fp_dim=2048)
        data_dict['target_smiles'].append(prod)
        data_dict['target_fps'].append(np.packbits(
            fp.reshape(1, -1).astype(int), axis=-1))
        data_dict['target_maxdepth'].append(maxdepth)

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

    # one_data_dict = get_maxDepth_dataset(merge_route_list[1]['tree'])
    data_dict_fname = os.path.join(
        merge_data_path, 'maxDepth_data_dic.pkl')
    # data_dict = defaultdict(list)
    # if not os.path.exists(data_dict_fname):
    #     print(f'Starting Collection.')
    #     run_record_route_list = merge_route_list
    # else:
    #     print(f'Re collection From {data_dict_fname}.')
    #     data_dict = torch.load(data_dict_fname)
    #     idx_run = data_dict['idx_run']
    #     rerun_idx = idx_run[-1]
    #     run_record_route_list = merge_route_list[rerun_idx:]
    #     print(f'Starting Collection from index {rerun_idx}, there are {len(run_record_route_list)} left')

    # for idx, route in tqdm(enumerate(run_record_route_list), total=len(run_record_route_list)):
    #     one_data_dict = get_maxDepth_dataset(route['tree'])
    #     for k in one_data_dict:
    #         data_dict[k] += one_data_dict[k]
    #     data_dict['idx_run'] += [idx]
    #     if idx % 10000 == 0:
    #         torch.save(data_dict, data_dict_fname)
    # torch.save(data_dict, data_dict_fname)
    data_dict = torch.load(data_dict_fname)
    
    depth_cnt_dict = defaultdict(set)
    for data_idx in tqdm(range(len(data_dict['target_smiles']))):
        smi = data_dict['target_smiles'][data_idx]
        depth_cnt_dict[smi].add(data_dict['target_maxdepth'][data_idx])

    depth_cnt_items = list(depth_cnt_dict.items())
    depth_cnt_items.sort(key=lambda x: len(x[1]), reverse=True)
    print([(x[0],len(x[1])) for x in depth_cnt_items[:100]])
    print(len(depth_cnt_items))

    data_filter_dict = defaultdict(list)
    for data_idx in tqdm(range(len(data_dict['target_smiles']))):
        if data_dict['target_maxdepth'][data_idx] > 0:
            data_filter_dict['target_smiles'].append(data_dict['target_smiles'][data_idx])
            data_filter_dict['target_maxdepth'].append(data_dict['target_maxdepth'][data_idx])
            data_filter_dict['target_fps'].append(data_dict['target_fps'][data_idx])
        else:
            continue
    data_filter_dic_fname = os.path.join(
        merge_data_path, 'filter_maxDepth_data_dic.pkl')
    torch.save(data_filter_dict, data_filter_dic_fname)
    filter_depth_cnt_dict = defaultdict(set)
    for data_idx in tqdm(range(len(data_filter_dict['target_smiles']))):
        smi = data_filter_dict['target_smiles'][data_idx]
        filter_depth_cnt_dict[smi].add(data_filter_dict['target_maxdepth'][data_idx])

    filter_depth_cnt_items = list(filter_depth_cnt_dict.items())
    filter_depth_cnt_items.sort(key=lambda x: len(x[1]), reverse=True)
    print('#####################################')
    print([(x[0],len(x[1])) for x in filter_depth_cnt_items[:100]])
    print(len(filter_depth_cnt_items))



    
