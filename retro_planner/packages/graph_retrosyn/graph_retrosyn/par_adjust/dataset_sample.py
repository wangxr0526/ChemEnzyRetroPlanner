import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from graph_retrosyn import dataset as dataset

seed = 123
sampling_rate = 0.05
multi_sampling_rate = 0.3

if __name__ == '__main__':
    debug = False

    save_sampling_data_path = os.path.join('sampling_data', 'raw')

    org_dataset_folder, data_fname = '../data/', 'USPTO-remapped_tpl_prod_react_v2'
    if debug:
        org_dataset_folder = org_dataset_folder.replace('data', 'debug_data')
        save_sampling_data_path = save_sampling_data_path.replace('data', 'debug_data')

    if not os.path.exists(save_sampling_data_path):
        os.makedirs(save_sampling_data_path)
    org_dataset = pd.read_csv(os.path.join(org_dataset_folder, 'raw', f'{data_fname}.csv'))
    org_dataset_split = torch.load(os.path.join(org_dataset_folder, 'raw', f'{data_fname}_split_index.pkl'))
    train_index, val_index, test_index = org_dataset_split
    # template_rules, rule2idx = torch.load(os.path.join(org_dataset_folder, 'raw', 'templates_index.pkl'))
    torch.save(torch.load(os.path.join(org_dataset_folder, 'raw', 'templates_index.pkl')),
               os.path.join(save_sampling_data_path,
                            'templates_index.pkl'))
    print('Select org train data and val data.')
    org_drop_test_dataset = org_dataset[train_index[0]: train_index[1]].append(org_dataset[val_index[0]: val_index[1]])
    org_total = len(org_drop_test_dataset)
    print('Selected # data:', org_total)
    sampling_total = int(sampling_rate * org_total)
    print('Sampling # data:', sampling_total)

    prod_react_list = list(zip(
        org_drop_test_dataset['prod_smiles'].tolist(),
        org_drop_test_dataset['react_smiles'].tolist()))
    templates = defaultdict(list)
    for ind, (tpl, prod_react) in enumerate(zip(org_drop_test_dataset['templates'].tolist(), prod_react_list)):
        templates[tpl].append(ind)
        # org_drop_test_dataset.loc[ind, 'template_id'] = str(template_rules[tpl])

    templates_items_list = list(templates.items())
    template_single = [x for x in templates_items_list if len(x[1]) == 1]
    template_multi = [x for x in templates_items_list if len(x[1]) > 1]
    total_multi = 0
    for template_item in template_multi:
        total_multi += len(template_item[1])
    print('total_multi #', total_multi)
    sampling_multi_num = int(sampling_rate * total_multi)
    print('total_single #', len(template_single))
    sampling_single_num = int(sampling_rate * len(template_single))

    print('Multi-data templates sampling # data', sampling_multi_num,
          'Single-data templates sampling # data', sampling_single_num,
          'Sampling # total', sampling_single_num + sampling_multi_num)

    sampling_index_pool = []
    rng = np.random.RandomState(seed)
    template_multi = rng.permutation(template_multi)
    for template_item in template_multi:
        num_this = len(template_item[1])
        # template = template_item[0]
        sampling_num_this = int(multi_sampling_rate * num_this)
        if len(sampling_index_pool) + sampling_num_this <= sampling_multi_num:
            sampling_index_pool.extend(np.random.choice(template_item[1], size=sampling_num_this, replace=False))
        else:
            break

    template_single_sampling = rng.choice(np.array([x[1] for x in template_single]).reshape(-1),
                                          size=(sampling_single_num + sampling_multi_num - len(sampling_index_pool)),
                                          replace=False)

    sampling_index_pool += template_single_sampling.tolist()
    sampling_index_pool = rng.permutation(sampling_index_pool)
    sampling_dataset = org_drop_test_dataset.loc[sampling_index_pool]
    sampling_dataset.to_csv(os.path.join(save_sampling_data_path, f'{data_fname}.csv'), index=False)

    dataset_split = [(0, int(0.8 * len(sampling_dataset))),
                     (int(0.8 * len(sampling_dataset)), int(0.9 * len(sampling_dataset))),
                     ((int(0.9 * len(sampling_dataset))), len(sampling_dataset))
                     ]
    torch.save(dataset_split, os.path.join(save_sampling_data_path, f'{data_fname}_split_index.pkl'))
