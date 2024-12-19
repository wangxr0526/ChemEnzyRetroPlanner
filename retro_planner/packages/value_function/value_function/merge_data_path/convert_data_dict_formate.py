import torch
import numpy as np
from retro_planner.common.parse_args import args
import random
from collections import defaultdict

if __name__ == '__main__':
    print('convert')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    fname = 'value_data_dic_142400.pkl'

    data_dict = torch.load(fname)

    data_dict_train = {}
    data_dict_val = {}
    

    data_dict['fps'] = np.concatenate(data_dict['fps'], axis=0)
    data_dict['values'] = torch.tensor(data_dict['values']).view(-1,1)
    shuffle_idx_positive = np.random.permutation(data_dict['fps'].shape[0])
    data_dict['fps'] = data_dict['fps'][shuffle_idx_positive]
    data_dict['values'] = data_dict['values'][shuffle_idx_positive]

    positive_split_idx = int(0.9*data_dict['fps'].shape[0])
    data_dict_train['fps'] = data_dict['fps'][:positive_split_idx]
    data_dict_train['values'] = data_dict['values'][:positive_split_idx]
    data_dict_val['fps'] = data_dict['fps'][positive_split_idx:]
    data_dict_val['values'] = data_dict['values'][positive_split_idx:]
    
    data_dict['reaction_costs'] = torch.tensor(data_dict['reaction_costs']).view(-1,1)
    data_dict['target_values'] = torch.tensor(data_dict['target_values']).view(-1,1)
    data_dict['reactant_fps'] = np.concatenate([x.reshape(-1, 3, 256) for x in data_dict['reactant_fps']], axis=0)
    data_dict['reactant_masks'] = torch.from_numpy(np.concatenate([x.reshape(-1,3) for x in data_dict['reactant_masks']], axis=0))
    shuffle_idx_negative = np.random.permutation(data_dict['reaction_costs'].shape[0])
    data_dict['reaction_costs'] = data_dict['reaction_costs'][shuffle_idx_negative]
    data_dict['target_values'] = data_dict['target_values'][shuffle_idx_negative]
    data_dict['reactant_fps'] = data_dict['reactant_fps'][shuffle_idx_negative]
    data_dict['reactant_masks'] = data_dict['reactant_masks'][shuffle_idx_negative]

    negative_split_idx = int(0.9*data_dict['reaction_costs'].shape[0])
    data_dict_train['reaction_costs'] = data_dict['reaction_costs'][:negative_split_idx]
    data_dict_train['target_values'] = data_dict['target_values'][:negative_split_idx]
    data_dict_train['reactant_fps'] = data_dict['reactant_fps'][:negative_split_idx]
    data_dict_train['reactant_masks'] = data_dict['reactant_masks'][:negative_split_idx]
    data_dict_val['reaction_costs'] = data_dict['reaction_costs'][negative_split_idx:]
    data_dict_val['target_values'] = data_dict['target_values'][negative_split_idx:]
    data_dict_val['reactant_fps'] = data_dict['reactant_fps'][negative_split_idx:]
    data_dict_val['reactant_masks'] = data_dict['reactant_masks'][negative_split_idx:]

    print('All data dict')
    for key in data_dict:
        try:
            print(f'{key}: {data_dict[key].shape}')
        except:
            print('index run:', len(data_dict[key]))
    print('Train data dict')
    for key in data_dict_train:
        print(f'{key}: {data_dict_train[key].shape}')
    print('Val data dict')
    for key in data_dict_val:
        print(f'{key}: {data_dict_val[key].shape}')

    torch.save(data_dict, '{}_convert.pkl'.format(fname.split('.')[0]))
    torch.save(data_dict_train, '{}_train_convert.pkl'.format(fname.split('.')[0]))
    torch.save(data_dict_val, '{}_val_convert.pkl'.format(fname.split('.')[0]))
    