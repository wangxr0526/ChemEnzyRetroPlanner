import torch
from tqdm import tqdm
from collections import defaultdict
import os

if __name__ == '__main__':
    debug = False

    org_maxDepth_data_path = '../maxDepth_data_dic.pkl'
    unique_maxDepth_data_path = '../unique_maxDepth_data_dic.pkl'

    org_maxDepth_data_dic = torch.load(org_maxDepth_data_path)

    print(org_maxDepth_data_dic.keys())
    for k in list(org_maxDepth_data_dic.keys()):
        if debug:
            org_maxDepth_data_dic[k] = org_maxDepth_data_dic[k][:1000]
        print(k, len(org_maxDepth_data_dic[k]))
    if not os.path.exists(unique_maxDepth_data_path):
        target_smiles_to_maxdepth = defaultdict(list)
        unique_maxDepth_data_dic = defaultdict(list)
        for index, smi in tqdm(enumerate(org_maxDepth_data_dic['target_smiles']), total=len(org_maxDepth_data_dic['target_smiles'])):
            target_smiles_to_maxdepth[smi].append(
                org_maxDepth_data_dic['target_maxdepth'][index])
            if smi not in unique_maxDepth_data_dic['target_smiles']:
                unique_maxDepth_data_dic['target_smiles'].append(smi)
                unique_maxDepth_data_dic['target_fps'].append(
                    org_maxDepth_data_dic['target_fps'][index])
        torch.save(target_smiles_to_maxdepth, './target_smiles_to_maxdepth.pkl')
        torch.save(unique_maxDepth_data_dic, unique_maxDepth_data_path)
    else:
        target_smiles_to_maxdepth = torch.load('./target_smiles_to_maxdepth.pkl')
        unique_maxDepth_data_dic = torch.load(unique_maxDepth_data_path)
    for index, smi in tqdm(enumerate(unique_maxDepth_data_dic['target_smiles']), total=len(unique_maxDepth_data_dic['target_smiles'])):
        maxdepth_list = target_smiles_to_maxdepth[smi]
        if len(maxdepth_list) > 1:
            if debug:
                print(len(maxdepth_list))
            pass
        maxdepth_list.sort()
        target_maxdepth = max(maxdepth_list, key=maxdepth_list.count)
        unique_maxDepth_data_dic['target_maxdepth'].append(target_maxdepth)
    assert len(unique_maxDepth_data_dic['target_maxdepth']) == \
        len(unique_maxDepth_data_dic['target_smiles']) == \
        len(unique_maxDepth_data_dic['target_fps'])
    # del unique_maxDepth_data_dic['target_maxdepth_list']
    print(unique_maxDepth_data_dic.keys())
    print(len(set(org_maxDepth_data_dic['target_smiles'])))
    for k in list(unique_maxDepth_data_dic.keys()):
        print(k, len(unique_maxDepth_data_dic[k]))

    torch.save(unique_maxDepth_data_dic, unique_maxDepth_data_path)
