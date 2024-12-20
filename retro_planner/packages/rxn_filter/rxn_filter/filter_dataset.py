import os
import random
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from drfp import DrfpEncoder
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)


def cal_fp(mol, radius=2, nbits=2048):
    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=int(nbits), useChirality=True)
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array.reshape([1, nbits])


def cal_rxn_fp(prod_fp, reactants, radius=2, nbits=2048):
    product_fp = prod_fp
    reactants_fp = sum(
        cal_fp(mol, radius=radius, nbits=nbits) for mol in reactants
    )
    return product_fp - reactants_fp


def reaction_to_fingerprint(reaction, radius=2, nbits=2048):
    reactants, product = reaction.split('>>')
    product = Chem.MolFromSmiles(product)
    reactants = [Chem.MolFromSmiles(reactant)
                 for reactant in reactants.split('.')]
    prod_fp = cal_fp(product, radius=radius, nbits=nbits)
    rxn_fp = cal_rxn_fp(prod_fp, reactants, radius=radius, nbits=nbits)
    prod_fp = torch.from_numpy(prod_fp).float()
    rxn_fp = torch.from_numpy(rxn_fp).float()
    return prod_fp, rxn_fp


def reactions_to_fingerprint(reaction_list, radius=2, nbits=2048):
    fps_prod = []
    fps_rxn = []
    for rxn in tqdm(reaction_list):
        fp_prod, fp_rxn = reaction_to_fingerprint(
            rxn,  radius=radius, nbits=nbits)
        fps_prod.append(fp_prod)
        fps_rxn.append(fp_rxn)
    fps_prod = torch.cat(fps_prod, dim=0)
    fps_rxn = torch.cat(fps_rxn, dim=0)
    return fps_prod, fps_rxn


def reactions_to_rxnfp(reaction_list):
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    fps_rxn = generate_fingerprints(
        reaction_list, fingerprint_generator=rxnfp_generator, batch_size=64)
    fps_rxn = torch.from_numpy(fps_rxn).float()
    dummy_fps = torch.zeros_like(fps_rxn).float()
    return dummy_fps, fps_rxn

def reactions_to_drfp(reaction_list):
    fps_rxn = DrfpEncoder.encode(reaction_list)
    fps_rxn = torch.tensor(fps_rxn).float()
    dummy_fps = torch.zeros_like(fps_rxn).float()
    return dummy_fps, fps_rxn


fp_function_dict = {
    'morganfp': reactions_to_fingerprint,
    'rxnfp': reactions_to_rxnfp,
    'drfp': reactions_to_drfp
}


class RXN2rxnfp:
    def __init__(self) -> None:
        self.model, self.tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_generator = RXNBERTFingerprintGenerator(
            self.model, self.tokenizer)

    def __call__(self, reaction_list):
        fps_rxn = generate_fingerprints(
            reaction_list, fingerprint_generator=self.rxnfp_generator, batch_size=1)
        '''
        rxnfp generate_fingerprints 函数有bug，如果要设置batch_size不能len(reaction_list)整除的数字，需要更改rxnfp源码， 所以这里嵌入多步调用的时候固定batch_size=1
        '''
        fps_rxn = torch.from_numpy(fps_rxn).float()
        dummy_fps = torch.zeros_like(fps_rxn).float()
        return dummy_fps, fps_rxn


class FilterDataset(Dataset):
    def __init__(self,
                 data_root,
                 raw_data_fname,
                 procssed_data_fname=None,
                 fp_function_name='rxnfp',
                 split_index_fname=None,):

        super(FilterDataset, self).__init__()
        self.data_root = data_root
        self.raw_data_path = os.path.join(self.data_root, raw_data_fname)
        if procssed_data_fname == None:
            self.procssed_data_path = os.path.join(self.data_root, raw_data_fname.split(
                '.')[0] + f'_{fp_function_name}.pkl')
        else:
            self.procssed_data_path = os.path.join(
                self.data_root, procssed_data_fname)
        print('Raw data path:', self.raw_data_path)
        print('Processed data path:', self.procssed_data_path)
        if not os.path.exists(self.procssed_data_path):
            print('Processing.')
            print(f'Using {fp_function_name}')
            raw_data_df = pd.read_csv(self.raw_data_path)
            rxn_smiles = raw_data_df['rxn_smiles'].tolist()
            self.labels = raw_data_df['labels'].tolist()
            assert len(rxn_smiles) == len(self.labels)
            fingerprint_func = fp_function_dict[fp_function_name]
            self.fps_prod, self.fps_rxn = fingerprint_func(rxn_smiles)
            self.labels = torch.tensor(self.labels, dtype=torch.float)
            self.data = [self[i] for i in range(len(self.fps_rxn))]
            torch.save((self.fps_prod, self.fps_rxn, self.labels),
                       self.procssed_data_path)
            print('Done')
        else:
            print(f'Loading processed dataset from {self.procssed_data_path}')
            self.fps_prod, self.fps_rxn, self.labels = torch.load(
                self.procssed_data_path)
            self.data = [self[i] for i in range(len(self.fps_rxn))]
        assert self.fps_prod.size(0) == self.fps_rxn.size(
            0) == self.labels.size(0)
        # self.transform = transform
        # self.target_transform = target_transform
        if split_index_fname:
            self.split_index_fname = os.path.join(
                self.data_root, split_index_fname)
            self.train, self.val, self.test = self.split_with_index()
        else:
            self.train, self.val, self.test = self.split()

    def split(self):
        len_data = len(self.data)
        data_indices = [i for i in range(len_data)]
        random.shuffle(data_indices)
        # data_indices = torch.tensor(data_indices)
        shuffle_dataset = [self[index] for index in data_indices]
        split_point_1 = int(0.8 * len_data)
        split_point_2 = int(0.9 * len_data)
        train = shuffle_dataset[:split_point_1]
        val = shuffle_dataset[split_point_1:split_point_2]
        test = shuffle_dataset[split_point_2:]
        return train, val, test

    def split_with_index(self):
        split_save_path = self.procssed_data_path.replace(
            '.pkl', '_split_dataset.pkl')
        if os.path.exists(split_save_path):
            print(f'Loading split dataset from {split_save_path}')
            train, val, test = torch.load(split_save_path)
            return train, val, test
        print(f'Read split index from {self.split_index_fname}')
        split_begin_end_list = torch.load(self.split_index_fname)
        split_data_list = []
        len_data = len(self.data)
        data_indices = [i for i in range(len_data)]
        for begin, end in split_begin_end_list:
            set_data = [self[index] for index in data_indices[begin:end]]
            split_data_list.append(set_data)

        train, val, test = split_data_list
        torch.save([train, val, test], split_save_path)
        train, val, test = torch.load(split_save_path)
        return train, val, test

    def __len__(self):
        return len(self.fps_rxn)

    def __getitem__(self, idx):
        fp_prod = self.fps_prod[idx]
        fp_rxn = self.fps_rxn[idx]
        label = self.labels[idx]
        return (fp_prod, fp_rxn), label

    def __repr__(self):
        try:
            return '\nProducts fingerprint shape: {}, \nReactions fingerprint shape: {}, \nLabels shape: {}'.format(
                self.fps_prod.size(),
                self.fps_rxn.size(),
                self.labels.size())
        except:
            return '\nReactions fingerprint shape: {}, \nLabels shape: {}'.format(
                self.fps_rxn.size(),
                self.labels.size())


if __name__ == '__main__':
    debug = True
    if debug:
        # split_index = [(0, 633), (633, 733), (733,)]
        dataset = FilterDataset(data_root='data',
                                raw_data_fname='random_gen_false_rxn.csv',
                                fp_function_name='rxnfp')
        test_loader = DataLoader(dataset.train, shuffle=True, batch_size=10)
        # print(dataset)
    else:
        dataset = FilterDataset(data_root='data',
                                raw_data_fname='random_gen_aizynth_filter_dataset.csv',
                                split_index_fname='random_gen_aizynth_filter_dataset_split_index.pkl')
        print(dataset)
