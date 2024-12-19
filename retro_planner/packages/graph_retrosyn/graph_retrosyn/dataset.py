import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from graph_retrosyn.utils import one_of_k_encoding, bond_features

try:
    import rdkit
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
    from rdkit.Chem.rdchem import HybridizationType as HT
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except:
    rdkit, Chem, RDConfig, MolFromSmiles, ChemicalFeatures, HT, BT = 7 * [None]
    print('Please install rdkit for train_all_dataset processing')
get_graph_raw = True

all_atoms_symbols = [
    'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
    'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Eu', 'Gd', 'Tb', 'Yb', 'Hf', 'Ta', 'W', 'Re', 'Pt',
    'Au', 'Hg', 'Pb', 'Bi'
]

all_atoms_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
all_atoms_valence = [0, 1, 2, 3, 4, 5, 6]
all_atoms_charges_to_int = {-1: 0, -2: 1, 1: 2, 2: 3, 0: 4, 3: 5, 4: 6, 5: 7}
all_atoms_charges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
all_atoms_num_hs = [0, 1, 2, 3, 4, 5]


def get_charge_int(charge):
    if charge not in all_atoms_charges_to_int:
        return 8
    else:
        return all_atoms_charges_to_int[charge]


def get_mol_nodes_edges(mol):
    # Read node features
    N = mol.GetNumAtoms()
    atom_type = []
    # atomic_number = []
    aromatic = []
    hybridization = []
    num_hs = []
    atom_charges = []
    atom_valence = []
    atom_degrees = []
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        # atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())
        num_hs.append(atom.GetTotalNumHs())
        atom_charges.append(get_charge_int(atom.GetFormalCharge()))
        atom_valence.append(atom.GetExplicitValence())
        atom_degrees.append(atom.GetTotalDegree())

    # Read edge features
    row, col, f_bond = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        # f_bond += 2 * [bond.GetBondType()]
        f_bond.append(bond_features(bond))
        f_bond.append(bond_features(bond))
    edge_index = torch.LongTensor([row, col])
    # edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in f_bond]
    edge_attr = torch.FloatTensor(f_bond)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    # row, col = edge_index

    # Concat node fetures
    # hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    # num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, all_atoms_symbols) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2]) for h in hybridization]
    x_num_hs = [one_of_k_encoding(nh, all_atoms_num_hs) for nh in num_hs]
    x_atom_charges = [one_of_k_encoding(x, all_atoms_charges) for x in atom_charges]
    x_atom_valence = [one_of_k_encoding(x, all_atoms_valence) for x in atom_valence]
    x_atom_degrees = [one_of_k_encoding(x, all_atoms_degrees) for x in atom_degrees]
    # x2 = torch.tensor([aromatic, num_hs], dtype=torch.float).t().contiguous()
    x = torch.cat([torch.FloatTensor(x_atom_type),
                   torch.FloatTensor(x_hybridization),
                   torch.FloatTensor(x_num_hs),
                   torch.FloatTensor(x_atom_charges),
                   torch.FloatTensor(x_atom_valence),
                   torch.FloatTensor(x_atom_degrees),
                   torch.tensor([aromatic], dtype=torch.float).t().contiguous()], dim=-1)

    return x, edge_index, edge_attr


def preprocess2fp(mol, fp_dim):
    # Compute fingerprint from mol to feature
    # mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return torch.from_numpy(arr).float()


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset='USPTO-remapped_tpl_prod_react', process2fp=False, fp_dim=2048,
                 multi_core_process=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None, debug=False):
        if debug:
            root = root.replace('data', 'debug_data')
        self.dataset = dataset  # random / random_nan / scaffold
        self.multi_core_process = multi_core_process
        self.process2fp = process2fp
        self.fp_dim = fp_dim
        super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        trn, val, test = self.split()
        self.train, self.val, self.test = trn, val, test
        self.mol_num_node_features = self[0].x.shape[1]
        self.mol_num_edge_features = self[0].edge_attr.shape[1]
        # self.num_tasks = len(self.tasks)

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['dataset_{}.pt'.format(self.dataset)]

    def process(self):
        # load csv

        df = pd.read_csv(self.raw_paths[0])
        template_rules, rule2idx = torch.load(
            os.path.join(self.raw_dir, 'templates_index.pkl'))
        templates = df['templates'].tolist()
        ground_truth = df['react_smiles'].tolist()
        prod_smiles = df['prod_smiles'].tolist()

        def get_unk_label(i):
            if templates[i] not in template_rules:
                return 99999999
            else:
                return template_rules[templates[i]]

        data_list = []
        for i, smi in enumerate(tqdm(prod_smiles)):
            if not self.is_valid_smiles(smi): continue
            mol = MolFromSmiles(smi)
            if mol is None: continue
            x, edge_index, edge_attr = get_mol_nodes_edges(mol)
            label = get_unk_label(i)
            y = torch.tensor(label, dtype=torch.long)
            # # label[np.isnan(label)] = 6
            # if self.dataset in ['esol', 'freesolv', 'lipophilicity']:
            #     y = torch.FloatTensor(label).unsqueeze(0)
            # elif self.dataset in ['bbbp', 'bace', 'sider', 'toxcast', 'toxcast', 'tox21']:
            #     label[np.isnan(label)] = -1  # Fill in -1 for those NaN labels
            #     y = torch.LongTensor(label).unsqueeze(0)
            if self.process2fp:
                fp = preprocess2fp(mol, self.fp_dim).view(1, -1)
                data = Data(x=x, fp=fp, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=y,
                            ground_truth=ground_truth[i])
            else:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=y, ground_truth=ground_truth[i])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def is_valid_smiles(smi):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        except:
            print("not successfully processed smiles: ", smi)
            return False
        return True

    def split(self):
        split_save_path = os.path.join(self.processed_dir, f'{self.dataset}_split_dataset.ckpt')
        if os.path.exists(split_save_path):
            train, val, test = torch.load(split_save_path)
            return train, val, test
        split_begin_end_list = torch.load(os.path.join(self.raw_dir, f'{self.dataset}_split_index.pkl'))
        split_data_list = []
        for begin, end in split_begin_end_list:
            split_data_list.append(self[begin:end])
        train, val, test = split_data_list
        torch.save([train, val, test], split_save_path)
        return train, val, test


def get_all_atom_types(prod_smiles):
    all_atom_types = {}
    for prod in tqdm(prod_smiles):
        mol = Chem.MolFromSmiles(prod)
        for atom in mol.GetAtoms():
            all_atom_types[atom.GetSymbol()] = atom.GetAtomicNum()
    return all_atom_types


def get_graph_raw_dataset(path, save_path, save_fname, train_all=False, debug=False):
    
    if debug:
        save_path = save_path.replace('data', 'debug_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not train_all:
        template_rules = {}
        with open(os.path.join(path, 'train_val_template_rules_1.dat'), 'r') as f:
            for i, l in tqdm(enumerate(f), desc='template rules'):
                rule = l.strip()
                template_rules[rule] = i
        idx2rule = {}
        for rule, idx in template_rules.items():
            idx2rule[idx] = rule
        torch.save((template_rules, idx2rule), os.path.join(save_path, 'templates_index.pkl'))
        df_frames = []
        dataset_begin_end_index = []
        begin = 0
        end = 0
        for flag in ['train', 'val', 'test']:
            templates_prods_reacts_df = pd.read_csv(os.path.join(path, '{}_templates.dat'.format(flag)), header=None)
            df_frames.append(templates_prods_reacts_df)
            end += len(templates_prods_reacts_df)
            dataset_begin_end_index.append((begin, end))
            begin = end
        if debug:
            dataset_begin_end_index = [(0, 4000), (4000, 4500), (4500, 5000)]
        torch.save(dataset_begin_end_index,
                   os.path.join(save_path, '{}_split_index.pkl').format(save_fname.split('.')[0]))
        all_templates_prods_reacts_df = pd.concat(df_frames)
        if debug:
            all_templates_prods_reacts_df = all_templates_prods_reacts_df.iloc[:5000, :]
        all_templates_prods_reacts_df.columns = ['templates', 'prod_smiles', 'react_smiles']
        all_templates_prods_reacts_df.to_csv(os.path.join(save_path, save_fname), index=False)
    else:
        template_rules = {}
        with open(os.path.join(path, 'template_rules_1.dat'), 'r') as f:
            for i, l in tqdm(enumerate(f), desc='template rules'):
                rule = l.strip()
                template_rules[rule] = i
        idx2rule = {}
        for rule, idx in template_rules.items():
            idx2rule[idx] = rule
        torch.save((template_rules, idx2rule), os.path.join(save_path, 'templates_index.pkl'))
        templates_prods_reacts_df = pd.read_csv(os.path.join(path, 'templates.dat'), header=None)
        templates_prods_reacts_df.columns = ['templates', 'prod_smiles', 'react_smiles']
        if debug:
            templates_prods_reacts_df = templates_prods_reacts_df.iloc[:5000, :]
        templates_prods_reacts_df.to_csv(os.path.join(save_path, 'USPTO-remapped_tpl_prod_react_train_all.csv'),
                                         index=False)


# if __name__ == '__main__':
#     prod_react_templates = pd.read_csv(os.path.join('../../mlp_retrosyn/mlp_retrosyn/train_all_dataset/templates.dat'), header=None,
#                                        sep='\t')
#     prod_smiles = prod_react_templates[1].tolist()
#     all_atom_types = get_all_atom_types(prod_smiles)
#     print('# atom types:', len(all_atom_types))
#     print(all_atom_types)
#     with open('./train_all_dataset/atom_types.txt', 'w') as f:
#         json.dump(all_atom_types, f)

if __name__ == '__main__':
    debug = True
    save_fname = 'Paroute_tpl_prod_react.csv'
    if get_graph_raw:
        get_graph_raw_dataset(os.path.join('../../single_step_datasets/PaRoutes_set-n5'),
                              os.path.join('./data_set-n5/raw'),
                              save_fname=save_fname,
                              train_all=False, debug=debug)

    reaction_template_dataset = Dataset(root=os.path.join('./data_set-n5'), dataset=save_fname.split('.')[0],
                                        process2fp=True, debug=debug)
    print(len(reaction_template_dataset.train),
          len(reaction_template_dataset.val), len(reaction_template_dataset.test))
