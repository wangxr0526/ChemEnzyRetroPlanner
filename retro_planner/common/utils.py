import csv
import os
from queue import Queue
import random
import signal
from collections import OrderedDict
import numpy as np
import pandas as pd

from rdkit import Chem
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, precision_score, recall_score, \
    f1_score, auc
import torch

class timeout:
    """
    Function for counting time. If a process takes too long to finish, it will be exited by this function.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds

        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def canonicalize_smiles(smi, clear_map=True):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if clear_map:
            [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        return Chem.MolToSmiles(mol)
    else:
        return ''

def proprecess_reactions(rxn_smiles):
    if '>' not in rxn_smiles:
        return ''
    else:
        try:
            reactants, _, products = rxn_smiles.split('>')
        except:
            return ''
        
        reactants = canonicalize_smiles(reactants)
        products = canonicalize_smiles(products)
        if '' in [reactants, products]:
            return ''
        else:
            return f'{reactants}>>{products}'
        
        
        
    
def max_reaction_depth(node):
    if node['type'] == 'reaction':
        # If the node is a reaction, we consider its depth and find the max depth of its children
        max_depth = 1  # Starting with 1 because the current node is a reaction
        if 'children' in node:
            children_depths = [max_reaction_depth(child) for child in node['children']]
            if children_depths:  # If there are child depths, add the maximum child depth to the current depth
                max_depth += max(children_depths)
        return max_depth
    elif 'children' in node:
        # If the node is not a reaction but has children, return the max depth of its children
        return max(max_reaction_depth(child) for child in node['children'])
    else:
        # If the node has no children, it's a leaf node, and its depth is 0
        return 0

def get_writer(output_name, header):
    # output_name = os.path.join(cmd_args.save_dir, fname)
    fout = open(output_name, 'w')
    writer = csv.writer(fout)
    writer.writerow(header)
    return fout, writer

def binary_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc_score = roc_auc_score(y_true, y_score)
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prauc = auc(precision_recall_curve(y_true, y_score)[1], precision_recall_curve(y_true, y_score)[0])
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    d = {'auc': auc_score, 'prauc': prauc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    return d

def multiclass_metrics(y_true, y_pred=None):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    d = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    return d


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 训练集变化不大时使训练加速

def predict_condition_for_dict_route(predictor, dict_route, topk=10):
    rxn2condition_dict = OrderedDict()
    node_queue = Queue()
    node_queue.put(dict_route)
    while not node_queue.empty():
        mol_node = node_queue.get()
        assert mol_node['type'] == 'mol'
        if 'children' not in mol_node: continue
        assert len(mol_node['children']) == 1
        reaction_node = mol_node['children'][0]
        reactants = []
        for c_mol_node in reaction_node['children']:
            reactants.append(c_mol_node['smiles'])
            node_queue.put(c_mol_node)
        reactants = '.'.join(reactants)
        rxn_smiles = '{}>>{}'.format(reactants, mol_node['smiles'])
        if predictor.model_name =='rcr':
            context_combos, context_combo_scores = predictor(rxn_smiles, topk, return_scores=True)
            condition_df = pd.DataFrame(context_combos)
            condition_df.columns = ['Temperature', 'Solvent', 'Reagent', 'Catalyst', 'null1', 'null2']
            condition_df['Score'] = [f"{num:.4f}" for num in context_combo_scores]
        elif predictor.model_name =='parrot':
            condition_df = predictor(rxn_smiles, topk, return_scores=True)
        rxn2condition_dict[rxn_smiles] = condition_df.round(2)
    return rxn2condition_dict
            




if __name__ == '__main__':
    import json
    from condition_predictor.condition_model import NeuralNetContextRecommender

    cont = NeuralNetContextRecommender()
    cont.load_nn_model(
        info_path='../packages/condition_predictor/condition_predictor/data/',
        weights_path='../packages/condition_predictor/condition_predictor/data/dict_weights.npy')
    with open('../test_data/n1-routes_sample_500.json','r', encoding='utf-8') as f:
        dict_routes = json.load(f)
    print(predict_condition_for_dict_route(cont.get_n_conditions, dict_routes[55]))

    
    
