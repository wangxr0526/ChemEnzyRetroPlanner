import logging

from torch.nn import functional as F, Sequential, Linear, ReLU, GRU
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Set2Set, NNConv

from collections import defaultdict
from rxn_filter.filter_models import FilterModel, FilterPolicy

import torch
from rdchiral.main import rdchiralRunText
from rdkit.Chem import MolFromSmiles


from graph_retrosyn.dataset import get_mol_nodes_edges, preprocess2fp
from graph_retrosyn.graph_layer import DMPNNLayer, DMPNNAlphaLayer

# from retro_planner.common.prepare_utils import prepare_filter_policy


class GCN(torch.nn.Module):
    def __init__(self, mol_in_dim=15, out_dim=2, dim=64):
        super(GCN, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = GCNConv(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class GCNFP(torch.nn.Module):
    def __init__(self, mol_in_dim=15, out_dim=2, dim=256, fp_dim=1024, fp_lindim=256,
                 dropout_rate=0.3):
        super(GCNFP, self).__init__()
        self.fp_dim = fp_dim
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(fp_dim, fp_lindim)
        self.bn1 = torch.nn.BatchNorm1d(fp_lindim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = GCNConv(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim + fp_lindim, out_dim)

    def forward(self, data):
        out_fp = F.dropout(F.elu(self.bn1(self.fc1(data.fp))), p=0.3, training=self.training)

        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index))
        out = self.set2set(out, data.batch)
        out = F.dropout(F.relu(self.lin1(out)), p=0.3, training=self.training)
        out = torch.cat([out, out_fp], dim=-1)
        out = self.lin2(out)
        return out


class MPNNFP(torch.nn.Module):
    def __init__(self, mol_in_dim=15, out_dim=2, dim=256, fp_dim=1024, fp_lindim=256,
                 dropout_rate=0.3):
        super(MPNNFP, self).__init__()
        self.fp_dim = fp_dim
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(fp_dim, fp_lindim)
        self.bn1 = torch.nn.BatchNorm1d(fp_lindim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        nn = Sequential(Linear(4, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim + fp_lindim, out_dim)
        # self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.lin3 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out_fp = F.dropout(F.elu(self.bn1(self.fc1(data.fp))), p=0.3, training=self.training)

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        out = F.dropout(F.relu(self.lin1(out)), p=0.3, training=self.training)
        out = torch.cat([out, out_fp], dim=-1)
        # out = self.bn2(F.dropout(F.relu(self.lin2(out)), p=0.3))
        out = self.lin2(out)
        return out


class DMPNN(torch.nn.Module):
    def __init__(self, mol_in_dim=15, out_dim=2, dim=64, f_ab_size=100):
        super(DMPNN, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        nn = torch.nn.Linear(f_ab_size, dim)
        self.conv = DMPNNLayer(dim, dim, dim, nn, dropout=0.3)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class DMPNNFP(torch.nn.Module):
    def __init__(self, mol_in_dim=15, out_dim=2, dim=256, b_in_dim=14, fp_lindim=64, fp_dim=1024, use_gru=False,
                 massage_depth=1,
                 dropout_rate=0.3):
        super(DMPNNFP, self).__init__()
        self.fp_dim = fp_dim
        self.dropout_rate = dropout_rate
        self.use_gru = use_gru
        self.massage_depth = massage_depth
        self.fc1 = torch.nn.Linear(fp_dim, fp_lindim)
        self.bn1 = torch.nn.BatchNorm1d(fp_lindim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = DMPNNAlphaLayer(dim, dim, dim, b_in_dim, dropout=0.3)
        if self.use_gru:
            self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim + fp_lindim, out_dim)

    def forward(self, data):
        out_fp = F.dropout(F.elu(self.bn1(self.fc1(data.fp))), p=0.3, training=self.training)
        out = F.relu(self.lin0(data.x))
        if self.use_gru:
            h = out.unsqueeze(0)
        for i in range(self.massage_depth):

            if self.use_gru:
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
            else:
                out = F.relu(self.conv(out, data.edge_index, data.edge_attr))

        # out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        out = self.set2set(out, data.batch)
        out = F.dropout(F.relu(self.lin1(out)), p=0.3, training=self.training)
        out = torch.cat([out, out_fp], dim=-1)
        out = self.lin2(out)
        return out


def process(smi):
    mol = MolFromSmiles(smi)
    x, edge_index, edge_attr = get_mol_nodes_edges(mol)
    fp = preprocess2fp(mol, 2048)
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, smi=smi, fp=fp)
    data.batch = torch.tensor([0], dtype=torch.long)
    return data


class GraphModel(object):
    def __init__(self,
                 graph_model,
                 idx2rules,
                 device=-1,
                 topk=None,
                 use_filter=False,
                 filter_config=None,
                 filter_path=None,
                 keep_score=False):
        super(GraphModel, self).__init__()
        self.net, self.idx2rules = graph_model, idx2rules
        self.net.eval()
        self.device = device
        self.use_filter = use_filter
        self.keep_score = keep_score
        self.topk = topk
        logging.info('use filter: {}'.format(self.use_filter))
        if self.use_filter:
            logging.info(f'Loding {filter_path}')
        self.net.to(device)
        if self.use_filter:
            assert filter_path != None
            filter_model = FilterModel(fp_dim=2048, dim=1024, dropout_rate=0.4)
            filter_model.load_state_dict(
                torch.load(filter_path, map_location='cpu'))
            filter = FilterPolicy(filter_model=filter_model)
            # # filter.load_from_filename('uspto', filter_path)
            # # filter.select(filter.items[0])
            
            # filter = prepare_filter_policy(filter_path=filter_path)
            
            self.filter = filter

    def run(self, data, topk=10, run_from_pred=False):
        if hasattr(self, 'topk'):
            topk = self.topk if self.topk else topk
        
        def run_one_rxn(x, rule_k):
            reactants = []
            scores = []
            templates = []
            for i, rule in enumerate(rule_k):
                out1 = []
                try:
                    out1 = rdchiralRunText(rule, x)
                    # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                    if len(out1) == 0:
                        continue
                    # if len(out1) > 1: print("more than two reactants."),print(out1)
                    out1 = sorted(out1)
                    for reactant in out1:
                        reactants.append(reactant)
                        scores.append(probs[0][i].item() / len(out1))
                        templates.append(rule)
                # out1 = rdchiralRunText(x, rule)
                except Exception as e:
                    # print(e)
                    pass
            if len(reactants) == 0:
                return None
            reactants_d = defaultdict(list)
            for r, s, t in zip(reactants, scores, templates):
                if '.' in r:
                    str_list = sorted(r.strip().split('.'))
                    reactants_d['.'.join(str_list)].append((s, t))
                else:
                    reactants_d[r].append((s, t))

            reactants, scores, templates, org_scores = self.merge_and_filter(
                reactants_d, x)
            if self.keep_score:
                total = sum(org_scores)
            else:
                total = sum(scores)
            if total == 0:
                return None
            scores = [s / total for s in scores]
            return {'reactants': reactants,
                    'scores': scores,
                    'template': templates}

        if run_from_pred:
            prod_list, preds = data
            preds = F.softmax(preds, dim=1)
            if self.device != 'cpu':
                preds = preds.cpu()
            probs, idxs = torch.topk(preds, k=topk)
            rule_k_list = [[self.idx2rules[ids] for ids in idx]
                           for idx in idxs.numpy().tolist()]
            assert len(prod_list) == len(rule_k_list)
            results = []
            for prod, rule_k in zip(prod_list, rule_k_list):
                results.append(run_one_rxn(prod, rule_k))
            return results
        else:
            data = process(data)
            x = data['smi']
            if isinstance(self.device, int):
                if self.device > 0:
                    self.device = torch.device(f'cuda:{self.device}')
                else:
                    self.device = torch.device('cpu')
            data = data.to(self.device)
            data.fp = data.fp.view(1, -1)
            try:
                preds = self.net(data)
            except:
                return None
            preds = F.softmax(preds, dim=1)
            if self.device != torch.device('cpu'):
                preds = preds.cpu()
            probs, idx = torch.topk(preds, k=topk)
            # probs = F.softmax(probs,dim=1)
            rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
            result = run_one_rxn(x, rule_k)
            return result
        # data.batch = torch.tensor([0], dtype=torch.long)

    def merge_and_filter(self, reactant_d, product):
        ret = []
        org_scores = []
        for reactant, l in reactant_d.items():
            try:
                ss, ts = zip(*l)
                if self.use_filter:
                    reaction_smiles = '{}>>{}'.format(reactant, product)
                    feasible = self.filter.is_feasible(reaction_smiles)
                    if feasible:
                        ret.append((reactant, sum(ss), list(ts)[0]))
                    else:
                        pass
                else:
                    ret.append((reactant, sum(ss), list(ts)[0]))
                if self.keep_score:
                    org_scores.append(sum(ss))
            except Exception as e:
                logging.info(e)
                pass

        if ret:
            reactants, scores, templates = zip(
                *sorted(ret, key=lambda item: item[1], reverse=True))
            return list(reactants), list(scores), list(templates), org_scores
        else:
            return [], [], [], []
