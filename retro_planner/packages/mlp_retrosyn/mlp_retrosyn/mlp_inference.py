from __future__ import print_function
import numpy as np
from rxn_filter.filter_models import FilterModel, FilterPolicy

import torch
import torch.nn.functional as F
from rdchiral.main import rdchiralRunText
from mlp_retrosyn.mlp_policies import load_parallel_model, preprocess
from collections import defaultdict

import logging

# from retro_planner.common.prepare_utils import prepare_filter_policy


def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(
        *sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, topk=None, fp_dim=2048, use_filter=False, filter_path=None,
                 keep_score=False):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(
            state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        self.use_filter = use_filter
        self.keep_score = keep_score
        self.topk = topk
        logging.info('use filter: {}'.format(self.use_filter))
        if self.use_filter:
            logging.info(f'Loding {filter_path}')
        if self.use_filter:
            assert filter_path != None
            filter_model = FilterModel(fp_dim=2048, dim=1024, dropout_rate=0.4)
            filter_model.load_state_dict(torch.load(filter_path,map_location='cpu'))
            filter = FilterPolicy(filter_model=filter_model)
            # filter.load_from_filename('uspto', filter_path)
            # filter.select(filter.items[0])
            # filter = prepare_filter_policy(filter_path=filter_path)
            self.filter = filter

    def run(self, x, topk=10):
        if hasattr(self, 'topk'):
            topk = self.topk if self.topk else topk
        
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr, [-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if isinstance(self.device, int):
            if self.device > 0:
                self.device = torch.device(f'cuda:{self.device}')
            else:
                self.device = torch.device('cpu')
        arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1)
        if self.device != torch.device('cpu'):
            preds = preds.cpu()
        probs, idx = torch.topk(preds, k=topk)
        # probs = F.softmax(probs,dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
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
                print(e)
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

    def merge_and_filter(self, reactant_d, product):
        ret = []
        org_scores = []
        for reactant, l in reactant_d.items():
            try:
                ss, ts = zip(*l)
                if self.use_filter:
                    reaction_smiles = '{}>>{}'.format(product, reactant)
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


if __name__ == '__main__':
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_rule_path', default='./train_all_dataset/train_test_dataset/train_val_template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='./model/saved_rollout_state_1_2048_2021-10-07_00_24_16.ckpt',
                        type=str, help='specify where the trained model is')
    parser.add_argument('--use_filter', action='store_true', default=False)
    args = parser.parse_args()
    use_filter = args.use_filter
    state_path = args.model_path
    template_path = args.template_rule_path
    if use_filter:
        from rxn_filter.filter_models import FilterPolicy
    model = MLPModel(state_path, template_path, device=0, fp_dim=2048, use_filter=use_filter, filter_path=None,
                     keep_score=False)
    x = '[F-:1]'
    # x = '[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]'
    # x = '[S:3](=[O:4])(=[O:5])([O:6][CH2:7][CH:8]([CH2:9][CH2:10][CH2:11][CH3:12])[CH2:13][CH3:14])[OH:15]'
    # x = 'OCC(=O)OCCCO'
    # x = 'CC(=O)NC1=CC=C(O)C=C1'
    x = 'S=C(Cl)(Cl)'
    # x = "NCCNC(=O)c1ccc(/C=N/Nc2ncnc3c2cnn3-c2ccccc2)cc1"
    # x = 'CCOC(=O)c1cnc2c(F)cc(Br)cc2c1O'
    # x = 'COc1cc2ncnc(Oc3cc(NC(=O)Nc4cc(C(C)(C)C(F)(F)F)on4)ccc3F)c2cc1OC'
    # x = 'COC(=O)c1ccc(CN2C(=O)C3(COc4cc5c(cc43)OCCO5)c3ccccc32)o1'
    x = 'O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3'
    # x = 'CO[C@H](CC(=O)O)C(=O)O'
    # x = 'O=C(O)c1cc(OCC(F)(F)F)c(C2CC2)cn1'
    y = model.run(x, 10)
    pprint(y)
