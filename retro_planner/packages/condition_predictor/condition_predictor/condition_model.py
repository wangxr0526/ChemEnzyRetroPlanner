from collections import OrderedDict
import os
import torch
from torch import nn
from torch.nn import functional as F
from scipy import stats
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
contextRecommender_loc = 'contextRecommender'


def create_rxn_Morgan2FP_separately(rsmi, psmi, rxnfpsize=16384, pfpsize=16384, useFeatures=False, calculate_rfp=True, useChirality=False):
    # Similar as the above function but takes smiles separately and returns pfp and rfp separately

    rsmi = rsmi.encode('utf-8')
    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print(e)
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=rxnfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(rxnfpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        return
    rfp = fp

    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=pfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(pfpsize, dtype='float32')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
    pfp = fp
    return [pfp, rfp]


class RXNConditionModel(nn.Module):
    def __init__(
        self,
        fp_dim=16384,
        h_dim=1000,
        dropout_rate=0.5,
        c1_dim=803,
        s1_dim=232,
        s2_dim=228,
        r1_dim=2240,
        r2_dim=1979,
    ):
        super(RXNConditionModel, self).__init__()
        self.fp_dim = fp_dim
        self.h_dim = h_dim
        self.dropout_rate = dropout_rate
        self.c1_dim = c1_dim
        self.s1_dim = s1_dim
        self.s2_dim = s2_dim
        self.r1_dim = r1_dim
        self.r2_dim = r2_dim

        self.lin_fp_transform1 = nn.Linear(self.fp_dim*2, self.h_dim)
        self.lin_fp_transform2 = nn.Linear(self.h_dim, self.h_dim)

        self.lin_c1_dense = nn.Linear(c1_dim, 100)
        self.lin_s1_dense = nn.Linear(s1_dim, 100)
        self.lin_s2_dense = nn.Linear(s2_dim, 100)
        self.lin_r1_dense = nn.Linear(r1_dim, 100)
        self.lin_r2_dense = nn.Linear(r2_dim, 100)

        self.lin_c1_h1 = nn.Linear(1000, 300)
        self.lin_s1_h1 = nn.Linear(1100, 300)
        self.lin_s2_h1 = nn.Linear(1200, 300)
        self.lin_r1_h1 = nn.Linear(1300, 300)
        self.lin_r2_h1 = nn.Linear(1400, 300)
        self.lin_T_h1 = nn.Linear(1500, 300)

        self.lin_c1_h2 = nn.Linear(300, 300)
        self.lin_r1_h2 = nn.Linear(300, 300)
        self.lin_r2_h2 = nn.Linear(300, 300)
        self.lin_s1_h2 = nn.Linear(300, 300)
        self.lin_s2_h2 = nn.Linear(300, 300)

        self.lin_c1 = nn.Linear(300, c1_dim)
        self.lin_r1 = nn.Linear(300, r1_dim)
        self.lin_r2 = nn.Linear(300, r2_dim)
        self.lin_s1 = nn.Linear(300, s1_dim)
        self.lin_s2 = nn.Linear(300, s2_dim)
        self.lin_T = nn.Linear(300, 1)

    def load_parameter_from_keras(self, fpath):
        param_dict = np.load(fpath, allow_pickle=True).item()
        state_dict = OrderedDict()
        for kname in param_dict:
            layer_flag = kname.split('/')[0]
            w_or_b_flag = 'weight' if 'kernel' in kname else 'bias'
            tname = 'lin_{}.{}'.format(layer_flag, w_or_b_flag)
            state_dict[tname] = torch.from_numpy(
                np.transpose(param_dict[kname]))

        self.load_state_dict(state_dict)

    def fp_func(self, pfp, rxnfp):
        fp_emb = torch.cat([pfp, rxnfp], dim=1)      # concatenate_1
        fp_emb = F.relu(self.lin_fp_transform1(
            fp_emb))          # fp_transform1
        fp_emb = F.relu(
            F.dropout(self.lin_fp_transform2(fp_emb), p=self.dropout_rate, training=self.training))
        return fp_emb

    def c1_func(self, fp_emb):
        out = F.relu(self.lin_c1_h1(fp_emb))
        out = torch.tanh(self.lin_c1_h2(out))
        out = F.softmax(self.lin_c1(out), dim=1)
        return out

    def s1_func(self, fp_emb, input_c1):
        c1_emb = F.relu(self.lin_c1_dense(input_c1))
        concat_fp_c1 = torch.cat([fp_emb, c1_emb], dim=1)
        out = F.relu(self.lin_s1_h1(concat_fp_c1))
        out = torch.tanh(self.lin_s1_h2(out))
        out = F.softmax(self.lin_s1(out), dim=1)
        return out

    def s2_func(self, fp_emb, input_c1, input_s1):
        c1_emb = F.relu(self.lin_c1_dense(input_c1))
        s1_emb = F.relu(self.lin_s1_dense(input_s1))
        concat_fp_c1_s1 = torch.cat([fp_emb, c1_emb, s1_emb], dim=1)
        out = F.relu(self.lin_s2_h1(concat_fp_c1_s1))
        out = torch.tanh(self.lin_s2_h2(out))
        out = F.softmax(self.lin_s2(out), dim=1)
        return out

    def r1_func(self, fp_emb, input_c1, input_s1, input_s2):
        c1_emb = F.relu(self.lin_c1_dense(input_c1))
        s1_emb = F.relu(self.lin_s1_dense(input_s1))
        s2_emb = F.relu(self.lin_s2_dense(input_s2))
        concat_fp_c1_s1_s2 = torch.cat([fp_emb, c1_emb, s1_emb, s2_emb], dim=1)
        out = F.relu(self.lin_r1_h1(concat_fp_c1_s1_s2))
        out = torch.tanh(self.lin_r1_h2(out))
        out = F.softmax(self.lin_r1(out), dim=1)
        return out

    def r2_func(self, fp_emb, input_c1, input_s1, input_s2, input_r1):
        c1_emb = F.relu(self.lin_c1_dense(input_c1))
        s1_emb = F.relu(self.lin_s1_dense(input_s1))
        s2_emb = F.relu(self.lin_s2_dense(input_s2))
        r1_emb = F.relu(self.lin_r1_dense(input_r1))
        concat_fp_c1_s1_s2_r1 = torch.cat(
            [fp_emb, c1_emb, s1_emb, s2_emb, r1_emb], dim=1)
        out = F.relu(self.lin_r2_h1(concat_fp_c1_s1_s2_r1))
        out = torch.tanh(self.lin_r2_h2(out))
        out = F.softmax(self.lin_r2(out), dim=1)
        return out

    def T_func(self, fp_emb, input_c1, input_s1, input_s2, input_r1, input_r2):
        c1_emb = F.relu(self.lin_c1_dense(input_c1))
        s1_emb = F.relu(self.lin_s1_dense(input_s1))
        s2_emb = F.relu(self.lin_s2_dense(input_s2))
        r1_emb = F.relu(self.lin_r1_dense(input_r1))
        r2_emb = F.relu(self.lin_r2_dense(input_r2))
        concat_fp_c1_s1_s2_r1_r2 = torch.cat(
            [fp_emb, c1_emb, s1_emb, s2_emb, r1_emb, r2_emb], dim=1)
        out = F.relu(self.lin_T_h1(concat_fp_c1_s1_s2_r1_r2))
        out = self.lin_T(out)
        return out


class NeuralNetContextRecommender():
    """Pytorch vision of 'Reaction condition predictor based on Nearest Neighbor method'
    Adapted from https://github.com/Coughy1991/Reaction_condition_recommendation/blob/master/scripts/neuralnetwork.py
    """

    def __init__(self, max_contexts=10, singleSlvt=True, with_smiles=True):
        """
        :param singleSlvt:
        :param with_smiles:
        """
        self.nnModel = None
        self.c1_dict = None
        self.s1_dict = None
        self.s2_dict = None
        self.r1_dict = None
        self.r2_dict = None
        self.num_cond = 1
        self.singleSlvt = singleSlvt
        self.with_smiles = with_smiles
        self.max_total_context = max_contexts
        self.max_context = 2
        self.fp_size = 2048

    def load_nn_model(self, info_path="", weights_path=""):
        if not info_path:
            print('Cannot load nerual net context recommender without a specific path to the model info. Exiting...')

        ###load model##############

        self.nnModel = RXNConditionModel()
        # load weights into new model
        self.nnModel.load_parameter_from_keras(weights_path)
        self.nnModel.eval()
        # get fp_size based on the model
        self.fp_size = self.nnModel.fp_dim
        
        r1_dict_file = os.path.join(info_path, "r1_dict.pickle")
        r2_dict_file = os.path.join(info_path, "r2_dict.pickle")
        s1_dict_file = os.path.join(info_path, "s1_dict.pickle")
        s2_dict_file = os.path.join(info_path, "s2_dict.pickle")
        c1_dict_file = os.path.join(info_path, "c1_dict.pickle")

        with open(r1_dict_file, "rb") as R1_DICT_F:
            self.r1_dict = pickle.load(R1_DICT_F)

        with open(r2_dict_file, "rb") as R2_DICT_F:
            self.r2_dict = pickle.load(R2_DICT_F)

        with open(s1_dict_file, "rb") as S1_DICT_F:
            self.s1_dict = pickle.load(S1_DICT_F)

        with open(s2_dict_file, "rb") as S2_DICT_F:
            self.s2_dict = pickle.load(S2_DICT_F)

        with open(c1_dict_file, "rb") as C1_DICT_F:
            self.c1_dict = pickle.load(C1_DICT_F)

        self.c1_dim = self.nnModel.c1_dim
        self.r1_dim = self.nnModel.r1_dim
        self.r2_dim = self.nnModel.r2_dim
        self.s1_dim = self.nnModel.s1_dim
        self.s2_dim = self.nnModel.s2_dim

        self.inverse_c1_dict = {v: k for k, v in self.c1_dict.items()}
        self.inverse_s1_dict = {v: k for k, v in self.s1_dict.items()}
        self.inverse_s2_dict = {v: k for k, v in self.s2_dict.items()}
        self.inverse_r1_dict = {v: k for k, v in self.r1_dict.items()}
        self.inverse_r2_dict = {v: k for k, v in self.r2_dict.items()}

        self.fp_func = self.nnModel.fp_func
        self.c1_func = self.nnModel.c1_func
        self.s1_func = self.nnModel.s1_func
        self.s2_func = self.nnModel.s2_func
        self.r1_func = self.nnModel.r1_func
        self.r2_func = self.nnModel.r2_func
        self.T_func = self.nnModel.T_func

    def get_n_conditions(self, rxn, n=10, singleSlvt=True, with_smiles=True, return_scores=False):
        """
        Reaction condition recommendations for a rxn (SMILES) from top n NN
        Returns the top n parseable conditions.
        """
        # print('started neuralnet')
        self.singleSlvt = singleSlvt
        self.with_smiles = with_smiles
        # print('rxn to recommend context for : {}'.format(rxn), contextRecommender_loc)
        try:
            rsmi = rxn.split('>>')[0]
            psmi = rxn.split('>>')[1]
            rct_mol = Chem.MolFromSmiles(rsmi)
            prd_mol = Chem.MolFromSmiles(psmi)
            [atom.ClearProp('molAtomMapNumber') for
             atom in rct_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
            [atom.ClearProp('molAtomMapNumber') for
             atom in prd_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
            rsmi = Chem.MolToSmiles(rct_mol, isomericSmiles=True)
            psmi = Chem.MolToSmiles(prd_mol, isomericSmiles=True)
            [pfp, rfp] = create_rxn_Morgan2FP_separately(
                rsmi, psmi, rxnfpsize=self.fp_size, pfpsize=self.fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
            pfp = pfp.reshape(1, self.fp_size)
            rfp = rfp.reshape(1, self.fp_size)
            rxnfp = pfp - rfp
            c1_input = []
            r1_input = []
            r2_input = []
            s1_input = []
            s2_input = []
            inputs = [pfp, rxnfp, c1_input, r1_input,
                      r2_input, s1_input, s2_input]

            (top_combos, top_combo_scores) = self.predict_top_combos(inputs=inputs)

            if return_scores:
                return (top_combos[:n], top_combo_scores[:n])
            else:
                return top_combos[:n]

        except Exception as e:
            print('Failed for reaction {} because {}. Returning None.'.format(
                rxn, e), contextRecommender_loc)
            return [[]]

    def path_condition(self, n, path):
        """Reaction condition recommendation for a reaction path with multiple reactions
            path: a list of reaction SMILES for each step
            return: a list of reaction context with n options for each step
        """
        rsmi_list = []
        psmi_list = []
        contexts = []
        for rxn in path:
            try:
                rsmi = rxn.split('>>')[0]
                psmi = rxn.split('>>')[1]

                rct_mol = Chem.MolFromSmiles(rsmi)
                prd_mol = Chem.MolFromSmiles(psmi)
                [atom.ClearProp('molAtomMapNumber')for
                 atom in rct_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
                [atom.ClearProp('molAtomMapNumber')for
                 atom in prd_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
                rsmi = Chem.MolToSmiles(rct_mol, isomericSmiles=True)
                psmi = Chem.MolToSmiles(prd_mol, isomericSmiles=True)
                [pfp, rfp] = create_rxn_Morgan2FP_separately(
                    rsmi, psmi, rxnfpsize=self.fp_size, pfpsize=self.fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
                pfp = pfp.reshape(1, self.fp_size)
                rfp = rfp.reshape(1, self.fp_size)
                rxnfp = pfp - rfp
                c1_input = []
                r1_input = []
                r2_input = []
                s1_input = []
                s2_input = []
                inputs = [pfp, rxnfp, c1_input, r1_input,
                          r2_input, s1_input, s2_input]
                top_combos = self.predict_top_combos(
                    inputs=inputs, c1_rank_thres=1, s1_rank_thres=3, s2_rank_thres=1, r1_rank_thres=4, r2_rank_thres=1)
                contexts.append(top_combos[:n])
            except Exception as e:
                print(
                    'Failed for reaction {} because {}. Returning None.'.format(rxn, e))
        return contexts

    def predict_top_combos(self, inputs, return_categories_only=False, c1_rank_thres=2, s1_rank_thres=3, s2_rank_thres=1, r1_rank_thres=3, r2_rank_thres=1):
        # this function predicts the top combos based on rank thresholds for
        # individual elements
        context_combos = []
        context_combo_scores = []
        num_combos = c1_rank_thres*s1_rank_thres * \
            s2_rank_thres*r1_rank_thres*r2_rank_thres
        [pfp, rxnfp, c1_input_user, r1_input_user,
            r2_input_user, s1_input_user, s2_input_user] = inputs
        pfp = torch.from_numpy(pfp).float()
        rxnfp = torch.from_numpy(rxnfp).float()
        # set s2 to be none ## specifically for the single sovlent case
        # s2_input_user = np.zeros(s2_dim,dtype = 'float32').reshape(1,s2_dim)
        # s2_input_user[0] = 1
        with torch.no_grad():
            fp_trans = self.fp_func(pfp, rxnfp)
            if c1_input_user == []:
                c1_inputs = fp_trans
                c1_pred = self.c1_func(c1_inputs)
                _, c1_cdts = c1_pred.squeeze().topk(c1_rank_thres)
                c1_cdts = c1_cdts.numpy()
            else:
                c1_cdts = np.nonzero(c1_input_user)[0]
            # find the name of catalyst
            for c1_cdt in c1_cdts:
                c1_name = self.c1_dict[c1_cdt]
                c1_input = np.zeros([1, self.c1_dim])
                c1_input[0, c1_cdt] = 1
                if c1_input_user == []:
                    c1_sc = c1_pred.squeeze().numpy()[c1_cdt]
                else:
                    c1_sc = 1
                if s1_input_user == []:
                    s1_pred = self.s1_func(
                        fp_trans, torch.from_numpy(c1_input).float())
                    _, s1_cdts = s1_pred.squeeze().topk(s1_rank_thres)
                    s1_cdts = s1_cdts.numpy()
                else:
                    s1_cdts = np.nonzero(s1_input_user)[0]
                for s1_cdt in s1_cdts:
                    s1_name = self.s1_dict[s1_cdt]
                    s1_input = np.zeros([1, self.s1_dim])
                    s1_input[0, s1_cdt] = 1
                    if s1_input_user == []:
                        s1_sc = s1_pred.squeeze().numpy()[s1_cdt]
                    else:
                        s1_sc = 1
                    if s2_input_user == []:
                        s2_pred = self.s2_func(fp_trans, torch.from_numpy(
                            c1_input).float(), torch.from_numpy(s1_input).float())
                        _, s2_cdts = s2_pred.squeeze().topk(s2_rank_thres)
                        s2_cdts = s2_cdts.numpy()
                    else:
                        s2_cdts = np.nonzero(s2_input_user)[0]
                    for s2_cdt in s2_cdts:
                        s2_name = self.s2_dict[s2_cdt]
                        s2_input = np.zeros([1, self.s2_dim])
                        s2_input[0, s2_cdt] = 1
                        if s2_input_user == []:
                            s2_sc = s2_pred.squeeze().numpy()[s2_cdt]
                        else:
                            s2_sc = 1
                        if r1_input_user == []:
                            r1_pred = self.r1_func(fp_trans, torch.from_numpy(c1_input).float(
                            ), torch.from_numpy(s1_input).float(), torch.from_numpy(s2_input).float())
                            _, r1_cdts = r1_pred.squeeze().topk(r1_rank_thres)
                            r1_cdts = r1_cdts.numpy()
                        else:
                            r1_cdts = np.nonzero(r1_input_user)[0]
                        for r1_cdt in r1_cdts:
                            r1_name = self.r1_dict[r1_cdt]
                            r1_input = np.zeros([1, self.r1_dim])
                            r1_input[0, r1_cdt] = 1
                            if r1_input_user == []:
                                r1_sc = r1_pred.squeeze().numpy()[r1_cdt]
                            else:
                                r1_sc = 1
                            if r2_input_user == []:
                                r2_pred = self.r2_func(fp_trans, torch.from_numpy(c1_input).float(), torch.from_numpy(
                                    s1_input).float(), torch.from_numpy(s2_input).float(), torch.from_numpy(r1_input).float())
                                _, r2_cdts = r2_pred.squeeze().topk(r2_rank_thres)
                                r2_cdts = r2_cdts.numpy()
                            else:
                                r2_cdts = np.nonzero(r2_input_user)[0]
                            for r2_cdt in r2_cdts:
                                r2_name = self.r2_dict[r2_cdt]
                                r2_input = np.zeros([1, self.r2_dim])
                                r2_input[0, r2_cdt] = 1
                                if r2_input_user == []:
                                    r2_sc = r2_pred.squeeze().numpy()[r2_cdt]
                                else:
                                    r2_sc = 1
                                T_pred = self.T_func(fp_trans, torch.from_numpy(c1_input).float(), torch.from_numpy(s1_input).float(
                                ), torch.from_numpy(s2_input).float(), torch.from_numpy(r1_input).float(),  torch.from_numpy(r2_input).float())
                                T_pred = T_pred.item()
                                # print(c1_name,s1_name,s2_name,r1_name,r2_name)
                                cat_name = [c1_name]
                                if r2_name == '':
                                    rgt_name = [r1_name]
                                else:
                                    rgt_name = [r1_name, r2_name]
                                if s2_name == '':
                                    slv_name = [s1_name]
                                else:
                                    slv_name = [s1_name, s2_name]
                                # if self.with_smiles:
                                #     rgt_name = [rgt for rgt in rgt_name if 'Reaxys' not in rgt]
                                #     slv_name = [slv for slv in slv_name if 'Reaxys' not in slv]
                                #     cat_name = [cat for cat in cat_name if 'Reaxys' not in cat]
                                # for testing purpose only, output order as training
                                if return_categories_only:
                                    context_combos.append(
                                        [c1_cdt, s1_cdt, s2_cdt, r1_cdt, r2_cdt, T_pred])
                                # esle ouptupt format compatible with the overall framework
                                else:
                                    context_combos.append(
                                        [T_pred, '.'.join(slv_name), '.'.join(rgt_name), '.'.join(cat_name), np.nan, np.nan])

                                context_combo_scores.append(
                                    c1_sc*s1_sc*s2_sc*r1_sc*r2_sc)
        context_ranks = list(
            num_combos+1 - stats.rankdata(context_combo_scores))

        context_combos = [context_combos[
            context_ranks.index(i+1)] for i in range(num_combos)]
        context_combo_scores = [context_combo_scores[
            context_ranks.index(i+1)] for i in range(num_combos)]

        return (context_combos, context_combo_scores)

    def category_to_name(self, chem_type, category):
        if chem_type == 'c1':
            return self.c1_dict[category]
        elif chem_type == 's1':
            return self.s1_dict[category]
        elif chem_type == 's2':
            return self.s2_dict[category]
        elif chem_type == 'r1':
            return self.r1_dict[category]
        elif chem_type == 'r2':
            return self.r2_dict[category]

    def name_to_category(self, chem_type, name):
        try:
            if chem_type == 'c1':
                return self.inverse_c1_dict[name]
            elif chem_type == 's1':
                return self.inverse_s1_dict[name]
            elif chem_type == 's2':
                return self.inverse_s2_dict[name]
            elif chem_type == 'r1':
                return self.inverse_r1_dict[name]
            elif chem_type == 'r2':
                return self.inverse_r2_dict[name]
        except:
            print('name {} not found!'.format(name))


if __name__ == '__main__':
    # condition_model = RXNConditionModel()
    # condition_model.load_parameter_from_keras('data/dict_weights.npy')
    # condition_model.eval()
    # rxn = 'CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1'
    # rsmi = rxn.split('>>')[0]
    # psmi = rxn.split('>>')[1]
    # [pfp, rfp] = create_rxn_Morgan2FP_separately(
    #     rsmi, psmi, rxnfpsize=16384, pfpsize=16384, useFeatures=False, calculate_rfp=True, useChirality=True)
    # pfp = torch.from_numpy(pfp.reshape(1, -1))
    # rfp = torch.from_numpy(rfp.reshape(1, -1))
    # rxnfp = pfp - rfp
    # with torch.no_grad():
    #     fp_emb = condition_model.fp_func(pfp, rxnfp)
    #     c1_pred = condition_model.c1_func(fp_emb)
    #     s1_pred = condition_model.s1_func(fp_emb, c1_pred)
    #     s2_pred = condition_model.s2_func(fp_emb, c1_pred, s1_pred)
    #     r1_pred = condition_model.r1_func(fp_emb, c1_pred, s1_pred, s2_pred)
    #     r2_pred = condition_model.r2_func(
    #         fp_emb, c1_pred, s1_pred, s2_pred, r1_pred)
    #     T_pred = condition_model.T_func(
    #         fp_emb, c1_pred, s1_pred, s2_pred, r1_pred, r2_pred)
    #     print(c1_pred, s1_pred, s2_pred, r1_pred, r2_pred, T_pred)

    cont = NeuralNetContextRecommender()
    cont.load_nn_model(info_path='./data/',
                       weights_path='./data/dict_weights.npy')

    context_combos, context_combo_scores = cont.get_n_conditions(
        'CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1', 10, return_scores=True)
    
    for cont, score in zip(context_combos, context_combo_scores):
        print('#################################################')
        print('Temperature, Solvent, Reagent, Catalyst')
        print('{:.2f}, {}, {}, {}, '.format(*cont[:4]))
        print(score)
        print()
