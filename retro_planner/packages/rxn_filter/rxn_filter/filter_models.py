import torch
from torch import nn
from torch.nn import functional as F

from rxn_filter.filter_dataset import FilterDataset, reaction_to_fingerprint

class FilterModel(nn.Module):
    def __init__(self, fp_dim=2048, dim=1024, dropout_rate=0.4):
        super(FilterModel, self).__init__()
        self.fp_dim = fp_dim
        self.dim = dim
        self.dropout_rate = dropout_rate

        self.lin_p = nn.Linear(self.fp_dim, self.dim)
        self.lin_r = nn.Linear(self.fp_dim, self.dim)
        # self.lin_o = nn.Linear(1, 1)

    def forward(self, fp_prod, fp_rxn):
        # fp_prod = data.fp_prod
        # fp_rxn = data.fp_rxn

        x_prod = F.dropout(self.lin_p(fp_prod), p=self.dropout_rate, training=self.training)
        x_prod = F.elu(x_prod).unsqueeze(1)

        x_rxn = self.lin_r(fp_rxn)
        x_rxn = F.elu(x_rxn).unsqueeze(1)

        out = torch.bmm(x_prod, x_rxn.transpose(1, 2))
        # out = self.lin_o(out)
        # out = torch.sigmoid(out)
        return out.view(-1)

class FilterModelRXNfp(nn.Module):
    def __init__(self, fp_dim=256, dim=1024, dropout_rate=0.4):
        super(FilterModelRXNfp, self).__init__()
        self.fp_dim = fp_dim
        self.dim = dim
        self.dropout_rate = dropout_rate

        self.lin_i = nn.Linear(self.fp_dim, self.dim)
        self.lin_o = nn.Linear(self.dim, 1)

    def forward(self, rxn_fp):

        x_rxn = F.dropout(self.lin_i(rxn_fp), p=self.dropout_rate, training=self.training)
        x_rxn = torch.tanh(x_rxn).unsqueeze(1)
        out = self.lin_o(x_rxn)
        return out.view(-1)


class FilterPolicy:
    def __init__(self, filter_model, cutoff=0.5, device='cpu'):
        self.device=device
        self.cutoff = cutoff
        self.filter_model = filter_model.to(device)
        self.filter_model.eval()

    def is_feasible(self, reaction, return_prob=False):

        prob = self.predict(reaction)
        feasible = (prob >= self.cutoff).item()
        if return_prob:
            return feasible, prob.item()
        else:
            return feasible

    def load(self, filter_fpath):
        checkpoint = torch.load(filter_fpath)
        self.filter_model.load_state_dict(checkpoint)
        self.filter_model.eval()

    def predict(self, reaction):
        prod_fp, rxn_fp = reaction_to_fingerprint(reaction, radius=2, nbits=2048)
        prod_fp, rxn_fp = prod_fp.to(self.device), rxn_fp.to(self.device)
        with torch.no_grad():
            y_score = torch.sigmoid(self.filter_model(prod_fp, rxn_fp))
        return y_score


if __name__ == '__main__':
    rxns = ['C=CCN1C[C@H](C)N([C@H](c2cccnc2)c2cccc(O)c2)C[C@H]1C.O=S(=O)(N(c1ccccc1)S(=O)(=O)C(F)(F)F)C(F)(F)F>>' + \
            'C=CCN1C[C@H](C)N([C@H](c2cccnc2)c2cccc(OS(=O)(=O)C(F)(F)F)c2)C[C@H]1C',
            'C=CCN1C[C@H](C)N([C@H](c2cccnc2)c2cccc(O)c2)C[C@H]1C.O=S(=O)(N(c1ccccc1)S(=O)(=O)C(F)(F)F)C(F)(F)F>>' + \
            'C=CCN1C[C@H](C)N([C@H](c2cccnc2)c2cccc(OS(=O)(=O)C(F)(F)F)c2)C[C@H]1C']
    # prod = rxn.split('>>')[-1]

    model = FilterModel(fp_dim=2048, dim=1024, dropout_rate=0.4)
    filter_policy = FilterPolicy(model, cutoff=0.9)
    filter_policy.is_feasible(rxns[0])

    # processed_data = [_reaction_to_fingerprint(x) for x in rxns]
    # processed_prod = [x[0] for x in processed_data]
    # processed_rxn = [x[1] for x in processed_data]
    # processed_prod = torch.cat([processed_prod[0], processed_prod[1]])
    # processed_rxn = torch.cat([processed_rxn[0], processed_rxn[1]])

    dataset = FilterDataset(raw_data_path='./data/toy/toy_data.csv',
                            procssed_data_path='./data/toy/toy_data.pkl')
    fps_prod, fps_rxn = dataset.fps_prod, dataset.fps_rxn

    preds = model(fps_prod, fps_rxn)
    print(preds)
