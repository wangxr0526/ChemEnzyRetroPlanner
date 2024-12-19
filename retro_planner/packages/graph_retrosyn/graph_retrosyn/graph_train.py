import os

import numpy as np
import torch

import torch.nn.functional as F
from torch import nn

from tqdm import tqdm, trange

from graph_retrosyn.graph_model import GCN, GCNFP, MPNNFP, DMPNN, DMPNNFP
from torch_geometric.loader import DataLoader
from graph_retrosyn.dataset import Dataset


def top_k_acc(preds, gt, k=1):
    # preds = preds.to(torch.device('cpu'))
    probs, idx = torch.topk(preds, k=k)
    idx = idx.cpu().numpy().tolist()
    gt = gt.cpu().numpy().tolist()
    num = preds.size(0)
    correct = 0
    for i in range(num):
        for id in idx[i]:
            if id == gt[i]:
                correct += 1
    return correct, num


def eval_one_epoch(model, val_loader, device):
    model.eval()
    eval_top1_correct, eval_top1_num = 0, 0
    eval_top10_correct, eval_top10_num = 0, 0
    eval_top50_correct, eval_top50_num = 0, 0
    loss = 0.0
    for data in tqdm(val_loader):
        data = data.to(device)
        with torch.no_grad():
            y_hat = model(data)
            loss += F.cross_entropy(y_hat, data.y).item()
            top_1_correct, num1 = top_k_acc(y_hat, data.y, k=1)
            top_10_correct, num10 = top_k_acc(y_hat, data.y, k=10)
            top_50_correct, num50 = top_k_acc(y_hat, data.y, k=50)
            eval_top1_correct += top_1_correct
            eval_top1_num += num1
            eval_top10_correct += top_10_correct
            eval_top10_num += num10
            eval_top50_correct += top_50_correct
            eval_top50_num += num50
    val_1 = eval_top1_correct / eval_top1_num
    val_10 = eval_top10_correct / eval_top10_num
    val_50 = eval_top50_correct / eval_top50_num
    loss = loss / (len(val_loader.dataset))
    return val_1, val_10, val_50, loss


def train_model(model, train_loader, loss_fn, optimizer, it, device):
    model.train()
    loss_all = 0
    losses = []
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
    return loss_all / len(train_loader.dataset)


if __name__ == '__main__':

    debug = False
    benchmark_paroutes = True
    if benchmark_paroutes:
        benchmark_data_name = 'data_set-n1'
    model_name = 'DMPNNFP'
    epochs = 100
    gpu = 1

    init_lr, \
    batch_size, \
    fp_lindim, \
    graph_lindim, \
    use_gru, \
    massage_depth = 0.001, 1024, 512, 512, False, 3

    if debug:
        batch_size = 8
        epochs = 5
        graph_lindim = 128
        fp_lindim = 128

    parameter_dim = {
        'lr': init_lr,
        # 'lr': [0.001],
        'batch_size': batch_size,
        'fp_lindim': fp_lindim,
        'graph_lindim': graph_lindim,
        'use_gru': use_gru,
        'massage_depth': massage_depth
    }
    print(parameter_dim)

    model_par_mark = f'init_lr_{str(init_lr)}_' + \
                     f'batch_size_{str(batch_size)}_' + \
                     f'fp_lindim_{str(fp_lindim)}_' + \
                     f'graph_lindim_{str(graph_lindim)}_' + \
                     f'use_gru_{str(use_gru)}_' + \
                     f'massage_depth_{str(massage_depth)}'
    if benchmark_paroutes:
        model_par_mark += f'_{benchmark_data_name}'
    mode_fname = f'saved_graph_rollout_state_1_{model_name}_' + model_par_mark + f'.ckpt'

    if model_name == 'GCN':
        process2fp = False
    elif model_name == 'GCNFP':
        process2fp = True
    elif model_name == 'MPNNFP':
        process2fp = True
    elif model_name == 'DMPNN':
        process2fp = False
    elif model_name == 'DMPNNFP':
        process2fp = True

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if not benchmark_paroutes:
        model_save_path = os.path.join('./model')   # org
        rxn_dataset = Dataset(root=os.path.join('./data'), dataset='USPTO-remapped_tpl_prod_react_v2',
                            process2fp=process2fp, debug=debug)
    else:
        model_save_path = os.path.join('./model_benchmark')
        rxn_dataset = Dataset(root=os.path.join(f'./{benchmark_data_name}'), dataset='Paroute_tpl_prod_react',
                            process2fp=process2fp, debug=debug)
    template_rules, rule2idx = torch.load(
        os.path.join(rxn_dataset.raw_dir, 'templates_index.pkl'))

    train_dataset, val_dataset, test_dataset = rxn_dataset.train, rxn_dataset.val, \
                                               rxn_dataset.test
    print('atom feature size:{}, bond feature size:{}, fingerprint size:{}'.format(train_dataset.data.x.shape[1],
                                                                                   train_dataset.data.edge_attr.shape[
                                                                                       1],
                                                                                   train_dataset.data.fp.shape[1]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if model_name == 'GCN':
        model = GCN(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim)
        model.name = 'GCN'
    elif model_name == 'GCNFP':
        model = GCNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                      fp_dim=1024, fp_lindim=fp_lindim)
        model.name = 'GCNFP'
    elif model_name == 'MPNNFP':
        model = MPNNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                       fp_dim=1024, fp_lindim=fp_lindim)
        model.name = 'MPNNFP'
    elif model_name == 'DMPNN':
        model = DMPNN(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                      f_ab_size=graph_lindim + rxn_dataset.data.edge_attr.shape[1])
        model.name = 'DMPNN'
    elif model_name == 'DMPNNFP':
        model = DMPNNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                        b_in_dim=rxn_dataset.data.edge_attr.shape[1],
                        fp_lindim=fp_lindim, fp_dim=2048, use_gru=use_gru,
                        massage_depth=massage_depth)
        model.name = 'DMPNNFP'
    else:
        raise ValueError
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.000001)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    it = trange(epochs)
    best = -1
    for epoch in it:
        # print('+++++++++++++++++++++++++{}+++++++++++++++++++++++++'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        it.set_description('Epoch: {}, lr: {}'.format(epoch, lr))
        train_model(model, train_loader, loss_fn, optimizer, it, device)
        val_1, val_10, val_50, loss = eval_one_epoch(model, val_loader, device)
        scheduler.step(loss)
        if best < val_1:
            best = val_1
            state = model.state_dict()
            torch.save((state, parameter_dim),
                       os.path.join(model_save_path, mode_fname))
        print("\n{}  Top 1: {}  ==> Top 10: {} ==> Top 50: {}, validation loss ==> {}".format(model_name, val_1, val_10,
                                                                                              val_50, loss))
