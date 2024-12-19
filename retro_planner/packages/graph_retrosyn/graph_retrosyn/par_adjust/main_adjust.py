import os

import numpy as np
import torch
import itertools
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm, trange

from graph_retrosyn.graph_model import GCN, GCNFP, MPNNFP, DMPNN, DMPNNFP
from torch_geometric.loader import DataLoader
from graph_retrosyn.dataset import Dataset
from graph_retrosyn.graph_train import train_model, eval_one_epoch

gpu = 0
epochs = 50
SUPER_PAR = {
    'lr': [0.001, 0.0001],
    # 'lr': [0.001],
    'batch_size': [512, 1024],
    'fp_lindim': [256, 512],
    'graph_lindim': [256, 512],
    'use_gru': [True, False],
    'massage_depth': [1, 3]
}

if __name__ == '__main__':
    debug = False
    model_name = 'DMPNNFP'

    model_save_path = os.path.join('./adj_model')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    rxn_dataset = Dataset(root=os.path.join('./sampling_data'), dataset='USPTO-remapped_tpl_prod_react_v2',
                          process2fp=True, debug=debug)

    train_dataset, val_dataset, test_dataset = rxn_dataset.train, rxn_dataset.val, \
                                               rxn_dataset.test
    if debug:
        epochs = 2
        train_dataset, val_dataset, test_dataset = train_dataset[:2], val_dataset[:2], test_dataset[:2]
        model_save_path = model_save_path.replace('model', 'debug_model')
        # device = torch.device('cpu')
    print('\n' + 'atom feature size:{}, bond feature size:{}, fingerprint size:{}'.format(train_dataset.data.x.shape[1],
                                                                                          train_dataset.data.edge_attr.shape[
                                                                                              1],
                                                                                          train_dataset.data.fp.shape[
                                                                                              1]))

    template_rules, rule2idx = torch.load(
        os.path.join(rxn_dataset.raw_dir, 'templates_index.pkl'))

    par_comb_index = list(itertools.product([0, 1], repeat=len(SUPER_PAR)))

    par_comb_set = set()
    for ind, par_comb_id in enumerate(par_comb_index):
        init_lr = SUPER_PAR['lr'][0]
        batch_size = SUPER_PAR['batch_size'][par_comb_id[1]]
        fp_lindim = SUPER_PAR['fp_lindim'][par_comb_id[2]]
        graph_lindim = SUPER_PAR['graph_lindim'][par_comb_id[3]]
        use_gru = SUPER_PAR['use_gru'][par_comb_id[4]]
        massage_depth = SUPER_PAR['massage_depth'][par_comb_id[5]]
        par_comb_set.add((
            init_lr,
            batch_size,
            fp_lindim,
            graph_lindim,
            use_gru,
            massage_depth
        ))
    par_comb_set = list(par_comb_set)
    par_comb_set.sort(key=lambda x: x[3], reverse=True)
    print('\n' + '+' * 30)
    print('ALL PAR COMB {}'.format(len(par_comb_set)))
    print('+' * 30)

    best_val_top1 = -1
    best_par_comb = None
    for ind, par_comb in enumerate(par_comb_set):
        init_lr, \
        batch_size, \
        fp_lindim, \
        graph_lindim, \
        use_gru, \
        massage_depth = par_comb
        print('\n' + '#' * 10 + f'Init SUPER_PAR comb {ind}' + '#' * 10)
        print('init_lr: {}\n'
              'batch_size: {}\n'
              'fp_lindim: {}\n'
              'graph_lindim: {}\n'
              'use_gru: {}\n'
              'massage_depth: {}\n'.format(
            init_lr,
            batch_size,
            fp_lindim,
            graph_lindim,
            use_gru,
            massage_depth

        ))
        parameter_dic = {
            'lr': init_lr,
            # 'lr': [0.001],
            'batch_size': batch_size,
            'fp_lindim': fp_lindim,
            'graph_lindim': graph_lindim,
            'use_gru': use_gru,
            'massage_depth': massage_depth
        }
        print('#' * 10 + 'Init Traing {}'.format(model_name) + '#' * 10)

        model = DMPNNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                        b_in_dim=rxn_dataset.data.edge_attr.shape[1],
                        fp_lindim=fp_lindim, fp_dim=2048, use_gru=use_gru,
                        massage_depth=massage_depth)

        model = model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=5,
                                                               min_lr=0.000001)

        print('#' * 10 + 'Traing' + '#' * 10)

        it = trange(epochs)
        best = -1

        for epoch in it:
            # print('+++++++++++++++++++++++++{}+++++++++++++++++++++++++'.format(epoch))
            lr = scheduler.optimizer.param_groups[0]['lr']
            it.set_description(
                'PAR COMB index:{} ALL COMB:{} Epoch: {}, lr: {}'.format(ind, len(par_comb_set), epoch, lr))
            train_model(model, train_loader, loss_fn, optimizer, it, device)
            val_1, val_10, val_50, loss = eval_one_epoch(model, val_loader, device)
            scheduler.step(loss)
            if best < val_1:
                best = val_1
                state = model.state_dict()
                parameter_dic['epoch'] = epoch
                torch.save((state, parameter_dic),
                           os.path.join(model_save_path,
                                        f'saved_graph_rollout_state_1_{model_name}_'
                                        f'init_lr_{str(init_lr)}_'
                                        f'batch_size_{str(batch_size)}_'
                                        f'fp_lindim_{str(fp_lindim)}_'
                                        f'graph_lindim_{str(graph_lindim)}_'
                                        f'use_gru_{str(use_gru)}_'
                                        f'massage_depth_{str(massage_depth)}'
                                        f'.ckpt'))
            print("\n{}  PAR COMB index:{} Top 1: {}  ==> Top 10: {} ==> Top 50: {}, validation loss ==> {}".format(
                model_name, ind, val_1,
                val_10,
                val_50, loss))
        if best > best_val_top1:
            best_val_top1 = best
            best_par_comb = parameter_dic
        del model
        del optimizer
        del scheduler
        del loss_fn
        del loss
        del val_1, val_10, val_50
        del train_loader, val_loader

    print('\n' + '@' * 40)
    end_massage = f'best par comb: {best_par_comb}, best validation top1 {best_val_top1}'
    print(end_massage)
    print('\n' + '@' * 40)
