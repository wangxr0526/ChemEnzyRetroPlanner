import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from graph_retrosyn.graph_model import GCN, GraphModel, GCNFP, MPNNFP, DMPNN, DMPNNFP
from retro_planner.common.utils import canonicalize_smiles
from graph_retrosyn.dataset import Dataset


def eval_model(data_or_loader, model, topk, fname_pred, device):
    # case_gen = rxn_data_gen(phase, model)
    # model.net.eval()
    import torch_geometric
    cnt = 0
    topk_scores = [0.0] * topk

    if isinstance(data_or_loader, torch_geometric.loader.dataloader.DataLoader):
        fpred = open(fname_pred, 'w')
        pbar = tqdm(data_or_loader)
        for data in pbar:
            prod_list = data['smi']
            react = data['ground_truth']
            data = data.to(device)
            with torch.no_grad():
                preds = model.net(data)
            pred_structs = model.run((prod_list, preds), topk=topk, run_from_pred=True)
            # pbar = tqdm(enumerate(zip(prod_list, pred_structs)))
            for idx_p, (prod, pred_struct) in enumerate(zip(prod_list, pred_structs)):
                if pred_struct is not None and len(pred_struct['reactants']):
                    predictions = pred_struct['reactants']
                else:
                    predictions = [prod]
                s = 0.0
                reactants = canonicalize_smiles(react[idx_p])
                for i in range(topk):
                    if i < len(predictions):
                        pred = predictions[i]
                        pred = canonicalize_smiles(pred)
                        predictions[i] = pred
                        cur_s = (pred == reactants)
                    else:
                        cur_s = s
                    s = max(cur_s, s)
                    topk_scores[i] += s
                cnt += 1
                if pred_struct is None or len(pred_struct['reactants']) == 0:
                    predictions = []
                fpred.write('{}>>{} {}\n'.format(react[idx_p], prod, len(predictions)))
                for i in range(len(predictions)):
                    fpred.write('{} {}\n'.format(pred_struct['template'][i], predictions[i]))
                msg = 'average score'
                for k in range(0, min(topk, 10), 3):
                    msg += ', t%d: %.4f' % (k + 1, topk_scores[k] / cnt)
                pbar.set_description(msg)
                pbar.update(1)
        h = '========%s results========' % len(data_or_loader.dataset)
        print(h)
        for k in range(topk):
            print('top %d: %.4f' % (k + 1, topk_scores[k] / cnt))
        print('=' * len(h))
        fpred.close()
        f_summary = '.'.join(fname_pred.split('.')[:-1]) + '.summary'
        with open(f_summary, 'w') as f:
            f.write('type overall\n')
            for k in range(topk):
                f.write('top %d: %.4f\n' % (k + 1, topk_scores[k] / cnt))
    elif isinstance(data_or_loader, dict):
        pbar = tqdm(data_or_loader['smi'])
        data = data_or_loader
        assert 'smi' in data_or_loader.keys()
        assert 'ground_truth' in data_or_loader.keys()
        fpred = open(fname_pred, 'w')
        for index, prod in enumerate(pbar):
            react = data['ground_truth'][index]
            pred_struct = model.run(prod, topk=topk, run_from_pred=False)
            if pred_struct is not None and len(pred_struct['reactants']):
                predictions = pred_struct['reactants']
            else:
                predictions = [prod]
            s = 0.0
            reactants = canonicalize_smiles(react)
            for i in range(topk):
                if i < len(predictions):
                    pred = predictions[i]
                    pred = canonicalize_smiles(pred)
                    predictions[i] = pred
                    cur_s = (pred == reactants)
                else:
                    cur_s = s
                s = max(cur_s, s)
                topk_scores[i] += s
            cnt += 1
            if pred_struct is None or len(pred_struct['reactants']) == 0:
                predictions = []
            fpred.write('{}>>{} {}\n'.format(react, prod, len(predictions)))
            for i in range(len(predictions)):
                fpred.write('{} {}\n'.format(pred_struct['template'][i], predictions[i]))
            msg = 'average score'
            for k in range(0, min(topk, 10), 3):
                msg += ', t%d: %.4f' % (k + 1, topk_scores[k] / cnt)
            pbar.set_description(msg)
        fpred.close()
        h = '========%s results========' % len(data['smi'])
        print(h)
        for k in range(topk):
            print('top %d: %.4f' % (k + 1, topk_scores[k] / cnt))
        print('=' * len(h))

        f_summary = '.'.join(fname_pred.split('.')[:-1]) + '.summary'
        with open(f_summary, 'w') as f:
            f.write('type overall\n')
            for k in range(topk):
                f.write('top %d: %.4f\n' % (k + 1, topk_scores[k] / cnt))
    else:
        raise ValueError


if __name__ == '__main__':
    debug = False
    benchmark_paroutes = False
    if benchmark_paroutes:
        benchmark_data_name = 'data_set-n5'


    init_lr, \
    batch_size, \
    fp_lindim, \
    graph_lindim, \
    use_gru, \
    massage_depth = 0.001, 1024, 512, 512, False, 3
    gpu = 0


    test_batch_size = 16

    model_name = 'DMPNNFP'

    if debug:
        # batch_size = 16
        batch_size = 8
        epochs = 5
        graph_lindim = 128
        fp_lindim = 128
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
    model_save_path = os.path.join('./model')

    # if debug:
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if not benchmark_paroutes:
        model_save_path = os.path.join('./model')   # org
        rxn_dataset = Dataset(root=os.path.join('./data'), dataset='USPTO-remapped_tpl_prod_react',
                            process2fp=process2fp, debug=debug)
    else:
        model_save_path = os.path.join('./model_benchmark')
        rxn_dataset = Dataset(root=os.path.join(f'./{benchmark_data_name}'), dataset='Paroute_tpl_prod_react',
                            process2fp=process2fp, debug=debug)

    template_rules, idx2rules = torch.load(
        os.path.join(rxn_dataset.raw_dir, 'templates_index.pkl'))

    test_dataset = rxn_dataset.test

    model_par_mark = f'init_lr_{str(init_lr)}_' + \
                     f'batch_size_{str(batch_size)}_' + \
                     f'fp_lindim_{str(fp_lindim)}_' + \
                     f'graph_lindim_{str(graph_lindim)}_' + \
                     f'use_gru_{str(use_gru)}_' + \
                     f'massage_depth_{str(massage_depth)}'
    if benchmark_paroutes:
        model_par_mark += f'_{benchmark_data_name}'
    mode_fname = f'saved_graph_rollout_state_1_{model_name}_' + model_par_mark + f'.ckpt'
    print('Loading', mode_fname)
    checkpoint, parameter_dim = torch.load(os.path.join(model_save_path, mode_fname))
    graph_lindim = parameter_dim['graph_lindim']
    fp_lindim = parameter_dim['fp_lindim']
    print(parameter_dim)
    # train_model_with_fp = True


    print('atom feature size:{}, bond feature size:{}, fingerprint size:{}'.format(test_dataset.data.x.shape[1],
                                                                                   test_dataset.data.edge_attr.shape[
                                                                                       1],
                                                                                   test_dataset.data.fp.shape[1]))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    if model_name == 'GCN':
        model = GCN(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim)
        model.name = 'GCN' + '_' + model_par_mark
    elif model_name == 'GCNFP':
        model = GCNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                      fp_dim=1024, fp_lindim=fp_lindim)
        model.name = 'GCNFP' + '_' + model_par_mark
    elif model_name == 'MPNNFP':
        model = MPNNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                       fp_dim=1024, fp_lindim=fp_lindim)
        model.name = 'MPNNFP' + '_' + model_par_mark
    elif model_name == 'DMPNN':
        model = DMPNN(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                      f_ab_size=graph_lindim + rxn_dataset.data.edge_attr.shape[1])
        model.name = 'DMPNN' + '_' + model_par_mark
    elif model_name == 'DMPNNFP':
        model = DMPNNFP(mol_in_dim=rxn_dataset.num_node_features, out_dim=len(template_rules), dim=graph_lindim,
                        b_in_dim=rxn_dataset.data.edge_attr.shape[1],
                        fp_lindim=fp_lindim, fp_dim=2048, use_gru=use_gru,
                        massage_depth=massage_depth)
        model.name = 'DMPNNFP' + '_' + model_par_mark
    else:
        raise ValueError

    model.load_state_dict(checkpoint)
    model = model.to(device)
    pred_model = GraphModel(model, idx2rules, device=device)

    # y = pred_model.run(test_dataset.data['smi'][3])
    # print(y)

    save_fname = f'./model/test_results_{model.name}.txt'
    print(f'Saving prediction results to {save_fname}')
    eval_model(test_loader, pred_model, 50,
               os.path.join(save_fname), device=device)
    # eval_model({'smi': test_dataset.data['smi'][4500:], 'ground_truth': test_dataset.data['ground_truth'][4500:]}, pred_model, 50,
    #            os.path.join('./model/test_results.txt'))
