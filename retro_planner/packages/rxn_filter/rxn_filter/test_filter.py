import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from rxn_filter.filter_models import FilterModel, FilterModelRXNfp
from rxn_filter.filter_dataset import FilterDataset
from rxn_filter.train_filter import eval_one_epoch
import yaml

if __name__ == '__main__':
    config = yaml.load(open(
        'model/filter_train_data_random_gen_aizynth_filter_dataset_drfp_2022-04-21_17h-00m-34s.yaml', "r"), Loader=yaml.FullLoader)
    print('\nconfig:')
    print(config)
    raw_data_fname = config['raw_data_fname']
    split_index_fname = config['split_index_fname']
    batch_size = config['batch_size']
    epochs = config['epochs']
    model_save_path = config['model_save_path']
    model_name = config['filter_model_name']
    fp_function_name = config['fp_function_name']
    device = torch.device('cuda:{}'.format(
        config['gpu']) if torch.cuda.is_available() else 'cpu')

    filter_dataset = FilterDataset(
        data_root=config['data_root'],
        raw_data_fname=raw_data_fname,
        split_index_fname=split_index_fname,
        fp_function_name=fp_function_name
    )

    test_set = filter_dataset.test

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_config = config['model_config'][model_name]
    if model_name == 'InScopeFilter':
        model = FilterModel(
            fp_dim=model_config['fp_dim'],
            dim=model_config['h_dim'],
            dropout_rate=model_config['dropout_rate']).to(device)
    elif config['filter_model_name'] == 'RXNFPFilter':
        model = FilterModelRXNfp(
            fp_dim=model_config['fp_dim'],
            dim=model_config['h_dim'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
    elif model_name == 'DRFPFilter':
        model = FilterModelRXNfp(
            fp_dim=model_config['fp_dim'],
            dim=model_config['h_dim'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
    else:
        raise ValueError(
            'Model spport: InScopeFilter, RXNFPFilter, DRFPFilter')
    # writer.add_graph(model, images)
    model_state_fname = config['model_state_fname']
    model.load_state_dict(torch.load(os.path.join(
        config['model_save_path'], model_state_fname), map_location=device))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    test_loss, dic_metrics, y_true_epoch, y_score_epoch = eval_one_epoch(
        model, test_loader, loss_fn, device, model_name=model_name,                                                                    return_score=True)
    torch.save({'y_true_epoch': y_true_epoch, 'y_score_epoch': y_score_epoch},
               f'./data/predict_score_dic_{model_state_fname}')
    print(f'Test Loss : {test_loss}')
    print(dic_metrics)
