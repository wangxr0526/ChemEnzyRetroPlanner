import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from retro_planner.common.utils import binary_metrics
from rxn_filter.filter_models import FilterModel, FilterModelRXNfp
from rxn_filter.filter_dataset import FilterDataset


def train_model(epoch, model, train_loader, loss_fn, optimizer, it, device, writer, model_name):
    model.train()
    loss_all = 0
    losses = []
    for i, data in tqdm(enumerate(train_loader)):
        X, y = data
        X_fps_prod, X_fps_rxn = X

        # data = data.to(device)
        optimizer.zero_grad()
        if model_name == 'InScopeFilter':
            X_fps_prod, X_fps_rxn, y = X_fps_prod.to(
                device), X_fps_rxn.to(device), y.to(device)
            loss = loss_fn(model(X_fps_prod, X_fps_rxn), y)
        elif model_name in ['RXNFPFilter', 'DRFPFilter']:
            X_fps_rxn, y = X_fps_rxn.to(device), y.to(device)
            loss = loss_fn(model(X_fps_rxn), y)
        else:
            raise ValueError(
                'Model spport: InScopeFilter, RXNFPFilter, DRFPFilter')
        loss.backward()
        loss_all += loss.item() * y.shape[0]
        # optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
        if i % 1000 == 999:  # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              loss_all / 1000,
                              epoch * len(train_loader) + i)

            running_loss = 0.0
    return loss_all / len(train_loader.dataset)


def eval_one_epoch(model, val_loader, loss_fn, device, model_name, return_score=False, ):
    model.eval()
    loss = 0.0
    # eval_number = 0
    y_true_epochs = []
    y_score_epochs = []
    for data in tqdm(val_loader):
        X, y = data
        X_fps_prod, X_fps_rxn = X
        with torch.no_grad():
            if model_name == 'InScopeFilter':
                X_fps_prod, X_fps_rxn, y = X_fps_prod.to(
                    device), X_fps_rxn.to(device), y.to(device)
                y_hat = model(X_fps_prod, X_fps_rxn)
            elif model_name in ['RXNFPFilter', 'DRFPFilter']:
                X_fps_rxn, y = X_fps_rxn.to(device), y.to(device)
                y_hat = model(X_fps_rxn)
            else:
                raise ValueError(
                    'Model spport: InScopeFilter, RXNFPFilter, DRFPFilter')
            loss += loss_fn(y_hat, y).item()
            y_score = torch.sigmoid(y_hat)
            y_score_epochs.append(y_score.cpu())
            y_true_epochs.append(y.cpu())
    y_true_epochs = torch.cat(y_true_epochs).numpy()
    y_score_epochs = torch.cat(y_score_epochs).numpy()
    loss = loss / (len(val_loader.dataset))
    dic_metrics = binary_metrics(y_true_epochs, y_score_epochs, threshod=0.5)
    if return_score:
        return loss, dic_metrics, y_true_epochs, y_score_epochs
    else:
        return loss, dic_metrics


if __name__ == '__main__':

    debug = True
    config = yaml.load(open('data/filter_config.yaml', "r"),
                       Loader=yaml.FullLoader)
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

    # default `log_dir` is "runs" - we'll be more specific here

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    time_str = time.strftime('%Y-%m-%d_%Hh-%Mm-%Ss',
                             time.localtime(time.time()))
    writer = SummaryWriter(f'runs/{time_str}')
    # model_state_fname = 'filter'

    if debug:
        raw_data_fname = 'random_gen_false_rxn.csv'
        split_index_fname = None
        batch_size = 128
        epochs = 1000
        config['raw_data_fname'] = raw_data_fname
        config['batch_size'] = batch_size
        config['epochs'] = epochs
        config['split_index_fname'] = split_index_fname
    data_name = raw_data_fname.split('/')[-1].split('.')[0]
    model_state_fname = f'filter_train_data_{data_name}_{fp_function_name}_{time_str}.pkl'

    filter_dataset = FilterDataset(
        data_root=config['data_root'],
        raw_data_fname=raw_data_fname,
        split_index_fname=split_index_fname,
        fp_function_name=fp_function_name
    )

    train_set, val_set, test_set = filter_dataset.train, filter_dataset.val, filter_dataset.test
    print(
        f'\nGet train data # : {len(train_set)}\nGet val data # : {len(val_set)}\nGet test data # : {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model_config = config['model_config'][model_name]
    if model_name == 'InScopeFilter':
        model = FilterModel(
            fp_dim=model_config['fp_dim'],
            dim=model_config['h_dim'],
            dropout_rate=model_config['dropout_rate']).to(device)
    elif model_name == 'RXNFPFilter':
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

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=5,
        min_lr=0.000001)

    it = trange(epochs)
    best = -1
    # runing_loss = 0.0
    for epoch in it:
        lr = scheduler.optimizer.param_groups[0]['lr']
        it.set_description('Epoch: {}, lr: {}'.format(epoch, lr))
        train_loss = train_model(
            epoch, model, train_loader, loss_fn, optimizer, it, device, writer, model_name=model_name)
        # print(train_loss)
        val_loss, dic_metrics = eval_one_epoch(
            model, val_loader, loss_fn, device, model_name=model_name)
        scheduler.step(val_loss)
        if best < dic_metrics['acc']:
            best = dic_metrics['acc']
            state = model.state_dict()
            model_state_path = os.path.join(model_save_path, model_state_fname)
            config['model_state_fname'] = model_state_fname
            torch.save(state,
                       model_state_path)
            with open(os.path.join(model_save_path, model_state_fname.replace('.pkl', '.yaml')), 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
        print(
            f"\nEpoch : {epoch}  validation loss ==> {val_loss}, auc ==> {dic_metrics['auc']}, accuracy ==> {dic_metrics['acc']}, precision ==> {dic_metrics['precision']},  recall ==> {dic_metrics['recall']}")
        for key in dic_metrics.keys():
            writer.add_scalar(key, dic_metrics[key], epoch)

    writer.close()
