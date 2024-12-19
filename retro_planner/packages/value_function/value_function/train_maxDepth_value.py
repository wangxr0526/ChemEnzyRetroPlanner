import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from value_function.value_mlp import MaxDepthValueMLP
from retro_planner.common.utils import binary_metrics, multiclass_metrics
from retro_planner.common import args
from retro_planner.utils import setup_logger
from value_function.value_data_loader import MaxDepthValueDataset, MaxDepthValueDataLoader, MaxDepthValueDatasetEmpty


def train_model(epoch, model, train_loader, loss_fn, optimizer, it, device, writer):
    model.train()
    loss_all = 0
    losses = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        X, y = data
        X, y = X.to(
            device), y.to(device).view(-1)
        # data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        loss_all += loss.item() * y.shape[0]
        optimizer.step()
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


def eval_one_epoch(model, val_loader, loss_fn, device, return_score=False):
    model.eval()
    loss = 0.0
    # eval_number = 0
    y_true_epochs = []
    y_pred_epochs = []
    for data in tqdm(val_loader):
        X, y = data
        X, y = X.to(
            device), y.to(device).view(-1)
        with torch.no_grad():
            y_hat = model(X)
            loss += loss_fn(y_hat, y).item()
            y_pred = y_hat.argmax(dim=1)
            y_pred_epochs.append(y_pred.cpu())
            y_true_epochs.append(y.cpu())
    y_true_epochs = torch.cat(y_true_epochs).numpy()
    y_pred_epochs = torch.cat(y_pred_epochs).numpy()
    loss = loss / (len(val_loader.dataset))
    dic_metrics = multiclass_metrics(
        y_true_epochs, y_pred_epochs)
    if return_score:
        return loss, dic_metrics, y_true_epochs, y_pred_epochs
    else:
        return loss, dic_metrics


if __name__ == '__main__':

    # train_all=True 不分割数据集全部训练，过拟合这部分数据，让value function记住这部分分子的路径长度
    debug = False
    train_all = False
    is_unique = True
    batch_size = 1024
    epochs = 100

    time_str = time.strftime('%Y-%m-%d_%Hh-%Mm-%Ss',
                             time.localtime(time.time()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_save_path = os.path.join('./depth_model_save_folder')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    writer = SummaryWriter(f'runs/{time_str}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if is_unique:
        unique_flag = '_unique'
        maxDepth_value_data_fname = os.path.join(
        './merge_data_path', 'unique_maxDepth_data_dic.pkl')
    else:
        unique_flag = ''
        maxDepth_value_data_fname = os.path.join(
        './merge_data_path', 'maxDepth_data_dic.pkl')

   
    if debug:
        maxDepth_value_data_fname = os.path.join(
            './merge_data_path', 'maxDepth_data_dic_debug.pkl')

    data_name = maxDepth_value_data_fname.split('/')[-1].split('.')[0]
    if not train_all:
        train_all_flag = '_train_val'
    else:
        train_all_flag = ''

    model_state_fname = f'filter_train_data_{data_name}{unique_flag}{train_all_flag}_{time_str}.pkl'

    dataset = MaxDepthValueDataset(
        fp_value_f=maxDepth_value_data_fname, depth_filter=1)
    ##### !!!!!!!!!!!!预测深度的时候一定要加这个depth_filter!!!!!!!!!!!!
    
    if train_all:
        train_loader = MaxDepthValueDataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True)
        val_loader = MaxDepthValueDataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        split_point = int(0.9*len(dataset))
        train_dataset, val_dataset = dataset[:
                                             split_point], dataset[split_point:]
        train_dataset = MaxDepthValueDatasetEmpty(*train_dataset)
        val_dataset = MaxDepthValueDatasetEmpty(*val_dataset)
        train_loader = MaxDepthValueDataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = MaxDepthValueDataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True)

    model = MaxDepthValueMLP(n_layers=1, fp_dim=2048, latent_dim=256, output_dim=len(
        dataset.all_depth), dropout_rate=0.4)
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.000001)

    it = trange(epochs)
    best = -1
    # runing_loss = 0.0
    for epoch in it:
        lr = scheduler.optimizer.param_groups[0]['lr']
        it.set_description('Epoch: {}, lr: {}'.format(epoch, lr))
        train_loss = train_model(
            epoch, model, train_loader, loss_fn, optimizer, it, device, writer)
        # print(train_loss)
        val_loss, dic_metrics = eval_one_epoch(
            model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        if best < dic_metrics['acc']:
            best = dic_metrics['acc']
            state = model.state_dict()
            torch.save(state,
                       os.path.join(model_save_path, model_state_fname))
        print(
            f"\nEpoch : {epoch}  validation loss ==> {val_loss}, accuracy ==> {dic_metrics['acc']}, precision ==> {dic_metrics['precision']},  recall ==> {dic_metrics['recall']}")
        for key in dic_metrics.keys():
            writer.add_scalar(key, dic_metrics[key], epoch)

    writer.close()

    # TODO
