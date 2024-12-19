import os
import numpy as np
import torch
import random
import pickle
import torch.nn.functional as F
import logging

from torch.utils import tensorboard
from retro_planner.common import args
from value_function.value_mlp import ValueMLP
from value_function.value_data_loader import ValueDataLoader
from value_function.trainer import Trainer
from retro_planner.utils import setup_logger
from torch.utils.tensorboard import SummaryWriter
import time

def train(debug):
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')
    print('device: ', device)

    model = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device
    )

    if debug:
        args.value_train = 'value_data_dic_sample_train_convert.pkl'
        args.value_val = 'value_data_dic_sample_val_convert.pkl'
        

    time_str = time.strftime('%Y-%m-%d_%Hh-%Mm-%Ss',
                             time.localtime(time.time()))
    writer = SummaryWriter(f'runs/{time_str}')
    assert os.path.exists('%s/%s' % (args.value_root, args.value_train))

    train_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_train),
        batch_size=args.batch_size
    )

    val_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_val),
        batch_size=args.batch_size
    )

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        n_epochs=args.value_fn_n_epochs,
        lr=args.lr,
        save_epoch_int=args.value_fn_save_epoch_int,
        model_folder=args.value_fn_save_folder,
        device=device,
        tensorboard_witer = writer
    )

    trainer.train()


if __name__ == '__main__':
    debug = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('train.log')

    train(debug)
