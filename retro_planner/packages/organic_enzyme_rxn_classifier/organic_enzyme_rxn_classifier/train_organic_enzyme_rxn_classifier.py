import json
import os
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class RXNClassificationDataset(Dataset):
    def __init__(self, rxnfps, labels, num_class) -> None:
        super(RXNClassificationDataset, self).__init__()
        self.rxnfps = rxnfps
        self.labels = labels
        self.fp_dim = rxnfps.size(1)
        self.num_class = num_class

    def __len__(self):
        return len(self.rxnfps)

    def __getitem__(self, index):
        return self.rxnfps[index], self.labels[index]

def load_dataset_with_random_shuffle(dataset_root):


    print('loading dataset csv file...')
    dataset_list = []
    for dataset_flag in ['train', 'val', 'test']:
        print(dataset_flag)
        dataset_df = pd.read_csv(os.path.join(dataset_root, f'{dataset_flag}.csv'))
        print('loading rxnfp...')
        dataset_df['rxnfp'] = np.load(os.path.join(dataset_root, f'{dataset_flag}.npz'))['fps'].tolist()
        rxnfps = torch.tensor(dataset_df['rxnfp'].tolist()).float()
        labels = torch.tensor(dataset_df['Label'].tolist()).long()
        dataset_cls = RXNClassificationDataset(rxnfps, labels, num_class=2)
        print(len(dataset_cls))
        dataset_list.append(dataset_cls)
    print('dataset loaded!')
    
    return dataset_list


    
class Classifier(nn.Module):
    def __init__(self, fp_dim=256, h_dim=1024, out_dim=128, dropout_rate=0.4):
        super(Classifier, self).__init__()
        self.fp_dim = fp_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.lin_i = nn.Linear(self.fp_dim, self.h_dim)
        self.lin_o = nn.Linear(self.h_dim, self.out_dim)

    def forward(self, rxn_fp):

        x_rxn = F.dropout(self.lin_i(rxn_fp), p=self.dropout_rate, training=self.training)
        x_rxn = torch.relu(x_rxn)
        out = self.lin_o(x_rxn)
        return out


def train_model(epoch, model, train_loader, loss_fn, optimizer, it, device):
    model.train()
    loss_all = 0
    losses = []
    for i, data in tqdm(enumerate(train_loader)):
        rxnfp, y = data

        # data = data.to(device)
        optimizer.zero_grad()

        rxnfp, y = rxnfp.to(device), y.to(device)
        loss = loss_fn(model(rxnfp), y)

        loss.backward()
        loss_all += loss.item() * y.shape[0]
        # optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
    return loss_all / len(train_loader.dataset)




def eval_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    loss = 0.0
    y_true_epochs = []
    y_score_epochs = []
    for data in tqdm(val_loader):
        rxnfp, y = data
        with torch.no_grad():
            rxnfp, y = rxnfp.to(device), y.to(device)
            y_hat = model(rxnfp)
            loss += loss_fn(y_hat, y).item()
            y_score = torch.softmax(y_hat, dim=1)
            y_score_epochs.append(y_score.cpu())
            y_true_epochs.append(y.cpu())
    y_true_epochs = torch.cat(y_true_epochs).numpy()
    y_score_epochs = torch.cat(y_score_epochs).numpy()
    y_pred_epochs = np.argmax(y_score_epochs, axis=1)

    # 计算额外的性能指标
    acc = (y_true_epochs == y_pred_epochs).sum() / len(y_pred_epochs)
    precision = precision_score(y_true_epochs, y_pred_epochs)
    recall = recall_score(y_true_epochs, y_pred_epochs)
    f1 = f1_score(y_true_epochs, y_pred_epochs)
    roc_auc = roc_auc_score(y_true_epochs, y_score_epochs[:, 1])  # 第二列是正类的概率

    loss = loss / (len(val_loader.dataset))

    return loss, acc, precision, recall, f1, roc_auc


def save_model(model_dir, config, model_state):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f)
    
    torch.save(model_state, os.path.join(model_dir, 'torch_model.pkl'))

def load_model(model_dir):
    model_state = torch.load(os.path.join(model_dir, 'torch_model.pkl'), map_location='cpu')
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return model_state, config


if __name__ == '__main__':
    (train_dataset, val_dataset, test_dataset) = load_dataset_with_random_shuffle(dataset_root='./dataset/preprocessed_data/unbalance_organic_enzyme_classification')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    train_config = {
        'fp_dim': train_dataset.fp_dim,
        'h_dim': 1024,
        'out_dim': train_dataset.num_class,
        'lr':0.00001,
        'batch_size':128,
        'epochs':100,
        'model_dir': './checkpoints/unbalance_organic_enzyme_classification'

    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_config['batch_size'])
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=train_config['batch_size'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=train_config['batch_size'])

    classifier = Classifier(fp_dim=train_config['fp_dim'], h_dim=train_config['h_dim'], out_dim=train_config['out_dim'])
    classifier = classifier.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=5,
        min_lr=0.000001)
    
    it = trange(train_config['epochs'])
    best_acc = -1
    for epoch in it:
        lr = scheduler.optimizer.param_groups[0]['lr']
        it.set_description('Epoch: {}, lr: {}'.format(epoch, lr))
        train_loss = train_model(
            epoch, classifier, train_dataloader, loss_fn, optimizer, it, device)
        # print(train_loss)
        val_loss, acc, precision, recall, f1, roc_auc= eval_one_epoch(
            classifier, val_dataloader, loss_fn, device)
        scheduler.step(val_loss)
        log_msg = (
            f"Epoch: {epoch}, "
            f"Validation Loss: {val_loss:.3f}, "
            f"Accuracy: {acc:.3f}, "
            f"Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, "
            f"F1 Score: {f1:.3f}, "
            f"ROC AUC: {roc_auc:.3f}"
        )
        print(log_msg)
        if acc > best_acc:
            best_acc = acc
            save_model(model_dir=train_config['model_dir'], config=train_config, model_state=classifier.state_dict())
