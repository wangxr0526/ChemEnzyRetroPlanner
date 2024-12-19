import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from train_ecreact_classifier import Classifier, load_dataset_with_random_shuffle, load_model

def scoring_test(model, test_loader, loss_fn, device, topk=(1,)):
    model.eval()
    loss = 0.0
    correct_topk = {k: 0 for k in topk}
    total = 0

    for data in tqdm(test_loader):
        rxnfp, y = data
        with torch.no_grad():
            rxnfp, y = rxnfp.to(device), y.to(device)
            y_hat = model(rxnfp)
            loss += loss_fn(y_hat, y).item()
            
            # 计算top-k准确率
            maxk = max(topk)
            y_score = torch.softmax(y_hat, dim=1)
            _, pred = y_score.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                correct_topk[k] += correct_k.item()
            total += y.size(0)

    loss /= len(test_loader.dataset)
    topk_acc = {k: correct_topk[k] / total for k in topk}

    return loss, topk_acc

if __name__ == '__main__':
    
    topk=[1, 3, 5, 10, 30, 50]
    
    (_, _, test_dataset) = load_dataset_with_random_shuffle(dataset_root='./dataset/preprocessed_data/ecreact_classification')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_state, config = load_model('./checkpoints/ecreact_classification')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'])
    classifier = Classifier(fp_dim=config['fp_dim'], h_dim=config['h_dim'], out_dim=config['out_dim'])
    classifier.load_state_dict(model_state)
    classifier = classifier.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss, topk_acc = scoring_test(
        classifier, test_dataloader, loss_fn, device, topk=topk)
    
    topk_msg = '\n'.join([f'Top-{k} Accuracy\t{topk_acc[k]}' for k in topk])

    log_msg = (
        "Scores\tValues\n"
        f"Test Loss\t{test_loss:.3f}\n"
         + topk_msg
    )
    print(log_msg)
    
    with open('./checkpoints/ecreact_classification/test_scores.csv', 'w') as f:
        f.write(log_msg)
