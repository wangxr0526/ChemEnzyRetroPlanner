import torch
from torch.utils.data import DataLoader
from train_organic_enzyme_rxn_classifier import Classifier, eval_one_epoch, load_dataset_with_random_shuffle, load_model



if __name__ == '__main__':
    (_, _, test_dataset) = load_dataset_with_random_shuffle(dataset_root='./dataset/preprocessed_data/unbalance_organic_enzyme_classification')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_state, config = load_model('./checkpoints/unbalance_organic_enzyme_classification')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'])
    classifier = Classifier(fp_dim=config['fp_dim'], h_dim=config['h_dim'], out_dim=config['out_dim'])
    classifier.load_state_dict(model_state)
    classifier = classifier.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss, acc, precision, recall, f1, roc_auc= eval_one_epoch(
        classifier, test_dataloader, loss_fn, device)

    log_msg = (
        "Scores\tValues\n"
        f"Test Loss\t{test_loss:.3f}\n"
        f"Accuracy\t{acc:.3f}\n"
        f"Precision\t{precision:.3f}\n"
        f"Recall\t{recall:.3f}\n"
        f"F1 Score\t{f1:.3f}\n"
        f"ROC AUC\t{roc_auc:.3f}\n"
    )
    print(log_msg)
    
    with open('./checkpoints/unbalance_organic_enzyme_classification/test_scores.csv', 'w') as f:
        f.write(log_msg)
