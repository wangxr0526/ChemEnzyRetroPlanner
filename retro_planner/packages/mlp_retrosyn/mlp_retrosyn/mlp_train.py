import os
from collections import defaultdict
from tqdm import tqdm
from mlp_retrosyn.mlp_policies import train_mlp
from pprint import pprint

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="train function for retrosynthesis Planner policies")
    parser.add_argument('--template_path', default='../../single_step_datasets/train_test_dataset/train_val_templates.dat',
                        type=str, help='Specify the path of the template.train_all_dataset')
    parser.add_argument('--template_rule_path', default='../../single_step_datasets/train_test_dataset/train_val_template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_dump_folder', default='./model',
                        type=str, help='specify where to save the trained models')
    parser.add_argument('--fp_dim', default=2048, type=int,
                        help="specify the fingerprint feature dimension")
    parser.add_argument('--batch_size', default=1024, type=int,
                        help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="specify the learning rate")
    parser.add_argument('--use_corrected_template', action='store_true', default=True,
                        help='Specify whether to use corrected templates.')
    args = parser.parse_args()
    template_path = args.template_path
    template_rule_path = args.template_rule_path
    model_dump_folder = args.model_dump_folder

    train_template_rule_fname = 'train_templates.dat'
    val_template_rule_fname = 'val_templates.dat'
    save_model_fname = 'saved_rollout_state_1'

    if args.use_corrected_template:
        print('Using corrected template...')
        template_rule_path = template_rule_path.replace(
            '.dat', '_corrected.dat')
        template_path = template_path.replace(
            '.dat', '_corrected.dat')
        train_template_rule_fname = train_template_rule_fname.replace(
            '.dat', '_corrected.dat')
        val_template_rule_fname = val_template_rule_fname.replace(
            '.dat', '_corrected.dat')
        save_model_fname = save_model_fname.replace('_1', '_1_corrected')
    fp_dim = args.fp_dim
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    lr = args.learning_rate
    print('Loading train_all_dataset...')
    prod_to_rules = defaultdict(set)
    # read the template train_all_dataset.
    with open(template_path, 'r') as f:
        for l in tqdm(f, desc="reading the mapping from prod to rules"):
            rule, prod, react = l.strip().split(',')
            prod_to_rules[prod].add(rule)
    if not os.path.exists(model_dump_folder):
        os.mkdir(model_dump_folder)
    pprint(args)
    train_mlp(prod_to_rules,
              template_rule_path,
              train_path=os.path.join(
                  '../../single_step_datasets/train_test_dataset', train_template_rule_fname),
              test_path=os.path.join(
                  '../../single_step_datasets/train_test_dataset', val_template_rule_fname),
              fp_dim=fp_dim,
              batch_size=batch_size,
              lr=lr,
              dropout_rate=dropout_rate,
              saved_model=os.path.join(model_dump_folder, 'saved_rollout_state_1'))
