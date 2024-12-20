"""
Modified version of:
<https://github.com/connorcoley/ochem_predict_nn/blob/master/data/generate_reaction_templates.py>
"""

import os
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
import pandas as pd
from retro_planner.common.utils import canonicalize_smiles

if __name__ == '__main__':
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Convert dataset formate to train")
    parser.add_argument('--raw_data_folder', default=os.path.join(
        '../../../../../../handle_multi_step_dataset/USPTO_remapped'),
        type=str)
    parser.add_argument('--rxn_template_fname', default='USPTO_remapped_remove_same_rxn_templates.csv',
                        type=str)
    parser.add_argument('--cooked_data_folder', default=os.path.join('../../../single_step_datasets/train_all_dataset'),
                        type=str)
    parser.add_argument('--use_corrected_template', action='store_true', default=True,
                        help='Specify whether to use corrected templates.')
    args = parser.parse_args()
    raw_data_folder = args.raw_data_folder
    rxn_template_fname = args.rxn_template_fname
    cooked_data_folder = args.cooked_data_folder
    use_corrected_template = args.use_corrected_template
    if use_corrected_template:
        rxn_template_fname = rxn_template_fname.split(
            '.')[0]+'_with_r_sorted_correct.csv'
        template_col_name = 'corrected_retro_template'

        save_data_lib_fname = 'templates_corrected.dat'
        save_template_lib_fname = 'template_rules_1_corrected.dat'
    else:
        template_col_name = 'retro_template'
        save_data_lib_fname = 'templates.dat'
        save_template_lib_fname = 'template_rules_1.dat'
    templates = defaultdict(tuple)
    transforms = []
    datafile = os.path.join(raw_data_folder, rxn_template_fname)
    df = pd.read_csv(datafile)
    rxn_smiles = df['droped_unmapped_rxn'].tolist()
    retro_templates = df[template_col_name].tolist()
    for i in tqdm(range(len(df))):
        rxn = rxn_smiles[i]
        rule = retro_templates[i]
        if not use_corrected_template:
            rule_split = rule.split('>>')
            rule = '(' + rule_split[0] + ')>>' + rule_split[1]
        product = canonicalize_smiles(
            rxn.strip().split('>')[-1], clear_map=True)
        reactant = canonicalize_smiles(
            rxn.strip().split('>')[0], clear_map=True)
        transforms.append((rule, product, reactant))
    print(len(transforms))
    with open(os.path.join(cooked_data_folder, save_data_lib_fname), 'w') as f:
        f.write('\n'.join(['\t'.join(rxn_prod) for rxn_prod in transforms]))

    # Generate rules for MCTS
    templates = defaultdict(int)
    for rule, _, _ in tqdm(transforms):
        templates[rule] += 1
    print("The number of templates is {}".format(len(templates)))
    # #
    template_rules = [rule for rule, cnt in templates.items() if cnt >= 1]
    print("all template rules with count >= 1: ", len(template_rules))
    with open(os.path.join(cooked_data_folder, save_template_lib_fname), 'w') as f:
        f.write('\n'.join(template_rules))
