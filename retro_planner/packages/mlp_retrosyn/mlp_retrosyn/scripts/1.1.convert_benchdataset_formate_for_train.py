"""
Modified version of:
<https://github.com/connorcoley/ochem_predict_nn/blob/master/data/generate_reaction_templates.py>
"""


'''
Convert PaRoutes' single step dataset format.    PaRoutes: https://github.com/MolecularAI/PaRoutes
'''
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
        '../../../../../../PaRoutes/data/'),
        type=str)
    parser.add_argument('--rxn_template_fname', default='uspto_rxn_n5_raw_template_library.csv',
                        type=str)
    parser.add_argument('--template_fname', default='uspto_rxn_n5_unique_templates.csv',
                        type=str)
    parser.add_argument('--cooked_data_folder', default=os.path.join('../../../single_step_datasets/PaRoutes_set-n5'),
                        type=str)
    args = parser.parse_args()
    raw_data_folder = args.raw_data_folder
    rxn_template_fname = args.rxn_template_fname
    template_fname = args.template_fname
    cooked_data_folder = args.cooked_data_folder
    
    if not os.path.exists(cooked_data_folder):
        os.makedirs(cooked_data_folder)

    template_col_name = 'retro_template'
    save_data_lib_fname = 'templates.dat'
    save_template_lib_fname = 'template_rules_1.dat'

    transforms = []
    datafile = os.path.join(raw_data_folder, rxn_template_fname)
    df = pd.read_csv(datafile)
    products_list = df['products'].tolist()
    reactants_list = df['reactants'].tolist()
    retro_templates = df[template_col_name].tolist()
    templates_lib = pd.read_csv(os.path.join(raw_data_folder, template_fname))
    template_rules = templates_lib['retro_template'].tolist()
    print("The number of templates is {}".format(len(template_rules)))
    # #
    print(f"all template rules with count >= {templates_lib['library_occurrence'].min()}: ", len(template_rules))
    with open(os.path.join(cooked_data_folder, save_template_lib_fname), 'w') as f:
        f.write('\n'.join(template_rules))

    for i in tqdm(range(len(df))):
        rule = retro_templates[i]
        if rule not in template_rules: continue
        product = canonicalize_smiles(
            products_list[i], clear_map=True)
        reactant = canonicalize_smiles(
            reactants_list[i], clear_map=True)
        transforms.append((rule, product, reactant))
    print(len(transforms))
    with open(os.path.join(cooked_data_folder, save_data_lib_fname), 'w') as f:
        f.write('\n'.join(['\t'.join(rxn_prod) for rxn_prod in transforms]))



