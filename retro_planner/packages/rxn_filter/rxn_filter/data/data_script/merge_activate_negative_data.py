import os.path

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random

from rxn_filter.utils import canonicalize_smiles

if __name__ == '__main__':
    from_aizynth_neg_data = pd.read_csv(
        os.path.join('../from_aizynthfinder_new_self/_template_library_false.csv'),
        header=None)
    from_aizynth_neg_data.columns = ["retro_template", "reactants", "products", "index"]
    print('negative data # :', len(from_aizynth_neg_data))

    filter_database = defaultdict(list)
    for _, retro_template, react, prod, index in tqdm(from_aizynth_neg_data.itertuples(),
                                                      total=len(from_aizynth_neg_data)):
        react = canonicalize_smiles(react, clear_map=True)
        prod = canonicalize_smiles(prod, clear_map=True)
        rxn_smiles = f'{react}>>{prod}'
        filter_database['rxn_smiles'].append(rxn_smiles)
        filter_database['labels'].append('0')
        filter_database['org_templates'].append(retro_template)

    positive_data = pd.read_csv(
        os.path.join('../../../../single_step_datasets/train_all_dataset/templates.dat'),
        header=None, sep='\t'
    )
    positive_data.columns = ['retro_template', 'products', 'reactants']
    for _, retro_template, prod, react in tqdm(positive_data.itertuples(), total=len(positive_data)):
        rxn_smiles = f'{react}>>{prod}'
        filter_database['rxn_smiles'].append(rxn_smiles)
        filter_database['labels'].append('1')
        filter_database['org_templates'].append(retro_template)

    print('positive data # :', len(positive_data))

    filter_database_df = pd.DataFrame(filter_database)
    # indices = [i for i in range(len(filter_database_df))]
    # random.shuffle(indices)
    # filter_database_df = filter_database_df.iloc[indices]
    filter_database_df.to_csv(os.path.join('../from_aizynthfinder_new_self/random_gen_aizynth_positive_false_data.csv'),
                              index=False)
