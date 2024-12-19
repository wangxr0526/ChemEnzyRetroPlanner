import csv
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rxn_filter.utils import RunRdRxn

MAX_TRY = 100
SEED = 1024


def run_task(task):
    select_template_array_save_folder = os.path.join('data/catch')
    index, org_tpl, rxn, template_cnt = task
    template_indices = [x for x in range(template_cnt)]
    random.shuffle(template_indices)
    select_indices = template_indices[:MAX_TRY]
    positive_data = [[rxn, '1', org_tpl]]
    select_template_array = torch.load(os.path.join(select_template_array_save_folder, 'catch_array.pkl'))[
        select_indices]
    negative_data = []
    for tpl in select_template_array:

        run_rxn = RunRdRxn(rxn, tpl)
        results = run_rxn.get_results()
        if results:
            results = list(results)
            for i, gen_rxn in enumerate(results):
                negative_data.append([gen_rxn, '0', tpl])
        else:
            continue
    return positive_data, negative_data


# extracted_pathways = Parallel(n_jobs=-1, verbose=1)(delayed(extract_one_patent)(data[key], key) for key in data.keys())
def random_gen_neg(new_datadf, org_template_df, save_path):
    import random
    random.seed(SEED)
    # gen_train_data = defaultdict(list)
    cnt_positive, cnt_negative = 0, 0
    fout = open(os.path.join(save_path), 'w')
    writer = csv.writer(fout)
    writer.writerow(['rxn_smiles', 'labels', 'org_templates'])
    template_cnt = len(org_template_df)
    for row in tqdm(new_datadf.itertuples(), total=len(new_datadf)):
        _, org_tpl, rxn = row
        writer.writerow([rxn, '1', org_tpl])
        cnt_positive += 1
        # gen_train_data['rxn_smiles'] += [rxn]
        # gen_train_data['labels'] += [1]
        # gen_train_data['org_templates'] += [org_tpl]
        template_indices = [x for x in range(template_cnt)]
        random.shuffle(template_indices)
        select_indices = template_indices[:MAX_TRY]
        template_df = org_template_df.iloc[select_indices]
        for _, tpl in template_df.itertuples():

            run_rxn = RunRdRxn(rxn, tpl)
            results = run_rxn.get_results()
            if results:
                results = list(results)
                # labels = [0 for _ in range(len(results))]
                # org_templates = [tpl for _ in range(len(results))]
                for i, gen_rxn in enumerate(results):
                    writer.writerow([gen_rxn, '0', tpl])
                    cnt_negative += 1
                # gen_train_data['rxn_smiles'] += results
                # gen_train_data['labels'] += [0 for _ in range(len(results))]
                # gen_train_data['org_templates'] += [tpl for _ in range(len(results))]
            else:
                continue
    # gen_train_datadf = pd.DataFrame(gen_train_data)
    # gen_train_datadf.to_csv(save_path)
    fout.close()
    return cnt_positive, cnt_negative


def get_select_template_array(save_path):
    template_indices = [x for x in range(template_cnt)]
    random.shuffle(template_indices)
    select_indices = template_indices[:MAX_TRY]
    select_template_array = org_template_array[select_indices]
    torch.save(select_template_array, save_path)
    return select_template_array


if __name__ == '__main__':
    debug = True

    multi_core = 0

    gen_neg_method = 'random'

    save_path = os.path.join(f'./data/{gen_neg_method}_gen_false_rxn.csv')
    all_dataset = pd.read_csv(os.path.join('../../single_step_datasets/train_all_dataset/templates.dat'), header=None,
                              sep='\t')
    all_dataset.columns = ['templates', 'prod_smiles', 'react_smiles']
    if debug:
        all_dataset = all_dataset.iloc[:500]

    assert gen_neg_method in ['random']

    print('Merge rxn smiles.')
    # rxn_smiles = []
    # new_datadf = pd.DataFrame()
    new_data = defaultdict(list)

    for row in tqdm(all_dataset.itertuples()):
        idx, tpl, prod, react = row
        new_data['templates'].append(tpl)
        new_data['rxn_smiles'].append(f'{react}>>{prod}')

    new_datadf = pd.DataFrame(new_data)
    org_template_df = pd.DataFrame({'templates': list(set(new_datadf['templates'].tolist()))})
    # template_df.drop_duplicates(subset=None, keep='first', inplace=False)
    if gen_neg_method == 'random':
        if not multi_core:
            cnt_positive, cnt_negative = random_gen_neg(new_datadf, org_template_df,
                                                        save_path)
            # cnt_positive = len(gen_train_datadf.loc[gen_train_datadf['labels'] == 1])
            # cnt_negative = len(gen_train_datadf.loc[gen_train_datadf['labels'] == 0])

            # gen_train_datadf.to_csv()
        else:
            import random

            pool = Pool(multi_core)
            random.seed(SEED)
            cnt_positive, cnt_negative = 0, 0
            fout = open(os.path.join(save_path), 'w')
            writer = csv.writer(fout)
            writer.writerow(['rxn_smiles', 'labels', 'org_templates'])
            org_template_array = np.asarray(org_template_df['templates'].tolist())
            template_cnt = len(org_template_array)
            select_template_array_save_folder = os.path.join('data/catch')
            if not os.path.exists(select_template_array_save_folder):
                os.mkdir(select_template_array_save_folder)
            print('Saving Catch Files!')
            # [
            #     get_select_template_array(os.path.join(select_template_array_save_folder, '{}.pkl'.format(i))) for i in
            #     tqdm(range(len(new_datadf)))]

            with open(os.path.join(select_template_array_save_folder, 'catch_array.pkl'), "wb") as writer:
                pickle.dump(org_template_array, writer, protocol=4)
            # torch.save(org_template_array, os.path.join(select_template_array_save_folder, 'catch_array.pkl'))

            tasks = [list(row) + [template_cnt] for i, row in
                     tqdm(enumerate(new_datadf.itertuples()))]
            cnt_positive, cnt_negative = 0, 0
            for results in tqdm(pool.imap_unordered(run_task, tasks), total=len(tasks)):
                positive_data, negative_data = results
                for p_data in positive_data:
                    writer.writerow(p_data)
                    cnt_positive += 1
                for n_data in negative_data:
                    writer.writerow(n_data)
                    cnt_negative += 1
            fout.close()
        print(f'Positive data #: {cnt_positive}\nNegative data #: {cnt_negative}')
