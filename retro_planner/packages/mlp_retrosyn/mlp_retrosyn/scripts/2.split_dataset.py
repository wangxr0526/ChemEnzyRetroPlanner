import os
from collections import defaultdict

import pandas as pd
import random

from tqdm import tqdm

random.seed(3060)


def get_template_lib(df, save_folder, flag, use_corrected_template):
    templates_list = df[0].tolist()
    templates = defaultdict(int)
    for rule in tqdm(templates_list):
        templates[rule] += 1
    print("The number of templates is {}".format(len(templates)))
    # #
    template_rules = [rule for rule, cnt in templates.items() if cnt >= 1]
    # print("all template rules with count >= 1: ", len(template_rules))

    save_template_lib_fname = '{}_template_rules_1.dat'.format(flag)
    if use_corrected_template:
        save_template_lib_fname = save_template_lib_fname.replace(
            '.dat', '_corrected.dat')

    with open(os.path.join(save_folder, save_template_lib_fname), 'w') as f:
        f.write('\n'.join(template_rules))
    return template_rules


if __name__ == '__main__':

    use_corrected_template = False

    # cooked_data_folder = '../../../single_step_datasets/train_all_dataset/'   # USPTO_remapped
    # save_path = '../../../single_step_datasets/train_test_dataset'
    cooked_data_folder = '../../../single_step_datasets/PaRoutes_set-n5/'     # PaRoutes set-n1
    save_path = cooked_data_folder

    '''
    PaRoutes set-n1 & set-n5 : train:val:test = 9:0.5:0.5   0.1~1  0.05~0.1  0~0.05
    USPTO_remapped : train:val:test = 8:1:1  0.2~1  0.1~0.2  0~0.1
    '''
    # split_num_list = [0.1, 0.2]
    split_num_list = [0.05, 0.1]
    assert split_num_list[1] > split_num_list[0]

    save_data_lib_fname = 'templates.dat'
    save_template_lib_fname = 'template_rules_1.dat'
    train_save_fname = 'train_templates.dat'
    val_save_fname = 'val_templates.dat'
    test_save_fname = 'test_templates.dat'
    if use_corrected_template:
        print('Using corrected template...')
        save_data_lib_fname = save_data_lib_fname.replace(
            '.dat', '_corrected.dat')
        save_template_lib_fname = save_template_lib_fname.replace(
            '.dat', '_corrected.dat')
        train_save_fname = train_save_fname.replace('.dat', '_corrected.dat')
        val_save_fname = val_save_fname.replace('.dat', '_corrected.dat')
        test_save_fname = test_save_fname.replace('.dat', '_corrected.dat')

    prod_templates = pd.read_csv(os.path.join(
        cooked_data_folder, save_data_lib_fname), sep='\t', header=None)
    rows, cols = prod_templates.shape
    database_index = [x for x in range(rows)]
    random.shuffle(database_index)
    prod_templates = prod_templates.iloc[database_index]


    split_index_1 = int(rows * split_num_list[0])
    split_index_2 = int(rows * split_num_list[1])

    data_test = prod_templates.iloc[0: split_index_1, :]
    data_val = prod_templates.iloc[split_index_1:split_index_2, :]
    data_train = prod_templates.iloc[split_index_2: rows, :]

    print('data_test:', len(data_test))
    print('data_validate:', len(data_val))
    print('data_train:', len(data_train))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data_test.to_csv(os.path.join(save_path, test_save_fname),
                     index=False, header=None)
    data_val.to_csv(os.path.join(save_path, val_save_fname),
                    index=False, header=None)
    data_train.to_csv(os.path.join(
        save_path, train_save_fname), index=False, header=None)

    test_rules = get_template_lib(
        data_test, save_path, 'test', use_corrected_template=use_corrected_template)
    val_rules = get_template_lib(
        data_val, save_path, 'val', use_corrected_template=use_corrected_template)
    train_rules = get_template_lib(
        data_train, save_path, 'train', use_corrected_template=use_corrected_template)

    test_in_train_number = 0
    for rule in tqdm(test_rules):
        if rule in train_rules:
            test_in_train_number += 1

    print('test set {:.2f}% rule in training set.'.format(
        100 * test_in_train_number / len(test_rules)))
    # test set 50.33% rule in training set.
    # using corrected template test set 48.13% rule in training set.
    val_in_train_number = 0
    for rule in tqdm(val_rules):
        if rule in train_rules:
            val_in_train_number += 1

    print('val set {:.2f}% rule in training set.'.format(
        100 * val_in_train_number / len(test_rules)))
    # val set 50.54% rule in training set.
    # using corrected template val set 48.27% rule in training set.
