import os.path
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def split_data_df(data, val_frac=0.1, test_frac=0.1, seed=1024):
    shuffle_indeces = [i for i in range(len(data))]
    random.seed(seed)
    random.shuffle(shuffle_indeces)
    data = data.iloc[shuffle_indeces].reset_index(drop=True)
    classes = sorted(np.unique(data['labels']))

    train_indeces = []
    val_indeces = []
    test_indeces = []
    for class_ in classes:
        class_indeces = data.loc[data['labels'] == class_].index
        N = len(class_indeces)
        print('{} rows with class value {}'.format(N, class_))

        # shuffle_func(indeces)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)
        train_indeces += class_indeces[:train_end].tolist()
        val_indeces += class_indeces[train_end:val_end].tolist()
        test_indeces += class_indeces[val_end:].tolist()
        # for i in tqdm(class_indeces[:train_end]):
        #     data.loc[i, 'dataset'] = 'train'
        # for i in tqdm(class_indeces[train_end:val_end]):
        #     data.loc[i, 'dataset'] = 'val'
        # for i in tqdm(class_indeces[val_end:]):
        #     data.loc[i, 'dataset'] = 'test'
    # print(data['dataset'].value_counts())
    random.shuffle(train_indeces)
    random.shuffle(val_indeces)
    random.shuffle(test_indeces)
    return data.iloc[train_indeces].reset_index(drop=True), \
           data.iloc[val_indeces].reset_index(drop=True),\
           data.iloc[test_indeces].reset_index(drop=True)


if __name__ == '__main__':

    raw_dataset_folder = os.path.join('../')
    save_fname = 'random_gen_aizynth_filter_dataset.csv'

    org_dataset = pd.read_csv(os.path.join('../from_aizynthfinder_new_self/random_gen_aizynth_positive_false_data.csv'))
    train_set, val_set, test_set = split_data_df(org_dataset)
    # train_set = org_dataset.iloc[org_dataset['dataset'] == 'train']
    # val_set = org_dataset.iloc[org_dataset['dataset'] == 'val']
    # test_set = org_dataset.iloc[org_dataset['dataset'] == 'test']
    catch_folder = os.path.join('../from_aizynthfinder_new_self/split_catch')
    if not os.path.exists(catch_folder):
        os.mkdir(catch_folder)
    train_set.to_csv(os.path.join(catch_folder, 'random_gen_aizynth_positive_false_data_train.csv'), index=False)
    val_set.to_csv(os.path.join(catch_folder, 'random_gen_aizynth_positive_false_data_val.csv'), index=False)
    test_set.to_csv(os.path.join(catch_folder, 'random_gen_aizynth_positive_false_data_test.csv'), index=False)

    df_frames = []
    dataset_begin_end_index = []
    begin = 0
    end = 0
    # new_df = pd.DataFrame()
    for flag in ['train', 'val', 'test']:
        set_df = pd.read_csv(os.path.join(catch_folder, 'random_gen_aizynth_positive_false_data_{}.csv'.format(flag)))
        df_frames.append(set_df)
        end += len(set_df)
        dataset_begin_end_index.append((begin, end))
        begin = end

    torch.save(dataset_begin_end_index,
               os.path.join(raw_dataset_folder, '{}_split_index.pkl').format(save_fname.split('.')[0]))
    new_df = pd.concat(df_frames)
    new_df.to_csv(os.path.join(raw_dataset_folder, save_fname))
