import os

import pandas as pd

if __name__ == '__main__':

    use_corrected_template = False
    # dataset_all_folder = '../../../single_step_datasets/train_test_dataset/'
    dataset_split_folder = '../../../single_step_datasets/PaRoutes_set-n5/'

    save_data_lib_fname = 'templates.dat'
    save_template_lib_fname = 'template_rules_1.dat'
    train_save_template_lib_fname = 'train_template_rules_1.dat'
    val_save_template_lib_fname = 'val_template_rules_1.dat'
    train_val_template_lib_fname = 'train_val_template_rules_1.dat'

    if use_corrected_template:
        print('Using corrected template...')
        save_template_lib_fname = save_template_lib_fname.replace('.dat', '_corrected.dat')
        train_save_template_lib_fname = train_save_template_lib_fname.replace('.dat', '_corrected.dat')
        val_save_template_lib_fname = val_save_template_lib_fname.replace('.dat', '_corrected.dat')
        train_val_template_lib_fname = train_val_template_lib_fname.replace('.dat', '_corrected.dat')

    train_templates = pd.read_csv(os.path.join(dataset_split_folder, train_save_template_lib_fname),
                                  header=None).values.flatten().tolist()
    val_templates = pd.read_csv(os.path.join(dataset_split_folder, val_save_template_lib_fname),
                                header=None).values.flatten().tolist()

    train_val_templates = list(set(train_templates+val_templates))

    with open(os.path.join(dataset_split_folder, train_val_template_lib_fname), 'w') as f:
        f.write('\n'.join(train_val_templates))