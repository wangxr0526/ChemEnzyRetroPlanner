import argparse
import os
import torch
import sys

parser = argparse.ArgumentParser()

# ===================== gpu _id ===================== #
parser.add_argument('--gpu', type=int, default=-1)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--test_routes',
                    default='dataset/routes_possible_test_hard.pkl')
parser.add_argument('--starting_molecules',
                    default='building_block_dataset/zinc_stock_2021_10_3_canonical_smiles_total_10312151_add_8546.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='merge_data_path')
parser.add_argument(
    '--value_train', default='value_data_dic_142400_train_convert.pkl')
parser.add_argument(
    '--value_val', default='value_data_dic_142400_val_convert.pkl')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--expansion_topk', type=int, default=50)
parser.add_argument('--exclude_target', action='store_true')
parser.add_argument('--keep_search', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--viz_dir', default='viz')
parser.add_argument('--output_name', type=str, default='plan')
parser.add_argument('--search_strategy', type=str,
                    default='DFS', help='DFS, MCTS, MCTS_STAR')

# ===================== one-step mlp_model ========== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--mlp_model_dump', type=str,
                    default='./packages/mlp_retrosyn/mlp_retrosyn/model/saved_rollout_state_1_2048_2021-10-07_00_24_16.ckpt')
parser.add_argument('--mlp_templates', type=str,
                    default='./packages/single_step_datasets/train_test_dataset/train_val_template_rules_1.dat')

# ===================== one-step graphfp_model ======= #
parser.add_argument('--use_graph_single', action='store_true')
parser.add_argument('--graph_model_dumb', type=str,
                    default='./packages/graph_retrosyn/graph_retrosyn/model/train_all_model/saved_graph_rollout_state_1_DMPNNFP_init_lr_0.001_batch_size_1024_fp_lindim_512_graph_lindim_512_use_gru_False_massage_depth_3.ckpt')
parser.add_argument('--graph_dataset_root', type=str,
                    default='./packages/graph_retrosyn/graph_retrosyn/data/raw')

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--value_fn_n_epochs', type=int, default=100)
parser.add_argument('--value_fn_save_epoch_int', type=int, default=10)
parser.add_argument('--value_fn_save_folder',
                    default='value_function/save_model')

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')
parser.add_argument('--test_file', default='',
                    help='CSV path for target molecules.')
parser.add_argument('--result_folder', default='results')

# ==================== depth value =================== #
parser.add_argument('--use_depth_value_fn', action='store_true')
parser.add_argument('--depth_output_dim', type=int, default=10)

# ==================== filter =================== #
parser.add_argument('--use_filter', action='store_true')
parser.add_argument(
    '--filter_path', default='./packages/rxn_filter/rxn_filter/model/filter_train_data_random_gen_aizynth_filter_dataset_2021-12-22_12h-15m-45s.pkl')
parser.add_argument('--keep_score', action='store_true')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
