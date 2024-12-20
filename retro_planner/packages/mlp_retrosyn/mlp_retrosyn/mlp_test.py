import os

import pandas as pd
from torch import use_deterministic_algorithms
from tqdm import tqdm

from mlp_retrosyn.mlp_inference import MLPModel
from retro_planner.common.utils import canonicalize_smiles


def eval_model(prods_reacts, model, topk, fname_pred):
    # case_gen = rxn_data_gen(phase, model)

    cnt = 0
    topk_scores = [0.0] * topk

    pbar = tqdm(prods_reacts, total=len(prods_reacts))

    fpred = open(fname_pred, 'w')
    for prod, react in pbar:
        pred_struct = model.run(prod, topk=topk)
        if pred_struct is not None and len(pred_struct['reactants']):
            predictions = pred_struct['reactants']
        else:
            predictions = [prod]
        s = 0.0
        reactants = canonicalize_smiles(react)
        for i in range(topk):
            if i < len(predictions):
                pred = predictions[i]
                pred = canonicalize_smiles(pred)
                predictions[i] = pred
                cur_s = (pred == reactants)
            else:
                cur_s = s
            s = max(cur_s, s)
            topk_scores[i] += s
        cnt += 1
        if pred_struct is None or len(pred_struct['reactants']) == 0:
            predictions = []
        fpred.write('{}>>{} {}\n'.format(react, prod, len(predictions)))
        for i in range(len(predictions)):
            fpred.write('{} {}\n'.format(
                pred_struct['template'][i], predictions[i]))
        msg = 'average score'
        for k in range(0, min(topk, 10), 3):
            msg += ', t%d: %.4f' % (k + 1, topk_scores[k] / cnt)
        pbar.set_description(msg)
    fpred.close()
    h = '========%s results========' % len(prods_reacts)
    print(h)
    for k in range(topk):
        print('top %d: %.4f' % (k + 1, topk_scores[k] / cnt))
    print('=' * len(h))

    f_summary = '.'.join(fname_pred.split('.')[:-1]) + '.summary'
    with open(f_summary, 'w') as f:
        f.write('type overall\n')
        for k in range(topk):
            f.write('top %d: %.4f\n' % (k + 1, topk_scores[k] / cnt))


if __name__ == '__main__':
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_rule_path', default=os.path.join(
        '../../single_step_datasets/train_test_dataset/train_val_template_rules_1.dat'),
        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='./model/saved_rollout_state_1_2048_2021-11-14_21_22_09.ckpt',
                        type=str, help='specify where the trained model is')
    parser.add_argument('--testset_path', default='../../single_step_datasets/train_test_dataset/test_templates.dat',
                        type=str, help='specify where the testset is')
    parser.add_argument('--test_results_path', default='../../single_step_datasets/train_test_dataset/test_results.txt',
                        type=str, help='specify where the test results to save.')
    parser.add_argument('--use_filter', action='store_true', default=False)
    parser.add_argument('--use_corrected_template', action='store_true', default=True,
                        help='Specify whether to use corrected templates.')
    args = parser.parse_args()
    use_filter = args.use_filter
    state_path = args.model_path
    template_path = args.template_rule_path
    testset_path = args.testset_path
    use_corrected_template = args.use_corrected_template
    test_results_path = args.test_results_path
    if use_corrected_template:
        print('Using corrected template...')
        template_path = template_path.replace(
            '.dat', '_corrected.dat')
        testset_path = testset_path.replace(
            '.dat', '_corrected.dat')
        test_results_path = test_results_path.replace(
            '.txt', '_corrected.txt')
    if use_filter:
        from rxn_filter.filter_models import FilterPolicy
    model = MLPModel(state_path, template_path, device=1, fp_dim=2048, use_filter=use_filter, filter_path=None,
                     keep_score=False)

    testset = pd.read_csv(
        args.testset_path, header=None)
    eval_model(list(zip(testset[1].tolist(), testset[2].tolist())), model, 50,
               test_results_path)
