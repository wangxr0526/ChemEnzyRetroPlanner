import json
import os
import pickle
import subprocess
import time
from condition_predictor.condition_model import NeuralNetContextRecommender
import numpy as np
from organic_enzyme_rxn_classifier.inference_api import EnzymeRXNClassifier, OrganicEnzymeRXNClassifier
import requests
from rxn_filter.filter_models import FilterModel, FilterPolicy
import torch
from collections import defaultdict
from graph_retrosyn.graph_model import GraphModel, GCN, GCNFP, MPNNFP, DMPNN, DMPNNFP
import pandas as pd
import logging
from copy import deepcopy
from mlp_retrosyn.mlp_inference import MLPModel
from onmt.bin.translate import load_model, run

from retro_planner.common.utils import canonicalize_smiles


dirpath = os.path.dirname(os.path.abspath(__file__))


def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        try:
            starting_mols = set(list(pd.read_csv(filename)['mol']))
        except:
            starting_mols = set(list(pd.read_csv(filename)['smiles']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    assert isinstance(starting_mols, set)
    return starting_mols


def prepare_starting_molecules_for_multi_stock(filenames: list):
    starting_mols = set()
    for filename in filenames:
        starting_mols.update(prepare_starting_molecules(filename))
    return starting_mols


def prepare_mlp_models(templates,
                       model_dump,
                       device=-1,
                       use_filter=False,
                       filter_path=None,
                       keep_score=True):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump,
                        templates,
                        device=device,
                        use_filter=use_filter,
                        filter_path=filter_path,
                        keep_score=keep_score)
    return one_step


def prepare_graphfp_models(graph_dataset_root,
                           graph_model_dumb,
                           device=-1,
                           topk=10,
                           use_filter=False,
                           filter_path=None,
                           keep_score=True):
    template_rules, idx2rules = torch.load(
        os.path.join(graph_dataset_root, 'templates_index.pkl'))
    fp_lindim, \
    graph_lindim, \
    use_gru, \
    massage_depth = 512, 512, False, 3

    graph_core = DMPNNFP(mol_in_dim=107,
                         out_dim=len(template_rules),
                         dim=graph_lindim,
                         b_in_dim=14,
                         fp_lindim=fp_lindim,
                         fp_dim=2048,
                         use_gru=use_gru,
                         massage_depth=massage_depth)
    logging.info(f'Loding Graph Model From {graph_model_dumb}.')
    checkpoint, _ = torch.load(graph_model_dumb,
                               map_location=torch.device('cpu'))
    graph_core.load_state_dict(checkpoint)
    one_step = GraphModel(graph_model=graph_core,
                          idx2rules=idx2rules,
                          device=device,
                          topk=topk,
                          use_filter=use_filter,
                          filter_path=filter_path,
                          keep_score=keep_score)
    return one_step


def prepare_onmt_models(model_path, beam_size, topk, device):

    class OnmtRunWrapper:

        def __init__(self, model_path, beam_size, topk, device) -> None:
            self.opt, self.translator = load_model(
                model_path=model_path,
                beam_size=beam_size,
                topk=topk,
                device=int(str(device).split(':')[-1])
                if str(device) != 'cpu' else -1,
                tokenizer='char')

        def run(self, target, topk=None):
            results = run(self.translator, self.opt, target)
            results = self._manual_rules_for_rxn_without_rdkit(target=target,
                                                               result=results)
            templates = [None for _ in range(len(results['scores']))]
            results['template'] = templates
            return results
        
        def _syntax_valid_filter(self, reactants, scores):
            new_reactants, new_scores = map(list, zip(*[(reactant, score) for reactant, score in zip(reactants, scores) if reactant]))

            return new_reactants, new_scores

        def _manual_rules_for_rxn_without_rdkit(self, target, result):
            reactants = result['reactants']
            scores = result['scores']
            
            assert len(reactants) == len(scores)
            reactants, scores = self._syntax_valid_filter(reactants, scores)

            for i in range(len(reactants)):
                reactant = reactants[i]
                # 目前只考虑单个反应物的情况
                if "." in reactant or '*' in reactant:
                    continue
                if self._get_carbon_num_by_string(
                        target) > 10 and self._get_carbon_num_by_string(
                            reactant) < 3:
                    scores[i] /= np.e**20

            reactants, scores = self._resort_reactant_and_score(
                reactants, scores)
            result['reactants'] = reactants
            result['scores'] = scores
            return result

        def _get_carbon_num_by_string(self, smi):
            smi = smi.upper()
            return smi.count('C')

        def _resort_reactant_and_score(self, reactants, scores):
            data = [(score, reactant)
                    for score, reactant in zip(scores, reactants)]
            data = sorted(data, key=lambda x: x[0], reverse=True)
            scores = [each[0] for each in data]
            reactants = [each[1] for each in data]
            return reactants, scores

    one_step = OnmtRunWrapper(model_path=model_path,
                              beam_size=beam_size,
                              device=device,
                              topk=topk)
    return one_step


def prepare_template_relevance_models(state_name, topk):

    class TemplateRelevanceWrapper:

        def __init__(self, state_name='reaxys', topk=10) -> None:
            self.state_name = state_name
            self.url = f'http://retro_template_relevance:9410/predictions/{self.state_name}'  # retro_template_relevance 容器的hostname
            self.url_beta = f'http://localhost:9410/predictions/{self.state_name}'  # 备用链接

            self.headers = {'Content-Type': 'application/json'}

            self.topk = topk
            self.active_url = self.url  # 默认使用主链接

            # 检查服务是否存在，不存在则初始化
            if not self._is_service_available():
                self._initialize_service()

        def _is_service_available(self):
            # 尝试主链接
            if self._check_url(self.url):
                self.active_url = self.url
                return True
            # 如果主链接不可用，尝试备用链接
            elif self._check_url(self.url_beta):
                self.active_url = self.url_beta
                return True
            return False

        def _check_url(self, url):
            """检查指定 URL 是否可用"""
            try:
                response = requests.get(url,
                                        timeout=5)
                return response.status_code == 503
            except requests.RequestException:
                return False

        def _initialize_service(self):
            # 初始化服务的逻辑
            try:
                # 使用subprocess在 ../docker 路径中执行启动脚本
                subprocess.run(
                    ["bash", "run_container.sh", "--only-run-backend"],
                    cwd="../docker",  # 指定脚本执行路径
                    check=True  # 若脚本执行失败则抛出异常
                )
                print("Service initialization script executed successfully.")

                # 等待服务启动
                time.sleep(10)  # 可以根据实际启动时间调整

            except subprocess.CalledProcessError as e:
                print(f"Failed to start the service: {e}")
                raise RuntimeError(
                    f"Failed to start the service '{self.state_name}'.")

            # 确认服务是否成功启动
            if not self._is_service_available():
                raise RuntimeError(
                    f"Failed to start the service '{self.state_name}'.")

        def run(self, target, topk=None):
            inputs = {
                'smiles': [target],
                'max_num_templates': self.topk if topk is None else topk,
            }

            data_json = json.dumps(inputs)
            try:
                # 使用活动的 URL 进行请求
                response = requests.post(self.active_url,
                                         headers=self.headers,
                                         data=data_json)
                response.raise_for_status()  # 如果请求失败，抛出异常
            except requests.RequestException:
                # 如果请求失败，尝试备用 URL
                if self.active_url == self.url and self._check_url(
                        self.url_beta):
                    self.active_url = self.url_beta
                elif self.active_url == self.url_beta and self._check_url(
                        self.url):
                    self.active_url = self.url
                else:
                    raise RuntimeError(
                        f"Both primary and backup services are unavailable.")

                # 再次尝试请求
                response = requests.post(self.active_url,
                                         headers=self.headers,
                                         data=data_json)
                response.raise_for_status()  # 如果再次失败，将抛出异常

            results = self._postprocess(response.json()[0])
            return results

        def _postprocess(self, outputs):
            results = {
                'reactants': outputs['reactants'],
                'scores': outputs['scores'],
                'template':
                [x['reaction_smarts'] for x in outputs['templates']],
            }
            return results

    # 返回封装类实例
    return TemplateRelevanceWrapper(state_name=state_name, topk=topk)

def handle_one_step_config(model_names, one_step_model_configs):
    selected_one_step_model_configs = []
    selected_model_subnames = []
    selected_one_step_model_types = []
    for selected_model_name in model_names:
        one_step_model_type, model_subname = selected_model_name.split('.')
        assert one_step_model_type in [
            'mlp_models', 'graphfp_models', 'onmt_models',
            'template_relevance'
        ]
        cur_config = one_step_model_configs[one_step_model_type][model_subname]
        cur_config['model_full_name'] = selected_model_name
        selected_one_step_model_configs.append(cur_config)
        selected_model_subnames.append(model_subname)
        selected_one_step_model_types.append(one_step_model_type)
        logging.info(f'Selected One Step Model: {selected_model_name}')
    return selected_one_step_model_configs, selected_model_subnames, selected_one_step_model_types

def handle_one_step_path(selected_one_step_model_types, selected_one_step_model_configs):
    handled_selected_one_step_model_configs = []
    for selected_one_step_model_type, selected_one_step_model_config in zip(
            selected_one_step_model_types,
            selected_one_step_model_configs):
        if selected_one_step_model_type == 'mlp_models':
            selected_one_step_model_config['mlp_templates'] = os.path.abspath(os.path.join(
                dirpath, 
                '..',
                selected_one_step_model_config['mlp_templates']))
            selected_one_step_model_config[
                'mlp_model_dump'] = os.path.abspath(os.path.join(
                    dirpath,
                    '..',
                    selected_one_step_model_config['mlp_model_dump']))
        elif selected_one_step_model_type == 'graphfp_models':
            selected_one_step_model_config[
                'graph_dataset_root'] = os.path.abspath(os.path.join(
                    dirpath,
                    '..',
                    selected_one_step_model_config['graph_dataset_root']))
            selected_one_step_model_config[
                'graph_model_dumb'] = os.path.abspath(os.path.join(
                    dirpath,
                    '..',
                    selected_one_step_model_config['graph_model_dumb']))
        elif selected_one_step_model_type == 'onmt_models':
            selected_one_step_model_config['model_path'] = [
                os.path.abspath(os.path.join(dirpath, '..', x))
                for x in selected_one_step_model_config['model_path']
            ]
        elif selected_one_step_model_type == 'template_relevance':
            pass
        else:
            raise ValueError()
        handled_selected_one_step_model_configs.append(
            selected_one_step_model_config)
    return handled_selected_one_step_model_configs


def prepare_single_step(
    one_step_model_type,
    model_configs,
    expansion_topk,
    device=-1,
    use_filter=False,
    filter_path=None,
    keep_score=True,
):

    if one_step_model_type == 'mlp_models':

        templates = model_configs['mlp_templates']
        mlp_model_dump = model_configs['mlp_model_dump']

        logging.info('Using MLP Model')
        logging.info('Templates: %s' % templates)
        logging.info('Loading trained mlp model from %s' % mlp_model_dump)
        # one_step = MLPModel(mlp_model_dump, templates, device=device, topk=expansion_topk,
        #                     use_filter=use_filter, filter_path=filter_path, keep_score=keep_score)

        one_step = prepare_mlp_models(templates=templates,
                                      model_dump=mlp_model_dump,
                                      device=device,
                                      use_filter=use_filter,
                                      keep_score=keep_score,
                                      filter_path=filter_path)

    elif one_step_model_type == 'graphfp_models':
        # graph_dataset_root = os.path.join(
        #     '../packages/graph_retrosyn/graph_retrosyn/data/raw')
        graph_dataset_root = model_configs['graph_dataset_root']
        graph_model_dumb = model_configs['graph_model_dumb']
        logging.info('Using GraphFP Model.')
        logging.info('Templates: %s' %
                     os.path.join(graph_dataset_root, 'templates_index.pkl'))

        one_step = prepare_graphfp_models(
            graph_dataset_root=graph_dataset_root,
            graph_model_dumb=graph_model_dumb,
            device=device,
            topk=expansion_topk,
            keep_score=keep_score,
            filter_path=filter_path,
            use_filter=use_filter)

    elif one_step_model_type == 'onmt_models':

        model_path = model_configs['model_path']
        beam_size = model_configs['beam_size']
        logging.info('Using Onmt Model')
        logging.info('Model Path: %s' % model_path)

        one_step = prepare_onmt_models(model_path=model_path,
                                       beam_size=beam_size,
                                       topk=expansion_topk,
                                       device=device)

    elif one_step_model_type == 'template_relevance':
        logging.info('Using MIT template relevance')
        # logging.info('Model Path: %s' % model_path)
        one_step = prepare_template_relevance_models(
            state_name=model_configs['state_name'], topk=expansion_topk)

    else:
        raise ValueError

    return one_step


def prepare_multi_single_step(model_configs, one_step_model_types,
                              expansion_topk, device, use_filter, keep_score,
                              filter_path):

    class MultiOneStepRunWrapper():

        def __init__(self,
                     singe_step_model_configs,
                     one_step_model_types,
                     expansion_topk,
                     device=device,
                     use_filter=use_filter,
                     keep_score=keep_score,
                     filter_path=filter_path) -> None:

            self.one_step_models = dict()
            for model_configs, model_type in zip(singe_step_model_configs,
                                                 one_step_model_types):
                if model_configs['model_full_name']=='onmt_models.bionav_one_step':
                    print()
                    pass
                try:
                    one_step = prepare_single_step(one_step_model_type=model_type,
                                                model_configs=model_configs,
                                                expansion_topk=expansion_topk,
                                                device=device,
                                                use_filter=use_filter,
                                                keep_score=keep_score,
                                                filter_path=filter_path)
                    
                    self.one_step_models[model_configs['model_full_name']] = one_step
                except Exception as e:
                    print(e)
                    print(model_configs['model_full_name'])
                    

        def run(self, target, topk=None, select_models=None):
            multi_one_step_results = defaultdict(list)
            for model_full_name in self.one_step_models:
                if select_models is not None:
                    if model_full_name not in select_models:
                        continue
                one_step = self.one_step_models[model_full_name]
                results = one_step.run(target, topk=topk)
                if model_full_name.split('.')[0] == 'onmt_models':
                    results['costs'] = 0.0 - np.log(
                        np.clip(np.array(results['scores']), 0, 1.0))
                else:
                    results['costs'] = 0.0 - np.log(
                        np.clip(np.array(results['scores']), 1e-3, 1.0))
                results['model_full_name'] = [model_full_name for _ in range(len(results['reactants']))]
                for k in results:
                    multi_one_step_results[k].extend(results[k])


            return multi_one_step_results

    one_step = MultiOneStepRunWrapper(
        singe_step_model_configs=model_configs,
        expansion_topk=expansion_topk,
        one_step_model_types=one_step_model_types,
        device=device,
        use_filter=use_filter,
        keep_score=keep_score,
        filter_path=filter_path)
    return one_step


def prepare_molstar_planner(one_step,
                            starting_mols,
                            expansion_topk,
                            value_fn,
                            iterations,
                            max_depth,
                            exclude_target=True,
                            viz=False,
                            viz_dir=None,
                            search_strategy='DFS',
                            keep_search=False):

    from retro_planner.search_frame.mcts_star.molmcts_star import mol_planner


    def expansion_handle(x):
        return one_step.run(x)

    def plan_handle(x, y=0):
        return mol_planner(target_mol=x,
                           target_mol_id=y,
                           starting_mols=starting_mols,
                           expand_fn=expansion_handle,
                           iterations=iterations,
                           max_depth=max_depth,
                           exclude_target=exclude_target,
                           viz=viz,
                           viz_dir=viz_dir,
                           value_fn=value_fn,
                           keep_search=keep_search)

    return plan_handle


def init_rcr(config, dirpath):

    class RCRInferenceWrapper:

        def __init__(self, config):
            info_path = os.path.join(dirpath, config['info_path'])
            weights_path = os.path.join(dirpath, config['weights_path'])
            self.condition_predictor = NeuralNetContextRecommender()
            self.condition_predictor.load_nn_model(info_path=info_path,
                                                   weights_path=weights_path)
            self.model_name = 'rcr'

        def __call__(self, rxn, topk, return_scores):
            return self.condition_predictor.get_n_conditions(
                rxn=rxn, n=topk, return_scores=return_scores)

    rcr = RCRInferenceWrapper(config)
    return rcr


def init_parrot(config, dirpath):

    class ParrotInferenceWrapper:

        def __init__(self) -> None:
            self.url = f'http://parrot_serve_container:9510/predictions/USPTO_condition'
            # self.url = f'http://0.0.0.0:9510/predictions/USPTO_condition'

            self.headers = {'Content-Type': 'application/json'}
            self.model_name = 'parrot'

        def __call__(self, target, topk=10):
            data = json.dumps([target])
            response = requests.post(self.url, headers=self.headers, data=data)
            condition_df = pd.read_json(response.json()[0])
            condition_df = condition_df[[
                'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2',
                'scores'
            ]]
            condition_df['Solvent'] = condition_df.apply(lambda row: '.'.join([
                s for s in [row['solvent1'], row['solvent2']]
                if canonicalize_smiles(s) != ''
            ]),
                                                         axis=1)
            condition_df['Reagent'] = condition_df.apply(lambda row: '.'.join([
                s for s in [row['reagent1'], row['reagent2']]
                if canonicalize_smiles(s) != ''
            ]),
                                                         axis=1)
            condition_df['Temperature'] = [
                '' for x in range(len(condition_df))
            ]
            condition_df = condition_df[[
                'Temperature', 'Solvent', 'Reagent', 'catalyst1', 'scores'
            ]]
            condition_df.columns = [
                'Temperature', 'Solvent', 'Reagent', 'Catalyst', 'Score'
            ]
            condition_df = condition_df.iloc[:topk]
            return condition_df

    parrot = ParrotInferenceWrapper()
    return parrot

def prepare_filter_policy(filter_path, cutoff=0.5, device='cpu'):
    filter_path = os.path.abspath(os.path.join(
                dirpath, 
                '..',
                filter_path))
    filter_model = FilterModel(fp_dim=2048, dim=1024, dropout_rate=0.4)
    filter_model.load_state_dict(
        torch.load(filter_path, map_location='cpu'))
    filter_policy = FilterPolicy(filter_model=filter_model, cutoff=cutoff, device=device)
    return filter_policy

def prepare_enzymatic_rxn_identifier(organic_enzyme_rxn_classifier_config, device='cpu'):
    enzymatic_rxn_identifier = OrganicEnzymeRXNClassifier(
                checkpoint_path=os.path.abspath(
                    os.path.join(
                        dirpath, 
                        '..',
                        organic_enzyme_rxn_classifier_config['checkpoint_path'])
                    ),
                device=device)
    return enzymatic_rxn_identifier

def prepare_enzyme_recommender(enzyme_rxn_classifier_config, device='cpu'):
    enzyme_recommender = EnzymeRXNClassifier(
                checkpoint_path=os.path.abspath(
                os.path.join(
                    dirpath,
                    '..',
                    enzyme_rxn_classifier_config['checkpoint_path'])
                ),
                device=device)
    return enzyme_recommender