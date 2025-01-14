from collections import OrderedDict, defaultdict
from copy import deepcopy
import json
import os
from queue import Queue

import numpy as np
import pandas as pd
import requests
import torch
import logging
import time
from tqdm import tqdm
import yaml
from retro_planner.common import (
    smiles_to_fp,
)
import ipywidgets as widgets
from rdkit import Chem
from rdkit.Chem import AllChem
from retro_planner.common.prepare_utils import (
    handle_one_step_config,
    handle_one_step_path,
    init_parrot,
    init_rcr,
    prepare_enzymatic_rxn_identifier,
    prepare_enzyme_recommender,
    prepare_molstar_planner,
    prepare_multi_single_step,
    prepare_tree_lstm_pathway_ranker,
    prepare_single_step,
    prepare_starting_molecules,
    prepare_starting_molecules_for_multi_stock,
)
from retro_planner.common.utils import (
    canonicalize_smiles,
    predict_condition_for_dict_route,
    max_reaction_depth,
)
from retro_planner.utils import setup_logger
from value_function.value_mlp import MaxDepthValueMLP, ValueMLP
from organic_enzyme_rxn_classifier.inference_api import (
    OrganicEnzymeRXNClassifier,
    EnzymeRXNClassifier,
)
from condition_predictor.condition_model import NeuralNetContextRecommender
from easifa.interface.utils import (
    EasIFAInferenceAPI,
    UniProtParserEC,
    full_swissprot_checkpoint_path,
    get_structure_html_and_active_data,
    uniprot_csv_path,
    pdb_cache_path,
    chebi_path,
    uniprot_rxn_path,
    uniprot_json_path,
)
from IPython.display import display, HTML
from ipywidgets import (
    HBox,
    Label,
    VBox,
    Text,
    Output,
    Checkbox,
    IntText,
    FloatText,
    Button,
    Dropdown,
    BoundedIntText,
    BoundedFloatText,
    SelectMultiple,
    Layout,
)

from retro_planner.viz_utils.route_tree import copy_route_tree

dirpath = os.path.dirname(os.path.abspath(__file__))


class RSPlanner:

    def __init__(self, config):

        self.config = config
        # search
        self.gpu = self.config["gpu"]
        self.expansion_topk = self.config["expansion_topk"]
        self.iterations = self.config["iterations"]
        self.max_depth = self.config["max_depth"]
        self.stocks = self.config["stocks"]
        self.exclude_target = self.config["exclude_target"]
        self.search_strategy = self.config["search_strategy"]
        self.keep_search = self.config["keep_search"]

        # single step
        # self.use_graph_single = self.config['use_graph_single']
        self.one_step_model_configs = self.config["one_step_model_configs"]

        # # mlp
        # self.mlp_models = self.config['mlp_models']
        # # graphfp
        # self.graphfp_models = self.config['graphfp_models']
        # # onmt
        # self.onmt_models = self.config['onmt_models']
        # self.graph_model_dumb = graph_model_dumb
        # self.graph_dataset_root = graph_dataset_root

        # rxn filter
        self.use_filter = self.config["use_filter"]
        self.filter_path = self.config["filter_path"]
        self.keep_score = self.config["keep_score"]

        # value function
        self.use_value_fn = self.config["use_value_fn"]
        self.n_layers = self.config["n_layers"]
        self.fp_dim = self.config["fp_dim"]
        self.value_fn_save_folder = self.config["value_fn_save_folder"]
        self.value_model = self.config["value_model"]
        self.latent_dim = self.config["latent_dim"]
        self.use_depth_value_fn = self.config["use_depth_value_fn"]
        self.depth_value_latent_dim = self.config["depth_value_latent_dim"]
        self.depth_output_dim = self.config["depth_output_dim"]

        # condition prediction
        self.pred_condition = self.config["pred_condition"]
        self.condition_config = self.config["condition_config"]

        # organic and enzyme reaction classifer
        self.organic_enzyme_rxn_classification = self.config[
            "organic_enzyme_rxn_classification"
        ]
        self.organic_enzyme_rxn_classifier_config = self.config[
            "organic_enzyme_rxn_classifier_config"
        ]
        # enzyme assign
        self.enzyme_assign = self.config["enzyme_assign"]
        self.enzyme_rxn_classifier_config = self.config["enzyme_rxn_classifier_config"]

        self.config["rxn_attributes"] = []

        for rxn_attribute in [
            "pred_condition",
            "organic_enzyme_rxn_classification",
            "enzyme_assign",
        ]:
            if self.config[rxn_attribute]:
                self.config["rxn_attributes"].append(rxn_attribute)

        # easifa
        self.easifa_config = self.config["easifa_config"]

        # pathway ranking
        self.pathway_ranking_config = self.config["pathway_ranking_config"]

        # viz
        self.viz = self.config["viz"]
        self.viz_dir = self.config["viz_dir"]

    def select_stock(self, stock_name):
        self.starting_molecules = self.stocks[stock_name]
        logging.info(f"Selected Stock: {stock_name}")

    def select_stocks(self, stock_names):
        self.starting_molecules = [self.stocks[x] for x in stock_names]
        logging.info(f"Selected Stock: {stock_names}")
    
    def select_pathway_ranker(self, ranker_name):
        self.pathway_ranker_name = self.pathway_ranking_config[ranker_name]
        if self.pathway_ranker_name == "tree_lstm_ranker":
            self.pathway_ranker = prepare_tree_lstm_pathway_ranker()
        elif self.pathway_ranker_name == "depth_based_ranker":
            pass


    # def select_one_step_model(self, model_names):
    #     '''
    #     model_show_name: graphfp_models.USPTO-full_remapped
    #                         onmt_models.bionav_one_step
    #     '''

    #     self.selected_model_subnames = []
    #     self.selected_one_step_model_types = []
    #     self.selected_one_step_model_configs = []

    #     if isinstance(model_names, str):
    #         model_names = [model_names]

    #     for selected_model_name in model_names:
    #         one_step_model_type, model_subname = selected_model_name.split('.')
    #         assert one_step_model_type in [
    #             'mlp_models', 'graphfp_models', 'onmt_models',
    #             'template_relevance'
    #         ]
    #         cur_config = self.one_step_model_configs[one_step_model_type][model_subname]
    #         cur_config['model_full_name'] = selected_model_name
    #         self.selected_one_step_model_configs.append(cur_config)
    #         self.selected_model_subnames.append(model_subname)
    #         self.selected_one_step_model_types.append(one_step_model_type)
    #         logging.info(f'Selected One Step Model: {selected_model_name}')

    #     self._handle_one_step_path()
    def select_one_step_model(self, model_names):
        """
        model_show_name: graphfp_models.USPTO-full_remapped
                            onmt_models.bionav_one_step
        """
        if isinstance(model_names, str):
            model_names = [model_names]

        (
            self.selected_one_step_model_configs,
            self.selected_model_subnames,
            self.selected_one_step_model_types,
        ) = handle_one_step_config(model_names, self.one_step_model_configs)
        self._handle_one_step_path()

    def select_condition_predictor(self, model_name):
        self.condition_config = self.condition_config[model_name]

    # def _handle_one_step_path(self):
    #     selected_one_step_model_configs = []
    #     for selected_one_step_model_type, selected_one_step_model_config in zip(
    #             self.selected_one_step_model_types,
    #             self.selected_one_step_model_configs):
    #         if selected_one_step_model_type == 'mlp_models':
    #             selected_one_step_model_config['mlp_templates'] = os.path.join(
    #                 dirpath, selected_one_step_model_config['mlp_templates'])
    #             selected_one_step_model_config[
    #                 'mlp_model_dump'] = os.path.join(
    #                     dirpath,
    #                     selected_one_step_model_config['mlp_model_dump'])
    #         elif selected_one_step_model_type == 'graphfp_models':
    #             selected_one_step_model_config[
    #                 'graph_dataset_root'] = os.path.join(
    #                     dirpath,
    #                     selected_one_step_model_config['graph_dataset_root'])
    #             selected_one_step_model_config[
    #                 'graph_model_dumb'] = os.path.join(
    #                     dirpath,
    #                     selected_one_step_model_config['graph_model_dumb'])
    #         elif selected_one_step_model_type == 'onmt_models':
    #             selected_one_step_model_config['model_path'] = [
    #                 os.path.join(dirpath, x)
    #                 for x in selected_one_step_model_config['model_path']
    #             ]
    #         elif selected_one_step_model_type == 'template_relevance':
    #             pass
    #         else:
    #             raise ValueError()
    #         selected_one_step_model_configs.append(
    #             selected_one_step_model_config)
    #     self.selected_one_step_model_configs = selected_one_step_model_configs

    def _handle_one_step_path(self):
        self.selected_one_step_model_configs = handle_one_step_path(
            self.selected_one_step_model_types, self.selected_one_step_model_configs
        )

    def prepare_plan(
        self,
        prepare_easifa=True,
        prepare_condition_predictor=True,
        prepare_enzyme_recommander=True,
    ):
        if not hasattr(self, "starting_molecules"):
            print(list(self.stocks.keys()))
            raise ValueError("Please select building block dataset")
        device = torch.device("cuda:%d" % self.gpu if self.gpu >= 0 else "cpu")
        if isinstance(self.starting_molecules, str):
            starting_molecules = os.path.join(dirpath, self.starting_molecules)
            starting_mols = prepare_starting_molecules(starting_molecules)
        elif isinstance(self.starting_molecules, list):
            starting_molecules = [
                os.path.join(dirpath, x) for x in self.starting_molecules
            ]
            starting_mols = prepare_starting_molecules_for_multi_stock(
                starting_molecules
            )
        else:
            raise ValueError("Stock error")
        filter_path = os.path.join(dirpath, self.filter_path)

        if len(self.selected_one_step_model_configs) == 1:
            one_step = prepare_single_step(
                one_step_model_type=self.selected_one_step_model_types[0],
                model_configs=self.selected_one_step_model_configs[0],
                device=device,
                use_filter=self.use_filter,
                filter_path=filter_path,
                expansion_topk=self.expansion_topk,
                keep_score=self.keep_score,
            )

        else:
            one_step = prepare_multi_single_step(
                one_step_model_types=self.selected_one_step_model_types,
                model_configs=self.selected_one_step_model_configs,
                device=device,
                use_filter=self.use_filter,
                filter_path=filter_path,
                expansion_topk=self.expansion_topk,
                keep_score=self.keep_score,
            )

        if self.use_value_fn and self.use_depth_value_fn:
            raise ValueError("Please select 'use_value_fn' or 'use_depth_value_fn'")
        elif self.use_value_fn:
            logging.info("use guiding function")
            model = ValueMLP(
                n_layers=self.n_layers,
                fp_dim=self.fp_dim,
                latent_dim=self.latent_dim,
                dropout_rate=0.1,
                device=device,
            ).to(device)
            value_fn_save_folder = os.path.join(dirpath, self.value_fn_save_folder)
            model_f = "%s/%s" % (value_fn_save_folder, self.value_model)
            logging.info("Loading value nn from %s" % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=self.fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v

        elif self.use_depth_value_fn:
            logging.info("use depth value guiding function")
            model = MaxDepthValueMLP(
                n_layers=self.n_layers,
                fp_dim=self.fp_dim,
                latent_dim=self.depth_value_latent_dim,
                output_dim=self.depth_output_dim,
                dropout_rate=0.4,
            ).to(device)
            value_fn_save_folder = os.path.join(dirpath, self.value_fn_save_folder)
            model_f = "%s/%s" % (value_fn_save_folder, self.value_model)
            logging.info("Loading value nn from %s" % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=self.fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                y_pred = model(fp).argmax(dim=1)
                depth = y_pred + 1
                return depth.float().item()

        else:

            def value_fn(x):
                return 0.0

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            starting_mols=starting_mols,
            expansion_topk=self.expansion_topk,
            iterations=self.iterations,
            exclude_target=self.exclude_target,
            viz=self.viz,
            viz_dir=self.viz_dir,
            max_depth=self.max_depth,
            search_strategy=self.search_strategy,
            value_fn=value_fn,
            keep_search=self.keep_search,
        )

        # info_path = os.path.join(dirpath , self.condition_config['info_path'])
        # weights_path = os.path.join(dirpath, self.condition_config['weights_path'])
        # condition_predictor = NeuralNetContextRecommender()
        # condition_predictor.load_nn_model(
        #     info_path=info_path,
        #     weights_path=weights_path
        # )
        # self.condition_predictor = condition_predictor.get_n_conditions

        if prepare_condition_predictor:
            if self.condition_config["model_name"] == "rcr":
                self.condition_predictor = init_rcr(
                    self.condition_config, dirpath=dirpath
                )
            elif self.condition_config["model_name"] == "parrot":
                self.condition_predictor = init_parrot(
                    self.condition_config, dirpath=dirpath
                )
            else:
                raise ValueError

        # print(os.path.abspath(self.organic_enzyme_rxn_classifier_config['checkpoint_path']))
        if prepare_enzyme_recommander:
            self.organic_enzyme_rxn_classifer = prepare_enzymatic_rxn_identifier(
                self.organic_enzyme_rxn_classifier_config, device="cpu"
            )
            self.enzyme_rxn_classifer = prepare_enzyme_recommender(
                self.enzyme_rxn_classifier_config, device="cpu"
            )

        if prepare_easifa:
            self.easifa_annotator = EasIFAInferenceAPI(
                model_checkpoint_path=os.path.join(
                    dirpath, self.easifa_config["checkpoint_path"]
                ),
                device="cuda:0",
            )
            self.uniprot_parser = UniProtParserEC(
        json_folder=uniprot_json_path,
        csv_folder=uniprot_csv_path,
        alphafolddb_folder=pdb_cache_path,
        chebi_path=chebi_path,
        rxn_folder=uniprot_rxn_path,
    )

    def predict_condition(self):
        assert self.result
        all_succ_routes = self.result["all_succ_routes"]
        all_succ_dict_routes = [route.route_to_dict() for route in all_succ_routes]
        rxn_with_condition = [
            predict_condition_for_dict_route(
                self.condition_predictor, route, self.condition_config["topk"]
            )
            for route in tqdm(all_succ_dict_routes)
        ]
        return rxn_with_condition

    def _predict_rxn_attribute_for_one_route(
        self,
        dict_route,
        rxn_attributes=[
            "pred_condition",
            "organic_enzyme_rxn_classification",
            "enzyme_assign",
        ],
    ):

        def meta_dict_to_json(meta_dict):
            new_meta_dict = defaultdict(dict)
            for meta in meta_dict:
                for k in meta_dict[meta]:
                    new_meta_dict[meta][k] = meta_dict[meta][k].to_json()
            return new_meta_dict

        rxn_attributes_dict = defaultdict(dict)
        node_queue = Queue()
        # dict_route_copy = deepcopy(dict_route)
        node_queue.put(dict_route)
        while not node_queue.empty():
            mol_node = node_queue.get()
            assert mol_node["type"] == "mol"
            if "children" not in mol_node:
                continue
            assert len(mol_node["children"]) == 1
            reaction_node = mol_node["children"][0]
            reactants = []
            for c_mol_node in reaction_node["children"]:
                reactants.append(c_mol_node["smiles"])
                node_queue.put(c_mol_node)
            reactants = ".".join(reactants)
            rxn_smiles = "{}>>{}".format(reactants, mol_node["smiles"])
            reaction_node["rxn_smiles"] = rxn_smiles

            for attribute in rxn_attributes:
                if attribute == "pred_condition":

                    if self.condition_predictor.model_name == "rcr":
                        context_combos, context_combo_scores = self.condition_predictor(
                            rxn_smiles,
                            self.condition_config["topk"],
                            return_scores=True,
                        )
                        condition_df = pd.DataFrame(context_combos)
                        condition_df.columns = [
                            "Temperature",
                            "Solvent",
                            "Reagent",
                            "Catalyst",
                            "null1",
                            "null2",
                        ]
                        condition_df["Score"] = [
                            f"{num:.4f}" for num in context_combo_scores
                        ]
                    elif self.condition_predictor.model_name == "parrot":
                        condition_df = self.condition_predictor(
                            rxn_smiles, self.condition_config["topk"]
                        )

                    rxn_attributes_dict[rxn_smiles]["condition"] = condition_df

                elif attribute == "organic_enzyme_rxn_classification":
                    _, organic_enzyme_rxn_confidence, organic_enzyme_rxn_names = (
                        self.organic_enzyme_rxn_classifer.predict(
                            [rxn_smiles], batch_size=32
                        )
                    )
                    rxn_attributes_dict[rxn_smiles][
                        "organic_enzyme_rxn_classification"
                    ] = pd.DataFrame(
                        {
                            "Reaction Type": [organic_enzyme_rxn_names[0]],
                            "Confidence": [organic_enzyme_rxn_confidence[0]],
                        }
                    )

                elif attribute == "enzyme_assign":
                    _, ecnumber_confidence, ecnumbers = (
                        self.enzyme_rxn_classifer.predict(
                            [rxn_smiles],
                            batch_size=32,
                            topk=self.enzyme_rxn_classifier_config["topk"],
                        )
                    )
                    rxn_attributes_dict[rxn_smiles]["enzyme_assign"] = pd.DataFrame(
                        {
                            "Ranks": [f"Top-{k+1}" for k in range(len(ecnumbers[0]))],
                            "EC Number": [ec for ec in ecnumbers[0]],
                            "Confidence": [conf for conf in ecnumber_confidence[0]],
                        }
                    )
                else:
                    raise ValueError

            reaction_node["rxn_attribute"] = meta_dict_to_json(rxn_attributes_dict)[
                rxn_smiles
            ]

        return rxn_attributes_dict

    def predict_rxn_attributes(self):
        assert self.result
        all_succ_routes = self.result["all_succ_routes"]
        all_succ_dict_routes = [route.dict_route for route in all_succ_routes]

        rxn_attributes_dicts = [
            self._predict_rxn_attribute_for_one_route(
                route, rxn_attributes=self.config["rxn_attributes"]
            )
            for route in tqdm(all_succ_dict_routes, disable=True)
        ]

        return rxn_attributes_dicts

    def download_enzyme_pdb_by_ec_and_pred_active_site(self, ec_number, rxn_smiles):
        structure_htmls = []
        active_data_list = []
        ec_list = []
        alphafolddb_id_list = []
        uniprot_df = self.uniprot_parser.query_enzyme_pdb_by_ec(
            ec_number=ec_number, size=1
        )
        if uniprot_df is not None:
            for alphafolddb_id, ec, pdb_fpath in zip(
                uniprot_df["AlphaFoldDB"].tolist(),
                uniprot_df["EC number"].tolist(),
                uniprot_df["pdb_fpath"].tolist(),
            ):
                pred_active_site_labels = self.easifa_annotator.inference(
                    rxn=rxn_smiles, enzyme_structure_path=pdb_fpath
                )
                structure_html, active_data = get_structure_html_and_active_data(
                    enzyme_structure_path=pdb_fpath,
                    site_labels=pred_active_site_labels,
                    view_size=(600, 600),
                )
                structure_htmls.append(structure_html)

                active_data_df = pd.DataFrame(
                    active_data,
                    columns=["Residue Index", "Residue Name", "Color", "Active Type"],
                )
                # active_data_df.columns = ['Residue Index', 'Residue Name', 'Color', 'Active Type']

                active_data_list.append(active_data_df)
                ec_list.append(ec)
                alphafolddb_id_list.append(alphafolddb_id)

            _results = list(
                zip(
                    list(range(len(structure_htmls))),
                    structure_htmls,
                    active_data_list,
                    ec_list,
                    alphafolddb_id_list,
                )
            )
            return _results
        else:
            return

    def _calculate_rxns_steps(self, succ_routes):
        return [max_reaction_depth(route.route_to_dict()) for route in succ_routes]
    
    def _depth_based_ranker(self, all_succ_routes):
        all_succ_routes_depth = self._calculate_rxns_steps(all_succ_routes)

        all_succ_routes_depth = np.array(all_succ_routes_depth)
        sorted_indices = np.argsort(all_succ_routes_depth)
        sorted_all_succ_routes = np.array(all_succ_routes)[sorted_indices].tolist()
        return sorted_all_succ_routes

    def _pathway_rank(self, all_succ_routes):
        if not hasattr(self, "pathway_ranker_name"):
            return self._depth_based_ranker(all_succ_routes)
        
        if self.pathway_ranker_name == 'depth_based_ranker':
            return self._depth_based_ranker(all_succ_routes)

        elif self.pathway_ranker_name == 'tree_lstm_ranker':
            try:
                all_succ_dict_routes = [
                        route.dict_route for route in all_succ_routes
                    ]
                all_succ_routes_scores, _ = self.pathway_ranker.predict_from_list(all_succ_dict_routes)
                all_succ_routes_scores = np.array(all_succ_routes_scores)
                sorted_indices = np.argsort(all_succ_routes_scores)[::-1]
                sorted_all_succ_routes = np.array(all_succ_routes)[sorted_indices].tolist()
            except:
                print('Warning tree lstm ranker not available, using depth-based ranker!')
                sorted_all_succ_routes = self._depth_based_ranker(all_succ_routes)
            return sorted_all_succ_routes


    def plan(self, target_mol):
        # target_mol = canonicalize_smiles(target_mol)
        t0 = time.time()
        succ, msg = self.plan_handle(target_mol)

        if succ:
            all_succ_routes = msg[2]
            sorted_all_succ_routes = self._pathway_rank(all_succ_routes)
            self.result = {
                "succ": succ,
                "time": time.time() - t0,
                "iter": msg[1],
                "routes": msg[0],
                "dict_routes": msg[0].route_to_dict(),
                "route_lens": msg[0].length,
                "all_succ_routes": sorted_all_succ_routes,
                "all_succ_dict_routes": [
                    route.dict_route for route in sorted_all_succ_routes
                ],
                "first_succ_time": msg[3],
            }
            return self.result

        else:
            logging.info(
                "Synthesis path for %s not found. Please try increasing "
                "the number of iterations." % target_mol
            )
            self.result = None
            return self.result


class RetroPlannerApp:

    def __init__(self, configfile) -> None:
        setup_logger()

        self.config = yaml.load(open(configfile, "r"), Loader=yaml.FullLoader)
        self.planner = None
        self.results = None
        self._input = dict()
        self._output = dict()
        self._buttons = dict()
        self.setup()

    def setup(self):
        self._create_input_widgets()
        self._create_search_widgets()
        self._create_route_widgets()
        self._create_enzyme_widgets()

    def _create_input_widgets(self):
        self._input["smiles"] = Text(
            description="SMILES",
            continuous_update=False,
            layout={
                "width": "99%",
                # "overflow": "auto",
            },
        )
        self._input["smiles"].observe(self._show_mol, names="value")
        display(self._input["smiles"])
        self._output["smiles"] = Output(
            layout={"border": "1px solid silver", "width": "50%", "height": "180px"}
        )
        self._input["keep_search"] = widgets.Checkbox(
            value=self.config["keep_search"],
            description="Keep search after solved one route",
            layout={
                "width": "99%",
                "overflow": "auto",
            },
        )
        self._input["use_filter"] = widgets.Checkbox(
            value=self.config["use_filter"],
            description="Use reaction plausibility evaluator",
            layout={"width": "auto", "description_width": "initial"},
        )
        self._input["use_depth_value_fn"] = widgets.Checkbox(
            value=self.config["use_depth_value_fn"],
            description="Use guiding function",
        )
        self._input["pred_condition"] = widgets.Checkbox(
            value=self.config["pred_condition"],
            description="Predict reaction condition",
        )
        self._input["organic_enzyme_rxn_classification"] = widgets.Checkbox(
            value=self.config["organic_enzyme_rxn_classification"],
            description="Identify enzymatic reactions",
        )

        self._input["enzyme_assign"] = widgets.Checkbox(
            value=self.config["enzyme_assign"],
            description="Recommend enzymes",
            layout={"width": "auto", "description_width": "initial"},
        )

        display(
            HBox(
                [
                    self._output["smiles"],
                    VBox(
                        [
                            self._input["keep_search"],
                            self._input["use_filter"],
                            self._input["use_depth_value_fn"],
                            self._input["pred_condition"],
                            self._input["organic_enzyme_rxn_classification"],
                            self._input["enzyme_assign"],
                        ]
                    ),
                ]
            )
        )

        # self._input["stocks"] = [
        #     Checkbox(
        #         value=False,
        #         description=key,
        #         style={"description_width": "initial"},
        #         layout={"justify": "left"},
        #     )
        #     for key in self.config['stocks'].keys()
        # ]

        # box_stocks = VBox(
        #     [Label("Stocks")] + self._input["stocks"],
        #     layout={"border": "1px solid silver"},
        # )
        init_value_stocks = (
            [list(self.config["stocks"].keys())[0]] if self.config["stocks"] else []
        )
        # print(init_value_stocks)

        self._input["stocks"] = SelectMultiple(
            options=list(self.config["stocks"].keys()),
            value=init_value_stocks,
            description="Select Stocks:",
            style={"description_width": "initial"},
            # rows=min(len(self.config['stocks'].keys()) + 1, 6),
            rows=6,
            layout={
                "width": "99%",
                "overflow": "auto",
            },
        )

        all_model_names = []
        for model_type in self.config["one_step_model_configs"]:
            for model_subname in self.config["one_step_model_configs"][model_type]:
                all_model_names.append(f"{model_type}.{model_subname}")

        init_value_one_step = [all_model_names[0]] if all_model_names else []
        # print(init_value_one_step)
        self._input["one_step"] = SelectMultiple(
            options=all_model_names,
            value=init_value_one_step,
            description="One Step Model:",
            style={"description_width": "initial"},
            # rows=min(len(all_model_names) + 1, 6),
            rows=6,
            layout={
                "width": "99%",
                "overflow": "auto",
            },
        )
        max_iter_box = self._make_slider_input("iterations", "Max Iterations", 10, 2000)
        self._input["iterations"].value = self.config["iterations"]

        max_search_depth = self._make_slider_input(
            "max_depth", "Max Search Depth", 3, 10
        )
        self._input["max_depth"].value = self.config["max_depth"]

        hbox = HBox(
            [
                self._input["stocks"],
                self._input["one_step"],
            ],
        )

        vbox = VBox(
            [
                hbox,
                max_iter_box,
                max_search_depth,
            ]
        )

        # box_options = HBox([box_stocks, vbox])
        # box_options = HBox([vbox])
        # display(box_options)
        display(vbox)

    def _create_search_widgets(self) -> None:
        self._buttons["execute"] = Button(
            description="Run Search",
            layout={"width": "auto", "description_width": "initial"},
        )
        self._buttons["execute"].on_click(self._on_exec_button_clicked)
        display(
            HBox(
                [self._buttons["execute"]],
                layout={
                    "display": "flex",
                    "justify_content": "center",
                    "align_items": "center",
                },
            )
        )
        self._output["tree_search"] = widgets.Output(
            layout={
                "border": "1px solid silver",
                "width": "99%",
                "height": "320px",
                "overflow": "auto",
            }
        )
        display(self._output["tree_search"])

    def _create_route_widgets(self) -> None:
        self._buttons["show_routes"] = Button(description="Show Reactions")
        self._buttons["show_routes"].on_click(self._on_display_route_button_clicked)
        self._input["route"] = Dropdown(
            options=[],
            description="Routes: ",
        )
        self._input["route"].observe(self._on_change_route_option)
        display(
            HBox(
                [
                    self._buttons["show_routes"],
                    self._input["route"],
                ],
                layout={
                    "display": "flex",
                    "justify_content": "center",
                    "align_items": "center",
                },
            )
        )

        self._output["routes"] = widgets.Output(
            layout={"border": "1px solid silver", "width": "99%", "min_height": "320px"}
        )
        display(self._output["routes"])

    def _create_enzyme_widgets(self):
        self._buttons["show_enzyme"] = Button(
            description="Download Enzyme and Show Active Site",
            layout={"width": "auto", "description_width": "initial"},
        )
        # layout=Layout(width='auto', description_width='initial'))
        self._buttons["show_enzyme"].on_click(self._one_display_enzyme_button_clicked)
        display(
            HBox(
                [
                    self._buttons["show_enzyme"],
                ],
                layout={
                    "display": "flex",
                    "justify_content": "center",
                    "align_items": "center",
                },
            )
        )
        self._output["enzyme"] = widgets.Output(
            layout={"border": "1px solid silver", "width": "99%"}
        )
        display(self._output["enzyme"])

    def _on_exec_button_clicked(self, _):
        self._toggle_button(False)
        self._search_results()
        self._toggle_button(True)

    def _on_change_route_option(self, change):
        if change["name"] != "index":
            return
        self._show_route(self._input["route"].index)

    def _on_display_route_button_clicked(self, _):
        self._toggle_button(False)
        self._input["route"].options = [
            # type: ignore
            f"Option {i}"
            for i, _ in enumerate(self.results["all_succ_routes"], 1)
        ]
        self._show_route(0)
        self._toggle_button(True)

    def _one_display_enzyme_button_clicked(self, _):
        self._toggle_button(False)
        self._download_enzyme_and_predict_active_sites()
        self._toggle_button(True)
        pass

    def _download_enzyme_and_predict_active_sites(self):
        if not hasattr(self, "showed_index"):
            self.showed_index = 0
        self._output["enzyme"].clear_output()
        with self._output["enzyme"]:

            rxn_attributes_items = list(
                self.route_with_attributes[self.showed_index].items()
            )
            rxn_attributes_items.reverse()
            for j, (rxn, _) in enumerate(rxn_attributes_items):
                display(HTML("<H3>Step {}".format(j + 1)))
                display(AllChem.ReactionFromSmarts(rxn, useSmiles=True))
                if "organic_enzyme_rxn_classification" in self.config["rxn_attributes"]:
                    Reaction_type_df = self.route_with_attributes[self.showed_index][
                        rxn
                    ]["organic_enzyme_rxn_classification"]
                    if (
                        "enzyme_assign" in self.config["rxn_attributes"]
                    ) and Reaction_type_df["Reaction Type"].tolist()[
                        0
                    ] == "Enzymatic Reaction":
                        Enzyme_type_df = self.route_with_attributes[self.showed_index][
                            rxn
                        ]["enzyme_assign"]
                        for ec in Enzyme_type_df["EC Number"].tolist():
                            display(HTML("<H4>Enzyme"))
                            display(HTML("<H4>EC number: {}".format(ec)))
                            enzyme_active_site_results = self.planner.download_enzyme_pdb_by_ec_and_pred_active_site(
                                ec_number=ec, rxn_smiles=rxn
                            )
                            if enzyme_active_site_results:
                                for (
                                    idx,
                                    structure_html,
                                    active_data_df,
                                    _,
                                    alphafolddb_id,
                                ) in enzyme_active_site_results:
                                    display(
                                        HTML(
                                            "<H4>UniProt ID: {}".format(alphafolddb_id)
                                        )
                                    )
                                    display(HTML(structure_html))
                                    display(
                                        HTML(
                                            f"<H4>Detailed active sites for {alphafolddb_id}"
                                        )
                                    )
                                    # active_data_df = active_data_df[['Residue Index', 'Residue Name', 'Active Type']]
                                    html_table_head = (
                                        "<thead><tr>"
                                        + "".join(
                                            [
                                                f'<th style="text-align: center;">{col}</th>'
                                                for col in active_data_df[
                                                    [
                                                        "Residue Index",
                                                        "Residue Name",
                                                        "Active Type",
                                                    ]
                                                ].columns
                                            ]
                                        )
                                        + "</tr></thead>"
                                    )
                                    html_rows = []
                                    for _, row in active_data_df.iterrows():
                                        html_rows.append(
                                            f"<tr><td style='text-align: center;'>{row['Residue Index']}</td><td style='text-align: center;'>{row['Residue Name']}</td><td style='text-align: center; color: {row['Color']};'>{row['Active Type']}</td></tr>"
                                        )
                                    html_table = f"<table>{html_table_head}<tbody>{''.join(html_rows)}</tbody></table>"
                                    display(HTML(html_table))

                    else:
                        display(HTML("<H4>It's not an enzymatic reaction"))

    def _show_route(self, index):

        self.showed_index = index

        if (
            index is None
            or self.results["all_succ_routes"] is None
            or index >= len(self.results["all_succ_routes"])
        ):
            return
        status = "Search Succeeded" if self.results["succ"] else "Search Failed"
        self._output["routes"].clear_output()
        with self._output["routes"]:
            display(HTML("<H2>%s" % status))
            display(HTML("<H2>Synthesis Route"))
            display(self.route_images[index])
            rxn_attributes_items = list(self.route_with_attributes[index].items())
            rxn_attributes_items.reverse()
            display(HTML("<H2>Reaction step by step"))

            for j, (rxn, _) in enumerate(rxn_attributes_items):
                display(HTML("<H3>Step {}".format(j + 1)))
                display(AllChem.ReactionFromSmarts(rxn, useSmiles=True))
                if "pred_condition" in self.config["rxn_attributes"]:
                    condition_df = self.route_with_attributes[index][rxn]["condition"]
                    display(HTML("<H4>Predicted Conditions:"))
                    condition_df["Ranks"] = [
                        f"Top-{k + 1}" for k in range(len(condition_df))
                    ]
                    condition_to_show = [
                        "Ranks",
                        "Temperature",
                        "Solvent",
                        "Reagent",
                        "Catalyst",
                        "Score",
                    ]

                    display(
                        HTML(f"{condition_df[condition_to_show].to_html(index=False)}")
                    )
                if "organic_enzyme_rxn_classification" in self.config["rxn_attributes"]:
                    Reaction_type_df = self.route_with_attributes[index][rxn][
                        "organic_enzyme_rxn_classification"
                    ]
                    display(HTML("<H4>Predicted Reaction Type:"))
                    display(HTML(f"{Reaction_type_df.to_html(index=False)}"))

                    if (
                        "enzyme_assign" in self.config["rxn_attributes"]
                    ) and Reaction_type_df["Reaction Type"].tolist()[
                        0
                    ] == "Enzymatic Reaction":
                        Enzyme_type_df = self.route_with_attributes[index][rxn][
                            "enzyme_assign"
                        ]
                        display(HTML("<H4>Enzyme Reaction Type:"))
                        display(HTML(f"{Enzyme_type_df.to_html(index=False)}"))

                        # self._create_enzyme_widgets()

    def _toggle_button(self, on_) -> None:
        for button in self._buttons.values():
            button.disabled = not on_

    def _search_results(self):
        self._output["tree_search"].clear_output()
        update_config = {
            "iterations": self._input["iterations"].value,
            "keep_search": self._input["keep_search"].value,
            "use_filter": self._input["use_filter"].value,
            "use_depth_value_fn": self._input["use_depth_value_fn"].value,
            "pred_condition": self._input["pred_condition"].value,
            "organic_enzyme_rxn_classification": self._input[
                "organic_enzyme_rxn_classification"
            ].value,
            "enzyme_assign": self._input["enzyme_assign"].value,
        }
        self.config.update(update_config)
        self.planner = RSPlanner(self.config)
        with self._output["tree_search"]:
            # for k, v in self.config.items():
            #     print(f'{k}:{v}')
            for k, v in update_config.items():
                logging.info(f"{k}:{v}")
            selected_stocks = self._input["stocks"].value[0]
            one_step_model = self._input["one_step"].value
            self.planner.select_stock(selected_stocks)
            self.planner.select_one_step_model(one_step_model)
            self.planner.prepare_plan()
            smiles = self._input["smiles"].value
            can_smiles = canonicalize_smiles(smiles)
            logging.info(f"Input Smiles: {smiles}\nCanonical Smiles: {can_smiles}")
            self.results = self.planner.plan(can_smiles)
            if not self.results:
                return
            logging.info(
                "{} routes were found!".format(len(self.results["all_succ_routes"]))
            )
            # if self.config['pred_condition']:
            #     logging.info('Prediting reaction condition...')
            #     self.route_with_condition = self.planner.predict_condition()
            if self.config["rxn_attributes"]:
                logging.info("Prediting reaction condition...")
                self.route_with_attributes = self.planner.predict_rxn_attributes()

            logging.info("Visualizing routes...")
            self.viz_routes()
            # print(self.results)
            # self.viz_routes()

    def _show_mol(self, change) -> None:
        self._output["smiles"].clear_output()
        with self._output["smiles"]:
            mol = Chem.MolFromSmiles(change["new"])
            display(mol)

    def _make_slider_input(self, label, description, min_val, max_val) -> HBox:
        label_widget = Label(description)
        slider = widgets.IntSlider(
            continuous_update=True,
            min=min_val,
            max=max_val,
            readout=False,
            layout={
                "width": "70%",
                "overflow": "auto",
            },
        )
        self._input[label] = IntText(continuous_update=True, layout={"width": "80px"})
        widgets.link((self._input[label], "value"), (slider, "value"))
        return HBox([label_widget, slider, self._input[label]])

    def viz_routes(self):
        if self.results == None:
            self.route_images = []
            return self.route_images
        all_succ_routes = self.results["all_succ_routes"]
        self.route_images = []
        for route in tqdm(all_succ_routes):
            route_cls_viz = copy_route_tree(route)
            route_img = route_cls_viz.viz_graph_route()
            self.route_images.append(route_img.to_image())
        return self.route_images


if __name__ == "__main__":

    smiles = "c2ccc(C1CCCCCC1)cc2"
    # smiles = 'CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1'
    setup_logger()
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    planner = RSPlanner(config)
    planner.select_stock("RetroStar-stock")
    planner.select_one_step_model(
        [
            "template_relevance.reaxys",
            # "graphfp_models.PaRoutes_benchmark_set-n1",
            "template_relevance.pistachio",
            "onmt_models.bionav_one_step",
        ]
    )
    planner.select_condition_predictor("rcr")
    planner.select_pathway_ranker('tree_lstm_ranker')
    planner.prepare_plan()
    result = planner.plan(smiles)
    print(result)
    # route_with_condition = planner.predict_condition()
    # print(route_with_condition)

    route_with_rxn_attributes = planner.predict_rxn_attributes()
    print(route_with_rxn_attributes)

    for route_attributes in route_with_rxn_attributes:
        for rxn_smiles, rxn_attributes in route_attributes.items():
            enzyme_assign_df = rxn_attributes["enzyme_assign"]
            top1_ec = enzyme_assign_df["EC Number"].tolist()[0]
            planner.download_enzyme_pdb_by_ec_and_pred_active_site(
                ec_number=top1_ec, rxn_smiles=rxn_smiles
            )

            pass
    # def viz_routes(results):
    #     if results == None:
    #         route_images = []
    #         return route_images
    #     all_succ_routes = results['all_succ_routes']
    #     route_images = []
    #     for route in tqdm(all_succ_routes):
    #         route_cls_viz = copy_route_tree(route)
    #         route_img = route_cls_viz.viz_graph_route()
    #         route_images.append(route_img.to_image())
    #     return route_images

    # viz_routes(result)
