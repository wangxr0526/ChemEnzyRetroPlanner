import os
import sys
import subprocess
import time
import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), "..")))
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch
import re
import json
from tqdm.auto import tqdm
from functools import partial
import py3Dmol
from rdkit.Chem import rdChemReactions, DataStructs
from torch.utils import data as torch_data
from easifa.common.utils import convert_fn, cuda, read_model_state
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from easifa.data_loaders.enzyme_dataloader import MyProtein
from easifa.data_loaders.enzyme_rxn_dataloader import (
    enzyme_rxn_collate_extract,
    atom_types,
    get_adm,
    ReactionFeatures,
)
from rdkit import Chem
from easifa.model_structure.enzyme_site_model import (
    EnzymeActiveSiteClsModel,
    EnzymeActiveSiteModel,
)

chebi_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "chebi", "structures.csv.gz")
)
pdb_cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pdb_cache"))
uniprot_csv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "uniprot_csv")
)
uniprot_json_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "uniprot_json")
)

uniprot_rxn_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "uniprot_rxn")
)
os.makedirs(pdb_cache_path, exist_ok=True)
os.makedirs(uniprot_csv_path, exist_ok=True)
os.makedirs(uniprot_json_path, exist_ok=True)
os.makedirs(uniprot_rxn_path, exist_ok=True)


dirpath = os.path.abspath(os.path.dirname(__file__))

default_ec_site_model_state_path = "../checkpoints/enzyme_site_type_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2023-06-14-11-04-55/global_step_70000"
default_ec_site_model_state_path = os.path.abspath(
    os.path.join(dirpath, default_ec_site_model_state_path)
)

full_swissprot_checkpoint_path = "../checkpoints/enzyme_site_type_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_3_at_2023-12-19-16-06-42/global_step_284000"
full_swissprot_checkpoint_path = os.path.abspath(
    os.path.join(dirpath, full_swissprot_checkpoint_path)
)

rxn_model_path = "../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25"
rxn_model_path = os.path.abspath(os.path.join(dirpath, rxn_model_path))

mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)


def calculate_rxn_diff_fps(rxn):
    rdrxn = rdChemReactions.ReactionFromSmarts(rxn)
    return rdChemReactions.CreateDifferenceFingerprintForReaction(rdrxn)


def calculate_similarity(x, y_list):
    return DataStructs.BulkTanimotoSimilarity(x, y_list)


label2active_type = {
    0: None,
    1: "Binding Site",
    2: "Catalytic Site",  # Active Site in UniProt
    3: "Other Site",
}


def cmd(command):
    subp = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    subp.wait()
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print(f"{command} Failure!")


def svg2file(fname, svg_text):
    with open(fname, "w") as f:
        f.write(svg_text)


def reaction2svg(reaction, path):
    # smi = ''.join(smi.split(' '))
    # mol = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DSVG(1500, 500)
    # opts = d.drawOptions()
    # opts.padding = 0  # 增加边缘的填充空间
    # opts.bondLength = -5  # 如果需要，可以调整键的长度
    # opts.atomLabelFontSize = 5  # 调整原子标签字体大小
    # opts.additionalAtomLabelPadding = 0  # 增加原子标签的额外填充空间

    d.DrawReaction(reaction)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', '').replace('y=\'0.0\'>', 'y=\'0.0\' fill=\'rgb(255,255,255,0)\'>')  # 使用replace将原始白色的svg背景变透明
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', 'rgb(255,255,255,0)')
    svg2 = svg.replace("svg:", "")
    svg2file(path, svg2)
    return "\n".join(svg2.split("\n")[8:-1])


def white_pdb(pdb_lines):
    save_path = os.path.join(pdb_cache_path, "input_pdb.pdb")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pdb_lines))


def get_3dmolviewer_id(html_line1):
    matches = re.search(r'id="([^"]+)"', html_line1)
    if matches:
        return matches.group(1)
    else:
        raise ValueError()
    



def get_structure_html_and_active_data(
    enzyme_structure_path,
    site_labels=None,
    view_size=(900, 900),
    res_colors={
        0: "#73B1FF",  # 非活性位点
        1: "#FF0000",  # Binding Site
        2: "#00B050",  # Active Site
        3: "#FFFF00",  # Other Site
    },
    show_active=True,
):
    with open(enzyme_structure_path) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(width=view_size[0], height=view_size[1])
    view.addModelsAsFrames(system)

    active_data = []

    if show_active and (site_labels is not None):
        i = 0
        res_idx = None
        for line in system.split("\n"):
            split = line.split()
            if len(split) == 0 or split[0] != "ATOM":
                continue
            if res_idx is None:
                first_res_idx = int(line[22:26].strip())
            res_idx = int(line[22:26].strip()) - first_res_idx
            color = res_colors[site_labels[res_idx]]
            view.setStyle({"model": -1, "serial": i + 1}, {"cartoon": {"color": color}})
            atom_name = line[12:16].strip()
            if (atom_name == "CA") and (site_labels[res_idx] != 0):
                residue_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                view.addLabel(
                    f"{residue_name} {res_idx + 1}",
                    {
                        "fontSize": 15,
                        "position": {"x": x, "y": y, "z": z},
                        "fontColor": color,
                        "fontOpacity": 1.0,
                        "backgroundColor": "white",
                        "bold": True,
                        "backgroundOpacity": 0.2,
                    },
                )
                active_data.append(
                    (
                        res_idx + 1,
                        residue_name,
                        color,
                        label2active_type[site_labels[res_idx]],
                    )
                )  # 设置label从1开始#

            i += 1
    else:
        view.setStyle({"model": -1}, {"cartoon": {"color": res_colors[0]}})
    # view.addSurface(py3Dmol.SAS, {'opacity': 0.5})
    view.zoomTo()
    # view.show()
    view.zoom(2.5, 600)
    return view.write_html(), active_data


class UniProtParser:
    def __init__(self, chebi_path, json_folder, rxn_folder, alphafolddb_folder):

        self.json_folder = json_folder
        os.makedirs(self.json_folder, exist_ok=True)
        self.rxn_foder = rxn_folder
        os.makedirs(self.rxn_foder, exist_ok=True)
        self.alphafolddb_folder = alphafolddb_folder
        os.makedirs(self.alphafolddb_folder, exist_ok=True)

        self.chebi_df = pd.read_csv(chebi_path)
        # self.query_uniprotkb_template = "curl  -o {} -H \"Accept: text/plain; format=tsv\" \"https://rest.uniprot.org/uniprotkb/search?query=accession:{}&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb,ft_binding,ft_act_site,ft_site\""
        self.query_uniprotkb_template = '/usr/bin/curl  -o {} -H "Accept: application/json" "https://rest.uniprot.org/uniprotkb/search?query=accession:{}&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb,ft_binding,ft_act_site,ft_site"'

        self.rhea_rxn_url_template = "/usr/bin/curl -o {} https://ftp.expasy.org/databases/rhea/ctfiles/rxn/{}.rxn"
        self.download_alphafolddb_url_template = "/usr/bin/curl -o {} https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

    def _qurey_smiles_from_chebi(self, chebi_id):
        this_id_data = self.chebi_df.loc[self.chebi_df["COMPOUND_ID"] == int(chebi_id)]
        this_id_data = this_id_data.loc[this_id_data["TYPE"] == "SMILES"]
        smiles = this_id_data["STRUCTURE"].tolist()[0]
        return smiles

    def _canonicalize_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return ""

    def _get_reactions(self, query_data):
        reaction_comments = [
            comment["reaction"]
            for comment in query_data["comments"]
            if comment["commentType"] == "CATALYTIC ACTIVITY"
        ]

        rxn_smiles_list = []
        ec_number_list = []
        for (
            rxn_comment
        ) in (
            reaction_comments
        ):  # (ˉ▽ˉ；)... 这个chebi_id的顺序并不一定按照name的顺序来记录，需要从交叉引用中直接找到反应信息，坑死
            if "reactionCrossReferences" in rxn_comment:
                enzyme_lib_info = [
                    ref["id"].split(":")
                    for ref in rxn_comment["reactionCrossReferences"]
                    if ref["database"] != "ChEBI"
                ]
                for enzyme_lib_name, enzyme_lib_id in enzyme_lib_info:
                    if enzyme_lib_name == "RHEA":
                        rxn_fpath = os.path.join(
                            self.rxn_foder, f"{str(int(enzyme_lib_id)+1)}.rxn"
                        )
                        if not os.path.exists(rxn_fpath):
                            cmd(
                                self.rhea_rxn_url_template.format(
                                    os.path.abspath(rxn_fpath),
                                    str(int(enzyme_lib_id) + 1),
                                )
                            )
                        try:
                            rxn = AllChem.ReactionFromRxnFile(rxn_fpath)
                            reaction_smiles = AllChem.ReactionToSmiles(rxn)
                            reactants_smiles, products_smiles = reaction_smiles.split(
                                ">>"
                            )
                            reactants_smiles = self._canonicalize_smiles(
                                reactants_smiles
                            )
                            products_smiles = self._canonicalize_smiles(products_smiles)
                            if "" not in [reactants_smiles, products_smiles]:
                                rxn_smiles = f"{reactants_smiles}>>{products_smiles}"
                                rxn_smiles_list.append(rxn_smiles)
                                if "ecNumber" in rxn_comment:
                                    ec_number_list.append(rxn_comment["ecNumber"])
                                else:
                                    ec_number_list.append("UNK")
                        except:
                            continue

        return [x for x in zip(ec_number_list, rxn_smiles_list)]

    def parse_from_uniprotkb_query(self, uniprot_id):

        uniprot_data_fpath = os.path.join(self.json_folder, f"{uniprot_id}.json")
        query_uniprotkb_cmd = self.query_uniprotkb_template.format(
            os.path.abspath(uniprot_data_fpath), uniprot_id
        )
        print(query_uniprotkb_cmd)
        if not os.path.exists(uniprot_data_fpath):
            cmd(query_uniprotkb_cmd)
        with open(uniprot_data_fpath, "r") as f:
            query_data = json.load(f)["results"][0]
        # query_data = pd.read_csv(f'test/{query_id}.tsv', sep='\t')

        try:
            ecNumbers = query_data["proteinDescription"]["recommendedName"]["ecNumbers"]
        except:
            ecNumbers = []
            # return None, 'Not Enzyme'
        try:
            alphafolddb_id = query_data["uniProtKBCrossReferences"][0]["id"]
        except:
            return None, "No Alphafolddb Structure"
        aa_length = query_data["sequence"]["length"]
        pdb_fpath = os.path.join(
            self.alphafolddb_folder, f"AF-{alphafolddb_id}-F1-model_v4.pdb"
        )
        if not os.path.exists(pdb_fpath):
            cmd(
                self.download_alphafolddb_url_template.format(
                    os.path.abspath(pdb_fpath), alphafolddb_id
                )
            )

        ec2rxn_smiles = self._get_reactions(query_data)

        if (len(ecNumbers) == 0) and (len(ec2rxn_smiles) == 0):
            return None, "Not Enzyme"
        elif (len(ec2rxn_smiles) == 0) and (len(ecNumbers) != 0):
            return None, "No recorded reaction catalyzed found"

        df = pd.DataFrame(ec2rxn_smiles, columns=["ec", "rxn_smiles"])
        df["pdb_fpath"] = [pdb_fpath for _ in range(len(df))]
        df["aa_length"] = [aa_length for _ in range(len(df))]

        return df, "Good"


class UniProtParserEC:
    def __init__(
        self,
        json_folder,
        csv_folder,
        alphafolddb_folder,
        download_size=5,
        download_time=60,
        chebi_path=None,
        rxn_folder=None,
    ) -> None:

        self.download_size = download_size
        self.download_time = download_time

        self.chebi_path = chebi_path
        if self.chebi_path:
            if not os.path.exists(self.chebi_path):
                self._download_chebi_file(self.chebi_path)

        self.json_folder = os.path.abspath(json_folder)
        os.makedirs(self.json_folder, exist_ok=True)
        self.csv_folder = os.path.abspath(csv_folder)
        os.makedirs(self.csv_folder, exist_ok=True)
        self.alphafolddb_folder = os.path.abspath(alphafolddb_folder)
        os.makedirs(self.alphafolddb_folder, exist_ok=True)
        if rxn_folder:
            self.rxn_folder = os.path.abspath(rxn_folder)
            os.makedirs(self.alphafolddb_folder, exist_ok=True)

        # self.query_uniport_template = '/usr/bin/curl -s -o {} -H \"Accept: text/plain; format=tsv\" \"https://rest.uniprot.org/uniprotkb/search?query=ec:{}+AND+existence:xref_alphafolddb&fields=accession,ec,sequence,xref_alphafolddb&size={}\"'
        self.query_uniport_template = '/usr/bin/curl -s -o {} -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/stream?query=ec:{}+AND+reviewed:true+AND+database:(alphafolddb)&fields=accession,ec,sequence,xref_alphafolddb&size={}"'
        self.query_uniport_json_template = '/usr/bin/curl  -o {} -H "Accept: application/json" "https://rest.uniprot.org/uniprotkb/search?query=ec:{}+AND+reviewed:true+AND+database:(alphafolddb)&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb"'
        self.download_alphafolddb_url_template = "/usr/bin/curl -s -o {} https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
        self.rhea_rxn_url_template = "/usr/bin/curl -o {} https://ftp.expasy.org/databases/rhea/ctfiles/rxn/{}.rxn"
        
    def _download_chebi_file(self, output_path):
        url = "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/structures.csv.gz"

        # 下载文件
        response = requests.get(url, stream=True)

        # 检查响应状态
        if response.status_code == 200:
            # 打开文件并写入内容
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("文件下载完成！")
        else:
            print(f"下载失败，状态码：{response.status_code}")

    def query_enzyme_pdb_by_ec(self, ec_number, size):

        select_data_df = pd.DataFrame()
        start_time = time.time()
        end_time = time.time()
        while (select_data_df.empty) and (end_time - start_time < self.download_time):

            uniprot_data_fpath = os.path.join(self.csv_folder, f"{ec_number}.csv")
            query_uniprotkb_cmd = self.query_uniport_template.format(
                os.path.abspath(uniprot_data_fpath), ec_number, self.download_size
            )
            if not os.path.exists(uniprot_data_fpath):
                cmd(query_uniprotkb_cmd)

            data_df = pd.read_csv(uniprot_data_fpath, sep="\t")
            try:
                data_df = data_df.sample(n=size * 2)
            except:
                data_df = data_df.sample(n=size)
            data_df = data_df.loc[~data_df["AlphaFoldDB"].isna()]
            data_df = data_df.loc[data_df["Sequence"].apply(lambda x: len(x)) <= 600]
            data_df["AlphaFoldDB"] = data_df["AlphaFoldDB"].apply(
                lambda x: x.split(";")[0]
            )

            select_data_df = data_df.sample(n=size) if not data_df.empty else data_df
            end_time = time.time()
            if not select_data_df.empty:
                break

        if select_data_df.empty:
            return

        pdb_fpath_list = []

        for alphafolddb_id in select_data_df["AlphaFoldDB"].tolist():
            pdb_fpath = os.path.join(
                self.alphafolddb_folder, f"AF-{alphafolddb_id}-F1-model_v4.pdb"
            )
            if not os.path.exists(pdb_fpath):
                cmd(
                    self.download_alphafolddb_url_template.format(
                        os.path.abspath(pdb_fpath), alphafolddb_id
                    )
                )
            pdb_fpath_list.append(pdb_fpath)

        select_data_df["pdb_fpath"] = pdb_fpath_list

        return select_data_df
    
    def _canonicalize_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return ""

    def _get_reactions(self, query_data):
        reaction_comments = [
            comment["reaction"]
            for comment in query_data["comments"]
            if comment["commentType"] == "CATALYTIC ACTIVITY"
        ]

        rxn_smiles_list = []
        ec_number_list = []
        for (
            rxn_comment
        ) in (
            reaction_comments
        ):  # (ˉ▽ˉ；)... 这个chebi_id的顺序并不一定按照name的顺序来记录，需要从交叉引用中直接找到反应信息，坑死
            if "reactionCrossReferences" in rxn_comment:
                enzyme_lib_info = [
                    ref["id"].split(":")
                    for ref in rxn_comment["reactionCrossReferences"]
                    if ref["database"] != "ChEBI"
                ]
                for enzyme_lib_name, enzyme_lib_id in enzyme_lib_info:
                    if enzyme_lib_name == "RHEA":
                        rxn_fpath = os.path.join(
                            self.rxn_folder, f"{str(int(enzyme_lib_id)+1)}.rxn"
                        )
                        if not os.path.exists(rxn_fpath):
                            cmd(
                                self.rhea_rxn_url_template.format(
                                    os.path.abspath(rxn_fpath),
                                    str(int(enzyme_lib_id) + 1),
                                )
                            )
                        try:
                            rxn = AllChem.ReactionFromRxnFile(rxn_fpath)
                            reaction_smiles = AllChem.ReactionToSmiles(rxn)
                            reactants_smiles, products_smiles = reaction_smiles.split(
                                ">>"
                            )
                            reactants_smiles = self._canonicalize_smiles(
                                reactants_smiles
                            )
                            products_smiles = self._canonicalize_smiles(products_smiles)
                            if "" not in [reactants_smiles, products_smiles]:
                                rxn_smiles = f"{reactants_smiles}>>{products_smiles}"
                                rxn_smiles_list.append(rxn_smiles)
                                if "ecNumber" in rxn_comment:
                                    ec_number_list.append(rxn_comment["ecNumber"])
                                else:
                                    ec_number_list.append("UNK")
                        except:
                            continue

        return [x for x in zip(ec_number_list, rxn_smiles_list)]

    def query_enzyme_pdb_by_ec_with_rxn_ranking(self, ec_number, rxn_smiles, topk=1):

        uniprot_data_fpath = os.path.join(self.json_folder, f"{ec_number}.json")
        query_uniprotkb_cmd = self.query_uniport_json_template.format(
            os.path.abspath(uniprot_data_fpath), ec_number
        )
        if not os.path.exists(uniprot_data_fpath):
            cmd(query_uniprotkb_cmd)
        with open(uniprot_data_fpath, "r") as f:
            query_data = json.load(f)["results"]
        # query_data = pd.read_csv(f'test/{query_id}.tsv', sep='\t')
        
        all_df = pd.DataFrame()
        
        for one_data in query_data:

                # return None, 'Not Enzyme'
            try:
                alphafolddb_id = one_data["uniProtKBCrossReferences"][0]["id"]
            except:
                continue
            aa_length = one_data["sequence"]["length"]
            pdb_fpath = os.path.join(
                self.alphafolddb_folder, f"AF-{alphafolddb_id}-F1-model_v4.pdb"
            )
            # if not os.path.exists(pdb_fpath):
            #     cmd(
            #         self.download_alphafolddb_url_template.format(
            #             os.path.abspath(pdb_fpath), alphafolddb_id
            #         )
            #     )

            ec2rxn_smiles = self._get_reactions(one_data)

            df = pd.DataFrame(ec2rxn_smiles, columns=["ec", "rxn_smiles"])
            df['EC number'] = [ec_number for _ in range(len(df))]
            df["pdb_fpath"] = [pdb_fpath for _ in range(len(df))]
            df["aa_length"] = [aa_length for _ in range(len(df))]
            df["AlphaFoldDB"] = [alphafolddb_id for _ in range(len(df))]
            all_df = pd.concat([all_df, df], axis=0)
        
        if all_df.empty:
            return all_df
        all_df = all_df.loc[all_df['aa_length']<600].reset_index(drop=True)
        all_df['rxn_similarity'] = calculate_similarity(calculate_rxn_diff_fps(rxn_smiles), all_df['rxn_smiles'].apply(lambda x:calculate_rxn_diff_fps(x)).tolist())
        all_df = all_df.sort_values('rxn_similarity', ascending=False).reset_index(drop=True)

        df_topk = all_df.iloc[:topk]
        for pdb_fpath in df_topk['pdb_fpath'].tolist():
            if not os.path.exists(pdb_fpath):
                cmd(
                    self.download_alphafolddb_url_template.format(
                        os.path.abspath(pdb_fpath), alphafolddb_id
                    )
                )
        return df_topk


class EasIFAInferenceAPI:
    def __init__(
        self,
        device="cpu",
        model_checkpoint_path=default_ec_site_model_state_path,
        max_enzyme_aa_length=600,
    ) -> None:
        self.max_enzyme_aa_length = max_enzyme_aa_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if model_checkpoint_path in [
            default_ec_site_model_state_path,
            full_swissprot_checkpoint_path,
        ]:
            self.convert_fn = lambda x: convert_fn(x)
        else:
            self.convert_fn = lambda x: x.tolist()
        model = EnzymeActiveSiteClsModel(
            rxn_model_path=rxn_model_path, num_active_site_type=4, from_scratch=True
        )

        model_state, _ = read_model_state(model_save_path=model_checkpoint_path)
        model.load_state_dict(model_state)
        print("Loaded checkpoint from {}".format(model_checkpoint_path))
        model.to(self.device)
        model.eval()
        self.model = model

    def _calculate_features(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fgraph = mol_to_graph(
            mol,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        )
        dgraph = get_adm(mol)

        return fgraph, dgraph

    def _calculate_rxn_features(self, rxn):
        try:
            react, prod = rxn.split(">>")

            react_features_tuple = self._calculate_features(react)
            prod_features_tuple = self._calculate_features(prod)

            return react_features_tuple, prod_features_tuple
        except:
            return None

    def _preprocess_one(self, rxn_smiles, enzyme_structure_path):

        protein = MyProtein.from_pdb(enzyme_structure_path)
        # protein = data.Protein.from_pdb(enzyme_structure_path)
        reaction_features = self._calculate_rxn_features(rxn_smiles)
        rxn_fclass = ReactionFeatures(reaction_features)
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {
            "protein_graph": protein,
            "reaction_graph": rxn_fclass,
            "protein_sequence": protein.to_sequence(),
        }
        return item

    def _calculate_one_data(self, rxn, enzyme_structure_path):
        data_package = self._preprocess_one(
            rxn_smiles=rxn, enzyme_structure_path=enzyme_structure_path
        )
        self.caculated_sequence = data_package["protein_sequence"]
        if len(self.caculated_sequence) > self.max_enzyme_aa_length:
            return None
        batch_one_data = enzyme_rxn_collate_extract([data_package])
        return batch_one_data

    @torch.no_grad()
    def inference(self, rxn, enzyme_structure_path):
        batch_one_data = self._calculate_one_data(rxn, enzyme_structure_path)
        if batch_one_data is None:
            return

        if self.device.type == "cuda":
            batch_one_data = cuda(batch_one_data, device=self.device)
        try:
            protein_node_logic, _ = self.model(batch_one_data)
        except:
            print(f"erro in this data")
            return
        pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
        pred = self.convert_fn(pred)
        return pred


class ECSiteBinInferenceAPI(EasIFAInferenceAPI):
    def __init__(
        self, device="cpu", model_checkpoint_path=default_ec_site_model_state_path
    ) -> None:
        model_state, model_args = read_model_state(
            model_save_path=model_checkpoint_path
        )
        need_convert = model_args.get("need_convert", False)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if (model_checkpoint_path == default_ec_site_model_state_path) or need_convert:
            self.convert_fn = lambda x: convert_fn(x)
        else:
            self.convert_fn = lambda x: x.tolist()
        model = EnzymeActiveSiteModel(rxn_model_path=rxn_model_path)

        model.load_state_dict(model_state)
        print("Loaded checkpoint from {}".format(model_checkpoint_path))
        model.to(self.device)
        model.eval()
        self.model = model


if __name__ == "__main__":
    # ECSitePred = EasIFAInferenceAPI(model_checkpoint_path=full_swissprot_checkpoint_path)

    uniprot_parser = UniProtParserEC(
        chebi_path=chebi_path,
        json_folder=uniprot_json_path,
        csv_folder=uniprot_csv_path,
        alphafolddb_folder=pdb_cache_path,
        rxn_folder=uniprot_rxn_path,
    )
    # uniprot_parser.query_enzyme_pdb_by_ec(ec_number="1.1.2.3", size=2)
    uniprot_parser.query_enzyme_pdb_by_ec_with_rxn_ranking(
        ec_number="1.1.2.3", rxn_smiles="CCCCCC>>CCCCC"
    )

    pass
