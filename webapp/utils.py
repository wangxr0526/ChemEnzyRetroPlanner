import json
import os
import random
import shutil
import zipfile
from Bio.PDB import PDBParser
from typing import Dict, List
import uuid
import pandas as pd
from images import get_structure_html_and_active_data, smitosvg_url, reactiontosvg_url

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

DEBUG = False
RXN_ATTR_NAMES = ["condition", "organic_enzyme_rxn_classification", "enzyme_assign"]
TABLE_FLOAT_ROUND = 4
EMPTY_ATTR = {
    "condition": pd.DataFrame(
        columns=["Temperature", "Solvent", "Reagent", "Catalyst", "Score"]
    ).to_json(),
    "organic_enzyme_rxn_classification": pd.DataFrame(
        columns=["Reaction Type", "Confidence"]
    ).to_json(),
    "enzyme_assign": pd.DataFrame(
        columns=["Ranks", "EC Number", "Confidence"]
    ).to_json(),
}


def condition_handle_fn(meta_dict):
    df = pd.read_json(meta_dict)
    df = df.round(TABLE_FLOAT_ROUND)
    df["Score"] = df["Score"].apply(lambda x: f"{x:.{TABLE_FLOAT_ROUND}f}")
    df = df[["Solvent", "Reagent", "Catalyst", "Temperature", "Score"]]
    return {
        "table": df.to_html(index=False).replace(
            '<table border="1" class="dataframe">\n',
            "<table>\n<caption>Reaction Condition Recommendation Results</caption>\n",
        )
    }
    # return {'table':df.to_html(index=False)}


def eznymatic_rxn_identification_handle_fn(meta_dict, debug=DEBUG):
    df = pd.read_json(meta_dict)
    df = df.round(TABLE_FLOAT_ROUND)
    if debug:
        debug_chage_to_enzymatic_rxn = random.choice([True, False])

        if debug_chage_to_enzymatic_rxn:
            df["Reaction Type"] = ["Enzymatic Reaction"] * len(df)

    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.{TABLE_FLOAT_ROUND}f}")
    try:
        reaction_type = df["Reaction Type"].tolist()[0]
        if reaction_type == "Organic Reaction":
            enzyme_recommend = False
        else:
            enzyme_recommend = True
    except:
        enzyme_recommend = False
    meta_data = {
        "table": df.to_html(index=False).replace(
            '<table border="1" class="dataframe">\n',
            "<table>\n<caption>Enzymatic Reaction Identification Results</caption>\n",
        ),
        "enzyme_recommend": enzyme_recommend,
    }
    return meta_data


def eznyme_recommendation_handle_fn(meta_dict):
    df = pd.read_json(meta_dict)
    df = df.round(TABLE_FLOAT_ROUND)
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.{TABLE_FLOAT_ROUND}f}")
    return {
        "table": df.to_html(index=False).replace(
            '<table border="1" class="dataframe">\n',
            "<table>\n<caption>Enzyme Recommendation Results</caption>\n",
        ),
        "ec": df["EC Number"].tolist(),
    }


ATTR_HANDLE_FUNCTION_DICT = {
    "condition": condition_handle_fn,
    "organic_enzyme_rxn_classification": eznymatic_rxn_identification_handle_fn,
    "enzyme_assign": eznyme_recommendation_handle_fn,
}


def easifa_placeholder_fn(rxn_smiles, pdb_fpath):
    parser = PDBParser()
    structure = parser.get_structure("PDB", pdb_fpath)

    # 假设我们只处理第一个模型
    model = structure[0]

    aa_idx = []
    sample_number = random.choice([3, 5, 7, 9, 11, 25])
    fake_active_data = []
    idx = 0
    for chain in model:
        for residue in chain:
            aa_idx.append(idx)
            idx += 1
    aa_active = random.sample(aa_idx, k=sample_number)
    idx = 0
    for chain in model:
        for residue in chain:
            if idx in aa_active:
                fake_active_data.append(random.randint(1, 3))
            else:
                fake_active_data.append(0)
            idx += 1

    return fake_active_data


def color_cell(str_col, color_col):
    return f'<div style="color:{color_col}">{str_col}</div>'


def strong_cell(str_col):
    return f"<strong>{str_col}</strong>"


def convert_easifa_results(site_labels, pdb_fpath, view_size=(790, 600), add_style=True):
    structure_html, active_data = get_structure_html_and_active_data(
        pdb_fpath, site_labels=site_labels, view_size=view_size
    )

    active_data_df = pd.DataFrame(
        active_data, columns=["Residue Index", "Residue Name", "Color", "Active Type"]
    )
    if not active_data_df.empty:
        if add_style:
            active_data_df["Active Type"] = active_data_df.apply(
                lambda row: color_cell(row["Active Type"], row["Color"]), axis=1
            )
        active_data_df = active_data_df[
            ["Residue Index", "Residue Name", "Active Type"]
        ]
        if add_style:
            for col in active_data_df.columns.tolist():
                active_data_df[col] = active_data_df[col].apply(lambda x: strong_cell(x))
    return structure_html, active_data_df


def enzyme_active_predict_placeholder(ec_fake_list, rxn_smiles):
    import random

    pdb_fnames = [x for x in os.listdir("data/test_enzyme") if x.endswith(".pdb")]

    active_data_placeholder = []
    pdb_fnames_sample = random.sample(pdb_fnames, k=3)

    # easifa_placeholder_fn = lambda x: []
    enzyme_data = []
    for idx, pdb_fname in enumerate(pdb_fnames_sample):
        pdb_fpath = os.path.abspath(os.path.join("data/test_enzyme", pdb_fname))

        alphafolddb_id = pdb_fname.replace("-F1-model_v4.pdb", "").replace("AF-", "")

        active_data_placeholder = easifa_placeholder_fn(rxn_smiles, pdb_fpath)
        structure_html, active_data = get_structure_html_and_active_data(
            pdb_fpath,
            site_labels=active_data_placeholder,
            view_size=(395, 300),
            debug=False,
        )

        active_data_df = pd.DataFrame(
            active_data,
            columns=["Residue Index", "Residue Name", "Color", "Active Type"],
        )

        # active_data_df['Residue Name'] = active_data_df.apply(lambda row: color_cell(row['Residue Name'], row['Color']), axis=1)
        active_data_df["Active Type"] = active_data_df.apply(
            lambda row: color_cell(row["Active Type"], row["Color"]), axis=1
        )

        active_data_df = active_data_df[
            ["Residue Index", "Residue Name", "Active Type"]
        ]

        for col in active_data_df.columns.tolist():
            active_data_df[col] = active_data_df[col].apply(lambda x: strong_cell(x))

        enzyme_data.append(
            {
                "id": idx + 1,
                "structure_html": structure_html,
                "active_data": active_data_df.to_html(
                    index=False, escape=False
                ).replace(
                    '<table border="1" class="dataframe">\n',
                    f"<table>\n<caption>Predicted Active Sites</caption>\n",
                ),
                "ec": ec_fake_list[0],
                "alphafolddb_id": alphafolddb_id,
            }
        )

    return enzyme_data


def reaction_attribute_to_meta(rxn_attributes: Dict):

    rxn_attributes_meta = {}
    if rxn_attributes == EMPTY_ATTR:
        rxn_attributes_meta["empty"] = True
    else:
        rxn_attributes_meta["empty"] = False
    for attr in RXN_ATTR_NAMES:
        meta_data = ATTR_HANDLE_FUNCTION_DICT[attr](rxn_attributes[attr])
        rxn_attributes_meta[attr] = meta_data
    return rxn_attributes_meta


def process_node(node, nodes, edges, parent_id=None, parent_smiles=None):
    node_id = str(uuid.uuid4())

    if node["type"] == "mol":
        if node.get("is_root", False):
            node_color = "#2B7CE9"
            node_size = 60
        elif not node["in_stock"]:
            node_color = "#FF4500"
            node_size = 40
        else:
            node_color = "#ADFF2F"
            node_size = 40

        nodes.append(
            {
                "id": node_id,
                "type": "mol",
                "in_stock": node["in_stock"],
                "shape": "circularImage",
                "image": smitosvg_url(node["smiles"], molSize=(300, 300)),
                "title": node["smiles"],
                "size": node_size,
                "widthConstraint": {"minimum": 100, "maximum": 100},
                "heightConstraint": {"minimum": 100, "maximum": 100},
                "borderWidth": 2,
                "color": node_color,
            }
        )

        # 更新 parent_smiles 以供子节点反应节点使用
        parent_smiles = node["smiles"]

    elif node["type"] == "reaction":
        # 处理反应节点
        reaction_smiles = node.get("rxn_smiles", None)

        if reaction_smiles is None:

            reactants_smiles = []
            product_smiles = parent_smiles

            for child in node.get("children", []):
                if child["type"] == "mol":
                    reactants_smiles.append(child["smiles"])

            reaction_smiles = "{}>>{}".format(
                ".".join(reactants_smiles), product_smiles
            )

        reaction_attributes = node.get("rxn_attribute", EMPTY_ATTR)
        reaction_attributes_meta = reaction_attribute_to_meta(reaction_attributes)

        enzyme_recommend = reaction_attributes_meta['organic_enzyme_rxn_classification']['enzyme_recommend']
        nodes.append(
            {
                "id": node_id,
                "type": "reaction",
                # 'label': reaction_smiles,
                "shape": "circle",
                "color": "#FFFF00" if not enzyme_recommend else "#54ff00",
                "size": 10,
                "reaction_smiles": reaction_smiles,
                "reaction_attribute": reaction_attributes_meta,
                "image": reactiontosvg_url(reaction_smiles, molSize=(900, 300)),
                # 'enzyme_data': enzyme_data,
            }
        )

    if parent_id:
        # 添加边从父节点到当前节点
        edges.append(
            {
                "from": parent_id,
                "to": node_id,
                "arrows": "to",
                "color": "#808080",
                "length": 10,
            }
        )

    # 递归处理所有子节点
    for child in node.get("children", []):
        process_node(
            child, nodes, edges, parent_id=node_id, parent_smiles=parent_smiles
        )


def route_to_network_meta(route: Dict):
    nodes = []
    edges = []
    route["is_root"] = True
    process_node(route, nodes, edges)
    return nodes, edges


def routes_to_network_meta(routes: List[Dict]):

    routes_meta = []
    for idx, route in enumerate(routes):
        nodes, edges = route_to_network_meta(route)
        routes_meta.append(
            {
                "id": idx + 1,  # 1 based
                "nodes": nodes,
                "edges": edges,
            }
        )
    return routes_meta


class IntereactionResults:
    def __init__(
        self, template_path="./data/interaction_results_template", network_data=None
    ) -> None:

        self.template_path = os.path.abspath(template_path)

        self.replace_point = "{{ routes_meta| tojson }}"

        self.network_data = network_data
        pass

    def write_html(self, html_save_path):
        self.html_save_path = html_save_path
        with open(
            os.path.join(self.template_path, "Interaction_results_tpl.html"), "r"
        ) as f:
            template_html = f.read()

        results_html = template_html.replace(
            self.replace_point, json.dumps(self.network_data)
        )

        with open(html_save_path, "w") as f:
            f.write(results_html)

    # def pack_results(self, zip_file_path):
    #     with zipfile.ZipFile(zip_file_path, "w") as zipf:
    #         zipf.write(self.html_save_path)
    #     os.remove(self.html_save_path)

    def get_intereaction_results(self, html_save_path):
        self.write_html(html_save_path)
        # self.pack_results(zip_file_path)


class RouteResultsPacker(IntereactionResults):
    def __init__(
        self, template_path="./data/interaction_results_template", route_save_path ='./data/synthesis_routes', results_id=None
    ) -> None:
        self.route_save_path = route_save_path
        self.results_id = results_id
        if os.path.exists(os.path.join(route_save_path, f'{results_id}.json')):
            self.route_json_path = os.path.join(route_save_path, f'{results_id}.json')
        else:
            self.route_json_path = os.path.join(route_save_path, f'locked_{results_id}.json')

        with open(self.route_json_path, "r") as f:
            routes = json.load(f)
        routes_meta = routes_to_network_meta(routes)
        super().__init__(template_path, routes_meta)

    def pack_route_results(self):
        cache_folder = os.path.join(os.path.dirname(self.route_json_path), f'.cache_for_{self.results_id}')
        os.makedirs(cache_folder, exist_ok=True)
        intereaction_results = os.path.join(cache_folder, "intereaction_results.html"
        )
        self.get_intereaction_results(intereaction_results)
        files_to_zip = [self.route_json_path, intereaction_results]
        with zipfile.ZipFile(os.path.join(self.route_save_path, f'{self.results_id}.zip'), 'w') as zipf:
            for file_path in files_to_zip:
                # 使用os.path.basename获取文件名
                file_name = os.path.basename(file_path)
                # 添加文件到ZIP，arcname设为文件名以去除文件原始的文件夹路径
                zipf.write(file_path, arcname=file_name)
        try:
            shutil.rmtree(cache_folder)
        except FileNotFoundError:
            print("Folder does not exist.")
        except PermissionError:
            print("You do not have permission to delete this folder.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
            
def is_valid_ec_number(ec_number: str) -> bool:
    """
    Validate the given EC number format, allowing for incomplete categorization with '-'.
    
    Args:
    ec_number (str): The EC number to validate.

    Returns:
    bool: True if the EC number is valid, False otherwise.
    """
    # Split the EC number by the dot
    parts = ec_number.split('.')
    
    # Check if it has exactly four parts
    if len(parts) != 4:
        return False
    
    # Check if each part is a non-negative integer or a hyphen (for the last part)
    for index, part in enumerate(parts):
        if not (part.isdigit() or (part == '-' and index == 3)):
            return False

    return True


def check_api_input(
        api_input: dict, 

        single_step_opt: list,
        stock_opt: list, 
        condition_opt: list
        ):
    # Validation for the 'retroplanner' API type

    required_keys = {'smiles', 'savedOptions'}
    if not required_keys.issubset(api_input.keys()):
        missing = required_keys - api_input.keys()
        return False, f"Missing keys in input: {', '.join(missing)}"

    # Check if options within 'savedOptions' are correct
    options = api_input.get('savedOptions', {})
    if not isinstance(options, dict):
        return False, "savedOptions must be a dictionary."
    
    for option, expected_values in [
        ('selectedModels', single_step_opt), 
        ('selectedStocks', stock_opt), 
        ('selectedConditionPredictor', condition_opt)
    ]:
        if option not in options:
            continue
        option_input = options[option]
        if isinstance(option_input, str):
            option_input = [option_input]
        if not all(item in expected_values for item in option_input):
            return False, f"Invalid value in {option}. Available options are {expected_values}."
    

    return True, "Input is valid."


# 用于生成RSA密钥对
def generate_rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem

def is_key_pair(private_key_data, public_key_pem):
    # 加载私钥并验证公钥
    try:
        private_key = serialization.load_pem_private_key(
            private_key_data,
            password=None,
            backend=default_backend()
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        if public_key == public_key_pem:
            return True
        else:
            return False
    except:
        return False


if __name__ == "__main__":
    # with open('data/test_routes/test_route0.json', 'r') as f:
    #     route = json.load(f)

    # nodes = []
    # edges = []
    # route['is_root'] = True
    # process_node(route, nodes, edges)
    # with open(
    #     "./data/synthesis_routes/db654ca6-6d51-4f8b-b986-d6bf998f31a6.json", "r"
    # ) as f:
    #     routes = json.load(f)
    # routes_meta = routes_to_network_meta(routes)

    # intereaction_handler = IntereactionResults(network_data=routes_meta)

    # test_save_path = os.path.join("./data/download/intereaction_routes", "test")
    # # os.makedirs(test_save_path, exist_ok=True)
    # # intereaction_handler.write_html(
    # #     os.path.join(test_save_path, "intereaction_results.html")
    # # )

    # intereaction_handler.get_intereaction_results(
    #     os.path.join(test_save_path, "test.zip")
    # )

    results_packer = RouteResultsPacker(results_id='db654ca6-6d51-4f8b-b986-d6bf998f31a6')
    results_packer.pack_route_results()
    pass
