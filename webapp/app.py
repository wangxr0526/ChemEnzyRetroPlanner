import datetime
import json
import os
import sqlite3
import uuid
import pytz
import pandas as pd
import requests
import torch
import yaml
from celery import Celery
from billiard.exceptions import SoftTimeLimitExceeded
from retro_planner.api import RSPlanner, dirpath

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)
from flask_cors import CORS

from utils import (
    RouteResultsPacker,
    check_api_input,
    color_cell,
    convert_easifa_results,
    enzyme_active_predict_placeholder,
    generate_rsa_key_pair,
    is_key_pair,
    is_valid_ec_number,
    route_to_network_meta,
    routes_to_network_meta,
    strong_cell,
)


from retro_planner.common.prepare_utils import (
    handle_one_step_config,
    handle_one_step_path,
    init_rcr,
    prepare_enzymatic_rxn_identifier,
    prepare_enzyme_recommender,
    prepare_filter_policy,
    prepare_multi_single_step,
    PrepareStockDatasetUsingFilter,
)
from retro_planner.common.utils import canonicalize_smiles, proprecess_reactions
from retro_planner.utils.logger import setup_logger

from easifa.interface.utils import (
    EasIFAInferenceAPI,
    UniProtParserEC,
    get_structure_html_and_active_data,
    pdb_cache_path,
    uniprot_csv_path,
    uniprot_json_path,
    uniprot_rxn_path,
    chebi_path,
)
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
app = Flask(__name__, 
            static_url_path='/retroplanner/static',  # 设置静态文件路径前缀
            static_folder='static'  # 静态文件的物理路径
            )
app.secret_key = "supersecretkey"  # 设置一个密钥，用于Flash消息

CORS(app)
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"

celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

_celery_soft_time_limit_s = int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT", "3600"))
if _celery_soft_time_limit_s > 0:
    celery.conf.task_soft_time_limit = _celery_soft_time_limit_s


app.synth_route = os.path.abspath(
    os.path.join(app.root_path, "data", "synthesis_routes")
)

app.saved_configs_path = os.path.abspath(
    os.path.join(app.root_path, "data", "saved_configs")
)

app.enzyme_structure_path = os.path.abspath(
    os.path.join(app.root_path, "data", "enzyme_structures")
)

app.models_registor_name_to_model_name = {
    "graphfp_models.USPTO-full_remapped": "GraphFP (USPTO-Full)",
    # "graphfp_models.PaRoutes_benchmark_set-n1": "GraphFP-Moldels (PaRoutes-Set-n1)",
    # "graphfp_models.PaRoutes_benchmark_set-n5": "GraphFP-Moldels (PaRoutes-Set-n5)",
    "onmt_models.bionav_one_step": "Transformer (USPTO-NPL+BioChem)",
    "template_relevance.pistachio": "Pistachio",
    "template_relevance.pistachio_ringbreaker": "Pistachio Ringbreaker",
    "template_relevance.reaxys": "Reaxys",
    'template_relevance.reaxys_biocatalysis': "Reaxys Biocatalysis",
    "template_relevance.bkms_metabolic": "BKMS Metabolic",
}

app.models_name_to_model_registor_name = {
    v: k for k, v in app.models_registor_name_to_model_name.items()
}
app.stocks_registor_name_to_stocks_name = {
    "Zinc_Fix-stock": "Zinc Buyable + USPTO Substrates",
    "RetroStar-stock": "eMolecules",
    "BioNav-stock": "BioNav stock (benchmark)",
    # "PaRotes_n1-stock": "PaRoutes-Set-n1-stock",
    # "PaRotes_n5-stock": "PaRoutes-Set-n5-stock",
}
app.stocks_name_to_stocks_registor_name = {
    v: k for k, v in app.stocks_registor_name_to_stocks_name.items()
}

app.condition_predictor_registor_name_to_condition_predictor_name = {
    "rcr": "Reaction Condition Recommander",
    "parrot": "Parrot",
}

app.condition_predictor_name_to_condition_predictor_registor_name = {
    v: k
    for k, v in app.condition_predictor_registor_name_to_condition_predictor_name.items()
}

os.makedirs(app.synth_route, exist_ok=True)
os.makedirs(app.enzyme_structure_path, exist_ok=True)
os.makedirs(app.saved_configs_path, exist_ok=True)


KEY_FOLDER = os.path.join(os.path.dirname(__file__), 'keys')
os.makedirs(KEY_FOLDER, exist_ok=True)

# 创建一个时区对象，这里是东八区
eastern_eight_zone = pytz.timezone('Asia/Shanghai')

# 数据库初始化
def init_db():
    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT,
            result TEXT,
            input_method TEXT,
            public_key BLOB,
            submitted_at DATETIME
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


@app.before_first_request
def first_request():
    app.planner_configs = yaml.load(
        open("../retro_planner/config/config.yaml", "r"), Loader=yaml.FullLoader
    )

    app.uniprot_parser = UniProtParserEC(
        json_folder=uniprot_json_path,
        csv_folder=uniprot_csv_path,
        alphafolddb_folder=pdb_cache_path,
        chebi_path=chebi_path,
        rxn_folder=uniprot_rxn_path,
    )
    app.easifa_annotator = EasIFAInferenceAPI(
        model_checkpoint_path=os.path.join(
            dirpath, app.planner_configs["easifa_config"]["checkpoint_path"]
        ),
        device=device,
    )
    selected_one_step_model_configs, _, selected_one_step_model_types = (
        handle_one_step_config(
            model_names=list(app.models_registor_name_to_model_name.keys()),
            one_step_model_configs=app.planner_configs["one_step_model_configs"],
        )
    )
    app.single_step_model_wrapper = prepare_multi_single_step(
        model_configs=handle_one_step_path(
            selected_one_step_model_configs=selected_one_step_model_configs,
            selected_one_step_model_types=selected_one_step_model_types,
        ),
        one_step_model_types=selected_one_step_model_types,
        expansion_topk=app.planner_configs["expansion_topk"],
        # device=device,
        device="cpu",
        use_filter=app.planner_configs["use_filter"],
        keep_score=app.planner_configs["keep_score"],
        filter_path=app.planner_configs["filter_path"],
        weights=[float(model_configs['weight']) for model_configs in selected_one_step_model_configs]
    )

    app.reaction_rater = prepare_filter_policy(
        filter_path=app.planner_configs["filter_path"], device="cpu"
    )

    app.organic_enzymatic_rxn_identifier = prepare_enzymatic_rxn_identifier(
        app.planner_configs["organic_enzyme_rxn_classifier_config"], device="cpu"
    )
    app.enzyme_recommender = prepare_enzyme_recommender(
        app.planner_configs["enzyme_rxn_classifier_config"], device="cpu"
    )

    app.condition_predictor = init_rcr(
        app.planner_configs["condition_config"]["rcr"], dirpath=dirpath
    )
    PrepareStockDatasetUsingFilter(
        stock_config=app.planner_configs["stocks"],
    )


@app.route("/retroplanner/")
def index():
    return render_template("index.html")


@app.route("/retroplanner/contact")
def contact():
    return render_template("contact.html")


@app.route("/retroplanner/help")
def help():
    return render_template("help.html")


@app.route("/retroplanner/queue")
def queue():
    return render_template("queue.html")


@app.route("/retroplanner/services")
def retrosynthesis_planner():
    all_model_names = []
    for model_type in app.planner_configs["one_step_model_configs"]:
        for model_subname in app.planner_configs["one_step_model_configs"][model_type]:
            all_model_names.append(f"{model_type}.{model_subname}")

    all_stock_names = [
        x
        for x in list(app.planner_configs["stocks"].keys())
        if x in app.stocks_registor_name_to_stocks_name
    ]

    all_condition_predictor_names = list(app.planner_configs["condition_config"].keys())

    multi_select_data = {
        "model_options": [
            app.models_registor_name_to_model_name[x] for x in all_model_names
        ],
        "stock_options": [
            app.stocks_registor_name_to_stocks_name[x] for x in all_stock_names
        ],
        "condition_predictor_options": [
            app.condition_predictor_registor_name_to_condition_predictor_name[x]
            for x in all_condition_predictor_names
        ],
    }

    return render_template("input.html", multi_select_data=multi_select_data)


@celery.task(bind=True)
def background_task(self, inputed_data, config:dict, input_method:str="WEB"):

    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()
    succ = False

    target_smiles = canonicalize_smiles(smi=inputed_data["smiles"], clear_map=False)
    if target_smiles == "":
        return_result = "Smiles is Not Valid!"
        c.execute(
            "UPDATE jobs SET result = ?, status = 'FAILED' WHERE job_id = ?",
            (return_result, self.request.id),
        )
        conn.commit()
        conn.close()
        return return_result

    inputed_configs:dict = inputed_data["savedOptions"]

    print(inputed_data)

    setup_logger()
    print("Init configs:")
    print(config)
    if inputed_configs:
        update_config = {
            "iterations": int(inputed_configs.get("iterationNumber", config["iterations"])),
            "keep_search": inputed_configs.get("Keep search after solved one route", config["keep_search"]),
            "use_filter": inputed_configs.get("Use reaction plausibility evaluator", config["use_filter"]),
            "use_depth_value_fn": inputed_configs.get("Use guiding function", config["use_depth_value_fn"]),
            "pred_condition": inputed_configs.get("Predict reaction condition", config["pred_condition"]),
            "organic_enzyme_rxn_classification": inputed_configs.get("Identify enzymatic reactions", config["organic_enzyme_rxn_classification"]),
            "enzyme_assign": inputed_configs.get("Recommend enzymes", config["enzyme_assign"]),
            "stock_limit_dict": inputed_configs.get("stockLimitDict", config["stock_limit_dict"]),
        }

        config.update(update_config)
    else:
        config.update({"iterations": 10})
    if not inputed_configs.get("selectedStocks", []):
        inputed_configs["selectedStocks"] = ["Zinc Buyable + USPTO Substrates"]
    if not inputed_configs.get("selectedModels", []):
        inputed_configs["selectedModels"] = ["Reaxys"]
    if not inputed_configs.get("selectedConditionPredictor", []):
        inputed_configs["selectedConditionPredictor"] = "Reaction Condition Recommander"

    inputed_configs["selectedStocks"] = [
        app.stocks_name_to_stocks_registor_name[x]
        for x in inputed_configs.get("selectedStocks", ["Zinc Buyable + USPTO Substrates"])
    ]
    inputed_configs["selectedModels"] = [
        app.models_name_to_model_registor_name[x]
        for x in inputed_configs.get(
            "selectedModels", ["Reaxys"]
        )
    ]

    inputed_configs["selectedConditionPredictor"] = (
        app.condition_predictor_name_to_condition_predictor_registor_name[
            inputed_configs["selectedConditionPredictor"]
        ]
    )
    try:
        planner = RSPlanner(config)
        planner.select_stocks(inputed_configs["selectedStocks"])
        planner.select_one_step_model(inputed_configs["selectedModels"])
        planner.select_condition_predictor(inputed_configs["selectedConditionPredictor"])
        planner.prepare_plan(prepare_easifa=False)

        result = planner.plan(target_smiles)
    except SoftTimeLimitExceeded:
        succ = False
        result = None
        return_result = "TOO_LARGE_LOCAL"
        c.execute(
            "UPDATE jobs SET result = ?, status = 'Timeout' WHERE job_id = ?",
            (return_result, self.request.id),
        )
        conn.commit()
        conn.close()
        return return_result
    except Exception:
        succ = False
        result = None
    if result:
        succ = True
        # print(result)
        # route_with_condition = planner.predict_condition()
        # print(route_with_condition)

        planner.predict_rxn_attributes()
        # print(route_with_rxn_attributes)

        dict_routes = [route.dict_route for route in result["all_succ_routes"]]

        results_id = self.request.id
        if input_method == "WEB":
            with open(os.path.join(app.synth_route, f"locked_{results_id}.json"), "w") as f:
                json.dump(dict_routes, f)
        elif input_method == "API":
            with open(os.path.join(app.synth_route, f"{results_id}.json"), "w") as f:
                json.dump(dict_routes, f)
        else:
            raise ValueError()

    if succ == True:

        return_result = "See Results"
        c.execute(
            "UPDATE jobs SET result = ?, status = 'SUCCESS' WHERE job_id = ?",
            (return_result, self.request.id),
        )
        conn.commit()
    else:
        # 如果有错误，更新任务为失败状态，并记录错误信息
        return_result = "Search Failed! Please increase the number of iterations."
        c.execute(
            "UPDATE jobs SET result = ?, status = 'FAILED' WHERE job_id = ?",
            (return_result, self.request.id),
        )
        conn.commit()

    conn.close()
    return return_result


@app.route("/retroplanner/calculation", methods=["POST"])
def calculation():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    inputed_data = {k: v for k, v in request.get_json().items()}
    config: dict = app.planner_configs
    task = background_task.delay(inputed_data, config)
    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()
    now = datetime.datetime.now(pytz.utc).astimezone(eastern_eight_zone)
    # 生成RSA密钥对
    private_key, public_key = generate_rsa_key_pair()
    # 保存私钥文件
    private_key_path = os.path.join(KEY_FOLDER, f"{task.id}.pem")
    with open(private_key_path, 'wb') as f:
        f.write(private_key)

    saved_configs = {
        "inputed_configs": inputed_data.get("savedOptions", {}),
        "planner_configs": config,
        "target_smiles": inputed_data.get("smiles", ""),
    }
    with open(os.path.join(app.saved_configs_path, f"{task.id}.json"), "w") as f:
        json.dump(saved_configs, f)
    
    
    c.execute(
        "INSERT INTO jobs (job_id, status, public_key, submitted_at, input_method) VALUES (?, ?, ?, ?, ?)",
        (task.id, "Submitted", public_key, now, "WEB"),
    )
    conn.commit()
    conn.close()
    

    log_data = {
        "status": "success",
        "message": "Data received",
        "results_id": task.id,
        "resultsLimit": inputed_data["savedOptions"].get("resultsLimit", 50),
        "private_key_path": f'/retroplanner/key_downloads/{task.id}',  # 下载链接
        "configs_path": f'/retroplanner/config_downloads/{task.id}',  # 下载链接
    }
    return jsonify(log_data), 202


@app.route("/retroplanner/status/<task_id>")
def task_status(task_id):
    task = background_task.AsyncResult(task_id)
    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()
    c.execute("SELECT status, result, input_method FROM jobs WHERE job_id = ?", (task_id,))
    job = c.fetchone()
    conn.close()
    return jsonify(
        {"results_id": task_id, "status": job[0], "result": job[1], "input_method": job[2]}
    )


# Update job listing to sort by submission time
@app.route("/retroplanner/jobs")
def list_jobs():
    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()

    # 1. 获取所有已提交的 WEB 任务
    c.execute("""
        SELECT job_id, submitted_at
        FROM jobs
        WHERE status = 'Submitted' AND input_method = 'WEB'
    """)
    submitted_jobs = c.fetchall()

    # 2. 找出超过两天的任务，收集其 job_id
    timeout_ids = []
    now = datetime.datetime.now(pytz.utc).astimezone(eastern_eight_zone)
    for job_id, submitted_at_str in submitted_jobs:
        try:
            submitted_time = datetime.datetime.fromisoformat(submitted_at_str)
            if now - submitted_time > datetime.timedelta(days=0.5):
                timeout_ids.append(job_id)
        except Exception as e:
            print(f"⛔ 时间解析失败: {submitted_at_str} ({e})")

    # 3. 更新状态为 'Timeout'
    if timeout_ids:
        c.executemany(
            "UPDATE jobs SET status = 'Timeout' WHERE job_id = ?",
            [(job_id,) for job_id in timeout_ids]
        )
        conn.commit()
        print(f"✅ 已更新 {len(timeout_ids)} 个超时任务为 'Timeout' 状态")

    # 4. 获取所有 WEB 任务供前端展示
    c.execute("""
        SELECT job_id, status, result, submitted_at, input_method
        FROM jobs
        ORDER BY submitted_at DESC
    """)
    jobs = [
        [job_id, status, result, submitted_at.split(".")[0]]
        for job_id, status, result, submitted_at, input_method in c.fetchall()
        if input_method == 'WEB'
    ]

    conn.close()
    return jsonify(jobs)


@app.route("/retroplanner/validate-key", methods=["POST"])
def validate_key():
    # 提取formData中的数据，例如你可能需要提取一个私钥
    private_key_file = request.files.get('privateKey')
    job_id = request.form.get('jobId')
    if private_key_file:
        # 从数据库获取任务的公钥信息
        with sqlite3.connect("jobs.db") as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT public_key, result FROM jobs WHERE job_id = ?', (job_id,))
            task_data = cursor.fetchone()
        # 检查任务是否存在
        if task_data:
            public_key_pem, _ = task_data
            private_key_data = private_key_file.read()
            if is_key_pair(private_key_data=private_key_data, public_key_pem=public_key_pem):
                os.rename(os.path.join(app.synth_route, f"locked_{job_id}.json"), os.path.join(app.synth_route, f"{job_id}.json"))
                return jsonify(success=True)
            else:
                return jsonify(success=False, message="Invalid private key"), 400
        else:
            return jsonify(success=False, message="Invalid private key"), 400
    

@app.route("/retroplanner/locked_results/<results_id>&resultsLimit-<resultsLimit>")
def locked_results(results_id, resultsLimit=None):
    if resultsLimit == None:
        resultsLimit = 50
    print(results_id)
    try:
        with open(os.path.join(app.synth_route, f"{results_id}.json"), "r") as f:
            routes = json.load(f)
        os.rename(os.path.join(app.synth_route, f"{results_id}.json"), os.path.join(app.synth_route, f"locked_{results_id}.json"))
        if resultsLimit == "all":
            routes_meta = routes_to_network_meta(routes)
        else:
            resultsLimit = int(resultsLimit)
            routes_meta = routes_to_network_meta(routes[:resultsLimit])
        return render_template(
            "results.html", routes_meta=routes_meta, results_id=results_id
        )
    except Exception as e:

        print(f"Error loading or processing results: {e}")

        return render_template(
            "error.html", error_message=str(e), results_id=results_id
        )
        
@app.route("/retroplanner/results/<results_id>&resultsLimit-<resultsLimit>")
def results(results_id, resultsLimit=None):
    if resultsLimit == None:
        resultsLimit = 50
    print(results_id)
    try:
        with open(os.path.join(app.synth_route, f"{results_id}.json"), "r") as f:
            routes = json.load(f)
        if resultsLimit == "all":
            routes_meta = routes_to_network_meta(routes)
        else:
            resultsLimit = int(resultsLimit)
            routes_meta = routes_to_network_meta(routes[:resultsLimit])
        return render_template(
            "results.html", routes_meta=routes_meta, results_id=results_id
        )
    except Exception as e:

        print(f"Error loading or processing results: {e}")

        return render_template(
            "error.html", error_message=str(e), results_id=results_id
        )



@app.route("/retroplanner/downloads/<results_id>")
def results_download(results_id):
    results_packer = RouteResultsPacker(results_id=results_id)
    results_packer.pack_route_results()
    return send_from_directory(app.synth_route, f"{results_id}.zip", as_attachment=True)

@app.route("/retroplanner/key_downloads/<task_id>")
def key_download(task_id):
    return send_from_directory(KEY_FOLDER, f"{task_id}.pem", as_attachment=True)

@app.route("/retroplanner/config_downloads/<task_id>")
def config_downloads(task_id):
    return send_from_directory(app.saved_configs_path, f"{task_id}.json", as_attachment=True)


@app.route("/retroplanner/process_node", methods=["POST"])
def process_node():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    node = request.get_json()
    reaction_attribute = node["reaction_attribute"]
    print(node["reaction_smiles"])
    if reaction_attribute["organic_enzyme_rxn_classification"]["enzyme_recommend"]:
        enzyme_data = []
        idx = 0
        try:
            for ec in reaction_attribute["enzyme_assign"]["ec"]:
                try:
                    print(f"Processing EC: {ec}")
                    uniprot_df = app.uniprot_parser.query_enzyme_pdb_by_ec(
                        ec_number=ec, size=1
                    )

                    if uniprot_df is not None:
                        for alphafolddb_id, ec, pdb_fpath in zip(
                            uniprot_df["AlphaFoldDB"].tolist(),
                            uniprot_df["EC number"].tolist(),
                            uniprot_df["pdb_fpath"].tolist(),
                        ):
                            predicted_active_labels = app.easifa_annotator.inference(
                                rxn=node["reaction_smiles"], enzyme_structure_path=pdb_fpath
                            )
                            structure_html, active_data_df = convert_easifa_results(
                                pdb_fpath=pdb_fpath,
                                site_labels=predicted_active_labels,
                                view_size=(395, 300),
                            )
                            if not active_data_df.empty:
                                enzyme_data.append(
                                    {
                                        "id": idx + 1,
                                        "structure_html": structure_html,
                                        "active_data": active_data_df.to_html(
                                            index=False, escape=False
                                        ).replace(
                                            '<table border="1" class="dataframe">\n',
                                            f"<table>\n<caption>Predicted Active Sites by EasIFA</caption>\n",
                                        ),
                                        "ec": ec,
                                        "alphafolddb_id": alphafolddb_id,
                                    }
                                )
                                idx += 1
                except Exception as e:
                    print(f"Error processing EC {ec}: {e}")
                    pass
        except:
            pass

        if len(enzyme_data) == 0:
            enzyme_data = None
    else:
        enzyme_data = None

    return (
        jsonify(
            {
                "status": "success",
                "message": "Data received",
                "enzyme_data": enzyme_data,
            }
        ),
        200,
    )


# @app.route("/retroplanner/api/retroplanner", methods=["POST"])
# def retroplanner_api():
#     if not request.is_json:
#         return jsonify({"error": "Missing JSON in request"}), 400

#     log_data = {
#         "status": "success",
#         "message": "Data received",
#     }
#     inputed_data = {k: v for k, v in request.get_json().items()}

#     target_smiles = canonicalize_smiles(
#         smi=inputed_data.get("smiles", ""), clear_map=False
#     )
#     if target_smiles == "":
#         return jsonify({"error": "Smiles is Not Valid!"}), 400



#     print(inputed_data)

#     valid, message = check_api_input(
#         inputed_data, 
#         single_step_opt=list(app.models_name_to_model_registor_name.keys()),
#         stock_opt=list(app.stocks_name_to_stocks_registor_name.keys()),
#         condition_opt=list(app.condition_predictor_name_to_condition_predictor_registor_name.keys()),
#         )
#     if not valid:
#         return (
#             jsonify(
#                 {
#                     "error": message,
#                 }
#             ),
#             400,
#         )
#     inputed_configs = inputed_data["savedOptions"]
#     setup_logger()
#     config: dict = app.planner_configs
#     print("Init configs:")
#     print(config)
#     if inputed_configs:
#         update_config = {
#             "iterations": inputed_configs.get(
#                 "iterationNumber", int(config["iterations"])
#             ),
#             "keep_search": inputed_configs.get(
#                 "Keep search after solved one route", config["keep_search"]
#             ),
#             "use_filter": inputed_configs.get(
#                 "Use reaction plausibility evaluator", config["use_filter"]
#             ),
#             "use_depth_value_fn": inputed_configs.get(
#                 "Use guiding function", config["use_depth_value_fn"]
#             ),
#             "pred_condition": inputed_configs.get(
#                 "Predict reaction condition",
#                 #   config["pred_condition"],
#                 False,
#             ),
#             "organic_enzyme_rxn_classification": inputed_configs.get(
#                 "Identify enzymatic reactions",
#                 #  config["organic_enzyme_rxn_classification"],
#                 False,
#             ),
#             "enzyme_assign": inputed_configs.get(
#                 "Recommend enzymes",
#                 #  config["enzyme_assign"],
#                 False,
#             ),
#         }

#         config.update(update_config)
#     else:
#         config.update({"iterations": 10})
#     if not inputed_configs.get("selectedStocks", []):
#         inputed_configs["selectedStocks"] = ["Zinc Buyable + USPTO Substrates"]
#     if not inputed_configs.get("selectedModels", []):
#         inputed_configs["selectedModels"] = ["Reaxys"]
#     if not inputed_configs.get("selectedConditionPredictor", []):
#         inputed_configs["selectedConditionPredictor"] = "Reaction Condition Recommander"

#     inputed_configs["selectedStocks"] = [
#         app.stocks_name_to_stocks_registor_name[x]
#         for x in inputed_configs.get("selectedStocks", ["Zinc Buyable + USPTO Substrates"])
#     ]
#     inputed_configs["selectedModels"] = [
#         app.models_name_to_model_registor_name[x]
#         for x in inputed_configs.get(
#             "selectedModels", ["Reaxys"]
#         )
#     ]

#     inputed_configs["selectedConditionPredictor"] = (
#         app.condition_predictor_name_to_condition_predictor_registor_name[
#             inputed_configs["selectedConditionPredictor"]
#         ]
#     )


    


#     planner = RSPlanner(config)
#     planner.select_stocks(inputed_configs["selectedStocks"])
#     planner.select_one_step_model(inputed_configs["selectedModels"])
#     planner.select_condition_predictor(inputed_configs["selectedConditionPredictor"])
#     planner.prepare_plan(prepare_easifa=False)

#     result = planner.plan(target_smiles)
#     if result:
#         # succ = True
#         # print(result)
#         # route_with_condition = planner.predict_condition()
#         # print(route_with_condition)

#         planner.predict_rxn_attributes()

#         dict_routes = [route.dict_route for route in result["all_succ_routes"]]

#         results_id = str(uuid.uuid4())
#         with open(os.path.join(app.synth_route, f"{results_id}.json"), "w") as f:
#             json.dump(dict_routes, f)

#         log_data.update(
#             {
#                 "results_id": results_id,
#                 # "resultsLimit": inputed_configs.get("resultsLimit", 50),
#                 "routes": dict_routes,
#             }
#         )
#         return jsonify(log_data), 200
#     else:
#         return (
#             jsonify(
#                 {
#                     "error": "Failure to search for the synthetic path, please expand the number of iterations!"
#                 }
#             ),
#             400,
#         )

@app.route("/retroplanner/api/retroplanner", methods=["POST"])
def retroplanner_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}
    config: dict = app.planner_configs
    config['input_method'] = "API"

    
    target_smiles = canonicalize_smiles(
        smi=inputed_data.get("smiles", ""), clear_map=False
    )
    if target_smiles == "":
        return jsonify({"error": "Smiles is Not Valid!"}), 400



    print(inputed_data)

    valid, message = check_api_input(
        inputed_data, 
        single_step_opt=list(app.models_name_to_model_registor_name.keys()),
        stock_opt=list(app.stocks_name_to_stocks_registor_name.keys()),
        condition_opt=list(app.condition_predictor_name_to_condition_predictor_registor_name.keys()),
        )
    if not valid:
        return (
            jsonify(
                {
                    "error": message,
                }
            ),
            400,
        )
        
    task = background_task.delay(inputed_data, config, input_method="API")
    conn = sqlite3.connect("jobs.db")
    c = conn.cursor()
    now = datetime.datetime.now(pytz.utc).astimezone(eastern_eight_zone)
    c.execute(
        "INSERT INTO jobs (job_id, status, public_key, submitted_at, input_method) VALUES (?, ?, ?, ?, ?)",
        (task.id, "Submitted", None, now, "API"),
    )
    conn.commit()
    conn.close()
    log_data.update(
        {
            "results_id": task.id,
        }
    )
    return jsonify(log_data), 200

@app.route("/retroplanner/api/retroplanner_results", methods=["POST"])
def retroplanner_api_results():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {}
    inputed_data = {k: v for k, v in request.get_json().items()}
    results_id = inputed_data.get('results_id', None)
    

    
    if results_id:
        conn = sqlite3.connect("jobs.db")
        c = conn.cursor()
        c.execute("SELECT status, result, input_method FROM jobs WHERE job_id = ?", (results_id,))
        job = c.fetchone()
        conn.close()
        try:
            with open(os.path.join(app.synth_route, f"{results_id}.json"), "r") as f:
                dict_routes = json.load(f)

            log_data.update(
                {
                    "routes": dict_routes,
                    "results_id": results_id,
                    "status": job[0]
                }
            )
        except:
            log_data.update(
                {
                    "results_id": results_id,
                    "status": job[0],
                    "error": "Search Failed! Please increase the number of iterations."
                }
            )
    else:
        log_data.update(
            {
                "results_id": results_id,
                "status": "FAILED",
                "error": "Search Failed! Please increase the number of iterations."
            }
        )
    return jsonify(log_data), 200



@app.route("/retroplanner/api/single_step", methods=["POST"])
def single_step_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}

    target_smiles = canonicalize_smiles(
        smi=inputed_data.get("smiles", ""), clear_map=False
    )
    if target_smiles == "":
        return jsonify({"error": "Smiles is Not Valid!"}), 400

    try:
        inputed_configs = inputed_data["savedOptions"]
        select_models = [ 
            app.models_name_to_model_registor_name[x]
            for x in inputed_configs.get(
                "oneStepModel", ["GraphFP-Moldels (PaRoutes-Set-n1)"]
            )
        ]
    except:

        input_example = '''
        curl -X POST http://localhost:8001/api/single_step \
            -H "Content-Type: application/json" \
            -d '{"smiles": "CCCCOCCCCC", "savedOptions":{"topk":10, "oneStepModel":["Reaxys"]}}'

        Available \'oneStepModel\' options: 
        '''
        input_example += ', '.join(list(app.models_name_to_model_registor_name.keys()))



        return jsonify({"error": f"Missing keys in input\ninput example:\n{input_example}"}, 400)
    # print(inputed_data)
    setup_logger()
    one_step_results = app.single_step_model_wrapper.run(
        target_smiles, topk=inputed_configs.get("topk", 50), select_models=select_models
    )
    log_data.update({"one_step_results": one_step_results})

    return jsonify(log_data), 200


@app.route("/retroplanner/api/condition_predictor", methods=["POST"])
def condition_predictor_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}

    target_reactions = proprecess_reactions(rxn_smiles=inputed_data.get("reaction", ""))
    if target_reactions == "":
        return jsonify({"error": "Reaction Smiles is Not Valid!"}), 400

    context_combos, context_combo_scores = app.condition_predictor(
        target_reactions,
        app.planner_configs["condition_config"]["rcr"]["topk"],
        return_scores=True,
    )

    use_cols = [
            "Temperature",
            "Solvent",
            "Reagent",
            "Catalyst",
            "Score",
        ]

    condition_df = pd.DataFrame(context_combos)
    condition_df.columns = [
        "Temperature",
        "Solvent",
        "Reagent",
        "Catalyst",
        "null1",
        "null2",
    ]
    condition_df["Score"] = [f"{num:.4f}" for num in context_combo_scores]
    condition_df = condition_df[
        use_cols
    ]
    condition_df["Temperature"] = condition_df["Temperature"].round(2)

    if inputed_data.get("return_dataframe", False):
        results_dict = {
            "condition_df": condition_df.to_json(),
        }
    else:
        results_dict = {
            "conditions": {col_name: condition_df[col_name].tolist() for col_name in use_cols},
        }

    log_data.update({"results": results_dict})

    return jsonify(log_data), 200


@app.route("/retroplanner/api/reaction_rater", methods=["POST"])
def reaction_rater_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}

    target_reactions = proprecess_reactions(rxn_smiles=inputed_data.get("reaction", ""))
    if target_reactions == "":
        return jsonify({"error": "Reaction Smiles is Not Valid!"}), 400

    feasible, confidence = app.reaction_rater.is_feasible(
        reaction=target_reactions, return_prob=True
    )

    results_dict = {
        "reaction_is_feasible": feasible,
        "confidence": confidence,
    }

    log_data.update({"results": results_dict})

    return jsonify(log_data), 200


@app.route("/retroplanner/api/enzymatic_rxn_identifier", methods=["POST"])
def enzymatic_rxn_identifier_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}

    target_reactions = proprecess_reactions(rxn_smiles=inputed_data.get("reaction", ""))
    if target_reactions == "":
        return jsonify({"error": "Reaction Smiles is Not Valid!"}), 400

    _, confidence, rxn_type = app.organic_enzymatic_rxn_identifier.predict(
        [target_reactions], batch_size=32
    )
    results_dict = {
        "reaction type": rxn_type[0],
        "confidence": confidence[0],
    }
    log_data.update({"results": results_dict})

    return jsonify(log_data), 200


@app.route("/retroplanner/api/enzyme_recommender", methods=["POST"])
def enzyme_recommender_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    log_data = {
        "status": "success",
        "message": "Data received",
    }
    inputed_data = {k: v for k, v in request.get_json().items()}

    target_reactions = proprecess_reactions(rxn_smiles=inputed_data.get("reaction", ""))
    if target_reactions == "":
        return jsonify({"error": "Reaction Smiles is Not Valid!"}), 400

    _, confidences, ec_numbers = app.enzyme_recommender.predict(
        [target_reactions], batch_size=32
    )
    results_dict = {
        "recommended enzyme type": ec_numbers,
        "confidence": confidences,
    }
    log_data.update({"results": results_dict})

    return jsonify(log_data), 200


@app.route("/retroplanner/api/easifa", methods=["POST"])
def easifa_api():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    inputed_data = {k: v for k, v in request.get_json().items()}

    target_reactions = proprecess_reactions(rxn_smiles=inputed_data.get("reaction", ""))
    if target_reactions == "":
        return jsonify({"error": "Reaction Smiles is Not Valid!"}), 400
    ec = inputed_data.get("EC number", "")
    if (not isinstance(ec, str)) or (not is_valid_ec_number(ec)):
        return jsonify({"error": "EC number is Not Valid!"}), 400

    uniprot_df: pd.DataFrame = (
        app.uniprot_parser.query_enzyme_pdb_by_ec_with_rxn_ranking(
            ec_number=ec, rxn_smiles=target_reactions, topk=1
        )
    )
    structure_htmls = {}
    enzyme_data = []
    idx = 0

    results_id = "easifa_" + str(uuid.uuid4())

    if not uniprot_df.empty:
        for alphafolddb_id, ec, pdb_fpath in zip(
            uniprot_df["AlphaFoldDB"].tolist(),
            uniprot_df["EC number"].tolist(),
            uniprot_df["pdb_fpath"].tolist(),
        ):
            predicted_active_labels = app.easifa_annotator.inference(
                rxn=target_reactions, enzyme_structure_path=pdb_fpath
            )
            structure_html, active_data_df = convert_easifa_results(
                pdb_fpath=pdb_fpath,
                site_labels=predicted_active_labels,
                view_size=(395, 300),
                add_style=False,
            )
            if not active_data_df.empty:
                if inputed_data.get("return_dataframe", False):
                    active_data = active_data_df.to_json()
                else: 
                    active_data = {col_name:active_data_df[col_name].tolist() for col_name in active_data_df.columns.tolist()}

                enzyme_data.append(
                    {
                        "id": idx + 1,
                        # "structure_html": structure_html,
                        "active_data": active_data,
                        "ec": ec,
                        "alphafolddb_id": alphafolddb_id,
                    }
                )
                idx += 1
                structure_htmls[alphafolddb_id] = structure_html
    else:
        return (
            jsonify(
                {
                    "status": "failure",
                    "message": "Failed to query the structure for the given EC number through the UniProt API.",
                }
            ),
            400,
        )
    if len(enzyme_data) == 0:
        return (
            jsonify(
                {
                    "status": "failure",
                    "message": "The structure retrieved could not be processed or the calculation failed.",
                }
            ),
            400,
        )
    else:
        results_save_path = os.path.join(
            app.enzyme_structure_path, f"{results_id}.json"
        )
        with open(results_save_path, "w") as f:
            json.dump(structure_htmls, f)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Data received",
                    "enzyme_data": enzyme_data,
                    "results_id": results_id,
                }
            ),
            200,
        )


@app.route("/retroplanner/api/enzyme_show/<results_id>&&&&<structure_id>")
def enzyem_show(results_id, structure_id):
    results_save_path = os.path.join(app.enzyme_structure_path, f"{results_id}.json")
    try:
        with open(results_save_path, "r") as f:
            structure_htmls = json.load(f)
    except:
        return render_template_string(f"No Results for {results_id}")
    return render_template_string(structure_htmls[structure_id])


@app.route("/retroplanner/agent")
def agent_interface():
    return render_template("agent.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
