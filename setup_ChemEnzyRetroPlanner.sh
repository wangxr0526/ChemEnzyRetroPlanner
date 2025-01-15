#!/bin/bash


# 设置环境变量
export CONDA_HOME=$(conda info --base)
export PLANNER_ROOT=$(pwd)
echo "Automatically configuring the ChemEnzyRetroPlanner environment..."
git clone https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance.git retro_planner/packages/template_relevance

cd retro_planner/packages/template_relevance
#!/bin/bash

# Define the URLs and file names of the models to be downloaded
declare -A models
models=(
    ["bkms_metabolic.mar"]="<URL_FOR_BKMS_METABOLIC>"
    ["pistachio.mar"]="<URL_FOR_PISTACHIO>"
    ["pistachio_ringbreaker.mar"]="<URL_FOR_PISTACHIO_RINGBREAKER>"
    ["reaxys_biocatalysis.mar"]="<URL_FOR_REAXYS_BIOCATALYSIS>"
    ["reaxys.mar"]="<URL_FOR_REAXYS>"
)

# Directory to store the downloaded models
MODEL_DIR="./mars"
mkdir -p "$MODEL_DIR"

# Loop through the models and check if they exist
for model in "${!models[@]}"; do
    if [ -f "$MODEL_DIR/$model" ]; then
        echo "$model already exists, skipping download."
    else
        echo "$model not found, downloading..."
        bash scripts/download_trained_models.sh
        if [ $? -eq 0 ]; then
            echo "$model downloaded successfully."
            break
        else
            echo "Failed to download $model. Please check the URL or your connection."
        fi
    fi
done

cd $PLANNER_ROOT
mkdir -p metadata


# # 安装 gdown 工具
# pip install gdown

mkdir -vp $CONDA_HOME/envs

# 检查并下载文件：ESM 模型
mkdir -vp ~/.cache/torch/hub/checkpoints/
if [ ! -f ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt ]; then
    echo "Downloading ESM model..."
    wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -O ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
else
    echo "ESM model already exists, skipping download."
fi

# 检查并下载 retro_planner_env.tar.gz
if [ ! -f $CONDA_HOME/envs/retro_planner_env.tar.gz ]; then
    echo "Downloading retro_planner_env.tar.gz..."
    curl -L -o $CONDA_HOME/envs/retro_planner_env.tar.gz "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/retro_planner_env.tar.gz?download=true"
else
    echo "retro_planner_env.tar.gz already exists, skipping download."
fi

# 解压 retro_planner_env.tar.gz 文件
if [ ! -d $CONDA_HOME/envs/retro_planner_env ]; then
    echo "Extracting retro_planner_env.tar.gz..."
    mkdir $CONDA_HOME/envs/retro_planner_env
    tar -xvf $CONDA_HOME/envs/retro_planner_env.tar.gz -C $CONDA_HOME/envs/retro_planner_env
fi

# 激活 conda 环境
source activate $CONDA_HOME/envs/retro_planner_env
conda unpack

# 检查并下载 building_block_dataset.zip
if [ ! -f $PLANNER_ROOT/metadata/building_block_dataset.zip ]; then
    echo "Downloading building_block_dataset.zip..."
    curl -L -o $PLANNER_ROOT/metadata/building_block_dataset.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/building_block_dataset.zip?download=true"
else
    echo "building_block_dataset.zip already exists, skipping download."
fi

# 检查并下载 condition_predictor_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/condition_predictor_metadata.zip ]; then
    echo "Downloading condition_predictor_metadata.zip..."
    curl -L -o $PLANNER_ROOT/metadata/condition_predictor_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/condition_predictor_metadata.zip?download=true"
else
    echo "condition_predictor_metadata.zip already exists, skipping download."
fi

# 检查并下载 easifa_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/easifa_metadata.zip ]; then
    echo "Downloading easifa_metadata.zip..."
    curl -L -o $PLANNER_ROOT/metadata/easifa_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/easifa_metadata.zip?download=true"
else
    echo "easifa_metadata.zip already exists, skipping download."
fi

# 检查并下载 graph_retrosyn_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/graph_retrosyn_metadata.zip ]; then
    echo "Downloading graph_retrosyn_metadata.zip ..."
    curl -L -o $PLANNER_ROOT/metadata/graph_retrosyn_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/graph_retrosyn_metadata.zip?download=true"
else
    echo "graph_retrosyn_metadata.zip already exists, skipping download."
fi

# 检查并下载 onmt_metadata.zip.zip
if [ ! -f $PLANNER_ROOT/metadata/onmt_metadata.zip.zip ]; then
    echo "Downloading onmt_metadata.zip.zip ..."
    curl -L -o $PLANNER_ROOT/metadata/onmt_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/onmt_metadata.zip?download=true"
else
    echo "onmt_metadata.zip already exists, skipping download."
fi

# 检查并下载 rxn_filter_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/rxn_filter_metadata.zip ]; then
    echo "Downloading rxn_filter_metadata.zip ..."
    curl -L -o $PLANNER_ROOT/metadata/rxn_filter_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/rxn_filter_metadata.zip?download=true"
else
    echo "rxn_filter_metadata.zip already exists, skipping download."
fi

# 检查并下载 value_fun_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/value_fun_metadata.zip ]; then
    echo "Downloading value_fun_metadata.zip ..."
    curl -L -o $PLANNER_ROOT/metadata/value_fun_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/value_fun_metadata.zip?download=true"
else
    echo "value_fun_metadata.zip already exists, skipping download."
fi

# 检查并下载 enzyme_cls_metadata.zip
if [ ! -f $PLANNER_ROOT/metadata/enzyme_cls_metadata.zip ]; then
    echo "Downloading enzyme_cls_metadata.zip ..."
    curl -L -o $PLANNER_ROOT/metadata/enzyme_cls_metadata.zip "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/enzyme_cls_metadata.zip?download=true"
else
    echo "enzyme_cls_metadata.zip already exists, skipping download."
fi

# 检查并下载 USPTO_condition.mars
if [ ! -f $PLANNER_ROOT/retro_planner/packages/parrot/mars/USPTO_condition.mar ]; then
    echo "Downloading USPTO_condition.mars ..."
    mkdir -vp $PLANNER_ROOT/retro_planner/packages/parrot/mars/
    curl -L -o $PLANNER_ROOT/retro_planner/packages/parrot/mars/USPTO_condition.mar "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/USPTO_condition.mar?download=true"
else
    echo "USPTO_condition.zip already exists, skipping download."
fi


# 检查并下载 structures.csv.gz
mkdir -vp $PLANNER_ROOT/retro_planner/packages/easifa/easifa/interface/chebi/
if [ ! -f $PLANNER_ROOT/retro_planner/packages/easifa/easifa/interface/chebi/structures.csv.gz ]; then
    echo "Downloading structures.csv.gz..."
    wget https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/structures.csv.gz -O $PLANNER_ROOT/retro_planner/packages/easifa/easifa/interface/chebi/structures.csv.gz
else
    echo "structures.csv.gz already exists, skipping download."
fi


cd $PLANNER_ROOT

chmod +x ./unpack_metadata.sh
bash unpack_metadata.sh

bash setup.sh