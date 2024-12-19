#!/bin/bash
echo "正在解压metadata..."
# 创建 metadata 文件夹
METADATA_DIR="metadata"

mkdir -p "$METADATA_DIR"

# 定义压缩文件名
BUILDING_BLOCK_ZIP="$METADATA_DIR/building_block_dataset.zip"
GRAPH_RETROSYN_ZIP="$METADATA_DIR/graph_retrosyn_metadata.zip"
ONMT_ZIP="$METADATA_DIR/onmt_metadata.zip"
CONDITION_PREDICTOR_ZIP="$METADATA_DIR/condition_predictor_metadata.zip"
EASIFA_ZIP="$METADATA_DIR/easifa_metadata.zip"

# 解压 building_block_dataset
if [ -f "$BUILDING_BLOCK_ZIP" ]; then
    unzip -o "$BUILDING_BLOCK_ZIP" 
    echo "$BUILDING_BLOCK_ZIP 解压完成."
else
    echo "$BUILDING_BLOCK_ZIP 文件不存在."
fi

# 解压 graph_retrosyn
if [ -f "$GRAPH_RETROSYN_ZIP" ]; then
    unzip -o "$GRAPH_RETROSYN_ZIP"
    echo "$GRAPH_RETROSYN_ZIP 解压完成."
else
    echo "$GRAPH_RETROSYN_ZIP 文件不存在."
fi

# 解压 onmt/checkpoints
if [ -f "$ONMT_ZIP" ]; then
    unzip -o "$ONMT_ZIP"
    echo "$ONMT_ZIP 解压完成."
else
    echo "$ONMT_ZIP 文件不存在."
fi

# 解压 condition_predictor/data
if [ -f "$CONDITION_PREDICTOR_ZIP" ]; then
    unzip -o "$CONDITION_PREDICTOR_ZIP" 
    echo "$CONDITION_PREDICTOR_ZIP 解压完成."
else
    echo "$CONDITION_PREDICTOR_ZIP 文件不存在."
fi

# 解压 easifa/checkpoints
if [ -f "$EASIFA_ZIP" ]; then
    unzip -o "$EASIFA_ZIP"
    echo "$EASIFA_ZIP 解压完成."
else
    echo "$EASIFA_ZIP 文件不存在."
fi

# 输出完成消息
echo "所有解压任务完成."
