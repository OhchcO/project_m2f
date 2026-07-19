#!/usr/bin/env bash
set -euo pipefail

# 每次换实验通常只需要改下面 4 个路径：
#   DATASET_DIR: 数据集根目录，要求里面有 val/models.json
#   RESULT_DIR: 单图训练输出目录，用来拼默认权重路径
#   WEIGHTS:    要评估的具体 .pth 权重，可以是 model_final.pth 或 model_00xxxxx.pth
#   OUTPUT_DIR: 评估结果保存目录，会生成 metrics/confusion/detail/pred_details
#
# 可选项：
#   CONDA_ENV: 默认 m2f
#   GPU_ID 或 CUDA_VISIBLE_DEVICES: 默认使用 0 号 GPU
#   --score_threshold: 预测实例分数阈值，当前 0.3
#   --min_size_test: 验证输入短边尺寸，当前 512
#   --fusion_mode score_weighted_view_vote:
#       单图模型先跑 14 张图，每个视角给可见面投一个类别，按预测置信度加权累计，再算整模型面级指标
#   --fusion_mode view_vote: 
#       单图模型先跑 14 张图，每个视角给可见面投一个类别，再算整模型面级指标
#   --fusion_mode pixel_vote: 
#       单图模型先跑 14 张图，每个像素给可见面投一个类别，再算整模型像素级指标（三选一）
#   --eval_class_mode both: 同时输出 24 细类和粗类指标

CONDA_ENV="${CONDA_ENV:-m2f}"
PROJECT_DIR="${PROJECT_DIR:-/data/m2f}"
MASK2FORMER_DIR="${MASK2FORMER_DIR:-${PROJECT_DIR}/Mask2Former}"
DATASET_DIR="${DATASET_DIR:-/mnt/e/wsl/datasets/MFRInstSegM2F_2100}"
RESULT_DIR="${RESULT_DIR:-/mnt/e/wsl/result/MFRInstSegM2F_2100_single}"
WEIGHTS="${WEIGHTS:-${RESULT_DIR}/model_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/e/wsl/result/eval_MFRInstSegM2F_2100_single_model_final}"

if [ ! -s "${WEIGHTS}" ]; then
  echo "[ERROR] Trained weights not found: ${WEIGHTS}"
  exit 1
fi
if [ ! -s "${DATASET_DIR}/val/models.json" ]; then
  echo "[ERROR] Dataset val/models.json not found under: ${DATASET_DIR}"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${MASK2FORMER_DIR}"

export MFR_SINGLEVIEW_DATASET="${DATASET_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

python "${PROJECT_DIR}/new_add/eval_mfr_singleview.py" \
  --config_file "${MASK2FORMER_DIR}/configs/mfr_singleview/maskformer2_R50_512.yaml" \
  --weights "${WEIGHTS}" \
  --val_dir "${DATASET_DIR}/val" \
  --output_dir "${OUTPUT_DIR}" \
  --score_threshold 0.3 \
  --min_size_test 512 \
  --fusion_mode score_weighted_view_vote \
  --eval_class_mode both
