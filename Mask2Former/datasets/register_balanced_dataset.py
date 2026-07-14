# -*- coding: utf-8 -*-
"""
注册 balanced_dataset 用于 Mask2Former 实例分割训练。

数据集结构:
    balanced_dataset/
    ├── train/
    │   ├── encoded_views/    # 输入图像
    │   ├── masks/            # 实例掩码 (用于预览，训练时使用 instances.json)
    │   ├── class_map.json    # 原始类别映射
    │   └── instances.json    # COCO 格式标注 (由 convert_balanced_to_coco.py 生成)
    └── val/
        ├── encoded_views/
        ├── masks/
        ├── class_map.json
        └── instances.json

用法:
    在训练脚本中 import 此模块即可自动注册数据集:
        import register_balanced_dataset

    然后在配置文件中使用:
        DATASETS:
            TRAIN: ("balanced_dataset_train",)
            TEST: ("balanced_dataset_val",)
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# ============================================================
# 23个加工特征类别 (class_map 中 ID 1-23 → 这里 ID 0-22)
# ============================================================
BALANCED_CATEGORIES = [
    {"id": 0,  "name": "through_hole",                "isthing": 1, "supercategory": "feature_type"},
    {"id": 1,  "name": "triangular_passage",          "isthing": 1, "supercategory": "feature_type"},
    {"id": 2,  "name": "rectangular_passage",         "isthing": 1, "supercategory": "feature_type"},
    {"id": 3,  "name": "6sides_passage",              "isthing": 1, "supercategory": "feature_type"},
    {"id": 4,  "name": "triangular_through_slot",     "isthing": 1, "supercategory": "feature_type"},
    {"id": 5,  "name": "rectangular_through_slot",    "isthing": 1, "supercategory": "feature_type"},
    {"id": 6,  "name": "circular_through_slot",       "isthing": 1, "supercategory": "feature_type"},
    {"id": 7,  "name": "rectangular_through_step",    "isthing": 1, "supercategory": "feature_type"},
    {"id": 8,  "name": "2sides_through_step",         "isthing": 1, "supercategory": "feature_type"},
    {"id": 9,  "name": "slanted_through_step",        "isthing": 1, "supercategory": "feature_type"},
    {"id": 10, "name": "Oring",                       "isthing": 1, "supercategory": "feature_type"},
    {"id": 11, "name": "blind_hole",                  "isthing": 1, "supercategory": "feature_type"},
    {"id": 12, "name": "triangular_pocket",           "isthing": 1, "supercategory": "feature_type"},
    {"id": 13, "name": "rectangular_pocket",          "isthing": 1, "supercategory": "feature_type"},
    {"id": 14, "name": "6sides_pocket",               "isthing": 1, "supercategory": "feature_type"},
    {"id": 15, "name": "circular_end_pocket",         "isthing": 1, "supercategory": "feature_type"},
    {"id": 16, "name": "rectangular_blind_slot",      "isthing": 1, "supercategory": "feature_type"},
    {"id": 17, "name": "v_circular_end_blind_slot",   "isthing": 1, "supercategory": "feature_type"},
    {"id": 18, "name": "h_circular_end_blind_slot",   "isthing": 1, "supercategory": "feature_type"},
    {"id": 19, "name": "triangular_blind_step",       "isthing": 1, "supercategory": "feature_type"},
    {"id": 20, "name": "circular_blind_step",         "isthing": 1, "supercategory": "feature_type"},
    {"id": 21, "name": "rectangular_blind_step",      "isthing": 1, "supercategory": "feature_type"},
    {"id": 22, "name": "round",                       "isthing": 1, "supercategory": "feature_type"},
]


def register_balanced_dataset(root: str = "datasets", dataset_name: str = "balanced_dataset", prefix: str = None):
    """
    注册数据集。

    使用 detectron2 内置的 register_coco_instances 注册 COCO 格式数据集。

    Args:
        root: 数据集根目录，默认为 "datasets"
        dataset_name: 数据集目录名，默认为 "balanced_dataset"
        prefix: 注册名称前缀，默认为 dataset_name
    """
    if prefix is None:
        prefix = dataset_name
    dataset_root = os.path.join(root, dataset_name)

    if not os.path.exists(dataset_root):
        print(f"[WARNING] 数据集目录不存在: {dataset_root}")
        return

    for split in ["train", "val"]:
        split_path = os.path.join(dataset_root, split)
        image_dir = os.path.join(split_path, "encoded_views")
        annotation_file = os.path.join(split_path, "instances.json")

        if not os.path.exists(image_dir):
            print(f"[WARNING] 图像目录不存在: {image_dir}")
            continue
        if not os.path.exists(annotation_file):
            print(f"[WARNING] COCO 标注文件不存在: {annotation_file}")
            print(f"  请先运行: python datasets/convert_balanced_to_coco.py")
            continue

        registered_name = f"{prefix}_{split}"

        # 检查是否已注册
        if registered_name in DatasetCatalog.list():
            print(f"[SKIP] 数据集已注册: {registered_name}")
            continue

        # 使用 detectron2 内置函数注册
        register_coco_instances(
            name=registered_name,
            metadata={},
            image_root=image_dir,
            json_file=annotation_file,
        )

        # 设置元数据
        MetadataCatalog.get(registered_name).set(
            thing_classes=[k["name"] for k in BALANCED_CATEGORIES],
            thing_dataset_id_to_contiguous_id={k["id"]: i for i, k in enumerate(BALANCED_CATEGORIES)},
            evaluator_type="coco",
            ignore_label=255,
        )

        print(f"[OK] 已注册数据集: {registered_name}")


# ============================================================
# 模块导入时自动注册
# ============================================================
# 获取 Mask2Former/datasets 目录的绝对路径
_datasets_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.getenv("DETECTRON2_DATASETS", _datasets_dir)

# 注册多个数据集
for _ds_name in ["balanced_dataset", "dataset_24class"]:
    register_balanced_dataset(_root, _ds_name)

# 注册 temp_data 目录下的数据集 (位于 project_m2f/temp_data)
_project_root = os.path.dirname(os.path.dirname(_datasets_dir))
_temp_root = os.path.join(_project_root, "temp_data")
for _ds_name in ["dataset_24class"]:
    register_balanced_dataset(_temp_root, _ds_name, "temp_data_24class")
