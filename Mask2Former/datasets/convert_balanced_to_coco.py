# -*- coding: utf-8 -*-
"""
将 balanced_dataset 转换为 COCO 格式的 JSON 文件。

转换后的文件结构:
    datasets/balanced_dataset/
    ├── train/
    │   ├── encoded_views/
    │   ├── masks/
    │   ├── class_map.json
    │   └── instances.json  # COCO 格式标注 (新生成)
    └── val/
        ├── encoded_views/
        ├── masks/
        ├── class_map.json
        └── instances.json  # COCO 格式标注 (新生成)

用法:
    python convert_balanced_to_coco.py
"""

import json
import os
import datetime
import numpy as np
from PIL import Image

# 23个加工特征类别 (class_map 中 ID 1-23 → 这里 ID 0-22)
# class 0 (chamfer) 从未生成，class 24 (stock) 为 ignore
CATEGORIES = [
    {"id": 0,  "name": "through_hole",                "supercategory": "feature_type"},
    {"id": 1,  "name": "triangular_passage",          "supercategory": "feature_type"},
    {"id": 2,  "name": "rectangular_passage",         "supercategory": "feature_type"},
    {"id": 3,  "name": "6sides_passage",              "supercategory": "feature_type"},
    {"id": 4,  "name": "triangular_through_slot",     "supercategory": "feature_type"},
    {"id": 5,  "name": "rectangular_through_slot",    "supercategory": "feature_type"},
    {"id": 6,  "name": "circular_through_slot",       "supercategory": "feature_type"},
    {"id": 7,  "name": "rectangular_through_step",    "supercategory": "feature_type"},
    {"id": 8,  "name": "2sides_through_step",         "supercategory": "feature_type"},
    {"id": 9,  "name": "slanted_through_step",        "supercategory": "feature_type"},
    {"id": 10, "name": "Oring",                       "supercategory": "feature_type"},
    {"id": 11, "name": "blind_hole",                  "supercategory": "feature_type"},
    {"id": 12, "name": "triangular_pocket",           "supercategory": "feature_type"},
    {"id": 13, "name": "rectangular_pocket",          "supercategory": "feature_type"},
    {"id": 14, "name": "6sides_pocket",               "supercategory": "feature_type"},
    {"id": 15, "name": "circular_end_pocket",         "supercategory": "feature_type"},
    {"id": 16, "name": "rectangular_blind_slot",      "supercategory": "feature_type"},
    {"id": 17, "name": "v_circular_end_blind_slot",   "supercategory": "feature_type"},
    {"id": 18, "name": "h_circular_end_blind_slot",   "supercategory": "feature_type"},
    {"id": 19, "name": "triangular_blind_step",       "supercategory": "feature_type"},
    {"id": 20, "name": "circular_blind_step",         "supercategory": "feature_type"},
    {"id": 21, "name": "rectangular_blind_step",      "supercategory": "feature_type"},
    {"id": 22, "name": "round",                       "supercategory": "feature_type"},
]

# class_map 中的 category_id 需要减1 (1-23 → 0-22)
CLASS_MAP_OFFSET = 1


def mask_to_coco_annotations(mask: np.ndarray, class_map: dict, image_file: str):
    """
    将实例掩码转换为 COCO 格式的标注。

    Args:
        mask: 实例掩码 (H, W), 值为 0(背景), 1,2,...(实例), 255(忽略)
        class_map: 该图像的 {face_id_str: category_id} 映射
        image_file: 图像文件名

    Returns:
        COCO 格式的标注列表
    """
    import cv2

    annotations = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[(instance_ids > 0) & (instance_ids < 255)]

    for instance_id in instance_ids:
        instance_id_int = int(instance_id)

        # 获取二值掩码
        instance_mask = (mask == instance_id_int).astype(np.uint8)
        instance_area = int(instance_mask.sum())

        if instance_area < 10:
            continue

        # 获取类别 ID
        face_id_str = str(instance_id_int - 1)
        if face_id_str not in class_map:
            continue

        # class_map 中的 ID 1-23 → COCO ID 0-22
        category_id = class_map[face_id_str] - CLASS_MAP_OFFSET

        # 计算边界框 [x, y, width, height]
        ys, xs = np.where(instance_mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        # 转换为 RLE 格式
        rle = mask_to_rle(instance_mask)

        annotations.append({
            "id": None,  # 稍后分配
            "image_id": None,  # 稍后分配
            "category_id": category_id,
            "segmentation": rle,
            "area": instance_area,
            "bbox": bbox,
            "iscrowd": 0,
        })

    return annotations


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    将二值掩码转换为 COCO RLE 格式（使用 pycocotools 编码）。

    Args:
        mask: 二值掩码 (H, W), 值为 0 或 1

    Returns:
        COCO RLE 字典
    """
    import pycocotools.mask as _mask_util
    # pycocotools 要求 Fortran order (column-major)
    rle = _mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    # counts 是 bytes 类型，转为 list 以保证 JSON 可序列化
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    rle["size"] = list(rle["size"])
    return rle


def convert_split(split_dir: str, output_path: str):
    """
    转换一个数据分割 (train 或 val) 为 COCO 格式。

    Args:
        split_dir: 分割目录路径 (如 datasets/balanced_dataset/train)
        output_path: 输出 JSON 文件路径
    """
    image_dir = os.path.join(split_dir, "encoded_views")
    mask_dir = os.path.join(split_dir, "masks")
    class_map_path = os.path.join(split_dir, "class_map.json")

    if not os.path.exists(class_map_path):
        print(f"class_map.json 不存在: {class_map_path}")
        return

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # 构建 COCO 格式
    coco_output = {
        "info": {
            "description": "balanced_dataset",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    annotation_id = 0

    for image_id, image_file in enumerate(image_files):
        # 添加图像信息
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height,
        })

        # 读取掩码
        mask_path = os.path.join(mask_dir, image_file)
        if not os.path.exists(mask_path):
            print(f"掩码不存在: {mask_path}, 跳过")
            continue

        mask = np.array(Image.open(mask_path))

        # 获取该图像的类别映射
        if image_file not in class_map:
            print(f"{image_file} 不在 class_map 中, 跳过")
            continue

        image_class_map = class_map[image_file]

        # 转换标注
        annotations = mask_to_coco_annotations(mask, image_class_map, image_file)

        for ann in annotations:
            ann["id"] = annotation_id
            ann["image_id"] = image_id
            coco_output["annotations"].append(ann)
            annotation_id += 1

    # 保存 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

    print(f"已转换: {output_path}")
    print(f"  图像数: {len(coco_output['images'])}")
    print(f"  标注数: {len(coco_output['annotations'])}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="balanced_dataset",
                        help="数据集目录名")
    args = parser.parse_args()

    dataset_root = os.path.join(os.path.dirname(__file__), args.dataset)

    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_root, split)
        output_path = os.path.join(split_dir, "instances.json")
        convert_split(split_dir, output_path)


if __name__ == "__main__":
    main()
