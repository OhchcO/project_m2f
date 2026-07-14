"""
面级别面积评估脚本。

直接从 masks/ 和 class_map.json 读取 GT，计算：
1. 面级别分类准确率
2. 面级别面积准确率（按面积加权）
3. 面级别 IoU

GT 数据格式：
- masks/xxx.png: 实例掩码，像素值=实例ID(1,2,3...)，0=背景
- class_map.json: {图片名: {实例序号(0,1,2...): 类别ID(0-6)}}

注意：masks 中的实例ID = class_map 中的序号 + 1

用法：
python eval_area_metrics.py \
  --val_dir /path/to/balanced_dataset/val \
  --config_file /path/to/config.yaml \
  --opts MODEL.WEIGHTS /path/to/model.pkl
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Mask2Former"))
from mask2former import add_maskformer2_config

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, NUM_CLASSES

from ins_inference_encoded import setup_cfg, detectron2_to_unified_format


def build_gt_from_mask_and_classmap(mask, class_map):
    """
    从 masks 和 class_map.json 构建 GT 面信息。

    Args:
        mask: 实例掩码 (H, W), 像素值=实例ID(1,2,3...), 0=背景, 255=忽略
        class_map: {实例序号(字符串): 类别ID}

    Returns:
        gt_faces: list of dict
    """
    gt_faces = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[(instance_ids > 0) & (instance_ids < 255)]

    for inst_id in instance_ids:
        inst_id_int = int(inst_id)
        # 实例ID = 序号 + 1，所以序号 = inst_id - 1
        seq_num = str(inst_id_int - 1)

        if seq_num not in class_map:
            continue

        category_id = class_map[seq_num]
        instance_mask = (mask == inst_id_int)
        area = int(instance_mask.sum())

        if area < 10:
            continue

        gt_faces.append({
            "instance_id": inst_id_int,
            "category_id": category_id,
            "mask": instance_mask,
            "area": area,
        })

    return gt_faces


def evaluate_single(gt_faces, pred_mask, pred_classes):
    """
    评估单张图的面级别结果。

    Args:
        gt_faces: list of dict, GT 面信息
        pred_mask: (H, W) 预测实例ID图 (0=背景)
        pred_classes: list of {"id": instance_id, "class": class_id, "score": score}

    Returns:
        results: list of dict
    """
    pred_info = {p["id"]: p for p in pred_classes}

    results = []

    for gt in gt_faces:
        gt_mask = gt["mask"]
        gt_class = gt["category_id"]
        gt_area = gt["area"]

        # 在该面区域内，统计各预测实例的像素数
        face_pred_ids = pred_mask[gt_mask]
        face_pred_nonzero = face_pred_ids[face_pred_ids != 0]

        if len(face_pred_nonzero) == 0:
            # 面内无预测
            results.append({
                "instance_id": gt["instance_id"],
                "gt_class": gt_class,
                "pred_class": -1,
                "is_correct": False,
                "area": gt_area,
                "face_iou": 0.0,
                "vote_ratio": 0.0,
                "score": 0.0,
            })
            continue

        # 投票：找出占比最大的预测实例
        unique_ids, counts = np.unique(face_pred_nonzero, return_counts=True)
        max_idx = int(np.argmax(counts))
        pred_id = int(unique_ids[max_idx])
        vote_ratio = float(counts[max_idx] / gt_area)

        # DEBUG: 打印调试信息（只打印第一张图）
        if not hasattr(evaluate_single, '_debug_done'):
            print(f"\n[DEBUG] pred_info keys: {list(pred_info.keys())[:5]}...")
            print(f"[DEBUG] pred_id={pred_id}, in pred_info: {pred_id in pred_info}")
            print(f"[DEBUG] pred_info_item: {pred_info.get(pred_id, 'NOT FOUND')}")
            evaluate_single._debug_done = True

        pred_info_item = pred_info.get(pred_id, {})
        pred_class = pred_info_item.get("label_id", -1)
        score = pred_info_item.get("score", 0.0)

        is_correct = (pred_class == gt_class)

        # 计算面级别的 IoU
        if pred_id > 0:
            pred_bin = (pred_mask == pred_id)
            intersection = int(np.sum(gt_mask & pred_bin))
            union = int(np.sum(gt_mask | pred_bin))
            face_iou = intersection / union if union > 0 else 0.0
        else:
            face_iou = 0.0

        results.append({
            "instance_id": gt["instance_id"],
            "gt_class": gt_class,
            "pred_class": pred_class,
            "is_correct": is_correct,
            "area": gt_area,
            "face_iou": face_iou,
            "vote_ratio": vote_ratio,
            "score": score,
        })

    return results


def compute_metrics(all_results, num_classes):
    """汇总所有图片的评估结果。"""
    total_faces = len(all_results)
    correct_faces = sum(1 for r in all_results if r["is_correct"])
    face_acc = correct_faces / total_faces if total_faces > 0 else 0.0

    total_area = sum(r["area"] for r in all_results)
    correct_area = sum(r["area"] for r in all_results if r["is_correct"])
    area_acc = correct_area / total_area if total_area > 0 else 0.0

    face_ious = [r["face_iou"] for r in all_results]
    mean_face_iou = np.mean(face_ious) if face_ious else 0.0

    # 各类别指标
    class_metrics = {}
    for cls_id in range(num_classes):
        cls_results = [r for r in all_results if r["gt_class"] == cls_id]
        cls_correct = [r for r in cls_results if r["is_correct"]]
        cls_area = sum(r["area"] for r in cls_results)
        cls_correct_area = sum(r["area"] for r in cls_correct)

        class_metrics[cls_id] = {
            "name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "total_faces": len(cls_results),
            "correct_faces": len(cls_correct),
            "accuracy": len(cls_correct) / len(cls_results) if cls_results else 0.0,
            "total_area": cls_area,
            "correct_area": cls_correct_area,
            "area_accuracy": cls_correct_area / cls_area if cls_area > 0 else 0.0,
            "mean_iou": np.mean([r["face_iou"] for r in cls_results]) if cls_results else 0.0,
        }

    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for r in all_results:
        gt = r["gt_class"]
        pred = r["pred_class"]
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            confusion[gt][pred] += 1

    return {
        "total_faces": total_faces,
        "correct_faces": correct_faces,
        "face_accuracy": face_acc,
        "total_area": total_area,
        "correct_area": correct_area,
        "area_accuracy": area_acc,
        "mean_face_iou": float(mean_face_iou),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
    }


def print_report(metrics, num_classes):
    """打印评估报告。"""
    print("\n" + "=" * 70)
    print("面级别面积评估报告")
    print("=" * 70)

    print(f"\n总体指标:")
    print(f"  总面数:          {metrics['total_faces']}")
    print(f"  正确面数:        {metrics['correct_faces']}")
    print(f"  面级别准确率:    {metrics['face_accuracy']:.4f} ({metrics['face_accuracy']*100:.2f}%)")
    print(f"  总面积:          {metrics['total_area']}")
    print(f"  面积加权准确率:  {metrics['area_accuracy']:.4f} ({metrics['area_accuracy']*100:.2f}%)")
    print(f"  平均面 IoU:      {metrics['mean_face_iou']:.4f}")

    print(f"\n{'类别':<12} {'面数':>6} {'准确率':>10} {'面积':>10} {'面积准确率':>12} {'平均IoU':>10}")
    print("-" * 70)
    for cls_id in range(num_classes):
        cm = metrics["class_metrics"][cls_id]
        print(f"  {cm['name']:<10} {cm['total_faces']:>6} {cm['accuracy']:>10.4f} "
              f"{cm['total_area']:>10} {cm['area_accuracy']:>12.4f} {cm['mean_iou']:>10.4f}")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=GT, 列=Pred):")
    header = "".join(f"{CLASS_NAMES.get(i, f'C{i}'):>8}" for i in range(num_classes))
    print(f"{'':>12}{header}")
    for gt_cls in range(num_classes):
        row = "".join(f"{metrics['confusion_matrix'][gt_cls][p]:>8}" for p in range(num_classes))
        print(f"  {CLASS_NAMES.get(gt_cls, f'C{gt_cls}'):<10}{row}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="面级别面积评估")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="验证集目录 (balanced_dataset/val/)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Mask2Former detectron2 配置文件")
    parser.add_argument("--output_dir", type=str, default="/data/project_m2f/temp_data/eval_output",
                        help="评估结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="实例置信度阈值")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径 (覆盖配置文件中的 MODEL.WEIGHTS)")

    args, unknown = parser.parse_known_args()

    # 读取 class_map.json
    class_map_path = os.path.join(args.val_dir, "class_map.json")
    print(f"加载 class_map: {class_map_path}")
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # class_map 格式: {图片名: {序号: 类别ID}}
    # 需要展平为 {(图片名, 序号): 类别ID}
    flat_class_map = {}
    for img_name, instances in class_map.items():
        for seq_str, cat_id in instances.items():
            flat_class_map[(img_name, seq_str)] = cat_id

    print(f"  图片数: {len(class_map)}")

    # 配置模型
    cfg = setup_cfg(args.config_file)
    if args.weights:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = args.weights
        cfg.freeze()
    print(f"模型权重: {cfg.MODEL.WEIGHTS}")
    predictor = DefaultPredictor(cfg)

    # 图像和掩码目录
    image_dir = os.path.join(args.val_dir, "encoded_views")
    mask_dir = os.path.join(args.val_dir, "masks")

    # 收集验证集图片
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if len(image_files) == 0:
        print(f"[ERROR] 未找到图片: {image_dir}")
        return

    print(f"验证集图片数: {len(image_files)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 逐图评估
    all_results = []
    num_classes = len(CLASS_NAMES)

    for img_file in tqdm(image_files, desc="评估进度"):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        # 读取图像
        image_np = cv2.imread(img_path)
        if image_np is None:
            continue

        # 读取 GT 掩码
        if not os.path.exists(mask_path):
            continue
        gt_mask = np.array(Image.open(mask_path))

        # 获取该图的类别映射
        if img_file not in class_map:
            continue
        img_class_map = class_map[img_file]

        # 构建 GT 面信息
        gt_faces = build_gt_from_mask_and_classmap(gt_mask, img_class_map)
        if len(gt_faces) == 0:
            continue

        # Mask2Former 推理
        outputs = predictor(image_np)
        raw_segmentation, segments_info = detectron2_to_unified_format(outputs, args.threshold)

        # 评估
        results = evaluate_single(gt_faces, raw_segmentation, segments_info)

        # 记录图片名
        for r in results:
            r["image"] = img_file

        all_results.extend(results)

    # 汇总
    metrics = compute_metrics(all_results, num_classes)

    # 打印报告
    print_report(metrics, num_classes)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存汇总指标
    metrics_path = os.path.join(args.output_dir, "eval_results.json")
    save_metrics = {
        "total_faces": metrics["total_faces"],
        "correct_faces": metrics["correct_faces"],
        "face_accuracy": metrics["face_accuracy"],
        "total_area": metrics["total_area"],
        "correct_area": metrics["correct_area"],
        "area_accuracy": metrics["area_accuracy"],
        "mean_face_iou": metrics["mean_face_iou"],
        "class_metrics": {str(k): v for k, v in metrics["class_metrics"].items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, ensure_ascii=False, indent=2)

    # 保存混淆矩阵 CSV
    csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["GT\\Pred"] + [CLASS_NAMES.get(i, f"C{i}") for i in range(num_classes)]
        writer.writerow(header)
        for gt_cls in range(num_classes):
            row = [CLASS_NAMES.get(gt_cls, f"C{gt_cls}")]
            row += [metrics["confusion_matrix"][gt_cls][p] for p in range(num_classes)]
            writer.writerow(row)

    # 保存逐图详细结果
    detail_path = os.path.join(args.output_dir, "eval_details.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存:")
    print(f"  汇总指标: {metrics_path}")
    print(f"  混淆矩阵: {csv_path}")
    print(f"  详细结果: {detail_path}")


if __name__ == "__main__":
    main()
