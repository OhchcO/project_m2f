r"""
实例分割推理脚本（编码图版本）- detectron2 Mask2Former 适配版。

面信息编码在图像通道中：
- R 通道：面的几何类型（0/50/100/150/200，对应平面/圆柱面/圆锥面/球面/其他面）
- G + B 通道：面 ID（G*256 + B，每个不同值代表一个面）

用法示例：
python ins_inference_encoded.py \
  --image /path/to/encoded_image.png \
  --config_file /path/to/mask2former/config.yaml \
  --opts MODEL.WEIGHTS /path/to/model.pkl

输出：
1. instance_mask.png：实例ID图
2. class_mask.png：类别ID图
3. class_map.json：实例ID → 类别ID + 面类型 + 置信度 + 面积 + bbox
4. visualization.png：可视化结果
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

# Mask2Former
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Mask2Former"))
from mask2former import add_maskformer2_config

# 本地配置
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLASS_NAMES, NUM_CLASSES,
    ENCODER_TYPE_GAP, ENCODER_TYPE_R_BASE, ENCODER_TYPE_NAMES,
    INFERENCE_THRESHOLD as DEFAULT_THRESHOLD, INFERENCE_MASK_THRESHOLD as DEFAULT_MASK_THRESHOLD,
    INFERENCE_MIN_RATIO as DEFAULT_MIN_RATIO, INFERENCE_MIN_FACE_AREA as DEFAULT_MIN_FACE_AREA,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 类别颜色（自动生成，用于可视化）
COLOR_PALETTE = [
    np.array([255, 0, 0], dtype=np.uint8),     # 红
    np.array([0, 255, 0], dtype=np.uint8),     # 绿
    np.array([0, 0, 255], dtype=np.uint8),     # 蓝
    np.array([255, 255, 0], dtype=np.uint8),   # 黄
    np.array([255, 0, 255], dtype=np.uint8),   # 品红
    np.array([0, 255, 255], dtype=np.uint8),   # 青
    np.array([128, 0, 255], dtype=np.uint8),   # 紫
    np.array([255, 128, 0], dtype=np.uint8),   # 橙
]
CLASS_COLORS = {cls_id: COLOR_PALETTE[i % len(COLOR_PALETTE)]
                for i, cls_id in enumerate(CLASS_NAMES.keys())}

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "instance_inference_encoded")


# ---------------------------------------------------------------------------
# R 通道 → 面的几何类型（范围匹配，从 config 生成）
# ---------------------------------------------------------------------------
def _build_r_ranges():
    ranges = []
    for type_id, r_base in sorted(ENCODER_TYPE_R_BASE.items()):
        r_max = r_base + ENCODER_TYPE_GAP - 1
        ranges.append((r_base, r_max, ENCODER_TYPE_NAMES[type_id]))
    return ranges

_R_RANGES = _build_r_ranges()


def get_face_type_name(r_value):
    for r_min, r_max, name in _R_RANGES:
        if r_min <= r_value <= r_max:
            return name
    return f"未知({r_value})"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_legend_patches():
    patches = []
    for class_id, class_name in CLASS_NAMES.items():
        color = CLASS_COLORS[class_id] / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{class_id}={class_name}"))
    return patches


def colorize_class_mask(class_mask):
    colored = np.full((*class_mask.shape, 3), 255, dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[class_mask == class_id] = color
    return colored


def colorize_instance_mask(instance_mask):
    colored = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
    instance_ids = [int(x) for x in np.unique(instance_mask) if x != 0]
    for instance_id in instance_ids:
        rng = np.random.default_rng(instance_id)
        colored[instance_mask == instance_id] = rng.integers(30, 256, size=3, dtype=np.uint8)
    return colored


def mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


# ---------------------------------------------------------------------------
# 从编码图提取面掩码
# ---------------------------------------------------------------------------
def extract_faces_from_encoded_image(image_rgb, min_area=10):
    if isinstance(image_rgb, torch.Tensor):
        arr = image_rgb.cpu().numpy()
    else:
        arr = image_rgb

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)

    h, w = arr.shape[:2]
    r_channel = arr[:, :, 0].astype(np.uint8)
    g_channel = arr[:, :, 1].astype(np.uint16)
    b_channel = arr[:, :, 2].astype(np.uint16)

    face_id_map = (g_channel << 8) | b_channel

    face_masks = {}
    face_type_counts = defaultdict(int)
    unique_ids = np.unique(face_id_map)

    for face_id_raw in unique_ids:
        face_mask = face_id_map == face_id_raw
        area = int(face_mask.sum())
        if area < min_area:
            continue

        face_r_values = r_channel[face_mask]
        r_unique, r_counts = np.unique(face_r_values, return_counts=True)
        dominant_face_type = int(r_unique[np.argmax(r_counts)])

        face_id = int(face_id_raw)
        face_masks[face_id] = {
            "mask": face_mask,
            "face_type": dominant_face_type,
            "face_type_name": get_face_type_name(dominant_face_type),
            "area": area,
        }
        face_type_counts[dominant_face_type] += 1

    print(f"从编码图提取到 {len(face_masks)} 个面 (min_area={min_area})")
    for ft, cnt in sorted(face_type_counts.items()):
        print(f"  {get_face_type_name(ft)}: {cnt} 个面")
    return face_masks


def print_face_summary(face_masks):
    print("\n编码图面信息汇总:")
    ft_counts = defaultdict(int)
    for face_info in face_masks.values():
        ft_counts[face_info["face_type"]] += 1
    print(f"  总面数: {len(face_masks)}")
    for ft, cnt in sorted(ft_counts.items()):
        print(f"  R={ft:>3} {get_face_type_name(ft):>6}: {cnt} 个面")


# ---------------------------------------------------------------------------
# detectron2 Mask2Former 推理 → 统一格式转换
# ---------------------------------------------------------------------------
def setup_cfg(config_file, opts=None):
    """构建 detectron2 配置"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def detectron2_to_unified_format(outputs, threshold):
    """
    将 detectron2 Mask2Former 输出转换为统一格式。

    Returns:
        raw_segmentation: (H, W) np.uint16, 每个像素为实例 ID（0=背景）
        segments_info: list of {"id": int, "label_id": int, "score": float}
    """
    instances = outputs["instances"].to("cpu")
    h, w = instances.image_size

    # 过滤低置信度
    valid = instances.scores >= threshold
    instances = instances[valid]

    raw_segmentation = np.zeros((h, w), dtype=np.uint16)
    segments_info = []

    for i in range(len(instances)):
        instance_id = i + 1
        mask = instances.pred_masks[i].numpy().astype(bool)
        class_id = int(instances.pred_classes[i])
        score = float(instances.scores[i])

        raw_segmentation[mask] = instance_id
        segments_info.append({
            "id": instance_id,
            "label_id": class_id,
            "score": score,
        })

    return raw_segmentation, segments_info


# ---------------------------------------------------------------------------
# 面后处理：按面边界裁剪 + 投票确定类别
# ---------------------------------------------------------------------------
def postprocess_with_encoded_faces(raw_segmentation, segments_info, face_masks, min_ratio=0.5):
    info_by_raw_id = {int(s["id"]): s for s in segments_info}
    instance_mask = np.zeros_like(raw_segmentation, dtype=np.uint16)
    class_mask = np.full_like(raw_segmentation, 255, dtype=np.uint8)
    class_map = {}

    empty_faces = 0
    low_ratio_faces = 0
    new_instance_id = 1

    for face_id, face_info in face_masks.items():
        face_mask_bool = face_info["mask"]
        face_type = face_info["face_type"]
        face_type_name = face_info["face_type_name"]

        face_pixels = raw_segmentation[face_mask_bool]
        face_pixels_nonzero = face_pixels[face_pixels != 0]

        if len(face_pixels_nonzero) == 0:
            empty_faces += 1
            continue

        unique_ids, counts = np.unique(face_pixels_nonzero, return_counts=True)
        max_idx = int(np.argmax(counts))
        raw_instance_id = int(unique_ids[max_idx])
        ratio = float(counts[max_idx] / face_mask_bool.sum())

        if ratio < min_ratio:
            low_ratio_faces += 1
            continue

        segment = info_by_raw_id.get(raw_instance_id)
        if segment is None:
            continue

        pred_class = int(segment["label_id"])
        score = float(segment.get("score", 0.0))

        instance_mask[face_mask_bool] = new_instance_id
        class_mask[face_mask_bool] = pred_class
        class_map[str(new_instance_id)] = {
            "class_id": pred_class,
            "class_name": CLASS_NAMES.get(pred_class, f"class_{pred_class}"),
            "score": score,
            "area": face_info["area"],
            "bbox": mask_to_bbox(face_mask_bool),
            "face_id": int(face_id),
            "face_type": face_type,
            "face_type_name": face_type_name,
            "vote_ratio": round(ratio, 4),
        }
        new_instance_id += 1

    print(f"后处理完成: {len(face_masks)} 个面 → {new_instance_id - 1} 个有效实例")
    print(f"  面内无预测: {empty_faces} 个面")
    print(f"  投票占比不足({min_ratio}): {low_ratio_faces} 个面")
    return instance_mask, class_mask, class_map


# ---------------------------------------------------------------------------
# 保存 & 可视化
# ---------------------------------------------------------------------------
def save_outputs(instance_mask, class_mask, class_map, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    instance_mask_path = os.path.join(output_dir, f"{prefix}_instance_mask.png")
    class_mask_path = os.path.join(output_dir, f"{prefix}_class_mask.png")
    class_map_path = os.path.join(output_dir, f"{prefix}_class_map.json")

    Image.fromarray(instance_mask).save(instance_mask_path)
    Image.fromarray(class_mask).save(class_mask_path)
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    return instance_mask_path, class_mask_path, class_map_path


def visualize(image, unc_image, raw_instance_mask, raw_class_mask,
              processed_instance_mask, processed_class_mask,
              output_dir, prefix):

    os.makedirs(output_dir, exist_ok=True)
    legend_patches = build_legend_patches()

    raw_instance_color = colorize_instance_mask(raw_instance_mask)
    raw_class_color = colorize_class_mask(raw_class_mask)
    raw_overlay = raw_class_color.astype(np.float32) / 255.0
    raw_overlay[raw_class_mask == 255] = np.nan

    if processed_instance_mask is None or processed_class_mask is None:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(raw_instance_color)
        axes[1].set_title("Raw Instance Mask")
        axes[1].axis("off")
        axes[2].imshow(raw_class_color)
        axes[2].set_title("Raw Class Mask")
        axes[2].axis("off")
        axes[2].legend(handles=legend_patches, loc="upper right", fontsize=9)
        axes[3].imshow(unc_image)
        axes[3].imshow(raw_overlay, alpha=0.7)
        axes[3].set_title("Raw Overlay")
        axes[3].axis("off")
        axes[3].legend(handles=legend_patches, loc="upper right", fontsize=9)
    else:
        proc_instance_color = colorize_instance_mask(processed_instance_mask)
        proc_class_color = colorize_class_mask(processed_class_mask)
        proc_overlay = proc_class_color.astype(np.float32) / 255.0
        proc_overlay[processed_class_mask == 255] = np.nan

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(raw_instance_color)
        axes[0, 1].set_title("Raw Instance Mask")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(raw_class_color)
        axes[0, 2].set_title("Raw Class Mask")
        axes[0, 2].axis("off")
        axes[0, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)
        axes[0, 3].imshow(unc_image)
        axes[0, 3].imshow(raw_overlay, alpha=0.7)
        axes[0, 3].set_title("Raw Overlay")
        axes[0, 3].axis("off")
        axes[0, 3].legend(handles=legend_patches, loc="upper right", fontsize=9)

        axes[1, 0].imshow(image)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(proc_instance_color)
        axes[1, 1].set_title("Processed Instance Mask")
        axes[1, 1].axis("off")
        axes[1, 2].imshow(proc_class_color)
        axes[1, 2].set_title("Processed Class Mask")
        axes[1, 2].axis("off")
        axes[1, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)
        axes[1, 3].imshow(unc_image)
        axes[1, 3].imshow(proc_overlay, alpha=0.7)
        axes[1, 3].set_title("Processed Overlay")
        axes[1, 3].axis("off")
        axes[1, 3].legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def format_class_stats(class_map, title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"检测到实例数: {len(class_map)}")

    class_counts = defaultdict(int)
    for instance_id, info in class_map.items():
        class_id = int(info["class_id"])
        class_counts[class_id] += 1
        face_type_str = info.get("face_type_name", "?")
        ratio_str = f"vote={info.get('vote_ratio', '?')}" if "vote_ratio" in info else ""
        print(
            f"  实例 {instance_id}: 类别 {class_id}({info['class_name']}), "
            f"面类型={face_type_str}, score={info['score']:.4f}, "
            f"area={info['area']}, bbox={info['bbox']}, {ratio_str}"
        )

    print("\n类别实例统计:")
    for class_id in sorted(CLASS_NAMES.keys()):
        print(f"  类别 {class_id}({CLASS_NAMES[class_id]}): {class_counts[class_id]} 个实例")


# ---------------------------------------------------------------------------
# 主推理函数
# ---------------------------------------------------------------------------
def run_inference(image_path, config_file, unc_image_path=None,
                  output_dir=DEFAULT_OUTPUT_DIR, threshold=DEFAULT_THRESHOLD,
                  min_ratio=DEFAULT_MIN_RATIO, min_face_area=DEFAULT_MIN_FACE_AREA,
                  opts=None, device_name="auto"):
    # Step 0: 配置模型
    print("配置 detectron2 Mask2Former...")
    cfg = setup_cfg(config_file, opts)
    print(f"模型权重: {cfg.MODEL.WEIGHTS}")

    predictor = DefaultPredictor(cfg)
    print(f"设备: GPU")

    # Step 1: 读取图像
    print(f"\n加载编码图像: {image_path}")
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return None
    image_pil = Image.open(image_path).convert("RGB")

    if unc_image_path and os.path.exists(unc_image_path):
        unc_image = Image.open(unc_image_path).convert("RGB")
    else:
        unc_image = image_pil.convert("L").convert("RGB")

    # Step 2: 从 R+GB 编码图提取面信息
    print("\n提取编码图面信息 (R=面几何类型, GB=面ID)...")
    face_masks = extract_faces_from_encoded_image(np.array(image_pil), min_area=min_face_area)
    print_face_summary(face_masks)

    # Step 3: Mask2Former 推理
    print("\n运行实例分割推理...")
    outputs = predictor(image_np)
    raw_segmentation, segments_info = detectron2_to_unified_format(outputs, threshold)

    print(f"检测到 {len(segments_info)} 个实例")

    # Step 4: 构建原始实例 mask
    raw_instance_mask = np.zeros_like(raw_segmentation, dtype=np.uint16)
    raw_class_mask = np.full_like(raw_segmentation, 255, dtype=np.uint8)
    raw_class_map = {}

    new_id = 1
    for segment in sorted(segments_info, key=lambda x: x["id"]):
        raw_id = int(segment["id"])
        class_id = int(segment["label_id"])
        score = float(segment.get("score", 0.0))
        bin_mask = raw_segmentation == raw_id
        area = int(bin_mask.sum())
        if area == 0:
            continue
        raw_instance_mask[bin_mask] = new_id
        raw_class_mask[bin_mask] = class_id
        raw_class_map[str(new_id)] = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "score": score,
            "area": area,
            "bbox": mask_to_bbox(bin_mask),
        }
        new_id += 1

    prefix = os.path.splitext(os.path.basename(image_path))[0]
    raw_output_dir = os.path.join(output_dir, "raw")
    raw_paths = save_outputs(raw_instance_mask, raw_class_mask, raw_class_map, raw_output_dir, prefix)

    # Step 5: 面后处理
    processed_instance_mask = None
    processed_class_mask = None
    processed_class_map = None
    processed_paths = None

    if len(face_masks) > 0:
        print("\n开始面边界后处理...")
        processed_instance_mask, processed_class_mask, processed_class_map = \
            postprocess_with_encoded_faces(
                raw_segmentation, segments_info, face_masks, min_ratio=min_ratio
            )
        processed_output_dir = os.path.join(output_dir, "processed")
        processed_paths = save_outputs(
            processed_instance_mask, processed_class_mask,
            processed_class_map, processed_output_dir, prefix
        )

    # Step 6: 可视化
    visualization_path = visualize(
        image=image_pil,
        unc_image=unc_image,
        raw_instance_mask=raw_instance_mask,
        raw_class_mask=raw_class_mask,
        processed_instance_mask=processed_instance_mask,
        processed_class_mask=processed_class_mask,
        output_dir=output_dir,
        prefix=prefix,
    )

    # Step 7: 统计
    format_class_stats(raw_class_map, "原始实例分割结果")
    if processed_class_map is not None:
        format_class_stats(processed_class_map, "面边界后处理实例分割结果")

    print("\n结果已保存:")
    print(f"  原始实例ID图: {raw_paths[0]}")
    print(f"  原始类别ID图: {raw_paths[1]}")
    print(f"  原始class_map: {raw_paths[2]}")
    if processed_paths:
        print(f"  后处理实例ID图: {processed_paths[0]}")
        print(f"  后处理类别ID图: {processed_paths[1]}")
        print(f"  后处理class_map: {processed_paths[2]}")
    print(f"  可视化图: {visualization_path}")

    return {
        "raw_instance_mask": raw_instance_mask,
        "raw_class_mask": raw_class_mask,
        "raw_class_map": raw_class_map,
        "processed_instance_mask": processed_instance_mask,
        "processed_class_mask": processed_class_mask,
        "processed_class_map": processed_class_map,
        "face_masks": face_masks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Former Instance Segmentation (detectron2, Encoded Image)")
    parser.add_argument("--image", type=str, required=True,
                        help="编码图像路径 (R=面几何类型, GB=面ID)")
    parser.add_argument("--unc_image", type=str, default=None,
                        help="叠加底图路径，不传则自动使用灰度图")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Mask2Former detectron2 配置文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="实例置信度阈值")
    parser.add_argument("--min_ratio", type=float, default=DEFAULT_MIN_RATIO,
                        help="面投票阈值，面内主实例占比低于此则忽略该面")
    parser.add_argument("--min_face_area", type=int, default=DEFAULT_MIN_FACE_AREA,
                        help="最小面面积（像素），小于此值的忽略")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="推理设备")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="修改配置选项，如 MODEL.WEIGHTS /path/to/model.pkl")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"图像不存在: {args.image}")
    elif not os.path.exists(args.config_file):
        print(f"配置文件不存在: {args.config_file}")
    else:
        run_inference(
            image_path=args.image,
            config_file=args.config_file,
            unc_image_path=args.unc_image,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_ratio=args.min_ratio,
            min_face_area=args.min_face_area,
            opts=args.opts,
            device_name=args.device,
        )
