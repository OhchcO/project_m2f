"""
Evaluate MFR single-view Mask2Former checkpoints on the video-format dataset.

The dataset format is the same as the multi-view/video route:
  val/models.json
  val/encoded_views/*.png
  val/face_id_maps/*.npy

For each CAD model, this script runs the 14 views independently with a
single-image Mask2Former checkpoint, projects predicted masks back to face ids,
votes per face across views, and reports the same face-level metrics as the
multi-view evaluator.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK2FORMER_DIR = os.path.join(PROJECT_DIR, "Mask2Former")
if MASK2FORMER_DIR not in sys.path:
    sys.path.insert(0, MASK2FORMER_DIR)

from mask2former import add_maskformer2_config  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from eval_area_metrics import (  # noqa: E402
    compute_metrics,
    print_report,
    remap_results_for_coarse_eval,
    save_eval_outputs,
)
from eval_mfr_multiview import (  # noqa: E402
    MFR_CLASS_NAMES,
    MFR_COARSE_CLASS_NAMES,
    MFR_FINE_TO_COARSE_CLASS,
    build_gt_faces,
    load_models,
    resolve_path,
)


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.DATASETS.TRAIN = ("mfr_singleview_train",)
    cfg.DATASETS.TEST = ("mfr_singleview_val",)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = args.score_threshold
    cfg.freeze()

    MetadataCatalog.get("mfr_singleview_train").set(thing_classes=list(MFR_CLASS_NAMES.values()))
    MetadataCatalog.get("mfr_singleview_val").set(thing_classes=list(MFR_CLASS_NAMES.values()))
    return cfg


def read_singleview_inputs(model_record, val_dir):
    frames = []
    face_id_maps = []
    views = sorted(model_record["views"], key=lambda item: item["view_id"])
    for view in views:
        image_path = resolve_path(val_dir, view["image"])
        face_map_path = resolve_path(val_dir, view["face_id_map"])
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        frames.append(image_bgr)
        face_id_maps.append(np.load(face_map_path).astype(np.int32))
    return frames, face_id_maps


def project_singleview_predictions_to_faces(outputs_per_view, face_id_maps, score_threshold, mask_threshold):
    face_label_votes = defaultdict(Counter)
    face_label_scores = defaultdict(float)
    face_label_sets = defaultdict(set)
    view_summaries = []

    for view_idx, (outputs, face_id_map) in enumerate(zip(outputs_per_view, face_id_maps), start=1):
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        labels = instances.pred_classes.numpy() if instances.has("pred_classes") else []
        view_summaries.append(
            {
                "view_id": view_idx,
                "num_instances": int(len(scores)),
                "labels": [int(label) for label in labels],
                "scores": [float(score) for score in scores],
            }
        )

        for score, label, pred_mask in zip(scores, labels, masks):
            score = float(score)
            if score < score_threshold:
                continue
            label = int(label)
            pred_mask = pred_mask > mask_threshold
            if pred_mask.shape != face_id_map.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (face_id_map.shape[1], face_id_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            face_ids = face_id_map[pred_mask]
            face_ids = face_ids[face_ids >= 0]
            if face_ids.size == 0:
                continue

            unique_face_ids, counts = np.unique(face_ids, return_counts=True)
            instance_face_set = {int(face_id) for face_id in unique_face_ids}
            for face_id, count in zip(unique_face_ids, counts):
                face_id = int(face_id)
                key = (face_id, label)
                face_label_votes[face_id][label] += int(count)
                face_label_scores[key] = max(face_label_scores[key], score)
                face_label_sets[key].update(instance_face_set)

    face_predictions = {}
    for face_id, votes in face_label_votes.items():
        label, pixels = votes.most_common(1)[0]
        total_pixels = sum(votes.values())
        key = (face_id, label)
        face_predictions[face_id] = {
            "label_id": int(label),
            "score": float(face_label_scores.get(key, 0.0)),
            "vote_pixels": int(pixels),
            "vote_ratio": pixels / total_pixels if total_pixels else 0.0,
            "pred_face_set": face_label_sets.get(key, set()),
        }
    return face_predictions, view_summaries


def evaluate_model(model_record, face_predictions, face_id_maps):
    gt_faces = build_gt_faces(model_record, face_id_maps)
    results = []
    for gt in gt_faces:
        face_id = int(gt["instance_id"])
        pred = face_predictions.get(face_id)
        gt_feature = {
            int(fid)
            for feature in model_record["features"]
            if int(feature["instance_id"]) == int(gt["feature_instance_id"])
            for fid in feature["face_ids"]
        }
        if pred is None:
            results.append(
                {
                    "model_id": model_record["model_id"],
                    "instance_id": face_id,
                    "feature_instance_id": gt["feature_instance_id"],
                    "gt_class": gt["gt_class"],
                    "pred_class": -1,
                    "is_correct": False,
                    "area": gt["area"],
                    "face_iou": 0.0,
                    "vote_ratio": 0.0,
                    "score": 0.0,
                }
            )
            continue

        pred_face_set = pred["pred_face_set"]
        union = gt_feature | pred_face_set
        inter = gt_feature & pred_face_set
        face_iou = len(inter) / len(union) if union else 0.0
        pred_class = int(pred["label_id"])
        results.append(
            {
                "model_id": model_record["model_id"],
                "instance_id": face_id,
                "feature_instance_id": gt["feature_instance_id"],
                "gt_class": gt["gt_class"],
                "pred_class": pred_class,
                "is_correct": pred_class == gt["gt_class"],
                "area": gt["area"],
                "face_iou": face_iou,
                "vote_ratio": pred["vote_ratio"],
                "score": pred["score"],
            }
        )
    return results


def save_prediction_summary(output_dir, model_record, view_summaries, face_predictions):
    pred_dir = os.path.join(output_dir, "pred_details")
    os.makedirs(pred_dir, exist_ok=True)
    payload = {
        "model_id": model_record["model_id"],
        "views": view_summaries,
        "face_predictions": {
            str(face_id): {
                "label_id": int(pred["label_id"]),
                "score": float(pred["score"]),
                "vote_ratio": float(pred["vote_ratio"]),
            }
            for face_id, pred in face_predictions.items()
        },
    }
    path = os.path.join(pred_dir, f"{model_record['model_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MFR single-view Mask2Former on multi-view dataset")
    parser.add_argument("--val_dir", default="/data/m2f/temp_data/multiview_feature_dataset/val")
    parser.add_argument("--config_file", default="/data/m2f/Mask2Former/configs/mfr_singleview/maskformer2_R50_512.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", default="/data/m2f/temp_data/eval_mfr_singleview")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--min_size_test", type=int, default=512)
    parser.add_argument("--eval_class_mode", choices=["fine", "coarse", "both"], default="both")
    parser.add_argument("--max_models", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    models = load_models(args.val_dir)
    if args.max_models:
        models = models[: args.max_models]

    all_results = []
    for model_record in tqdm(models, desc="Evaluating MFR singleview"):
        frames, face_id_maps = read_singleview_inputs(model_record, args.val_dir)
        outputs_per_view = [predictor(frame) for frame in frames]
        face_predictions, view_summaries = project_singleview_predictions_to_faces(
            outputs_per_view,
            face_id_maps,
            args.score_threshold,
            args.mask_threshold,
        )
        save_prediction_summary(args.output_dir, model_record, view_summaries, face_predictions)
        all_results.extend(evaluate_model(model_record, face_predictions, face_id_maps))

    saved_paths = []
    if args.eval_class_mode in ("fine", "both"):
        fine_metrics = compute_metrics(all_results, MFR_CLASS_NAMES)
        print_report(fine_metrics, MFR_CLASS_NAMES, "MFR single-view fine face-level report")
        suffix = "fine" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(args.output_dir, fine_metrics, MFR_CLASS_NAMES, all_results, suffix))

    if args.eval_class_mode in ("coarse", "both"):
        coarse_results = remap_results_for_coarse_eval(all_results, MFR_FINE_TO_COARSE_CLASS)
        coarse_metrics = compute_metrics(coarse_results, MFR_COARSE_CLASS_NAMES)
        print_report(coarse_metrics, MFR_COARSE_CLASS_NAMES, "MFR single-view coarse face-level report")
        suffix = "coarse" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(args.output_dir, coarse_metrics, MFR_COARSE_CLASS_NAMES, coarse_results, suffix))

    print("\n结果已保存:")
    for metrics_path, csv_path, detail_path in saved_paths:
        print(f"  汇总指标: {metrics_path}")
        print(f"  混淆矩阵: {csv_path}")
        print(f"  详细结果: {detail_path}")
    print(f"  预测详情: {os.path.join(args.output_dir, 'pred_details')}")


if __name__ == "__main__":
    main()
