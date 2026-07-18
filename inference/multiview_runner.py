# -*- coding: utf-8 -*-
"""Headless multi-view inference helpers shared by the web UI.

The Qt desktop UI owns windowing only.  This module owns STEP rendering,
Detectron2 VideoMaskFormer inference, face-level voting, and mesh serialization.
"""

import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import pyvista as pv
from PIL import Image

from inference_config import OUTPUT_HEIGHT, OUTPUT_WIDTH, TYPE_R_BASE, class_color, class_name
from core.color_encoder import FaceColorEncoder, build_reverse_gb_map
from core.geometry import get_cube_14_view_directions, get_parallel_scale, get_viewup, rgb_to_float
from core.step_io import load_step_faces
from inference.postprocess import extract_faces_from_encoded_image, postprocess_with_encoded_faces


DEFAULT_M2F_ROOT = "/data/m2f/Mask2Former"
DEFAULT_VIDEO_CONFIG = os.path.join(
    DEFAULT_M2F_ROOT, "configs", "mfr_multiview", "video_maskformer2_R50_bs1_14view.yaml"
)
DEFAULT_CONFIG = DEFAULT_VIDEO_CONFIG
DEFAULT_SINGLEVIEW_CONFIG = os.path.join(
    DEFAULT_M2F_ROOT, "configs", "mfr_singleview", "maskformer2_R50_512.yaml"
)
DEFAULT_VIDEO_WEIGHT_CANDIDATES = [
    "/data/m2f/temp_data/mfr_multiview_server_bs1_512_train2k_output/model_final.pth",
    "/data/m2f/result/mfr_multiview_server_bs1_512_output/model_final.pth",
    "/hy-tmp/mfr_multiview_MFRInstSegM2F_2100_bs1_512_ep50_output/model_final.pth",
]
DEFAULT_SINGLEVIEW_WEIGHT_CANDIDATES = [
    "/hy-tmp/mfr_singleview_MFRInstSegM2F_2100_bs1_512_ep50_output/model_final.pth",
    "/mnt/e/wsl/result/MFRInstSegM2F_2100_singleview_512_output/model_final.pth",
]
DEFAULT_WEIGHTS = next(
    (path for path in DEFAULT_VIDEO_WEIGHT_CANDIDATES if os.path.exists(path)),
    DEFAULT_VIDEO_WEIGHT_CANDIDATES[0],
)
DEFAULT_SINGLEVIEW_WEIGHTS = next(
    (path for path in DEFAULT_SINGLEVIEW_WEIGHT_CANDIDATES if os.path.exists(path)),
    DEFAULT_SINGLEVIEW_WEIGHT_CANDIDATES[0],
)
_PREDICTOR_CACHE = {}
_PREDICTOR_CACHE_LOCK = threading.Lock()


def _elapsed(start_time):
    return f"{time.perf_counter() - start_time:.2f}s"


def render_encoded_views(step_data, output_dir, directions, progress=None):
    """Render 14 RGB encoded views: R=surface type/area, GB=face id."""
    encoded_dir = os.path.join(output_dir, "encoded_views")
    os.makedirs(encoded_dir, exist_ok=True)

    faces = step_data["faces"]
    bounds = step_data["bounds"]
    max_area = max((float(face["mesh"].area) for face in faces), default=1.0)
    encoder = FaceColorEncoder(len(faces), shuffle=False)

    colors = []
    for face in faces:
        type_id = {
            "Plane": 0,
            "Cylinder": 1,
            "Cone": 2,
            "Sphere": 3,
        }.get(face["face_type"], 4)
        area_ratio = float(face["mesh"].area) / max_area if max_area > 0 else 0.0
        r = min(255, TYPE_R_BASE[type_id] + int(round(area_ratio * 50)))
        g, b = encoder.gb_mapping[face["face_id"]]
        colors.append((r, g, b))

    mapping_path = os.path.join(output_dir, "encoding_map.json")
    with open(mapping_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "encoder": {
                    "faces": {
                        str(face["face_id"]): {"G": colors[idx][1], "B": colors[idx][2]}
                        for idx, face in enumerate(faces)
                    }
                }
            },
            file,
            indent=2,
        )

    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    center = (
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    )
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    dist = max(max_dim, 1.0) * 3.0

    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter.disable_anti_aliasing()
    plotter.set_background("white")
    plotter.camera.SetParallelProjection(True)
    for face, color in zip(faces, colors):
        plotter.add_mesh(face["mesh"], color=rgb_to_float(color), lighting=False, smooth_shading=False)

    image_paths = []
    for index, direction in enumerate(directions, start=1):
        if progress:
            progress(f"渲染编码视图 {index}/{len(directions)}")
        viewup = get_viewup(direction)
        camera_position = (
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        plotter.camera_position = [camera_position, center, viewup]
        parallel_scale = max(
            get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=aspect_ratio, margin=1.10),
            0.01,
        )
        plotter.camera.SetParallelScale(parallel_scale)
        plotter.render()
        image_path = os.path.join(encoded_dir, f"{index:06d}.png")
        plotter.screenshot(image_path)
        image_paths.append(image_path)

    plotter.close()
    return encoded_dir, image_paths, mapping_path


def load_video_predictor(m2f_root, config_path, weights_path, device):
    root = os.path.abspath(m2f_root)
    config_path = os.path.abspath(config_path)
    weights_path = os.path.abspath(weights_path)
    cache_key = (root, config_path, weights_path, device)
    with _PREDICTOR_CACHE_LOCK:
        cached = _PREDICTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached, True

    if root not in sys.path:
        sys.path.insert(0, root)
    demo_video_dir = os.path.join(root, "demo_video")
    if demo_video_dir not in sys.path:
        sys.path.insert(0, demo_video_dir)

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from mask2former_video import add_maskformer2_video_config
    from demo_video.predictor import VideoPredictor

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    predictor = VideoPredictor(cfg)
    with _PREDICTOR_CACHE_LOCK:
        _PREDICTOR_CACHE[cache_key] = predictor
    return predictor, False


def load_singleview_predictor(m2f_root, config_path, weights_path, device):
    root = os.path.abspath(m2f_root)
    config_path = os.path.abspath(config_path)
    weights_path = os.path.abspath(weights_path)
    cache_key = ("singleview", root, config_path, weights_path, device)
    with _PREDICTOR_CACHE_LOCK:
        cached = _PREDICTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached, True

    if root not in sys.path:
        sys.path.insert(0, root)

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    with _PREDICTOR_CACHE_LOCK:
        _PREDICTOR_CACHE[cache_key] = predictor
    return predictor, False


def masks_to_segmentation(frame_masks, scores, labels, threshold, height, width):
    segmentation = np.zeros((height, width), dtype=np.int32)
    segments_info = []
    order = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
    instance_id = 1
    for prediction_index in order:
        score = float(scores[prediction_index])
        if score < threshold:
            continue
        mask = frame_masks[prediction_index]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask).astype(bool)
        if mask.shape != segmentation.shape:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        segmentation[mask] = instance_id
        segments_info.append({"id": instance_id, "label_id": int(labels[prediction_index]), "score": score})
        instance_id += 1
    return segmentation, segments_info


def colorize_segments(segmentation, segments_info):
    colored = np.full((*segmentation.shape, 3), 255, dtype=np.uint8)
    label_by_segment = {int(item["id"]): int(item["label_id"]) for item in segments_info}
    for segment_id, label_id in label_by_segment.items():
        colored[segmentation == segment_id] = class_color(label_id)
    return colored


def instances_to_segmentation(outputs, threshold, height, width):
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return np.zeros((height, width), dtype=np.int32), []

    order = sorted(range(len(instances)), key=lambda idx: float(instances.scores[idx]), reverse=True)
    segmentation = np.zeros((height, width), dtype=np.int32)
    segments_info = []
    instance_id = 1
    for index in order:
        score = float(instances.scores[index])
        if score < threshold:
            continue
        mask = instances.pred_masks[index].numpy().astype(bool)
        if mask.shape != segmentation.shape:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        segmentation[mask] = instance_id
        segments_info.append(
            {"id": instance_id, "label_id": int(instances.pred_classes[index]), "score": score}
        )
        instance_id += 1
    return segmentation, segments_info


def build_view_face_labels(step_data, class_map):
    face_labels = {
        face["face_id"]: {"class_id": -1, "class_name": "background", "score": 0.0, "votes": 0}
        for face in step_data["faces"]
    }
    for info in class_map.values():
        face_id = int(info["face_id"])
        class_id = int(info["class_id"])
        face_labels[face_id] = {
            "class_id": class_id,
            "class_name": class_name(class_id),
            "score": float(info.get("score", 0.0)),
            "votes": 1,
            "color": class_color(class_id),
        }
    return face_labels


def build_face_labels(step_data, face_predictions):
    face_labels = {}
    for face in step_data["faces"]:
        face_id = face["face_id"]
        votes = face_predictions.get(face_id, [])
        if not votes:
            face_labels[face_id] = {"class_id": -1, "class_name": "background", "score": 0.0, "votes": 0}
            continue
        class_scores = defaultdict(float)
        class_counts = Counter()
        for class_id, score in votes:
            class_scores[class_id] += score
            class_counts[class_id] += 1
        best_class_id = max(class_scores, key=class_scores.get)
        face_labels[face_id] = {
            "class_id": int(best_class_id),
            "class_name": class_name(best_class_id),
            "score": round(class_scores[best_class_id] / max(class_counts[best_class_id], 1), 4),
            "votes": int(class_counts[best_class_id]),
        }
    return face_labels


def run_detectron2_multiview(
    step_data,
    m2f_root,
    config_path,
    weights_path,
    output_dir,
    device,
    score_threshold,
    min_ratio,
    min_face_area,
    progress=None,
):
    total_start = time.perf_counter()
    directions = get_cube_14_view_directions()
    render_start = time.perf_counter()
    encoded_dir, image_paths, mapping_path = render_encoded_views(step_data, output_dir, directions, progress)
    gb_to_fid = build_reverse_gb_map(mapping_path)
    if progress:
        progress(f"14 视角渲染完成，用时 {_elapsed(render_start)}")

    if progress:
        progress("加载 Detectron2 VideoMaskFormer 权重")
    load_start = time.perf_counter()
    predictor, cache_hit = load_video_predictor(m2f_root, config_path, weights_path, device)
    if progress:
        progress(f"模型{'缓存命中' if cache_hit else '加载完成'}，用时 {_elapsed(load_start)}")

    frames = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("编码视图读取失败")

    if progress:
        progress("运行 14 视角视频推理")
    infer_start = time.perf_counter()
    predictions = predictor(frames)
    if progress:
        progress(f"模型推理完成，用时 {_elapsed(infer_start)}")
    scores = list(predictions.get("pred_scores", []))
    labels = list(predictions.get("pred_labels", []))
    pred_masks = list(predictions.get("pred_masks", []))

    height, width = frames[0].shape[:2]
    face_predictions = {face["face_id"]: [] for face in step_data["faces"]}
    result_dir = os.path.join(output_dir, "frame_results")
    os.makedirs(result_dir, exist_ok=True)

    project_start = time.perf_counter()
    for frame_index, image_path in enumerate(image_paths):
        if progress:
            progress(f"回投第 {frame_index + 1}/{len(image_paths)} 个视角")
        frame_masks = []
        for instance_masks in pred_masks:
            if frame_index < len(instance_masks):
                frame_masks.append(instance_masks[frame_index])
        segmentation, segments_info = masks_to_segmentation(
            frame_masks, scores, labels, score_threshold, height, width
        )
        encoded_rgb = np.array(Image.open(image_path).convert("RGB"))
        face_masks = extract_faces_from_encoded_image(encoded_rgb, min_area=min_face_area, gb_to_fid=gb_to_fid)
        _, _, class_map = postprocess_with_encoded_faces(segmentation, segments_info, face_masks, min_ratio=min_ratio)
        view_dir = os.path.join(result_dir, f"{frame_index + 1:06d}")
        os.makedirs(view_dir, exist_ok=True)
        np.save(os.path.join(view_dir, "segmentation.npy"), segmentation)
        Image.fromarray(colorize_segments(segmentation, segments_info)).save(
            os.path.join(view_dir, "prediction_color.png")
        )
        with open(os.path.join(view_dir, "segments_info.json"), "w", encoding="utf-8") as file:
            json.dump(segments_info, file, ensure_ascii=False, indent=2)
        with open(os.path.join(view_dir, "class_map.json"), "w", encoding="utf-8") as file:
            json.dump(class_map, file, ensure_ascii=False, indent=2)
        with open(os.path.join(view_dir, "view_face_labels.json"), "w", encoding="utf-8") as file:
            json.dump(build_view_face_labels(step_data, class_map), file, ensure_ascii=False, indent=2)
        for info in class_map.values():
            face_id = int(info["face_id"])
            face_predictions.setdefault(face_id, []).append((int(info["class_id"]), float(info["score"])))

    face_labels = build_face_labels(step_data, face_predictions)
    if progress:
        progress(f"14 视角回投融合完成，用时 {_elapsed(project_start)}")

    label_path = os.path.join(output_dir, "face_labels.json")
    with open(label_path, "w", encoding="utf-8") as file:
        json.dump(face_labels, file, ensure_ascii=False, indent=2)

    result = {
        "output_dir": output_dir,
        "encoded_dir": encoded_dir,
        "mapping_path": mapping_path,
        "label_path": label_path,
        "face_labels": face_labels,
        "raw_predictions": {
            "scores": [float(score) for score in scores],
            "labels": [int(label) for label in labels],
            "num_instances": len(scores),
        },
    }
    if progress:
        progress(f"多视角推理总用时 {_elapsed(total_start)}")
    return result


def run_detectron2_singleview(
    step_data,
    m2f_root,
    config_path,
    weights_path,
    output_dir,
    device,
    score_threshold,
    min_ratio,
    min_face_area,
    progress=None,
):
    total_start = time.perf_counter()
    directions = get_cube_14_view_directions()
    render_start = time.perf_counter()
    encoded_dir, image_paths, mapping_path = render_encoded_views(step_data, output_dir, directions, progress)
    gb_to_fid = build_reverse_gb_map(mapping_path)
    if progress:
        progress(f"14 视角渲染完成，用时 {_elapsed(render_start)}")

    if progress:
        progress("加载 Detectron2 单图 Mask2Former 权重")
    load_start = time.perf_counter()
    predictor, cache_hit = load_singleview_predictor(m2f_root, config_path, weights_path, device)
    if progress:
        progress(f"模型{'缓存命中' if cache_hit else '加载完成'}，用时 {_elapsed(load_start)}")

    frames = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("编码视图读取失败")

    height, width = frames[0].shape[:2]
    face_predictions = {face["face_id"]: [] for face in step_data["faces"]}
    result_dir = os.path.join(output_dir, "frame_results")
    os.makedirs(result_dir, exist_ok=True)
    raw_prediction_summary = []

    infer_start = time.perf_counter()
    for frame_index, (image_path, frame) in enumerate(zip(image_paths, frames)):
        if progress:
            progress(f"单图推理第 {frame_index + 1}/{len(frames)} 个视角")
        outputs = predictor(frame)
        segmentation, segments_info = instances_to_segmentation(outputs, score_threshold, height, width)
        raw_prediction_summary.append(
            {
                "frame": frame_index + 1,
                "num_instances": len(segments_info),
                "labels": [int(item["label_id"]) for item in segments_info],
                "scores": [float(item["score"]) for item in segments_info],
            }
        )

        encoded_rgb = np.array(Image.open(image_path).convert("RGB"))
        face_masks = extract_faces_from_encoded_image(encoded_rgb, min_area=min_face_area, gb_to_fid=gb_to_fid)
        _, _, class_map = postprocess_with_encoded_faces(segmentation, segments_info, face_masks, min_ratio=min_ratio)
        view_dir = os.path.join(result_dir, f"{frame_index + 1:06d}")
        os.makedirs(view_dir, exist_ok=True)
        np.save(os.path.join(view_dir, "segmentation.npy"), segmentation)
        Image.fromarray(colorize_segments(segmentation, segments_info)).save(
            os.path.join(view_dir, "prediction_color.png")
        )
        with open(os.path.join(view_dir, "segments_info.json"), "w", encoding="utf-8") as file:
            json.dump(segments_info, file, ensure_ascii=False, indent=2)
        with open(os.path.join(view_dir, "class_map.json"), "w", encoding="utf-8") as file:
            json.dump(class_map, file, ensure_ascii=False, indent=2)
        with open(os.path.join(view_dir, "view_face_labels.json"), "w", encoding="utf-8") as file:
            json.dump(build_view_face_labels(step_data, class_map), file, ensure_ascii=False, indent=2)
        for info in class_map.values():
            face_id = int(info["face_id"])
            face_predictions.setdefault(face_id, []).append((int(info["class_id"]), float(info["score"])))

    if progress:
        progress(f"14 视角单图推理与回投完成，用时 {_elapsed(infer_start)}")

    face_labels = build_face_labels(step_data, face_predictions)
    label_path = os.path.join(output_dir, "face_labels.json")
    with open(label_path, "w", encoding="utf-8") as file:
        json.dump(face_labels, file, ensure_ascii=False, indent=2)

    result = {
        "output_dir": output_dir,
        "encoded_dir": encoded_dir,
        "mapping_path": mapping_path,
        "label_path": label_path,
        "face_labels": face_labels,
        "raw_predictions": {
            "mode": "singleview",
            "frames": raw_prediction_summary,
            "num_instances": sum(item["num_instances"] for item in raw_prediction_summary),
        },
    }
    if progress:
        progress(f"单图投票推理总用时 {_elapsed(total_start)}")
    return result


def mesh_to_payload(step_data, face_labels=None):
    face_labels = face_labels or {}
    faces_payload = []
    for face in step_data["faces"]:
        mesh = face["mesh"].triangulate()
        points = np.asarray(mesh.points, dtype=np.float32)
        raw_faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = raw_faces.reshape((-1, 4))[:, 1:4].astype(np.int32)
        face_id = int(face["face_id"])
        label = face_labels.get(face_id) or face_labels.get(str(face_id)) or {}
        class_id = int(label.get("class_id", -1))
        faces_payload.append(
            {
                "face_id": face_id,
                "face_type": face["face_type"],
                "class_id": class_id,
                "class_name": class_name(class_id),
                "score": float(label.get("score", 0.0)),
                "votes": int(label.get("votes", 0)),
                "color": class_color(class_id),
                "points": points.reshape(-1).round(6).tolist(),
                "triangles": triangles.reshape(-1).tolist(),
            }
        )
    return {"bounds": [float(x) for x in step_data["bounds"]], "faces": faces_payload}


def summarize_labels(face_labels):
    counts = Counter(int(info.get("class_id", -1)) for info in face_labels.values())
    return [
        {"class_id": class_id, "class_name": class_name(class_id), "faces": count, "color": class_color(class_id)}
        for class_id, count in sorted(counts.items())
    ]
