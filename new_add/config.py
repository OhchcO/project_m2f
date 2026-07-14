# -*- coding: utf-8 -*-
"""
统一配置文件 — 推理相关配置的唯一来源
"""
import os

# ============================================================
# 1. 模型路径
# ============================================================
if os.name == "nt":
    MODEL_DIR = r"E:\soft\code\Mask2former"
else:
    HOME = os.path.expanduser("~")
    MODEL_DIR = os.path.join(HOME, "mask2former", "mask2former-feature-recognition")

# ============================================================
# 2. 特征类别（7类）
# ============================================================
NUM_CLASSES = 7

CLASS_NAMES = {
    0: "hole",
    1: "closed_pocket",
    2: "closed_slot",
    3: "open_pocket",
    4: "open_slot",
    5: "wide_slot",
    6: "oring_slot",
}

CLASS_NAMES_CN = {
    0: "孔",
    1: "封闭型腔",
    2: "封闭槽",
    3: "开放型腔",
    4: "开放槽",
    5: "宽体槽",
    6: "O形槽",
}

# ============================================================
# 3. 编码器配置（STP面类型 R 通道编码）
# ============================================================
TYPE_GAP = 51
TYPE_R_BASE = {
    0: 0,     # 平面 (Plane)
    1: 51,    # 圆柱面 (Cylinder)
    2: 102,   # 圆锥面 (Cone)
    3: 153,   # 球面 (Sphere)
    4: 204,   # 其他面 (Other)
}

TYPE_NAMES_EN = {
    0: "Plane",
    1: "Cylinder",
    2: "Cone",
    3: "Sphere",
    4: "Other",
}

TYPE_NAMES_CN = {
    0: "平面",
    1: "圆柱面",
    2: "圆锥面",
    3: "球面",
    4: "其他面",
}

# 兼容别名
ENCODER_TYPE_GAP = TYPE_GAP
ENCODER_TYPE_R_BASE = TYPE_R_BASE
ENCODER_TYPE_NAMES = TYPE_NAMES_CN

# ============================================================
# 4. 渲染配置
# ============================================================
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
DEFAULT_FACE_COLOR = (210, 210, 210)

# ============================================================
# 5. 推理默认参数
# ============================================================
INFERENCE_THRESHOLD = 0.5
INFERENCE_MASK_THRESHOLD = 0.5
INFERENCE_MIN_RATIO = 0.5
INFERENCE_MIN_FACE_AREA = 10
DEFAULT_OUTPUT_DIR = r"E:\soft\code\Mask2former\results\visualizations\instance_inference_encoded"

# ============================================================
# 6. 可视化颜色
# ============================================================
CLASS_COLORS_3D = {
    0: (255, 0, 0),
    1: (255, 255, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (255, 128, 0),
    5: (128, 0, 255),
    6: (0, 200, 200),
}

COLOR_PALETTE = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 255],
]

CLASS_COLORS = {i: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(NUM_CLASSES)}
