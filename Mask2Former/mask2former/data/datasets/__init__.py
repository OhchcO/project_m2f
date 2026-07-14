# Copyright (c) Facebook, Inc. and its affiliates.
from . import (
    register_ade20k_full,
    register_ade20k_panoptic,
    register_coco_stuff_10k,
    register_mapillary_vistas,
    register_coco_panoptic_annos_semseg,
    register_ade20k_instance,
    register_mapillary_vistas_panoptic,
)

# 自定义数据集注册
import os
import sys
_datasets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets")
if os.path.isdir(_datasets_dir) and _datasets_dir not in sys.path:
    sys.path.insert(0, _datasets_dir)
try:
    import register_balanced_dataset
except ImportError:
    pass
