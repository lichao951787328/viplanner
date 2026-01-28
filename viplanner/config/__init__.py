'''
Author: lichao951787328 951787328@qq.com
Date: 2025-12-31 14:52:20
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-20 20:08:53
FilePath: /viplanner/viplanner/config/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .coco_sem_meta import _COCO_MAPPING, get_class_for_id
from .costmap_cfg import (
    CostMapConfig,
    GeneralCostMapConfig,
    ReconstructionCfg,
    SemCostMapConfig,
    TsdfCostMapConfig,
)
from .learning_cfg_myself import DataCfg, TrainCfg
from .viplanner_sem_meta import OBSTACLE_LOSS, VIPlannerSemMetaHandler

__all__ = [
    # configs
    "ReconstructionCfg",
    "SemCostMapConfig",
    "TsdfCostMapConfig",
    "CostMapConfig",
    "GeneralCostMapConfig",
    "TrainCfg",
    "DataCfg",
    # mapping
    "VIPlannerSemMetaHandler",
    "OBSTACLE_LOSS",
    "get_class_for_id",
    "_COCO_MAPPING",
]

# EoF
