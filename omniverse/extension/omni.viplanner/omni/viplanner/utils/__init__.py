# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .unreal_importer import UnRealImporter
from .unreal_importer_cfg import UnRealImporterCfg

# Matterport camera utilities are optional; guard their import to avoid
# forcing the omni.isaac.matterport dependency in non-matterport runs.
try:
    from .viplanner_matterport_raycast_camera import (
        VIPlannerMatterportRayCasterCamera,
        VIPlannerMatterportRayCasterCameraCfg,
    )
    _HAS_MATTERPORT = True
except Exception:  # ModuleNotFoundError or any runtime import error
    VIPlannerMatterportRayCasterCamera = None  # type: ignore
    VIPlannerMatterportRayCasterCameraCfg = None  # type: ignore
    _HAS_MATTERPORT = False

__all__ = ["UnRealImporter", "UnRealImporterCfg"]
if _HAS_MATTERPORT:
    __all__ += [
        "VIPlannerMatterportRayCasterCamera",
        "VIPlannerMatterportRayCasterCameraCfg",
    ]
