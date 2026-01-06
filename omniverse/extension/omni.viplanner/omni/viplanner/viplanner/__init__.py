# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from .vip_anymal import VIPlanner
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

# 注意：不要在此处导入 VIPlannerAlgo 以避免与配置模块产生循环依赖。
# 如需使用，请直接从具体模块路径导入：
# from omni.viplanner.viplanner.viplanner_algo import VIPlannerAlgo

__all__ = ["DATA_DIR"]
