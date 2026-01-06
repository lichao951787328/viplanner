'''
Author: lichao951787328 951787328@qq.com
Date: 2025-12-31 14:52:20
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-06 14:27:27
FilePath: /viplanner/omniverse/extension/omni.viplanner/omni/viplanner/viplanner/mdp/commands/path_follower_command_generator_cfg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

import math
from dataclasses import MISSING
from typing import Tuple

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass
from typing_extensions import Literal

from .path_follower_command_generator import PathFollowerCommandGenerator


@configclass
class PathFollowerCommandGeneratorCfg(CommandTermCfg):
    class_type: PathFollowerCommandGenerator = PathFollowerCommandGenerator
    """Name of the command generator class."""

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""

    path_frame: Literal["world", "robot"] = "world"
    """Frame in which the path is defined.
    - ``world``: the path is defined in the world frame. Also called ``odom``.
    - ``robot``: the path is defined in the robot frame. Also called ``base``.
    """

    lookAheadDistance: float = MISSING
    """The lookahead distance for the path follower."""
    two_way_drive: bool = False
    """Allow robot to use reverse gear."""
    switch_time_threshold: float = 1.0
    """Time threshold to switch between the forward and backward drive."""
    maxSpeed: float = 0.75
    """Maximum speed of the robot."""
    maxAccel: float = 2.5 / 100.0
    """Maximum acceleration of the robot."""
    joyYaw: float = 1.0
    """TODO: add description"""
    yawRateGain: float = 7.0  # 3.5
    """Gain for the yaw rate."""
    stopYawRateGain: float = 7.0  # 3.5
    """"""
    maxYawRate: float = 90.0 * math.pi / 360
    dirDiffThre: float = 0.7
    stopDisThre: float = 0.2
    slowDwnDisThre: float = 0.3
    slowRate1: float = 0.25
    slowRate2: float = 0.5
    noRotAtGoal: bool = True
    autonomyMode: bool = False

    dynamic_lookahead: bool = False
    min_points_within_lookahead: int = 3

    # IsaacLab ManagerBasedRLEnv expects command terms to define a resampling window
    # to decide when to refresh the command. Provide a sensible default to pass validation.
    resampling_time_range: Tuple[float, float] = (0.2, 0.2)
