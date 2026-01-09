# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import types as _types
import omni
try:
    import omni.isaac.core.utils.prims as prim_utils  # type: ignore
except ModuleNotFoundError:
    # Minimal USD-based fallback; goal checks will be disabled if goal prim is missing
    def _get_prim_at_path(path: str):
        try:
            stage = omni.usd.get_context().get_stage()
            return stage.GetPrimAtPath(path)
        except Exception:
            return None

    prim_utils = _types.SimpleNamespace(get_prim_at_path=_get_prim_at_path)
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def at_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate the planner when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # extract goal position
    goal_prim = prim_utils.get_prim_at_path("/World/goal")
    if goal_prim is None or not hasattr(goal_prim, "GetAttribute"):
        # No goal prim available; never terminate on goal in this run
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    goal_pos_attr = goal_prim.GetAttribute("xformOp:translate")
    if goal_pos_attr is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    goals = torch.tensor(goal_pos_attr.Get(), device=env.device).repeat(env.num_envs, 1)

    # Check conditions for termination
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goals[:, :2], dim=1, p=2)
    return distance_goal < distance_threshold
