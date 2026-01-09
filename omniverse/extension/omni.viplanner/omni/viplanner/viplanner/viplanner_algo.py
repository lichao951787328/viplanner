# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import carb
import isaaclab.utils.math as math_utils
import torch
import torchvision.transforms as transforms
from typing import List

try:
    import omni.usd  # Isaac Sim USD context
    from pxr import Usd, UsdGeom, Gf
except Exception:
    omni = None


# Train configuration import (global) so other methods can use it
try:
    from viplanner.config import TrainCfg
except Exception:
    try:
        from omni.viplanner.config import TrainCfg
    except Exception:
        TrainCfg = None  # Will be validated at runtime

# 首选直接从顶层包 viplanner 导入（仓库根作为 Python 包安装或加入 sys.path 时可用）
try:
    from viplanner.plannernet.autoencoder import AutoEncoder, DualAutoEncoder
except Exception:
    # 回退：若顶层包未安装，再尝试同扩展内的相对路径（通常不存在）
    try:
        from ..plannernet.autoencoder import AutoEncoder, DualAutoEncoder
    except Exception:
        # 最后回退：尝试旧的裸包名（极少数环境）
        from plannernet.autoencoder import AutoEncoder, DualAutoEncoder

# 轨迹优化同样优先从顶层 viplanner 包导入
try:
    from viplanner.traj_cost_opt.traj_opt import TrajOpt
except Exception:
    # 兼容旧路径（如有自定义打包到 omni.viplanner 名称空间的情况）
    from omni.viplanner.traj_cost_opt.traj_opt import TrajOpt

"""
VIPlanner Helpers
"""


class VIPlannerAlgo:
    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda"):
        """Apply VIPlanner Algorithm

        Args:
            model_dir (str): Directory that include model.pt and model.yaml
        """
        super().__init__()

        assert os.path.exists(model_dir), "Model directory does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"

        # params
        self.fear_threshold = fear_threshold
        self.device = device

        # load model
        if TrainCfg is None:
            raise RuntimeError("TrainCfg not available; ensure viplanner.config is importable")
        self.train_config: TrainCfg = None
        self.load_model(model_dir)

        # get transforms for images
        self.transform = transforms.Resize(self.train_config.img_input_size, antialias=None)

        # init trajectory optimizer
        self.traj_generate = TrajOpt()

        # setup waypoint display in Isaac (support new and old debug_draw APIs)
        self.draw = None
        self._draw_warned = False
        self._draw_acquired_once = False
        self._try_acquire_debug_draw()
        self.color_fear = [(1.0, 0.4, 0.1, 1.0)]  # red
        self.color_path = [(0.4, 1.0, 0.1, 1.0)]  # green
        self.size = [5.0]

    def load_model(self, model_dir: str):
        # load train config
        self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        carb.log_info(
            f"Model loaded using sem: {self.train_config.sem}, rgb: {self.train_config.rgb}, knodes: {self.train_config.knodes}, in_channel: {self.train_config.in_channel}"
        )

        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth

        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        # get model and load weights
        try:
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), weights_only=True)
        except ValueError:
            model_state_dict = torch.load(os.path.join(model_dir, "model.pt"), weights_only=True)
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if self.device.lower() == "cpu":
            carb.log_warn("CUDA not available, VIPlanner will run on CPU")
            self.cuda_avail = False
        else:
            self.net = self.net.cuda()
            self.cuda_avail = True
        return

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """transform goal into camera frame"""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = math_utils.quat_apply(math_utils.quat_inv(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(
        self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor
    ) -> torch.Tensor:
        """transform path from camera frame to world frame"""
        return math_utils.quat_apply(
            cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame
        ) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        # transform images
        image = self.transform(image)
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0  # set all inf or nan values to 0
        return image

    ###
    # Planning
    ###

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(image), goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        # transform input
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    ###
    # Debug Draw
    ###

    def debug_draw(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        if getattr(self, "draw", None) is None:
            # Lazy acquire after viewport creation
            self._try_acquire_debug_draw()
            if self.draw is None:
                # Fallback: draw as USD BasisCurves when DebugDraw is unavailable
                self._fallback_draw_curves(paths, fear, goal)
                return
        # Native DebugDraw path
        self.draw.clear_lines()
        self.draw.clear_points()

        def draw_single_traj(traj, color, size):
            traj[:, 2] = torch.mean(traj[:, 2])
            self.draw.draw_lines(traj[:-1].tolist(), traj[1:].tolist(), color * len(traj[1:]), size * len(traj[1:]))

        for idx, curr_path in enumerate(paths):
            if fear[idx] > self.fear_threshold:
                draw_single_traj(curr_path, self.color_fear, self.size)
                self.draw.draw_points(goal.tolist(), self.color_fear * len(goal), self.size * len(goal))
            else:
                draw_single_traj(curr_path, self.color_path, self.size)
                self.draw.draw_points(goal.tolist(), self.color_path * len(goal), self.size * len(goal))

    def _fallback_draw_curves(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        # If USD context is not available, skip
        try:
            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            if stage is None:
                return
        except Exception:
            return
        # Ensure a parent Xform exists
        root_path = "/World/DebugDraw"
        try:
            xform = UsdGeom.Xform.Get(stage, root_path)
            if not xform:
                xform = UsdGeom.Xform.Define(stage, root_path)
        except Exception:
            return
        # For each path, update or create a BasisCurves prim
        for idx, curr_path in enumerate(paths):
            pts: List[Gf.Vec3f] = [Gf.Vec3f(p[0].item(), p[1].item(), p[2].item()) for p in curr_path]
            prim_path = f"{root_path}/path_{idx}"
            try:
                curves = UsdGeom.BasisCurves.Get(stage, prim_path)
                if not curves:
                    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
                    curves.CreateTypeAttr("linear")
                    curves.CreateWrapAttr("nonPeriodic")
                # Update attributes
                curves.CreateCurveVertexCountsAttr().Set([len(pts)])
                curves.CreatePointsAttr().Set(pts)
                curves.CreateWidthsAttr().Set([0.02] * len(pts))
                color = self.color_fear[0] if fear[idx] > self.fear_threshold else self.color_path[0]
                curves.CreateDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])
            except Exception:
                # Skip on any USD error
                continue

    def _try_acquire_debug_draw(self):
        # Prefer Isaac Sim 5.x API
        try:
            import isaacsim.util.debug_draw as sim_debug_draw
            if hasattr(sim_debug_draw, "acquire_debug_draw_interface"):
                self.draw = sim_debug_draw.acquire_debug_draw_interface()
            elif hasattr(sim_debug_draw, "get_debug_draw_interface"):
                self.draw = sim_debug_draw.get_debug_draw_interface()
        except Exception:
            pass
        # Fallback to legacy omni.isaac.debug_draw
        if self.draw is None:
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw
                self.draw = omni_debug_draw.acquire_debug_draw_interface()
            except Exception:
                pass
        if self.draw is not None and not self._draw_acquired_once:
            print("[INFO] DebugDraw interface acquired.")
            self._draw_acquired_once = True
        elif self.draw is None and not self._draw_warned:
            print("[WARNING] DebugDraw interface not available (extension not loaded or headless).")
            self._draw_warned = True
