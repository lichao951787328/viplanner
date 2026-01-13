# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import os
import pickle
import random
import time
from pathlib import Path  # 移到顶部导入

import cv2
import numpy as np
import isaaclab.utils.math as math_utils
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext

from viplanner.config import VIPlannerSemMetaHandler

# 导入 myself 版本的地形分析
from .terrain_analysis_myself import TerrainAnalysis

# 尝试导入 myself 版本的配置，若不存在则回退
try:
    from .viewpoint_sampling_cfg_myself import ViewpointSamplingCfg
except ImportError:
    from .viewpoint_sampling_cfg import ViewpointSamplingCfg


class ViewpointSampling:
    """视点采样器 (Myself Variant)：负责采样相机位姿并渲染保存图像/标注。"""

    def __init__(self, cfg: ViewpointSamplingCfg, scene: InteractiveScene):
        self.cfg = cfg
        self.scene = scene
        self.sim = SimulationContext.instance()
        self.terrain_analyser = TerrainAnalysis(self.cfg.terrain_analysis, scene=self.scene)
        self.viplanner_sem_meta = VIPlannerSemMetaHandler()

    def sample_viewpoints(self, nbr_viewpoints: int, seed: int = 1) -> torch.Tensor:
        """按数量与随机种子采样视点，返回 [N,7]=[x,y,z,qw,qx,qy,qz]。"""
        filename = f"viewpoints_seed{seed}_samples{nbr_viewpoints}.pkl"
        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        filename = os.path.join(filedir, filename)
        
        if os.path.isfile(filename):
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                print(f"[INFO] Loaded {nbr_viewpoints} with seed {seed}.")
                return data
            except Exception:
                print(f"[WARNING] Cache file {filename} corrupted, resampling.")

        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        random.seed(seed)
        print(f"[INFO] Start sampling {nbr_viewpoints} viewpoints.")

        # 保护机制：如果点太少
        if self.terrain_analyser.points.shape[0] == 0:
            print("[ERROR] No valid points in terrain analysis.")
            return torch.zeros((0, 7))

        nbr_samples_per_point = int(np.ceil(nbr_viewpoints / self.terrain_analyser.points.shape[0]).item())
        sample_locations = torch.zeros((nbr_samples_per_point * self.terrain_analyser.points.shape[0], 2))
        sample_locations_count = 0
        curr_point_idx = 0
        
        while sample_locations_count < nbr_viewpoints:
            sample_idx = self.terrain_analyser.samples[:, 0] == curr_point_idx
            valid_count = sample_idx.sum()
            
            if valid_count > 0:
                count_to_take = min(nbr_samples_per_point, nbr_viewpoints - sample_locations_count)
                # 使用 torch.randperm 随机选择邻居
                sample_idx_select = torch.randperm(valid_count)[:count_to_take]
                
                # 获取选中的样本行 (start_idx, neighbor_idx)
                selected = self.terrain_analyser.samples[sample_idx][sample_idx_select, :2]
                
                sample_locations[sample_locations_count : sample_locations_count + selected.shape[0]] = selected
                sample_locations_count += selected.shape[0]
            
            curr_point_idx += 1
            if curr_point_idx >= self.terrain_analyser.points.shape[0]:
                curr_point_idx = 0
                if sample_locations_count == 0:
                    print("[ERROR] Infinite loop detected in sampling (graph disconnected?).")
                    break

        sample_locations = sample_locations[:sample_locations_count].type(torch.int64)

        neighbor_direction = (
            self.terrain_analyser.points[sample_locations[:, 0]] - self.terrain_analyser.points[sample_locations[:, 1]]
        )
        z_angles = torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).to("cpu")

        x_angles = math_utils.sample_uniform(
            self.cfg.x_angle_range[0], self.cfg.x_angle_range[1], sample_locations_count, device="cpu"
        )
        y_angles = math_utils.sample_uniform(
            self.cfg.y_angle_range[0], self.cfg.y_angle_range[1], sample_locations_count, device="cpu"
        )
        x_angles = torch.deg2rad(x_angles)
        y_angles = torch.deg2rad(y_angles)

        samples = torch.zeros((sample_locations_count, 7))
        samples[:, :3] = self.terrain_analyser.points[sample_locations[:, 0]]
        samples[:, 3:] = math_utils.quat_from_euler_xyz(x_angles, y_angles, z_angles)

        print(f"[INFO] Sampled {sample_locations_count} viewpoints.")

        os.makedirs(filedir, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] Saved viewpoints to {filename}.")

        if self.cfg.debug_viz:
            self._visualize_samples(samples)

        return samples

    def _visualize_samples(self, samples):
        env_render_steps = 100
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/viewpoints"
        marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
        self.visualizer = VisualizationMarkers(marker_cfg)
        self.visualizer.visualize(samples[:, :3], samples[:, 3:])

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print(f"[INFO] Visualizing samples...")
            for _ in range(env_render_steps):
                self.sim.render()
            self.visualizer.set_visibility(False)

    def render_viewpoints(self, samples: torch.Tensor):
        """按给定视点渲染各相机通道并保存到磁盘。"""
        print(f"[INFO] Start rendering {samples.shape[0]} images.")

        num_envs = self.scene.num_envs
        num_rounds = int(np.ceil(samples.shape[0] / num_envs))
        image_idx = [0] * len(self.cfg.cameras)

        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        os.makedirs(os.path.join(filedir, "semantics"), exist_ok=True)
        os.makedirs(os.path.join(filedir, "depth"), exist_ok=True)
        if "rgb" in self.cfg.cameras.values():
            os.makedirs(os.path.join(filedir, "rgb"), exist_ok=True)

        intrinsics = np.zeros((len(self.cfg.cameras), 3, 4))
        for cam_idx, cam in enumerate(self.cfg.cameras.keys()):
            intrinsics[cam_idx][:3, :3] = self.scene.sensors[cam].data.intrinsic_matrices[0].cpu().numpy()
        np.savetxt(os.path.join(filedir, "intrinsics.txt"), intrinsics.reshape(-1, 12), delimiter=",")

        np.savetxt(
            os.path.join(filedir, "camera_extrinsic.txt"),
            samples[:, [0, 1, 2, 4, 5, 6, 3]].cpu().numpy(),
            delimiter=",",
        )

        samples = samples.to(self.scene.device)
        start_time = time.time()
        for i in range(num_rounds):
            samples_idx = torch.arange(i * num_envs, min((i + 1) * num_envs, samples.shape[0]))
            
            for cam in self.cfg.cameras.keys():
                self.scene.sensors[cam].set_world_poses(
                    positions=samples[samples_idx, :3],
                    orientations=samples[samples_idx, 3:],
                    env_ids=torch.arange(samples_idx.shape[0], device=self.scene.device),
                    convention="world",
                )
            self.scene.write_data_to_sim()
            
            if any([isinstance(self.scene.sensors[cam], Camera) for cam in self.cfg.cameras.keys()]):
                for _ in range(5): # Reduce render steps for speed
                    self.sim.render()
            self.scene.update(self.sim.get_physics_dt())

            for cam_idx, (cam, annotator) in enumerate(self.cfg.cameras.items()):
                image_data_np = self.scene.sensors[cam].data.output[annotator].clone().cpu().numpy()
                
                # Handle NaNs/Infs
                image_data_np[np.isnan(image_data_np)] = 0
                image_data_np[np.isinf(image_data_np)] = 0

                for idx in range(samples_idx.shape[0]):
                    global_id = image_idx[cam_idx]
                    
                    if annotator == "semantic_segmentation" or annotator == "rgb":
                        if image_data_np.shape[-1] == 1:
                            info = self.scene.sensors[cam].data.info[idx][annotator]["idToLabels"]
                            # Simplified mapping logic
                            mapping = np.zeros((256, 3), dtype=np.uint8)
                            for k, v in info.items():
                                if int(k) < 256:
                                    mapping[int(k)] = self.viplanner_sem_meta.class_color.get(v["class"], (0,0,0))
                            output = mapping[image_data_np[idx].squeeze(-1).astype(np.uint8)]
                        else:
                            output = image_data_np[idx]

                        cv2.imwrite(
                            os.path.join(
                                filedir,
                                "semantics" if annotator == "semantic_segmentation" else "rgb",
                                f"{global_id:04d}.png",
                            ),
                            cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR),
                        )
                    else:
                        cv2.imwrite(
                            os.path.join(filedir, "depth", f"{global_id:04d}.png"),
                            np.uint16(image_data_np[idx] * self.cfg.depth_scale),
                        )
                        np.save(
                            os.path.join(filedir, "depth", f"{global_id:04d}.npy"),
                            image_data_np[idx] * self.cfg.depth_scale,
                        )

                    image_idx[cam_idx] += 1
            
            if i % 10 == 0:
                 print(f"[INFO] Rendered batch {i}/{num_rounds}...")

    # =========================================================================
    # [IMPORTANT FIX] Robust Path Handling in collect_rotated_from_samples
    # =========================================================================
    def collect_rotated_from_samples(
        self,
        save_dir: str | None = None,
        max_samples: int | None = None,
        grid_res: float | None = None,
        size: float = 8.0,
        camera_height: float = 8.0,
        save_rgb: bool = True,
    ) -> int:
        from importlib import util as importlib_util
        import sys as _sys
        import math
        import numpy as np

        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        filedir = save_dir or (self.cfg.save_path if self.cfg.save_path else self._get_save_filedir())
        Path(filedir).mkdir(parents=True, exist_ok=True)

        # --- Robust Path Logic Start ---
        # 1. 尝试定位到 'viplanner' 根目录 (假设我们在 extension/.../collectors/ 中)
        # parents[5] 通常是 repo_root/extension，或者直接是 repo_root
        current_file = Path(__file__).resolve()
        
        # 常见结构: .../viplanner/omniverse/standalone
        # 策略：向上查找直到找到 'omniverse' 文件夹
        found_standalone = False
        standalone_path = None
        
        # 从当前目录向上搜寻 10 层
        search_path = current_file
        for _ in range(10):
            search_path = search_path.parent
            # 检查 omniverse/standalone
            candidate = search_path / "omniverse" / "standalone" / "traversability_grid_rotated_view.py"
            if candidate.exists():
                standalone_path = candidate
                found_standalone = True
                break
            # 检查 standalone (旧结构)
            candidate_old = search_path / "standalone" / "traversability_grid_rotated_view.py"
            if candidate_old.exists():
                standalone_path = candidate_old
                found_standalone = True
                break
            # 到达根目录停止
            if search_path.parent == search_path:
                break
        
        if not found_standalone:
            # 硬编码回退 (基于你之前的 logs)
            fallback = Path("/home/eai/VLN/viplanner/omniverse/standalone/traversability_grid_rotated_view.py")
            if fallback.exists():
                standalone_path = fallback
            else:
                raise FileNotFoundError(f"Could not find traversability_grid_rotated_view.py starting search from {current_file}")
        
        print(f"[INFO] Using rotated helper at: {standalone_path}")
        # --- Robust Path Logic End ---

        spec = importlib_util.spec_from_file_location("traversed_rot", standalone_path.as_posix())
        assert spec and spec.loader, f"Cannot load {standalone_path}"
        mod = importlib_util.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        compute_rotated_maps = getattr(mod, "compute_rotated_maps")
        render_rotated_rgb = getattr(mod, "render_rotated_rgb")

        if grid_res is None and hasattr(self.cfg, "terrain_analysis"):
            grid_res = getattr(self.cfg.terrain_analysis, "grid_resolution", 0.1)
        grid_res = float(grid_res)

        samples = self.terrain_analyser.samples.cpu()
        points = self.terrain_analyser.points.cpu()

        n_total = samples.shape[0]
        if n_total == 0:
            print("[WARNING] No valid samples to collect.")
            return 0

        n_processed = 0
        limit = min(n_total, max_samples) if max_samples else n_total
        idx_iter = range(limit)

        for out_idx, i in enumerate(idx_iter):
            start_idx = int(samples[i, 0].item())
            neighbor_idx = int(samples[i, 1].item())

            start_xy = points[start_idx, :2].numpy()
            neighbor_xy = points[neighbor_idx, :2].numpy()

            dx = neighbor_xy[0] - start_xy[0]
            dy = neighbor_xy[1] - start_xy[1]
            yaw_rad = math.atan2(dy, dx)
            yaw_deg = float(np.degrees(yaw_rad))

            sample_dir = Path(filedir) / f"sample_{out_idx+1:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Processing sample {out_idx+1}/{limit}: start={start_idx}, neighbor={neighbor_idx}, yaw={yaw_deg:.1f}°")

            try:
                result = compute_rotated_maps(self.scene, float(start_xy[0]), float(start_xy[1]), float(size), float(grid_res), float(yaw_deg))
            except Exception as e:
                print(f"[WARNING] compute_rotated_maps failed for sample {out_idx+1}: {e}")
                continue

            np.save((sample_dir / "height.npy").as_posix(), result["height"])
            np.save((sample_dir / "sem_mask.npy").as_posix(), result["sem_mask"])

            camera_pose = {
                "position": np.array([float(start_xy[0]), float(start_xy[1]), float(camera_height)]),
                "yaw_deg": float(yaw_deg),
                "pitch_deg": 0.0,
                "roll_deg": 90.0,
            }
            np.save((sample_dir / "camera_pose.npy").as_posix(), camera_pose)

            if save_rgb:
                rgb_path = (sample_dir / "top_down_rgb.png").as_posix()
                try:
                    sim = getattr(self, "sim", None)
                    if sim is None:
                        sim = SimulationContext.instance()
                    render_rotated_rgb(sim, self.scene, float(start_xy[0]), float(start_xy[1]), float(camera_height), float(yaw_deg), rgb_path)
                except Exception as e:
                    print(f"[WARNING] render_rotated_rgb failed for sample {out_idx+1}: {e}")

            # Visualizations
            try:
                
                # binary_mask = np.zeros_like(result["height"], dtype=np.uint8)
                # binary_mask[np.isfinite(result["height"])] = 255
                # cv2.imwrite(os.path.join(sample_dir, "height_valid_mask.png"), binary_mask)
                
                hg = result["height"]
                finite = np.isfinite(hg)
                if finite.any():
                    hmin = np.nanmin(hg[finite])
                    hmax = np.nanmax(hg[finite])
                    denom = (hmax - hmin) if (hmax > hmin) else 1.0
                    norm = np.clip((hg - hmin) / denom, 0.3, 1.0)
                    norm[~finite] = 0.0
                    height_img = (norm * 255).astype(np.uint8)
                    cv2.imwrite((sample_dir / "height.png").as_posix(), height_img)

                sm = (result["sem_mask"] * 255).astype(np.uint8)
                cv2.imwrite((sample_dir / "sem_mask.png").as_posix(), sm)
            except Exception:
                pass

            with open((sample_dir / "SUCCESS.txt").as_posix(), "w") as f:
                f.write("OK\n")

            n_processed += 1

        print(f"[INFO] Finished processing {n_processed} samples, saved to {filedir}")
        return n_processed

    def _get_save_filedir(self) -> str:
        if hasattr(self.scene.terrain.cfg, "obj_filepath"):
            terrain_file_path = self.scene.terrain.cfg.obj_filepath
        elif hasattr(self.scene.terrain.cfg, "usd_path") and isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
             # Fallback logic to avoid crash if terrain type is different
             terrain_file_path = "./output_data/default_env"
             
        env_name = os.path.splitext(os.path.basename(terrain_file_path))[0]
        filedir = os.path.join(os.path.dirname(terrain_file_path), env_name)
        os.makedirs(filedir, exist_ok=True)
        return filedir