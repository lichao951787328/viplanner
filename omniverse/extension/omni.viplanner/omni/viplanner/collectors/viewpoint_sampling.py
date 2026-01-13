# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# 说明（中文）：
# 本文件实现视点采样与渲染的核心逻辑，用于数据采集：
# 1) 通过 TerrainAnalysis 在地形上构建可通行图（点/边）；
# 2) 从图中按指定数量采样相机位姿（位置+朝向）；
# 3) 批量设置相机姿态，渲染并保存深度/语义/RGB等输出及相机内外参。
# 仅添加注释，不改变原有行为。

import builtins
import os
import pickle
import random
import time

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

from .terrain_analysis import TerrainAnalysis
from .viewpoint_sampling_cfg import ViewpointSamplingCfg


class ViewpointSampling:
    """视点采样器：负责采样相机位姿并渲染保存图像/标注。"""

    def __init__(self, cfg: ViewpointSamplingCfg, scene: InteractiveScene):
        # 保存配置与场景句柄
        self.cfg = cfg
        self.scene = scene

        # 获取仿真上下文（用于渲染步进）
        self.sim = SimulationContext.instance()

        # 地形分析器：按需构建高度图/采样点/可通行图
        self.terrain_analyser = TerrainAnalysis(self.cfg.terrain_analysis, scene=self.scene)

        # 语义配色映射（用于把分割 id 映射成可视化颜色）
        self.viplanner_sem_meta = VIPlannerSemMetaHandler()

# 邻居的“可通行性”和连线过滤是在地形分析阶段一次性完成的，后续取 yaw 时不再重复判定。
# 地形分析阶段（先做一次，全局准备）
# 点集构建：采样 XY → 过滤墙体/安全距离/语义 → 得到合法 points（Z=地面+机器人高度）。
# 边集构建：对每点做 kNN → 过滤穿墙/大高度差/语义代价过高的边 → 得到“无碰撞可行图”。
# 距离摘要：在图上跑 Dijkstra（截断 max_path_length）→ 记录所有合法的（起点, 可达邻居, 路径长度）到 terrain_analyser.samples。
# 视点采样阶段（按需抽样，不再做碰撞判定）
# 选对儿：从 terrain_analyser.samples 随机抽取（起点, 邻居）对儿，邻居与连线已在上一步过滤为“可达/无碰撞/距截断内”。
# yaw：用“起点→邻居”的方向算 yaw（atan2）。
# roll/pitch：在 x_angle_range/y_angle_range 范围内做均匀扰动。
# 位姿：位置取起点，朝向由 yaw+扰动生成四元数；形成最终采样视点并保存/渲染。
# 核心差异点
# 邻居判定与连线可通行性是在“建图阶段”统一完成的；采样时只是“从已验证的对儿中挑选并生成姿态”，不会再做碰撞检查。
# max_path_length 控制“离得不远且可达”的对儿集合，保证采样方向有意义（面向通路/空旷区）。
    def sample_viewpoints(self, nbr_viewpoints: int, seed: int = 1) -> torch.Tensor:
        """按数量与随机种子采样视点，返回 [N,7]=[x,y,z,qw,qx,qy,qz]。"""
        # 若存在同配置缓存，直接加载
        filename = f"viewpoints_seed{seed}_samples{nbr_viewpoints}.pkl"
        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        filename = os.path.join(filedir, filename)
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(f"[INFO] Loaded {nbr_viewpoints} with seed {seed}.")
            return data
        else:
            print(f"[INFO] No viewpoint samples found for seed {seed} and {nbr_viewpoints} samples.")

        # 地形分析（若未完成）
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # 固定随机种子
        random.seed(seed)
        print(f"[INFO] Start sampling {nbr_viewpoints} viewpoints.")

        # 规划从每个点抽取的样本数量（向上取整）
        nbr_samples_per_point = int(np.ceil(nbr_viewpoints / self.terrain_analyser.points.shape[0]).item())
        sample_locations = torch.zeros((nbr_samples_per_point * self.terrain_analyser.points.shape[0], 2))
        sample_locations_count = 0
        curr_point_idx = 0
        while sample_locations_count < nbr_viewpoints:
            # 选择当前点的所有候选边
            sample_idx = self.terrain_analyser.samples[:, 0] == curr_point_idx
            # 随机选择一部分（避免偏置），保证总量不超过目标
            sample_idx_select = torch.randperm(sample_idx.sum())[\
                : min(nbr_samples_per_point, nbr_viewpoints - sample_locations_count)
            ]
            sample_locations[sample_locations_count : sample_locations_count + sample_idx_select.shape[0]] = (
                self.terrain_analyser.samples[sample_idx][sample_idx_select, :2]
            )
            sample_locations_count += sample_idx_select.shape[0]
            curr_point_idx += 1
            if curr_point_idx >= self.terrain_analyser.points.shape[0]:
                curr_point_idx = 0

        sample_locations = sample_locations[:sample_locations_count].type(torch.int64)

        # 由“起点->邻居”的方向推得 z 轴朝向
        # direction from start -> neighbor so camera faces the neighbor
        neighbor_direction = (
            self.terrain_analyser.points[sample_locations[:, 1]] - self.terrain_analyser.points[sample_locations[:, 0]]
        )
        z_angles = torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).to("cpu")

        # 在给定范围内对 x/y 轴角度做均匀扰动
        x_angles = math_utils.sample_uniform(
            self.cfg.x_angle_range[0], self.cfg.x_angle_range[1], sample_locations_count, device="cpu"
        )
        y_angles = math_utils.sample_uniform(
            self.cfg.y_angle_range[0], self.cfg.y_angle_range[1], sample_locations_count, device="cpu"
        )
        x_angles = torch.deg2rad(x_angles)
        y_angles = torch.deg2rad(y_angles)

        # 组装输出：位置 + 四元数
        samples = torch.zeros((sample_locations_count, 7))
        samples[:, :3] = self.terrain_analyser.points[sample_locations[:, 0]]
        samples[:, 3:] = math_utils.quat_from_euler_xyz(x_angles, y_angles, z_angles)

        print(f"[INFO] Sampled {sample_locations_count} viewpoints.")

        # 保存采样结果
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] Saved {sample_locations_count} viewpoints with seed {seed} to {filename}.")

        # 调试可视化（绿色箭头）
        if self.cfg.debug_viz:
            env_render_steps = 1000
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/viewpoints"
            marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
            self.visualizer = VisualizationMarkers(marker_cfg)
            self.visualizer.visualize(samples[:, :3], samples[:, 3:])

            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing {sample_locations_count} samples for {env_render_steps} render steps...")
                for _ in range(env_render_steps):
                    self.sim.render()
                self.visualizer.set_visibility(False)
                print("[INFO] Done visualizing.")

        return samples

    def render_viewpoints(self, samples: torch.Tensor):
        """按给定视点渲染各相机通道并保存到磁盘。"""
        print(f"[INFO] Start rendering {samples.shape[0]} images.")

        # 每回合可并行渲染的数量 = 环境数
        num_envs = self.scene.num_envs
        num_rounds = int(np.ceil(samples.shape[0] / num_envs))
        image_idx = [0] * len(self.cfg.cameras)

        # 目录与相机配置
        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        os.makedirs(os.path.join(filedir, "semantics"), exist_ok=True)
        os.makedirs(os.path.join(filedir, "depth"), exist_ok=True)
        if "rgb" in self.cfg.cameras.values():
            os.makedirs(os.path.join(filedir, "rgb"), exist_ok=True)

        print(f"[INFO] Saving camera configurations to {filedir}.")
        intrinsics = np.zeros((len(self.cfg.cameras), 3, 4))  # ROS Projection matrix 3x4
        for cam_idx, cam in enumerate(self.cfg.cameras.keys()):
            intrinsics[cam_idx][:3, :3] = self.scene.sensors[cam].data.intrinsic_matrices[0].cpu().numpy()
        np.savetxt(os.path.join(filedir, "intrinsics.txt"), intrinsics.reshape(-1, 12), delimiter=",")

        # 外参顺序：x y z qx qy qz qw（注意与采样时四元数顺序不同）
        np.savetxt(
            os.path.join(filedir, "camera_extrinsic.txt"),
            samples[:, [0, 1, 2, 4, 5, 6, 3]].cpu().numpy(),
            delimiter=",",
        )

        # 渲染与保存
        samples = samples.to(self.scene.device)
        start_time = time.time()
        for i in range(num_rounds):
            samples_idx = torch.arange(i * num_envs, min((i + 1) * num_envs, samples.shape[0]))
            # 批量设置相机位姿
            for cam in self.cfg.cameras.keys():
                self.scene.sensors[cam].set_world_poses(
                    positions=samples[samples_idx, :3],
                    orientations=samples[samples_idx, 3:],
                    env_ids=torch.arange(samples_idx.shape[0]),
                    convention="world",
                )
            self.scene.write_data_to_sim()
            if any([isinstance(self.scene.sensors[cam], Camera) for cam in self.cfg.cameras.keys()]):
                for _ in range(10):
                    self.sim.render()
            self.scene.update(self.sim.get_physics_dt())

            # 逐相机导出
            for cam_idx, (cam, annotator) in enumerate(self.cfg.cameras.items()):
                image_data_np = self.scene.sensors[cam].data.output[annotator].clone().cpu().numpy()
                image_data_np[np.isnan(image_data_np)] = 0
                image_data_np[np.isinf(image_data_np)] = 0

                for idx in range(samples_idx.shape[0]):
                    if annotator == "semantic_segmentation" or annotator == "rgb":
                        if image_data_np.shape[-1] == 1:
                            info = self.scene.sensors[cam].data.info[idx][annotator]["idToLabels"]
                            # 语义 id -> 颜色
                            info = {
                                int(k): (
                                    self.viplanner_sem_meta.class_color["static"]
                                    if v["class"] in ("BACKGROUND", "UNLABELLED")
                                    else self.viplanner_sem_meta.class_color.get(
                                        v["class"], self.viplanner_sem_meta.class_color["static"]
                                    )
                                )
                                for k, v in info.items()
                            }
                            unique_data_ids = np.unique(image_data_np)
                            unique_data_ids.sort()
                            mapping = np.zeros(
                                (max(unique_data_ids.max() + 1, max(info.keys()) + 1), 3), dtype=np.uint8
                            )
                            mapping[list(info.keys())] = np.array(list(info.values()), dtype=np.uint8)
                            output = mapping[image_data_np[idx].squeeze(-1)]
                        else:
                            output = image_data_np[idx]

                        assert cv2.imwrite(
                            os.path.join(
                                filedir,
                                "semantics" if annotator == "semantic_segmentation" else "rgb",
                                f"{image_idx[cam_idx]}".zfill(4) + ".png",
                            ),
                            cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR),
                        )
                    else:
                        # 深度：PNG16 + NPY（按 depth_scale 量化/保存）
                        assert cv2.imwrite(
                            os.path.join(filedir, "depth", f"{image_idx[cam_idx]}".zfill(4) + ".png"),
                            np.uint16(image_data_np[idx] * self.cfg.depth_scale),
                        )
                        np.save(
                            os.path.join(filedir, "depth", f"{image_idx[cam_idx]}".zfill(4) + ".npy"),
                            image_data_np[idx] * self.cfg.depth_scale,
                        )

                    image_idx[cam_idx] += 1
                    if sum(image_idx) % 100 == 0:
                        print(f"[INFO] Rendered {sum(image_idx)} images in {(time.time() - start_time):.4f}s.")

    def collect_rotated_from_samples(
        self,
        save_dir: str | None = None,
        max_samples: int | None = None,
        grid_res: float | None = None,
        size: float = 8.0,
        camera_height: float = 8.0,
        save_rgb: bool = True,
    ) -> int:
        """
        使用 TerrainAnalysis 的 samples 集合，按每个 (start, neighbor) 对生成旋转视角的高度图与语义可通行性图。

        行为要点：
        - yaw = atan2(neighbor_y - start_y, neighbor_x - start_x) （相机朝向邻居）
        - 不对 x/y 角度做扰动（固定俯视）
        - 不保存相机内参（按你的要求）
        - 每个样本保存到 `save_dir/sample_00001/` 子目录下，包含 `height.npy`, `sem_mask.npy`, `camera_pose.npy`，以及可选的 `top_down_rgb.png` 和可视化 PNG

        返回：成功处理的样本数量
        """
        from importlib import util as importlib_util
        from pathlib import Path
        import sys as _sys
        import math
        import numpy as np

        # ensure terrain analysis has been run
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # determine save dir
        filedir = save_dir or (self.cfg.save_path if self.cfg.save_path else self._get_save_filedir())
        Path(filedir).mkdir(parents=True, exist_ok=True)

        # load rotated helper from standalone script
        # 在运行时从一个具体文件路径动态加载并执行那个 Python 脚本，把它当成一个模块来用
        # 找到当前 Python 文件的绝对路径并向上定位到祖先目录（向上 5 层）。
        repo_root = Path(__file__).resolve().parents[5]
        # 在上面的 repo_root 下拼接出目标脚本的完整路径（Path 对象）。
        standalone_path = repo_root / "standalone" / "traversability_grid_rotated_view.py"
        # 作用：使用 importlib 的工具从指定文件路径创建一个模块规范（ModuleSpec）
        spec = importlib_util.spec_from_file_location("traversed_rot", standalone_path.as_posix())
        # 检查 spec 是否成功创建且有可用的 loader；若任一为假则断言失败并抛出异常（提示无法加载该文件）。
        assert spec and spec.loader, f"Cannot load {standalone_path}"
        # 基于 ModuleSpec 创建一个空的 module 对象（相当于 types.ModuleType(spec.name)），但尚未执行模块内的代码。
        mod = importlib_util.module_from_spec(spec)
        # 将刚创建的 module 对象放到 sys.modules（或 _sys.modules，代码里实际是用别名 _sys）中，键为 spec.name（这里是 "traversed_rot"）。
        _sys.modules[spec.name] = mod
        # 通过 spec 的 loader 去执行模块源代码，并把模块的全局名称空间填入 mod（即把文件中的顶层代码运行一遍，模块内定义的函数/类/变量都会出现在 mod 中）。
        spec.loader.exec_module(mod)

        compute_rotated_maps = getattr(mod, "compute_rotated_maps")
        render_rotated_rgb = getattr(mod, "render_rotated_rgb")

        # determine grid_res default
        if grid_res is None and hasattr(self.cfg, "terrain_analysis"):
            grid_res = getattr(self.cfg.terrain_analysis, "grid_resolution", 0.1)
        grid_res = float(grid_res)

        samples = self.terrain_analyser.samples.cpu()
        points = self.terrain_analyser.points.cpu()

        # iterate samples (each row: start_idx, neighbor_idx, path_length)
        # samples即 terrain_analyser.samples，包含每个采样点的起点索引、邻居索引和路径长度
        n_total = samples.shape[0]
        n_processed = 0
        # 采样队列
        idx_iter = range(n_total) if max_samples is None else range(min(n_total, max_samples))

        for out_idx, i in enumerate(idx_iter):
            start_idx = int(samples[i, 0].item())
            neighbor_idx = int(samples[i, 1].item())

            start_xy = points[start_idx, :2].numpy()
            neighbor_xy = points[neighbor_idx, :2].numpy()

            dx = neighbor_xy[0] - start_xy[0]
            dy = neighbor_xy[1] - start_xy[1]
            yaw_rad = math.atan2(dy, dx)
            yaw_deg = float(np.degrees(yaw_rad))

            # create per-sample directory
            sample_dir = Path(filedir) / f"sample_{out_idx+1:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Processing sample {out_idx+1}/{min(n_total, max_samples) if max_samples else n_total}: start={start_idx}, neighbor={neighbor_idx}, yaw={yaw_deg:.1f}°")

            # call compute_rotated_maps (returns dict with height & sem_mask)
            try:
                result = compute_rotated_maps(self.scene, float(start_xy[0]), float(start_xy[1]), float(size), float(grid_res), float(yaw_deg))
            except Exception as e:
                print(f"[WARNING] compute_rotated_maps failed for sample {out_idx+1}: {e}")
                continue

            # save arrays
            np.save((sample_dir / "height.npy").as_posix(), result["height"])
            np.save((sample_dir / "sem_mask.npy").as_posix(), result["sem_mask"])

            # save minimal camera pose
            camera_pose = {
                "position": np.array([float(start_xy[0]), float(start_xy[1]), float(camera_height)]),
                "yaw_deg": float(yaw_deg),
                "pitch_deg": 0.0,
                "roll_deg": 90.0,
            }
            np.save((sample_dir / "camera_pose.npy").as_posix(), camera_pose)

            # optional RGB render
            if save_rgb:
                rgb_path = (sample_dir / "top_down_rgb.png").as_posix()
                try:
                    # render_rotated_rgb expects sim and scene from the AppLauncher; use self.sim if available
                    sim = getattr(self, "sim", None)
                    if sim is None:
                        from isaaclab.sim import SimulationContext
                        sim = SimulationContext.instance()
                    render_rotated_rgb(sim, self.scene, float(start_xy[0]), float(start_xy[1]), float(camera_height), float(yaw_deg), rgb_path)
                except Exception as e:
                    print(f"[WARNING] render_rotated_rgb failed for sample {out_idx+1}: {e}")

            # save visualizations (height.png, sem_mask.png)
            try:
                import cv2
                hg = result["height"]
                finite = np.isfinite(hg)
                if finite.any():
                    hmin = np.nanmin(hg[finite])
                    hmax = np.nanmax(hg[finite])
                    denom = (hmax - hmin) if (hmax > hmin) else 1.0
                    norm = np.clip((hg - hmin) / denom, 0.0, 1.0)
                    norm[~finite] = 0.0
                    height_img = (norm * 255).astype(np.uint8)
                    cv2.imwrite((sample_dir / "height.png").as_posix(), height_img)

                sm = (result["sem_mask"] * 255).astype(np.uint8)
                cv2.imwrite((sample_dir / "sem_mask.png").as_posix(), sm)
            except Exception:
                pass

            # mark success
            with open((sample_dir / "SUCCESS.txt").as_posix(), "w") as f:
                f.write("OK\n")

            n_processed += 1

        print(f"[INFO] Finished processing {n_processed} samples, saved to {filedir}")
        return n_processed


    def _get_save_filedir(self) -> str:
        """推导默认保存目录：基于地形资源路径构造目录。

        - 如果地形来自 OBJ/PLY（matterport），读取 `obj_filepath`；
        - 如果来自 USD，读取 `usd_path`；
        目录规则：<terrain_file_path>/<env_name_without_ext>
        """
        if hasattr(self.scene.terrain.cfg, "obj_filepath"):
            terrain_file_path = self.scene.terrain.cfg.obj_filepath
        elif hasattr(self.scene.terrain.cfg, "usd_path") and isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            raise KeyError("Only implemented for terrains loaded from usd and matterport")
        env_name = os.path.splitext(terrain_file_path)[0]
        filedir = os.path.join(terrain_file_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return filedir
