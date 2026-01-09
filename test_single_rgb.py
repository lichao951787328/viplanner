#!/usr/bin/env python3
"""
Test single RGB capture using data_collect_myself.py approach
"""
import sys
from pathlib import Path

# Setup paths
_EXT_ROOT = Path(__file__).resolve().parent / "extension"
_OMNI_VIPLANNER_PKG = (_EXT_ROOT / "omni.viplanner").as_posix()
if _OMNI_VIPLANNER_PKG not in sys.path:
    sys.path.insert(0, _OMNI_VIPLANNER_PKG)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils
import types as _types
import importlib.util

# Setup module stubs
_pkg_omni_dir = (_EXT_ROOT / "omni.viplanner" / "omni").as_posix()
_pkg_viplanner_dir = (_EXT_ROOT / "omni.viplanner" / "omni" / "viplanner").as_posix()
_pkg_config_dir = (_EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config").as_posix()

if "omni" not in sys.modules:
    _omni = _types.ModuleType("omni")
    _omni.__path__ = [_pkg_omni_dir]
    sys.modules["omni"] = _omni
if "omni.viplanner" not in sys.modules:
    _ov = _types.ModuleType("omni.viplanner")
    _ov.__path__ = [_pkg_viplanner_dir]
    sys.modules["omni.viplanner"] = _ov
if "omni.viplanner.config" not in sys.modules:
    _ovc = _types.ModuleType("omni.viplanner.config")
    _ovc.__path__ = [_pkg_config_dir]
    sys.modules["omni.viplanner.config"] = _ovc

# Load warehouse config
_cfg_dir = _EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config"
_wh_path = (_cfg_dir / "warehouse_cfg.py").as_posix()
_spec = importlib.util.spec_from_file_location("omni.viplanner.config.warehouse_cfg", _wh_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
WarehouseTerrainSceneCfg = getattr(_mod, "TerrainSceneCfg")

# Create scene
scene_cfg = WarehouseTerrainSceneCfg(1, env_spacing=1.0)
scene_cfg.robot = None
scene_cfg.height_scanner = None
scene_cfg.contact_forces = None
scene_cfg.depth_camera = None  # Disable depth camera

# Keep only semantic camera with RGB
print(f"[INFO] Semantic camera data types: {scene_cfg.semantic_camera.data_types}")

sim_cfg = sim_utils.SimulationCfg()
sim = SimulationContext(sim_cfg)
scene = InteractiveScene(scene_cfg)
sim.reset()

camera = scene.sensors['semantic_camera']
print(f"[INFO] Camera available outputs: {list(camera.data.output.keys())}")

# Test 1: Original position (flat view)
print("\n" + "="*60)
print("TEST 1: Flat horizontal view at (5, 5, 1.5)")
print("="*60)
pos1 = torch.tensor([[5.0, 5.0, 1.5]], dtype=torch.float32, device="cuda:0")
# Small pitch down (like data_collect_myself default: -2 to 5 deg range, let's use 0)
orient1 = math_utils.quat_from_euler_xyz(
    torch.tensor([0.0]).deg2rad(), 
    torch.tensor([0.0]).deg2rad(), 
    torch.tensor([0.0]).deg2rad()
).to("cuda:0")

camera.set_world_poses(positions=pos1, orientations=orient1, env_ids=torch.tensor([0], device="cuda:0"), convention="world")
scene.write_data_to_sim()
for _ in range(10):
    sim.render()
scene.update(sim.get_physics_dt())

rgb1 = camera.data.output["rgb"][0].cpu().numpy()
print(f"RGB shape: {rgb1.shape}, dtype: {rgb1.dtype}, range: [{rgb1.min()}, {rgb1.max()}]")
from PIL import Image
Image.fromarray(rgb1).save("/home/eai/VLN/viplanner/test_flat_view.png")
print("✓ Saved: test_flat_view.png")

# Test 2: Top-down view
print("\n" + "="*60)
print("TEST 2: Top-down view at (5, 5, 6) - Y-axis rotation")
print("="*60)
pos2 = torch.tensor([[5.0, 5.0, 6.0]], dtype=torch.float32, device="cuda:0")
# Rotate around Y axis -90 degrees (not X!)
orient2 = math_utils.quat_from_euler_xyz(
    torch.tensor([0.0]).deg2rad(),     # x: no pitch
    torch.tensor([-90.0]).deg2rad(),   # y: rotate to look down
    torch.tensor([0.0]).deg2rad()      # z: no yaw
).to("cuda:0")

camera.set_world_poses(positions=pos2, orientations=orient2, env_ids=torch.tensor([0], device="cuda:0"), convention="world")
scene.write_data_to_sim()
for _ in range(10):
    sim.render()
scene.update(sim.get_physics_dt())

rgb2 = camera.data.output["rgb"][0].cpu().numpy()
print(f"RGB shape: {rgb2.shape}, dtype: {rgb2.dtype}, range: [{rgb2.min()}, {rgb2.max()}]")
Image.fromarray(rgb2).save("/home/eai/VLN/viplanner/test_topdown_view.png")
print("✓ Saved: test_topdown_view.png")

print("\n" + "="*60)
print("Tests complete! Check:")
print("  - test_flat_view.png (horizontal)")
print("  - test_topdown_view.png (top-down)")
print("="*60)

simulation_app.close()
