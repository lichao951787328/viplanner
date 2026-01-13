#!/usr/bin/env python3
"""
Small-batch rotated-view data collector.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess
import csv
import math

# 1. 导入 AppLauncher (现在我们在 Isaac 环境下，这行不会报错了)
try:
    from isaaclab.app import AppLauncher
except ImportError:
    # 如果这行报错，说明 bash 脚本里还是没用 isaaclab.sh -p 运行
    print("[ERROR] Failed to import isaaclab.app. Please run this script using './isaaclab.sh -p script.py'")
    sys.exit(1)

# make omni.viplanner importable
_ROOT = Path(__file__).resolve().parent.parent / "extension"
_PKG = (_ROOT / "omni.viplanner").as_posix()
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

def build_parser():
    p = argparse.ArgumentParser("Small rotated-view collector (subprocess mode)")
    p.add_argument("--scene", default="warehouse", choices=["warehouse", "carla", "matterport"])
    p.add_argument("--num_samples", type=int, default=8, help="number of viewpoints to sample (Isaac mode)")
    p.add_argument("--max_samples", type=int, default=None, help="max samples to process")
    p.add_argument("--out", type=str, default=None, help="output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--samples_file", type=str, default=None, help="CSV file with start_x,start_y,yaw_deg per line (fallback mode)")
    p.add_argument("--grid_res", type=float, default=0.1)
    p.add_argument("--capture_size", type=float, default=8.0)
    p.add_argument("--camera_height", type=float, default=8.0)
    
    # 允许接收 --headless，并将其传递给 AppLauncher
    # 注意：AppLauncher 会自动处理 --headless，但我们需要在这里定义它以防 parse_args 报错
    # 或者我们可以让 AppLauncher 添加它的参数
    AppLauncher.add_app_launcher_args(p)
    
    return p

def call_rotated_script(standalone_path: Path, scene: str, start_x: float, start_y: float, yaw_deg: float, out_dir: str, grid_res: float, capture_size: float, camera_height: float):
    # 构建子进程调用命令
    # 关键：子进程也必须用当前运行的 Python 解释器（即 Isaac Lab 的 python）
    cmd = [sys.executable, standalone_path.as_posix(),
           '--scene', scene,
           '--enable_cameras',
           '--grid_res', str(grid_res),
           '--capture_size', str(capture_size),
           '--offset_x', str(start_x),
           '--offset_y', str(start_y),
           '--camera_height', str(camera_height),
           '--yaw_deg', str(yaw_deg),
           '--save_dir', out_dir,
           '--headless'] # 强制子进程也是 headless

    print(f"[CMD] Processing sample...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] rotated script failed: {e}")
        return False

def main():
    print("[INFO] Running small rotated-view data collector (Isaac Lab context)...")
    
    # 解析参数
    parser = build_parser()
    args = parser.parse_args()
    
    # 显式设置 enable_cameras，因为 data_collect_myself 往往需要它
    args.enable_cameras = True

    out_root = args.out or "./rotated_out"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    sample_list = []

    # 1. 尝试读取 CSV
    if args.samples_file and Path(args.samples_file).exists():
        print(f"[INFO] Reading samples from {args.samples_file}")
        with open(args.samples_file, 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and not row[0].startswith('#'):
                    try:
                        sample_list.append((float(row[0]), float(row[1]), float(row[2])))
                    except: pass
    
    # 2. 如果没有样本，启动 Isaac Sim 进行内联采样
    simulation_app = None
    if not sample_list:
        print("[INFO] No samples provided. Initializing SimulationApp for inline sampling...")
        
        # 启动仿真器
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        
        # 只有在 App 启动后才能导入 Isaac Sim 核心库
        import isaaclab.sim as sim_utils
        from omniverse.standalone.data_collect_myself import ViewpointSamplingCfg, ViewpointSampling
        from isaaclab.scene import InteractiveScene
        
        try:
            # 加载产生 make_scene 的模块
            import importlib.util as _il
            tv_path = Path(__file__).resolve().parent / "traversability_grid_rotated_view.py"
            if not tv_path.exists():
                 tv_path = Path(__file__).resolve().parents[1] / 'omniverse' / 'standalone' / 'traversability_grid_rotated_view.py'
            
            spec = _il.spec_from_file_location('rotated_view_standalone', tv_path.as_posix())
            mod = _il.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            print('[INFO] Creating scene...')
            # 注意：mod.make_scene 可能会尝试创建 SimulationContext
            # 由于我们已经初始化了 app，我们需要确保 make_scene 兼容
            # 通常 make_scene 会做 InteractiveScene(cfg)
            scene, sim = mod.make_scene(args.scene)
            
            # 采样
            cfg = ViewpointSamplingCfg()
            cfg.save_path = out_root
            vs = ViewpointSampling(cfg, scene=scene)
            
            print(f"[INFO] Sampling {args.num_samples} viewpoints...")
            samples = vs.sample_viewpoints(args.num_samples, seed=args.seed)
            
            # 处理采集
            vs.collect_rotated_from_samples(save_dir=out_root, max_samples=args.max_samples, 
                                          grid_res=args.grid_res, size=args.capture_size, 
                                          camera_height=args.camera_height, save_rgb=True)
            
            print("[INFO] Inline sampling complete.")
            simulation_app.close()
            return # 完成后直接退出，不需要走 subprocess

        except Exception as e:
            print(f"[ERROR] Inline sampling failed: {e}")
            import traceback
            traceback.print_exc()
            if simulation_app: simulation_app.close()
            sys.exit(1)

    # 3. 如果是从 CSV 读取的，则需要关闭 App (如果它被意外启动了) 或者直接运行 subprocess
    # 由于我们在 if not sample_list 块外，这里还没启动 App，这很好。
    # 这样我们可以反复调用 subprocess 而不产生多个 App 实例冲突（Isaac Sim 不支持同一进程多次启动 App）

    standalone_path = Path(__file__).resolve().parent / "traversability_grid_rotated_view.py"
    if not standalone_path.exists():
        standalone_path = Path(__file__).resolve().parents[1] / 'omniverse' / 'standalone' / 'traversability_grid_rotated_view.py'

    if args.max_samples:
        sample_list = sample_list[:args.max_samples]

    print(f"[INFO] Processing {len(sample_list)} samples via subprocess...")
    for idx, (sx, sy, yaw) in enumerate(sample_list):
        sample_out = Path(out_root) / f"sample_{idx+1:05d}"
        sample_out.mkdir(parents=True, exist_ok=True)
        print(f"--- Sample {idx+1}/{len(sample_list)} ---")
        call_rotated_script(standalone_path, args.scene, sx, sy, yaw, sample_out.as_posix(), 
                          args.grid_res, args.capture_size, args.camera_height)

if __name__ == '__main__':
    main()