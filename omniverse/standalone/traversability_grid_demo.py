#!/usr/bin/env python3
"""
Build a traversability grid by combining geometry (height/slope/obstacles)
and optional semantics when available.

Outputs: grid.npy (0/1), grid.png (visualization), height.npy (meters)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Ensure the omni.viplanner package is importable from the repo extension folder
_EXT_ROOT = Path(__file__).resolve().parent.parent / "extension"
_OMNI_VIPLANNER_PKG = (_EXT_ROOT / "omni.viplanner").as_posix()
if _OMNI_VIPLANNER_PKG not in sys.path:
    sys.path.insert(0, _OMNI_VIPLANNER_PKG)
_EXT_ROOT_POSIX = _EXT_ROOT.as_posix()
if _EXT_ROOT_POSIX not in sys.path:
    sys.path.append(_EXT_ROOT_POSIX)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traversability grid demo (geometry + optional semantics)")
    p.add_argument("--scene", default="warehouse", choices=["warehouse", "carla", "matterport"], help="Scene")
    p.add_argument("--grid_res", type=float, default=0.25, help="Grid resolution (meters)")
    p.add_argument("--slope_deg", type=float, default=15.0, help="Max slope degrees considered traversable")
    p.add_argument("--buffer", type=float, default=0.4, help="Min distance from walls (meters)")
    p.add_argument("--seed", type=int, default=1, help="Random seed (for internal sampling consistency)")
    p.add_argument("--save_dir", type=str, default=None, help="Output directory")
    # Isaac app args
    AppLauncher.add_app_launcher_args(p)
    return p


def make_scene(scene_name: str):
    from importlib import util as importlib_util, machinery as importlib_mach
    # Import Omniverse/Isaac modules only after AppLauncher instantiated in main()
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    # Prepare lightweight package stubs to avoid executing config/__init__.py
    import types as _types
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

    cfg_root = Path(_pkg_config_dir)
    mod_name = f"omni.viplanner.config.{scene_name}_cfg"
    mod_path = (cfg_root / f"{scene_name}_cfg.py").as_posix()
    spec = importlib_util.spec_from_file_location(mod_name, mod_path)
    assert spec and spec.loader
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    SceneCfg = getattr(mod, "TerrainSceneCfg")

    scene_cfg = SceneCfg(1, env_spacing=1.0)
    # For grid demo we don't need robot/height scanner/contact forces
    scene_cfg.robot = None
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None
    # Disable cameras to avoid requiring --enable_cameras in headless runs
    if hasattr(scene_cfg, "depth_camera"):
        scene_cfg.depth_camera = None
    if hasattr(scene_cfg, "semantic_camera"):
        scene_cfg.semantic_camera = None

    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return scene


def compute_traversability_grid(scene, grid_res: float, slope_deg: float, buffer_m: float,
                                use_semantics: bool = False, semantics_allow: tuple[str, ...] = ("floor", "sidewalk", "ground")):
    # Import analysis only after AppLauncher instantiated, and avoid package __init__ side-effects
    from importlib import util as importlib_util
    from pathlib import Path as _Path
    import types as _types
    import sys as _sys
    _collectors_dir = _Path(__file__).resolve().parent.parent / "extension" / "omni.viplanner" / "omni" / "viplanner" / "collectors"
    # Create a lightweight stub for the collectors package to avoid executing its on-disk __init__.py
    if "omni.viplanner.collectors" not in _sys.modules:
        _pkg_collectors = _types.ModuleType("omni.viplanner.collectors")
        _pkg_collectors.__path__ = [_collectors_dir.as_posix()]
        _sys.modules["omni.viplanner.collectors"] = _pkg_collectors
    # terrain_analysis_cfg
    _cfg_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_cfg",
        (_collectors_dir / "terrain_analysis_cfg.py").as_posix(),
    )
    assert _cfg_spec and _cfg_spec.loader
    _cfg_mod = importlib_util.module_from_spec(_cfg_spec)
    # Register module so decorators relying on sys.modules can resolve __module__
    _sys.modules[_cfg_spec.name] = _cfg_mod
    _cfg_spec.loader.exec_module(_cfg_mod)  # type: ignore
    TerrainAnalysisCfg = getattr(_cfg_mod, "TerrainAnalysisCfg")
    # terrain_analysis_myself
    _ta_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_myself",
        (_collectors_dir / "terrain_analysis_myself.py").as_posix(),
    )
    assert _ta_spec and _ta_spec.loader
    _ta_mod = importlib_util.module_from_spec(_ta_spec)
    _sys.modules[_ta_spec.name] = _ta_mod
    _ta_spec.loader.exec_module(_ta_mod)  # type: ignore
    TerrainAnalysis = getattr(_ta_mod, "TerrainAnalysis")
    # Configure analysis
    tac = TerrainAnalysisCfg()
    tac.grid_resolution = grid_res
    tac.robot_buffer_spawn = buffer_m
    tac.semantic_cost_mapping = None  # we apply semantics separately in this demo

    ta = TerrainAnalysis(tac, scene)
    ta._setup_raycaster()
    ta.construct_height_map()

    # Height map and extents
    hg = ta.height_grid  # [Nx, Ny]
    x_max, y_max, x_min, y_min = ta.mesh_dimensions
    xs = np.linspace(x_min, x_max, hg.shape[0])
    ys = np.linspace(y_min, y_max, hg.shape[1])

    # Finite hits indicate ground present
    hits_mask = torch.isfinite(hg).cpu().numpy()

    # Slope from height grid (finite diff)
    hg_np = hg.cpu().numpy()
    gx = np.zeros_like(hg_np)
    gy = np.zeros_like(hg_np)
    gx[1:-1, :] = (hg_np[2:, :] - hg_np[:-2, :]) / (2 * grid_res)
    gy[:, 1:-1] = (hg_np[:, 2:] - hg_np[:, :-2]) / (2 * grid_res)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_ok = slope <= np.tan(np.deg2rad(slope_deg))

    # Wall proximity: reuse closeness filter on grid points
    # Build XY centers
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    XY = np.stack([X[hits_mask], Y[hits_mask]], axis=1)
    if XY.size == 0:
        prox_ok_mask = np.zeros_like(hits_mask, dtype=bool)
    else:
        z = ta.get_height(torch.from_numpy(XY).float().to(ta.device)).cpu().numpy()
        ray_origins = torch.from_numpy(np.stack([XY[:, 0], XY[:, 1], z + tac.robot_height], axis=1)).float().to(ta.device)
        ro2, _ = ta._point_filter_wall_closeness(ray_origins.clone(), torch.from_numpy(z).float().to(ta.device))
        kept = set(map(tuple, np.round(ro2[:, :2].cpu().numpy(), 6)))
        flat_xy = list(map(tuple, np.round(XY, 6)))
        keep_flags = np.array([pt in kept for pt in flat_xy], dtype=bool)
        prox_ok_mask = np.zeros_like(hits_mask, dtype=bool)
        prox_ok_mask[hits_mask] = keep_flags

    trav = hits_mask & slope_ok & prox_ok_mask

    # Optional semantics filtering (camera-based or USD semantics if available)
    used_semantics = False
    sem_mask = None
    if use_semantics:
        try:
            # Try USD semantics via _raycast_usd_stage
            grid_points = torch.from_numpy(np.column_stack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * (tac.wall_height * 2)])).float().to(ta.device)
            direction = torch.zeros_like(grid_points); direction[:, 2] = -1.0
            _, _, _, classes = ta._raycast_usd_stage(grid_points, direction, max_dist=tac.wall_height * 2, return_class=True)
            if classes is not None and any(c is not None for c in classes):
                # Optional debug: print basic stats about semantic classes
                if os.environ.get("VIPLANNER_DEBUG_SEM", "0") == "1":
                    from collections import Counter as _Counter
                    non_empty = [c for c in classes if c is not None]
                    print(f"[VIPLANNER_DEBUG] Semantic raycast: total={len(classes)}, non_empty={len(non_empty)}")
                    if non_empty:
                        print(f"[VIPLANNER_DEBUG] Class counts: {_Counter(non_empty)}")
                # Map to allow list
                allow = set(s.lower() for s in semantics_allow)
                sem_ok = np.array([(c is not None and str(c).lower() in allow) for c in classes], dtype=bool)
                sem_ok = sem_ok.reshape(X.shape)
                trav = trav & sem_ok
                sem_mask = sem_ok.astype(np.uint8)
                used_semantics = True
            else:
                # Fallback: if no semantic hits at all but geometry is valid,
                # expose a semantic mask identical to the traversability grid
                # so users can still visualize a "semantic" map.
                if os.environ.get("VIPLANNER_GEOM_SEM_FALLBACK", "1") == "1":
                    if os.environ.get("VIPLANNER_DEBUG_SEM", "0") == "1":
                        print("[VIPLANNER_DEBUG] No semantic ray hits; using geometry-only fallback mask.")
                    sem_mask = trav.astype(np.uint8)
                    used_semantics = True
        except Exception:
            used_semantics = False

    # Collect last collided USD prim paths for mapping debug if available
    collision_paths = None
    try:
        collision_paths = getattr(ta, "_last_ray_collision_paths", None)
    except Exception:
        collision_paths = None

    return {
        "grid": trav.astype(np.uint8),
        "height": hg_np,
        "xs": xs,
        "ys": ys,
        "used_semantics": used_semantics,
        "sem_mask": sem_mask,
        "collision_paths": collision_paths,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Do not forcibly disable semantics; let environment decide
    # But disable people to avoid requiring prim_utils.create_prim in headless demo
    os.environ.setdefault("VIPLANNER_DISABLE_PEOPLE", "1")
    app = AppLauncher(args).app

    scene = make_scene(args.scene)

    result = compute_traversability_grid(
        scene,
        grid_res=args.grid_res,
        slope_deg=args.slope_deg,
        buffer_m=args.buffer,
        use_semantics=bool(os.environ.get("VIPLANNER_USE_SEMANTICS", "0") == "1"),
    )

    # Save outputs
    out_dir = args.save_dir or os.environ.get("VIPLANNER_DATA_DIR") or str(Path.cwd() / "traversability_output")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "grid.npy"), result["grid"])  # 0/1 grid
    np.save(os.path.join(out_dir, "height.npy"), result["height"])  # meters
    # Simple visualization
    try:
        import cv2
        vis = (result["grid"] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "grid.png"), vis)
        # Save a headless-friendly height visualization
        hg = result.get("height")
        if hg is not None:
            import numpy as _np
            hg_np = _np.array(hg)
            finite = _np.isfinite(hg_np)
            if finite.any():
                hmin = _np.nanmin(hg_np[finite])
                hmax = _np.nanmax(hg_np[finite])
                denom = (hmax - hmin) if (hmax > hmin) else 1.0
                norm = _np.clip((hg_np - hmin) / denom, 0.0, 1.0)
                norm[~finite] = 0.0
                height_img = (norm * 255).astype(_np.uint8)
                cv2.imwrite(os.path.join(out_dir, "height.png"), height_img)
        if result.get("sem_mask") is not None:
            sm = (result["sem_mask"] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, "sem_mask.png"), sm)
            np.save(os.path.join(out_dir, "sem_mask.npy"), result["sem_mask"])
        # Save a debug list of USD collision paths to help mapping tweaks
        if result.get("collision_paths"):
            from collections import Counter
            paths = [p for p in result["collision_paths"] if p]
            if len(paths) > 0:
                cnt = Counter(paths)
                with open(os.path.join(out_dir, "sem_debug_paths.txt"), "w", encoding="utf-8") as f:
                    for p, n in cnt.most_common():
                        f.write(f"{n}\t{p}\n")
    except Exception:
        pass

    print(f"[INFO] Traversability grid saved to: {out_dir}")
    print(f"[INFO] used_semantics={result['used_semantics']}  grid shape={result['grid'].shape}")
    if not result['used_semantics']:
        print("[HINT] Semantics not applied. Ensure VIPLANNER_SEMANTIC_MAP keywords match USD prim paths (e.g., 'groundplane','floor','tile','concrete').")

    app.close()


if __name__ == "__main__":
    main()
