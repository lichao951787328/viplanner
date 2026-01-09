'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-08 16:37:22
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-08 17:03:35
FilePath: /viplanner/omniverse/standalone/add_floor_colliders_warehouse.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
"""Batch-add PhysX CollisionAPI and enable collision for floor/groundplane meshes
in the warehouse USD, without needing to open the GUI.

Usage (in env_isaaclab):

    python omniverse/standalone/add_floor_colliders_warehouse.py
"""

from pxr import Usd, UsdGeom, UsdPhysics, Sdf


# Hard-coded parameters for your setup. Edit here if needed.
USD_PATH = "/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd"
# Substring keywords to match floor-like meshes.
KEYWORDS = ["sm_floor", "groundplane"]


def add_colliders(usd_path: str, keywords: list[str]) -> int:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    keywords = [k.lower() for k in keywords]

    num_touched = 0
    print(f"[INFO] Processing USD: {usd_path}")
    print(f"[INFO] Keywords: {keywords}")

    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue

        path_str = p.GetPath().pathString
        path_lower = path_str.lower()

        if not any(kw in path_lower for kw in keywords):
            continue

        print(f"Set collision ON for: {path_str}")
        # Apply CollisionAPI
        UsdPhysics.CollisionAPI.Apply(p)

        # Create / get bool physics:collisionEnabled
        attr = p.GetAttribute("physics:collisionEnabled")
        if not attr:
            attr = p.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(True)

        num_touched += 1

    stage.GetRootLayer().Save()
    print(f"[INFO] Done. Updated meshes: {num_touched}")
    return num_touched


def verify(usd_path: str, keywords: list[str]) -> None:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open stage for verification: {usd_path}")

    keywords = [k.lower() for k in keywords]

    print(f"[INFO] Verifying USD: {usd_path}")
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue

        path_str = p.GetPath().pathString
        path_lower = path_str.lower()
        if not any(kw in path_lower for kw in keywords):
            continue

        coll_api = UsdPhysics.CollisionAPI(p)
        has_api = coll_api.GetPrim().IsValid()
        attr = p.GetAttribute("physics:collisionEnabled")
        val = attr.Get() if attr else None
        print(f"Mesh: {path_str}")
        print(f"  Has CollisionAPI: {has_api}")
        print(f"  Attr exists: {bool(attr)}")
        if attr:
            print(f"  Attr type: {attr.GetTypeName()}")
            print(f"  Attr value: {val}")


if __name__ == "__main__":
    add_colliders(USD_PATH, KEYWORDS)
    verify(USD_PATH, KEYWORDS)
