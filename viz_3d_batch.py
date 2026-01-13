#!/usr/bin/env python3
"""
Batch convert 2D height maps to 3D visualization (HTML + PNG)

功能：
遍历指定根目录下的所有 sample_XXXXX 文件夹，
生成交互式 HTML 和静态 PNG 预览图。
"""
import argparse
import numpy as np
from pathlib import Path
import sys
import time

# 尝试导入 plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def create_visualization(height_map, sem_mask, grid_res=0.1, output_html_path="output.html", save_png=True):
    """
    生成 Plotly 3D 对象，并保存为 HTML 和 PNG
    """
    if not PLOTLY_AVAILABLE:
        print("[ERROR] plotly not installed. Run: pip install plotly")
        return False
    
    H, W = height_map.shape
    xs = np.arange(W) * grid_res
    ys = np.arange(H) * grid_res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    # 颜色映射
    colors = np.where(sem_mask == 1, 0.8, 0.2)
    
    # 创建 Figure
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=height_map,
        surfacecolor=colors,
        colorscale=[[0, 'red'], [1, 'green']],
        showscale=True,
        colorbar=dict(
            title="Traversability",
            tickvals=[0.2, 0.8],
            ticktext=["Non-traversable", "Traversable"],
            len=0.5
        ),
        name="Height Map"
    )])
    
    # 获取样本名称
    sample_name = Path(output_html_path).parent.name
    
    # 设置视角和布局
    # eye 控制相机位置: x,y 是水平方向, z 是高度
    camera_eye = dict(x=1.2, y=-1.2, z=0.8) 
    
    fig.update_layout(
        title=f"Sample: {sample_name}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Height (m)",
            aspectmode='data', # 保持真实比例
            camera=dict(eye=camera_eye)
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50) # 减少留白
    )
    
    # 1. 保存 HTML
    fig.write_html(str(output_html_path))
    
    # 2. 保存 PNG (如果需要)
    if save_png:
        output_png_path = str(output_html_path).replace(".html", ".png")
        try:
            # engine="kaleido" 是推荐的静态导出引擎
            # scale=2 可以让图片更清晰 (类似 Retina 屏截图)
            fig.write_image(output_png_path, engine="kaleido", width=1000, height=800, scale=1.5)
            return True, True # HTML OK, PNG OK
        except Exception as e:
            # 如果 kaleido 没安装或报错，不影响 HTML 的生成
            error_msg = str(e)
            if "requires the kaleido package" in error_msg:
                print(f"  [WARNING] Cannot save PNG: 'kaleido' is missing. Run: pip install -U kaleido")
            else:
                print(f"  [WARNING] Failed to save PNG: {e}")
            return True, False # HTML OK, PNG Failed

    return True, False

def create_height_only_visualization(height_map, grid_res=0.1, output_html_path="height_3d.html", save_png=True):
    if not PLOTLY_AVAILABLE:
        print("[ERROR] plotly not installed. Run: pip install plotly")
        return False

    H, W = height_map.shape
    xs = np.arange(W) * grid_res
    ys = np.arange(H) * grid_res
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=height_map,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Height (m)", len=0.5),
        name="Height Map"
    )])

    sample_name = Path(output_html_path).parent.name
    camera_eye = dict(x=1.2, y=-1.2, z=0.8)
    fig.update_layout(
        title=f"Sample: {sample_name} (Height Only)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Height (m)",
            aspectmode='data',
            camera=dict(eye=camera_eye)
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.write_html(str(output_html_path))
    if save_png:
        output_png_path = str(output_html_path).replace(".html", ".png")
        try:
            fig.write_image(output_png_path, engine="kaleido", width=1000, height=800, scale=1.5)
            return True, True
        except Exception as e:
            print(f"  [WARNING] Failed to save PNG: {e}")
            return True, False
    return True, False

def process_single_sample(sample_dir, args):
    height_path = sample_dir / "height.npy"
    sem_mask_path = sample_dir / "sem_mask.npy"
    
    if not height_path.exists() or not sem_mask_path.exists():
        return False
    
    try:
        height_map = np.load(height_path)
        sem_mask = np.load(sem_mask_path)
        
        html_path = sample_dir / "map_3d.html"
        
        # 调用生成函数
        html_ok, png_ok = create_visualization(
            height_map, 
            sem_mask, 
            args.grid_res, 
            html_path, 
            save_png=True # 强制开启保存 PNG
        )
        
        height_html_path = sample_dir / "height_3d.html"
        create_height_only_visualization(
            height_map,
            args.grid_res,
            height_html_path,
            save_png=True
        )
        
        return html_ok
        
    except Exception as e:
        print(f"[ERROR] Failed processing {sample_dir.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch convert 2D maps to 3D HTML and PNG")
    parser.add_argument("--root_dir", type=str, default="/home/eai/VLN/viplanner/rotated_out",
                       help="Root directory containing sample_XXXXX subfolders")
    parser.add_argument("--grid_res", type=float, default=0.1)
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    if not root_path.exists():
        print(f"[ERROR] Root directory not found: {root_path}")
        sys.exit(1)
        
    sample_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    total = len(sample_dirs)
    
    print(f"Starting PNG/HTML generation for {total} samples in {root_path}...")
    
    success_count = 0
    for idx, sample_dir in enumerate(sample_dirs):
        print(f"[{idx+1}/{total}] Processing {sample_dir.name}...", end="\r")
        if process_single_sample(sample_dir, args):
            success_count += 1
            
    print(f"\nDone! Processed {success_count}/{total} samples.")
    print(f"Check the folders for 'map_3d.png' files.")

if __name__ == "__main__":
    main()