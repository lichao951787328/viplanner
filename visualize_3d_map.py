#!/usr/bin/env python3
"""
Convert 2D height map and semantic mask to 3D visualization

支持多种3D格式输出：
- PLY点云（可用CloudCompare/MeshLab打开）
- OBJ网格（可用Blender/MeshLab打开）
- 交互式HTML（可用浏览器打开）
"""
import argparse
import numpy as np
from pathlib import Path


def create_3d_point_cloud(height_map, sem_mask, grid_res=0.1, output_path="output_3d.ply"):
    """
    生成PLY格式的3D点云
    
    Args:
        height_map: (H, W) 高程数组
        sem_mask: (H, W) 语义mask（0=不可通行, 1=可通行）
        grid_res: 网格分辨率（米）
        output_path: 输出文件路径
    """
    H, W = height_map.shape
    
    # 创建网格坐标
    xs = np.arange(W) * grid_res
    ys = np.arange(H) * grid_res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    # 展平数组
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = height_map.flatten()
    sem_flat = sem_mask.flatten()
    
    # 过滤无效点（NaN高度）
    valid_mask = np.isfinite(z_flat)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    sem_valid = sem_flat[valid_mask]
    
    # 根据可通行性着色
    # 绿色=可通行, 红色=不可通行
    colors = np.zeros((len(sem_valid), 3), dtype=np.uint8)
    colors[sem_valid == 1] = [0, 255, 0]    # 绿色：可通行
    colors[sem_valid == 0] = [255, 0, 0]    # 红色：不可通行
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(x_valid)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(x_valid)):
            f.write(f"{x_valid[i]:.4f} {y_valid[i]:.4f} {z_valid[i]:.4f} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
    
    print(f"✓ PLY point cloud saved: {output_path}")
    print(f"  Points: {len(x_valid)}")
    print(f"  Green: traversable, Red: non-traversable")


def create_3d_mesh(height_map, sem_mask, grid_res=0.1, output_path="output_3d.obj"):
    """
    生成OBJ格式的3D网格
    
    Args:
        height_map: (H, W) 高程数组
        sem_mask: (H, W) 语义mask
        grid_res: 网格分辨率（米）
        output_path: 输出文件路径
    """
    H, W = height_map.shape
    
    # 创建顶点网格
    xs = np.arange(W) * grid_res
    ys = np.arange(H) * grid_res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    vertices = []
    vertex_map = np.full((H, W), -1, dtype=int)  # 记录每个网格点的顶点索引
    
    # 收集有效顶点
    for i in range(H):
        for j in range(W):
            if np.isfinite(height_map[i, j]):
                vertex_map[i, j] = len(vertices)
                vertices.append((X[i, j], Y[i, j], height_map[i, j]))
    
    # 生成三角形面
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            # 获取四个角点的顶点索引
            v00 = vertex_map[i, j]
            v10 = vertex_map[i+1, j]
            v01 = vertex_map[i, j+1]
            v11 = vertex_map[i+1, j+1]
            
            # 如果四个角点都有效，生成两个三角形
            if v00 >= 0 and v10 >= 0 and v01 >= 0 and v11 >= 0:
                # 三角形1: (i,j) - (i+1,j) - (i,j+1)
                faces.append((v00, v10, v01))
                # 三角形2: (i+1,j) - (i+1,j+1) - (i,j+1)
                faces.append((v10, v11, v01))
    
    # 写入OBJ文件
    with open(output_path, 'w') as f:
        f.write("# OBJ file generated from height map\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        
        f.write("\n")
        
        # 写入面（OBJ索引从1开始）
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✓ OBJ mesh saved: {output_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")


def create_interactive_html(height_map, sem_mask, grid_res=0.1, output_path="output_3d.html"):
    """
    生成交互式HTML 3D可视化（使用Plotly）
    
    Args:
        height_map: (H, W) 高程数组
        sem_mask: (H, W) 语义mask
        grid_res: 网格分辨率（米）
        output_path: 输出文件路径
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARNING] plotly not installed. Run: pip install plotly")
        return False
    
    H, W = height_map.shape
    
    # 创建网格坐标
    xs = np.arange(W) * grid_res
    ys = np.arange(H) * grid_res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    # 创建颜色数组（绿色=可通行，红色=不可通行）
    colors = np.where(sem_mask == 1, 0.8, 0.2)  # 0.8=浅色(可通行), 0.2=深色(不可通行)
    
    # 创建3D表面图
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
            ticktext=["Non-traversable", "Traversable"]
        ),
        name="Height Map"
    )])
    
    # 设置布局
    fig.update_layout(
        title="3D Height Map with Traversability",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Height (meters)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=800,
    )
    
    # 保存HTML文件
    fig.write_html(output_path)
    print(f"✓ Interactive HTML saved: {output_path}")
    print(f"  Open in browser to view 3D visualization")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert 2D maps to 3D visualization")
    parser.add_argument("--input_dir", type=str, default="output_with_semantics",
                       help="Input directory with height.npy and sem_mask.npy")
    parser.add_argument("--output_dir", type=str, default="output_3d",
                       help="Output directory for 3D files")
    parser.add_argument("--grid_res", type=float, default=0.1,
                       help="Grid resolution in meters")
    parser.add_argument("--format", type=str, default="all", 
                       choices=["ply", "obj", "html", "all"],
                       help="Output format: ply (point cloud), obj (mesh), html (interactive), or all")
    args = parser.parse_args()
    
    # 加载数据
    input_dir = Path(args.input_dir)
    height_path = input_dir / "height.npy"
    sem_mask_path = input_dir / "sem_mask.npy"
    
    if not height_path.exists():
        print(f"[ERROR] Height map not found: {height_path}")
        return
    if not sem_mask_path.exists():
        print(f"[ERROR] Semantic mask not found: {sem_mask_path}")
        return
    
    height_map = np.load(height_path)
    sem_mask = np.load(sem_mask_path)
    
    print(f"[INFO] Loaded data:")
    print(f"  Height shape: {height_map.shape}")
    print(f"  Height range: [{np.nanmin(height_map):.2f}, {np.nanmax(height_map):.2f}] meters")
    print(f"  Traversable: {sem_mask.sum()}/{sem_mask.size} ({100*sem_mask.sum()/sem_mask.size:.1f}%)")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成3D文件
    print(f"\n[INFO] Generating 3D visualizations...")
    
    if args.format in ["ply", "all"]:
        ply_path = output_dir / "map_3d.ply"
        create_3d_point_cloud(height_map, sem_mask, args.grid_res, ply_path)
    
    if args.format in ["obj", "all"]:
        obj_path = output_dir / "map_3d.obj"
        create_3d_mesh(height_map, sem_mask, args.grid_res, obj_path)
    
    if args.format in ["html", "all"]:
        html_path = output_dir / "map_3d.html"
        create_interactive_html(height_map, sem_mask, args.grid_res, html_path)
    
    print(f"\n[SUCCESS] 3D files saved to: {output_dir}")
    print(f"\nHow to view:")
    print(f"  - PLY: Open with MeshLab, CloudCompare")
    print(f"  - OBJ: Open with Blender, MeshLab, online viewers")
    print(f"  - HTML: Open with any web browser (interactive!)")


if __name__ == "__main__":
    main()
