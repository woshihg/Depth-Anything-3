import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
import sys

def gs_to_mesh_poisson(gs_ply, output_mesh_path, poisson_depth=9, density_threshold=0.01):
    """
    直接从 3DGS PLY 文件（点云）生成 Mesh，不进行渲染。
    使用泊松重建 (Poisson Reconstruction) 算法。
    """
    print(f"Loading 3DGS point cloud from {gs_ply}...")
    plydata = PlyData.read(gs_ply)
    v = plydata['vertex']
    
    # 1. 提取点坐标 (means)
    points = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # 2. 提取颜色 (从 SH 系数的 DC 项转换)
    # 3DGS 中 f_dc 与 RGB 的简单转换关系 (SH 0阶系数): RGB = 0.5 + 0.282 * f_dc
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = (0.5 + 0.28209479177387814 * f_dc).clip(0, 1)

    # 3. 过滤透明度 (可选)
    # 如果点太模糊（opacity 低），泊松重建会产生很多杂质
    if 'opacity' in v.data.dtype.names:
        opacity = v['opacity']
        # 激活函数通常是 sigmoid，这里简单过滤掉低权重值
        mask = opacity > 0.1 
        points = points[mask]
        colors = colors[mask]
        print(f"Filtered points by opacity: {len(points)} remaining.")

    # 4. 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 估计法线 (泊松重建必须有法线)
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # 6. 泊松重建
    print(f"Running Poisson Reconstruction (depth={poisson_depth})...")
    # depth 越高，细节越多，但计算越慢
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    # 7. 剪枝 (根据采样密度移除虚假的面片)
    print("Pruning low-density areas...")
    densities = np.asarray(densities)
    # 移除密度低于阈值的部分（泊松重建通常会产生一个闭合的壳，需要剪枝）
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Saving mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson Reconstruction from 3DGS PLY directly")
    parser.add_argument("--gs_ply", required=True, help="Path to 3DGS .ply file")
    parser.add_argument("--output", default="gs_mesh_poisson.ply", help="Output mesh path")
    parser.add_argument("--depth", type=int, default=10, help="Poisson reconstruction depth (8-12)")
    parser.add_argument("--density_thr", type=float, default=0.05, help="Density threshold for pruning (0-1)")

    args = parser.parse_args()
    gs_to_mesh_poisson(args.gs_ply, args.output, args.depth, args.density_thr)
