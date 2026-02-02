import os
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys
from PIL import Image as PILImage

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from depth_anything_3.utils.colmap_loader import load_and_filter_colmap_data
except ImportError:
    print("Warning: Could not import load_and_filter_colmap_data from src. Make sure paths are correct.")
    raise

def poisson_reconstruction(colmap_dir, image_dir, depth_dir, output_mesh_path, 
                          poisson_depth=9, voxel_size=0.01, 
                          depth_trunc=10.0, depth_scale=1000.0,
                          density_threshold=0.05):
    """
    使用泊松重建方法从深度图和姿态生成 mesh。
    
    Args:
        colmap_dir: 包含 cameras.bin 和 images.bin 的文件夹
        image_dir: 图像文件夹
        depth_dir: 深度图文件夹
        output_mesh_path: 输出 mesh 路径 (.ply)
        poisson_depth: 泊松重建的八叉树深度。越大细节越多，但计算量按指数增长（建议 8-10）。
        voxel_size: 预处理点云时的下采样体素大小
        depth_trunc: 截断深度，超过该距离的深度将被忽略
        depth_scale: 深度图缩放因子
        density_threshold: 密度阈值百分比（0-1），用于移除泊松重建产生的虚假闭合面。
    """
    print(f"Loading COLMAP data from {colmap_dir}...")
    intrinsics, extrinsics, image_names, _, _ = load_and_filter_colmap_data(colmap_dir, image_dir, process_res=-1)
    
    if len(image_names) == 0:
        print("Error: No images found.")
        return

    combined_pcd = o3d.geometry.PointCloud()

    print(f"Processing {len(image_names)} frames into a unified point cloud...")
    for i in tqdm(range(len(image_names))):
        img_name = image_names[i]
        
        # 1. 加载 RGB
        rgb_path = os.path.join(image_dir, img_name)
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(image_dir, os.path.basename(img_name))
        color_raw = o3d.io.read_image(rgb_path)
        
        # 2. 加载 Depth
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        depth_path = None
        for ext in ['.png', '.npy', '.npz']:
            p = os.path.join(depth_dir, base_name + ext)
            if os.path.exists(p):
                depth_path = p
                break
        
        if depth_path is None:
            continue

        if depth_path.endswith('.npy'):
            depth_np = np.load(depth_path).astype(np.float32)
            depth = o3d.geometry.Image(depth_np)
        elif depth_path.endswith('.npz'):
            data = np.load(depth_path)
            depth_np = data['depth'].astype(np.float32) if 'depth' in data else data[data.files[0]].astype(np.float32)
            depth = o3d.geometry.Image(depth_np)
        else:
            depth = o3d.io.read_image(depth_path)

        # 3. 尺寸适配与缩放
        depth_np = np.asarray(depth)
        color_np = np.asarray(color_raw)
        h_d, w_d = depth_np.shape[:2]
        h_c, w_c = color_np.shape[:2]

        if h_c != h_d or w_c != w_d:
            color_pil = PILImage.fromarray(color_np)
            color_pil = color_pil.resize((w_d, h_d), PILImage.BILINEAR)
            color = o3d.geometry.Image(np.array(color_pil))
            
            K = intrinsics[i].copy()
            K[0, 0] *= (w_d / w_c)
            K[1, 1] *= (h_d / h_c)
            K[0, 2] *= (w_d / w_c)
            K[1, 2] *= (h_d / h_c)
        else:
            color = color_raw
            K = intrinsics[i]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(w_d, h_d, K[0,0], K[1,1], K[0,2], K[1,2])
        extrinsic = extrinsics[i]

        # 4. 生成单帧点云
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # 将点云转入世界坐标系
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        combined_pcd += pcd

        # 每隔一定帧数进行一次下采样，防止内存溢出
        if i % 50 == 0:
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Final point cloud cleanup: {len(combined_pcd.points)} points.")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 5. 估计法线 (泊松重建的核心)
    print("Estimating normals...")
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )
    # 确保法线方向一致（尝试指向相机原点或向外）
    combined_pcd.orient_normals_consistent_tangent_plane(k=15)

    # 6. 运行泊松重建
    print(f"Performing Poisson Surface Reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined_pcd, depth=poisson_depth
    )

    # 7. 移除低密度区域 (泊松重建会产生虚假的闭合结构，需要根据密度裁剪)
    print("Trimming low-density areas...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Saving Poisson mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson Surface Reconstruction from Depth and COLMAP")
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output", default="poisson_mesh.ply")
    parser.add_argument("--poisson_depth", type=int, default=9, help="Octree depth (higher=more detail, 8-11 recommended)")
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--depth_trunc", type=float, default=10.0)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--density_threshold", type=float, default=0.05, help="Quantile to remove low-density mesh parts")

    args = parser.parse_args()
    poisson_reconstruction(args.colmap_dir, args.image_dir, args.depth_dir, args.output, 
                           args.poisson_depth, args.voxel_size, args.depth_trunc, 
                           args.depth_scale, args.density_threshold)
