import os
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from depth_anything_3.utils.colmap_loader import load_and_filter_colmap_data
except ImportError:
    print("Warning: Could not import load_and_filter_colmap_data from src. Make sure paths are correct.")
    # Fallback or error
    raise

def fuse_tsdf(colmap_dir, image_dir, depth_dir, output_mesh_path, voxel_size=0.01, depth_trunc=10.0, depth_scale=1.0, recenter=True):
    """
    使用 TSDF Fusion 方法从深度图和姿态生成 mesh。
    
    Args:
        colmap_dir: 包含 cameras.bin 和 images.bin 的文件夹
        image_dir: 图像文件夹 (用于对齐 COLMAP 名字)
        depth_dir: 深度图文件夹
        output_mesh_path: 输出 mesh 路径 (.ply 或 .obj)
        voxel_size: 体素大小，单位通常与深度一致（通常是米）
        depth_trunc: 截断深度，超过该距离的深度将被忽略
        depth_scale: 深度图的缩放因子。如果深度图单位是毫米，需要设置为 1000.0 转为米
        recenter: 是否将场景重定位到第一台相机
    """
    print(f"Loading COLMAP data from {colmap_dir}...")
    # load_and_filter_colmap_data 会内部过滤掉不在 image_dir 中的图片
    # 并返回 Intrinsics 和 Extrinsics (World-to-Camera)
    # 注意: process_res 设置为 -1 表示使用原始分辨率
    intrinsics, extrinsics, image_names, _, _ = load_and_filter_colmap_data(colmap_dir, image_dir, process_res=-1, recenter=recenter)
    
    if len(image_names) == 0:
        print("Error: No images found after matching COLMAP and image directory.")
        return

    # 初始化可选规模的 TSDF 体积 (ScalableTSDFVolume)
    # voxel_length: 体素边长
    # sdf_trunc: 截断距离，通常取 2-5 倍的 voxel_length
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"Integrating {len(image_names)} frames into TSDF volume...")
    for i in tqdm(range(len(image_names))):
        img_name = image_names[i]
        
        # 1. 加载 RGB
        rgb_path = os.path.join(image_dir, img_name)
        if not os.path.exists(rgb_path):
            # 尝试在子目录找（COLMAP 名字有时包含相对路径）
            basename = os.path.basename(img_name)
            rgb_path = os.path.join(image_dir, basename)
            
        color = o3d.io.read_image(rgb_path)
        
        # 2. 加载 Depth
        # 假设深度图文件名与图像相同，但目录不同，尝试常见后缀
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        depth_path = None
        for ext in ['.png', '.npy', '.npz']:
            p = os.path.join(depth_dir, base_name + ext)
            if os.path.exists(p):
                depth_path = p
                break
        
        if depth_path is None:
            # 尝试使用带子路径的名字
            sub_path_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.npy', '.npz']:
                p = os.path.join(depth_dir, sub_path_name + ext)
                if os.path.exists(p):
                    depth_path = p
                    break

        if depth_path is None:
            print(f"Warning: Depth map not found for {img_name}, skipping.")
            continue

        if depth_path.endswith('.npy'):
            depth_np = np.load(depth_path).astype(np.float32)
            depth = o3d.geometry.Image(depth_np)
        elif depth_path.endswith('.npz'):
            data = np.load(depth_path)
            if 'depth' in data:
                depth_np = data['depth'].astype(np.float32)
            else:
                depth_np = data[data.files[0]].astype(np.float32)
            depth = o3d.geometry.Image(depth_np)
        else:
            depth = o3d.io.read_image(depth_path)

        # 获取深度图和真彩色图的尺寸
        depth_np = np.asarray(depth)
        color_np = np.asarray(color)
        h_d, w_d = depth_np.shape[:2]
        h_c, w_c = color_np.shape[:2]

        # 3. 如果尺寸不匹配，缩放 RGB 图片以适配深度图
        if h_c != h_d or w_c != w_d:
            # 使用 Open3D 的 resize 或先转为浮点数缩放再转回
            color_pcd_temp = o3d.geometry.Image(color_np)
            # 注意: o3d.geometry.Image 并没有直接的 resize 方法
            # 我们使用 PIL 辅助缩放
            from PIL import Image as PILImage
            color_pil = PILImage.fromarray(color_np)
            color_pil = color_pil.resize((w_d, h_d), PILImage.BILINEAR)
            color = o3d.geometry.Image(np.array(color_pil))
            
            # 同时缩放相机内参 K
            K = intrinsics[i].copy()
            K[0, 0] *= (w_d / w_c)  # fx
            K[1, 1] *= (h_d / h_c)  # fy
            K[0, 2] *= (w_d / w_c)  # cx
            K[1, 2] *= (h_d / h_c)  # cy
        else:
            K = intrinsics[i]

        # 创建相机内参对象
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w_d, h_d, K[0,0], K[1,1], K[0,2], K[1,2])
        
        # extrinsics[i] 已经是 World-to-Camera (W2C)
        extrinsic = extrinsics[i]

        # 4. 合成 RGBD 并 Integrate
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(rgbd, intrinsic, extrinsic)

    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # 可选: 翻转坐标系 (如果结果是倒着的)
    # mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    print(f"Saving mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF Fusion from Depth and COLMAP Poses")
    parser.add_argument("--colmap_dir", required=True, help="Path to COLMAP sparse directory (containing bin files)")
    parser.add_argument("--image_dir", required=True, help="Path to RGB images directory")
    parser.add_argument("--depth_dir", required=True, help="Path to depth maps directory")
    parser.add_argument("--output", default="output_mesh.ply", help="Path to save the output mesh")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for TSDF (e.g., 0.01 for 1cm)")
    parser.add_argument("--depth_trunc", type=float, default=10.0, help="Max depth distance to integrate")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor to divide depth values (e.g., 1000 for mm-to-m)")
    parser.add_argument("--recenter", action="store_true", help="Recenter the scene to the first camera")

    args = parser.parse_args()

    fuse_tsdf(
        args.colmap_dir,
        args.image_dir,
        args.depth_dir,
        args.output,
        voxel_size=args.voxel_size,
        depth_trunc=args.depth_trunc,
        depth_scale=args.depth_scale,
        recenter=args.recenter
    )
