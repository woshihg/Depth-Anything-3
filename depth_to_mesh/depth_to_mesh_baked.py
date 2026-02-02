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
    print("Warning: Could not import load_and_filter_colmap_data from src.")
    raise

def bake_texture_to_mesh(colmap_dir, image_dir, depth_dir, input_mesh_path, output_mesh_path,
                         voxel_size=0.01, depth_trunc=10.0, depth_scale=1000.0,
                         run_optimization=True):
    """
    使用 Open3D 的 ColorMap Optimization 将多帧图像烘焙到 Mesh 上。
    
    Args:
        input_mesh_path: 你之前生成的没有颜色或颜色模糊的 .ply 模型
    """
    # 1. 加载模型
    print(f"Loading base mesh from {input_mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    
    # 2. 读取相机数据
    intrinsics_np, extrinsics_np, image_names, _, _ = load_and_filter_colmap_data(colmap_dir, image_dir, process_res=-1)
    
    # 3. 准备 RGBD 序列和相机轨迹
    rgbd_images = []
    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    
    print("Preparing RGBD data and camera trajectory...")
    for i in tqdm(range(len(image_names))):
        img_name = image_names[i]
        
        # 加载并缩放图像与内参 (逻辑与之前一致)
        rgb_path = os.path.join(image_dir, img_name)
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(image_dir, os.path.basename(img_name))
        
        # 获取深度图路径
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        depth_path = None
        for ext in ['.png', '.npy', '.npz']:
            p = os.path.join(depth_dir, base_name + ext)
            if os.path.exists(p):
                depth_path = p
                break
        
        if depth_path is None: continue

        # 读取数据 (简化的读取逻辑)
        color_p = o3d.io.read_image(rgb_path)
        if depth_path.endswith('.npy'):
            depth_p = o3d.geometry.Image(np.load(depth_path).astype(np.float32))
        else:
            depth_p = o3d.io.read_image(depth_path)

        # 尺寸适配
        w_d, h_d = np.asarray(depth_p).shape[1], np.asarray(depth_p).shape[0]
        w_c, h_c = np.asarray(color_p).shape[1], np.asarray(color_p).shape[0]
        
        if w_c != w_d or h_c != h_d:
            color_pil = PILImage.fromarray(np.asarray(color_p))
            color_p = o3d.geometry.Image(np.array(color_pil.resize((w_d, h_d), PILImage.BILINEAR)))
            
            K = intrinsics_np[i].copy()
            K[0,0] *= (w_d/w_c); K[1,1] *= (h_d/h_c); K[0,2] *= (w_d/w_c); K[1,2] *= (h_d/h_c)
        else:
            K = intrinsics_np[i]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_p, depth_p, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        rgbd_images.append(rgbd)

        # 创建相机参数节点
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(w_d, h_d, K[0,0], K[1,1], K[0,2], K[1,2])
        params.extrinsic = extrinsics_np[i]
        camera_trajectory.parameters.append(params)

    # 4. ColorMap 优化
    # 这是关键步骤：它会计算每个像素对网格的贡献，消除重叠区域的重影和颜色不一致
    if run_optimization:
        print("Running ColorMap Optimization (this may take a while)...")
        # Open3D 0.19+ 的参数必须在构造函数中设置
        # 注意：maximum_allowable_depth 默认是 2.5，如果你的场景较大，建议设大一点
        option = o3d.pipelines.color_map.RigidOptimizerOption(
            maximum_iteration=15,
            maximum_allowable_depth=depth_trunc
        )
        
        o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory, option
        )

    print(f"Saving optimized mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input mesh from TSDF or Poisson")
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output", default="baked_mesh.ply")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    args = parser.parse_args()

    bake_texture_to_mesh(args.colmap_dir, args.image_dir, args.depth_dir, args.mesh, args.output, depth_scale=args.depth_scale)
