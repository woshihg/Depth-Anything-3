import os
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys
import torch
from plyfile import PlyData

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from depth_anything_3.utils.colmap_loader import load_and_filter_colmap_data
    from depth_anything_3.specs import Gaussians
    from depth_anything_3.model.utils.gs_renderer import render_3dgs
except ImportError:
    print("Warning: Could not import necessary modules from src. Make sure paths are correct.")
    raise

def load_gaussian_ply(path, device='cuda'):
    """
    Load a 3DGS .ply file into a Gaussians object.
    """
    print(f"Loading Gaussians from {path}...")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    means = np.stack([v['x'], v['y'], v['z']], axis=-1)
    
    # Opacity
    opacities = v['opacity']
    
    # Scales (stored as log scales in 3DGS)
    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)
    scales = np.exp(scales)
    
    # Rotations (stored as quaternions)
    rotations = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)
    # 3DGS PLY usually stores WXYZ or XYZW. Need to be careful. 
    # Depth-Anything-3 uses WXYZ based on specs.py
    
    # SH Coefficients
    # f_dc_0, f_dc_1, f_dc_2
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1) # (N, 3)
    
    # f_rest_0 ... f_rest_N
    rest_names = [name for name in v.data.dtype.names if name.startswith('f_rest_')]
    if len(rest_names) > 0:
        # Sort rest_names numerically
        rest_names = sorted(rest_names, key=lambda x: int(x.split('_')[-1]))
        f_rest = np.stack([v[name] for name in rest_names], axis=-1) # (N, 45) for degree 3
        # Reshape to (N, 3, 15)
        f_rest = f_rest.reshape(f_rest.shape[0], 3, -1)
        harmonics = np.concatenate([f_dc[:, :, np.newaxis], f_rest], axis=-1) # (N, 3, 16)
    else:
        harmonics = f_dc[:, :, np.newaxis] # (N, 3, 1)

    # Convert to torch and add batch dim
    gaussian = Gaussians(
        means=torch.from_numpy(means).float().to(device).unsqueeze(0),
        scales=torch.from_numpy(scales).float().to(device).unsqueeze(0),
        rotations=torch.from_numpy(rotations).float().to(device).unsqueeze(0),
        harmonics=torch.from_numpy(harmonics).float().to(device).unsqueeze(0),
        opacities=torch.from_numpy(opacities).float().to(device).unsqueeze(0)
    )
    
    return gaussian

def gs_to_mesh_tsdf(gs_ply, colmap_dir, output_mesh_path, voxel_size=0.01, depth_trunc=10.0, render_res=-1):
    """
    Render GS to RGBD and fuse into Mesh using TSDF.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load GS
    gaussian = load_gaussian_ply(gs_ply, device=device)
    
    # 2. Load Cameras
    print(f"Loading COLMAP data from {colmap_dir}...")
    from depth_anything_3.utils.colmap_loader import read_cameras_binary, read_images_binary, get_intrinsic_matrix
    from scipy.spatial.transform import Rotation as R
    
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")
    
    if not os.path.exists(cameras_file):
        # Try finding it in sparse/0
        cameras_file = os.path.join(colmap_dir, "sparse", "0", "cameras.bin")
        images_file = os.path.join(colmap_dir, "sparse", "0", "images.bin")

    if not os.path.exists(cameras_file):
        raise FileNotFoundError(f"Could not find COLMAP files in {colmap_dir}")

    cameras = read_cameras_binary(cameras_file)
    images_raw = read_images_binary(images_file)
    
    image_ids = sorted(images_raw.keys(), key=lambda k: images_raw[k].name)
    
    # 3. Initialize TSDF
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"Integrating {len(image_ids)} rendered frames into TSDF volume...")
    for img_id in tqdm(image_ids):
        img_data = images_raw[img_id]
        cam_data = cameras[img_data.camera_id]
        
        # Get W2C
        qvec = img_data.qvec
        r = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
        rot_mat = r.as_matrix()
        
        w2c = np.eye(4)
        w2c[:3, :3] = rot_mat
        w2c[:3, 3] = img_data.tvec
        
        # Get K
        K = get_intrinsic_matrix(cam_data)
        W, H = cam_data.width, cam_data.height
        
        if render_res > 0:
            scale = render_res / max(W, H)
            W = int(W * scale)
            H = int(H * scale)
            K[0, 0] *= scale; K[1, 1] *= scale
            K[0, 2] *= scale; K[1, 2] *= scale

        # 4. Render RGB and Depth from GS
        ex_t = torch.from_numpy(w2c).float().to(device).unsqueeze(0)
        in_t = torch.from_numpy(K).float().to(device).unsqueeze(0)
        
        with torch.no_grad():
            color, depth = render_3dgs(
                extrinsics=ex_t,
                intrinsics=in_t,
                image_shape=(H, W),
                gaussian=gaussian,
                use_sh=True,
                color_mode="RGB+D"
            )
        
        # depth from render_3dgs is [1, H, W]
        # color is [1, 3, H, W] in [0, 1]
        depth_np = depth[0].cpu().numpy()
        color_np = (color[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # 5. Integrate into Open3D
        o3d_color = o3d.geometry.Image(color_np)
        o3d_depth = o3d.geometry.Image(depth_np)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0, # rendered depth is in meters
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(rgbd, intrinsic, w2c)

    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    print(f"Saving mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF Fusion from 3DGS PLY")
    parser.add_argument("--gs_ply", required=True, help="Path to 3DGS .ply file")
    parser.add_argument("--colmap_dir", required=True, help="Path to COLMAP sparse directory")
    parser.add_argument("--output", default="gs_mesh.ply", help="Output mesh path")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for TSDF")
    parser.add_argument("--depth_trunc", type=float, default=10.0, help="Max depth distance")
    parser.add_argument("--render_res", type=int, default=-1, help="Resolution to render (max dim)")

    args = parser.parse_args()

    gs_to_mesh_tsdf(
        args.gs_ply,
        args.colmap_dir,
        args.output,
        voxel_size=args.voxel_size,
        depth_trunc=args.depth_trunc,
        render_res=args.render_res
    )
