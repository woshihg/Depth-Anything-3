import trimesh
import numpy as np
from plyfile import PlyData
import argparse
import os

def align_gs_to_glb(glb_path, ply_path, output_path):
    """
    Read alignment matrix from GLB metadata and apply it to a GS PLY file.
    Saves the aligned GS centers as a new GLB file.
    """
    print(f"Loading GLB from {glb_path}...")
    scene = trimesh.load(glb_path)
    
    # 1. Extract alignment matrix
    alignment_matrix = None
    if isinstance(scene, trimesh.Scene) and scene.metadata and "hf_alignment" in scene.metadata:
        alignment_matrix = np.array(scene.metadata["hf_alignment"])
        print("Found 'hf_alignment' matrix in GLB metadata.")
    else:
        print("Warning: No 'hf_alignment' found in GLB metadata. Using identity matrix.")
        alignment_matrix = np.eye(4)

    # 2. Load GS PLY
    print(f"Loading GS PLY from {ply_path}...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    
    # Extract positions
    gs_pts = np.stack([v['x'], v['y'], v['z']], axis=1)
    print(f"Loaded {len(gs_pts)} Gaussians.")

    # 3. Apply transformation to positions
    # Add homogeneous coordinate (w=1)
    gs_pts_homo = np.hstack([gs_pts, np.ones((gs_pts.shape[0], 1))])
    # Apply matrix: X' = X @ A.T
    gs_pts_aligned = (gs_pts_homo @ alignment_matrix.T)[:, :3]

    # 4. Extract colors (DC component of SH)
    if 'f_dc_0' in v:
        # SH DC to RGB approximation: 0.5 + 0.28209 * f_dc
        f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
        gs_colors = np.clip(0.5 + 0.28209 * f_dc, 0, 1)
        # Convert to uint8 for trimesh
        gs_colors_u8 = (gs_colors * 255).astype(np.uint8)
    else:
        gs_colors_u8 = None

    # 5. Create a new PointCloud and save as GLB
    print(f"Creating aligned PointCloud...")
    aligned_pc = trimesh.points.PointCloud(vertices=gs_pts_aligned, colors=gs_colors_u8)
    
    # We can also preserve the metadata in the new GLB
    new_scene = trimesh.Scene(aligned_pc)
    new_scene.metadata["hf_alignment"] = alignment_matrix.tolist()
    
    print(f"Exporting to {output_path}...")
    new_scene.export(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align GS PLY to GLB coordinate system")
    parser.add_argument("glb", help="Path to the reference .glb file (containing hf_alignment)")
    parser.add_argument("ply", help="Path to the source .ply GS file")
    parser.add_argument("--output", help="Path to save the aligned .glb (default: input_aligned.glb)")

    args = parser.parse_args()

    if not args.output:
        base, _ = os.path.splitext(args.ply)
        args.output = base + "_aligned.glb"

    align_gs_to_glb(args.glb, args.ply, args.output)
