import trimesh
import numpy as np
from plyfile import PlyData
import argparse
import os

def reverse_align(glb_path, input_path, output_path):
    """
    Read hf_alignment from GLB, calculate its inverse, and apply to the input point cloud/mesh.
    """
    print(f"Loading reference GLB from {glb_path}...")
    scene = trimesh.load(glb_path)
    
    # 1. Extract and invert the alignment matrix
    if isinstance(scene, trimesh.Scene) and scene.metadata and "hf_alignment" in scene.metadata:
        alignment_matrix = np.array(scene.metadata["hf_alignment"])
        try:
            inv_matrix = np.linalg.inv(alignment_matrix)
            print("Found 'hf_alignment' matrix. Inverse calculated successfully.")
        except np.linalg.LinAlgError:
            print("Error: Alignment matrix is singular and cannot be inverted.")
            return
    else:
        print("Error: No 'hf_alignment' found in GLB metadata. Cannot perform reverse transformation.")
        return

    # 2. Load input data
    print(f"Loading input file from {input_path}...")
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.ply':
        # Special handling for GS PLY to preserve colors/points if needed
        # but if it's a standard PLY, trimesh is fine too.
        # Let's check if it's a GS PLY by reading header
        plydata = PlyData.read(input_path)
        v = plydata['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1)
        
        # Apply inverse transformation
        print(f"Applying reverse transformation to {len(pts)} points...")
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_reversed = (pts_homo @ inv_matrix.T)[:, :3]

        # Try to get colors
        colors = None
        if 'f_dc_0' in v:
            f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
            colors = np.clip(0.5 + 0.28209 * f_dc, 0, 1)
            colors = (colors * 255).astype(np.uint8)
        elif 'red' in v:
            colors = np.stack([v['red'], v['green'], v['blue']], axis=1)

        print(f"Exporting reversed point cloud to {output_path}...")
        reversed_pc = trimesh.points.PointCloud(vertices=pts_reversed, colors=colors)
        reversed_pc.export(output_path)

    else:
        # For GLB and other mesh formats, preserve the structure
        data = trimesh.load(input_path)
        print(f"Applying reverse transformation to {input_path}...")
        data.apply_transform(inv_matrix)
        
        print(f"Exporting to {output_path}...")
        data.export(output_path)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse transform a point cloud or mesh using hf_alignment matrix from a GLB")
    parser.add_argument("glb", help="Path to the reference .glb file containing hf_alignment")
    parser.add_argument("input", help="Path to the input file to be reversed (.ply, .glb, etc.)")
    parser.add_argument("--output", help="Path to save the reversed output (default: matches input extension)")

    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_reversed" + ext

    reverse_align(args.glb, args.input, args.output)
