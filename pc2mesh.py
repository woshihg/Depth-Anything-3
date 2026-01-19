import trimesh
import open3d as o3d
import numpy as np
import argparse
import os

def pc_to_mesh(input_path, output_path, depth=9, density_threshold=0.01, visualize=False, save_combined=False, gs_path=None):
    """
    Convert a point cloud (from .glb or other formats) to a mesh using Poisson reconstruction.
    
    Args:
        input_path (str): Path to the input point cloud file (.glb, .ply, etc.)
        output_path (str): Path to save the output mesh file (.obj, .ply, .glb, etc.)
        depth (int): The depth of the octree used for Poisson reconstruction. Higher means more detail.
        density_threshold (float): Quantile threshold to remove low-density vertices (noise).
        visualize (bool): Whether to open a window to visualize the point cloud and mesh together.
        save_combined (bool): Whether to save both point cloud and mesh into a single .glb file.
        gs_path (str): Path to a Gaussian Splatting .ply file to visualize alongside.
    """
    print(f"Loading point cloud from {input_path}...")
    # Load using trimesh to handle .glb easily
    scene = trimesh.load(input_path)
    
    # Get alignment matrix if it exists in metadata (DA3 specific)
    alignment_matrix = None
    if isinstance(scene, trimesh.Scene) and scene.metadata and "hf_alignment" in scene.metadata:
        alignment_matrix = np.array(scene.metadata["hf_alignment"])
        print("Found alignment matrix in .glb metadata.")

    # Extract points and colors from the scene
    if isinstance(scene, trimesh.Scene):
        # Filter only PointCloud geometries to avoid camera wireframes (Path3D)
        pc_geoms = [
            g.apply_transform(scene.graph[name][0]) 
            for name, g in scene.geometry.items() 
            if isinstance(g, trimesh.PointCloud)
        ]
        
        if not pc_geoms:
            # Fallback: try to get anything that has vertices if no PointCloud is explicitly found
            flattened = scene.to_geometry()
            points = flattened.vertices
            if hasattr(flattened, "colors") and flattened.colors is not None:
                colors = flattened.colors[:, :3] / 255.0
            elif hasattr(flattened, "visual") and hasattr(flattened.visual, "vertex_colors"):
                colors = flattened.visual.vertex_colors[:, :3] / 255.0
            else:
                colors = None
        else:
            # Concatenate all point clouds found
            points = np.concatenate([g.vertices for g in pc_geoms], axis=0)
            all_colors = []
            for g in pc_geoms:
                if g.colors is not None:
                    all_colors.append(g.colors[:, :3])
            colors = np.concatenate(all_colors, axis=0) / 255.0 if all_colors else None
        
        if points is None or len(points) == 0:
            raise ValueError("No point cloud found in the input file.")
    else:
        points = scene.vertices
        colors = scene.colors[:, :3] / 255.0 if scene.colors is not None else None

    print(f"Loaded {len(points)} points.")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 1. Estimate normals (required for Poisson)
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    # 2. Poisson Surface Reconstruction
    print(f"Performing Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 3. Post-processing: Remove low-density vertices (outliers/noise)
    print("Cleaning up mesh...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4. Save the mesh
    print(f"Saving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)

    # 5. Optional: Load Gaussian Splatting if provided (needed for combined save or visualization)
    gs_pcd_data = None
    if gs_path:
        print(f"Loading Gaussian Splatting from {gs_path}...")
        from plyfile import PlyData
        plydata = PlyData.read(gs_path)
        v = plydata['vertex']
        gs_pts = np.stack([v['x'], v['y'], v['z']], axis=1)
        
        # Apply alignment if found in .glb
        if alignment_matrix is not None:
            print("Applying .glb alignment to Gaussians...")
            gs_pts_homo = np.hstack([gs_pts, np.ones((gs_pts.shape[0], 1))])
            gs_pts = (gs_pts_homo @ alignment_matrix.T)[:, :3]
        
        # Extract colors (DC component)
        if 'f_dc_0' in v:
            f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
            # Simple SH to RGB approximation: 0.5 + 0.28209 * f_dc
            gs_colors = np.clip(0.5 + 0.28209 * f_dc, 0, 1)
        else:
            gs_colors = None
        
        gs_pcd_data = {"points": gs_pts, "colors": gs_colors}

    # 6. Optional: Save combined .glb
    if save_combined:
        combined_path = os.path.splitext(output_path)[0] + "_combined.glb"
        print(f"Saving combined point cloud, mesh, and GS to {combined_path}...")
        
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) * 255.0 if mesh.has_vertex_colors() else None
        
        tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, metadata={"name": "Reconstructed Mesh"})
        tm_pc = trimesh.points.PointCloud(vertices=points, colors=colors * 255.0 if colors is not None else None, metadata={"name": "Original PointCloud"})
        
        geometries = [tm_mesh, tm_pc]
        
        if gs_pcd_data:
            tm_gs = trimesh.points.PointCloud(
                vertices=gs_pcd_data["points"], 
                colors=gs_pcd_data["colors"] * 255.0 if gs_pcd_data["colors"] is not None else None,
                metadata={"name": "Gaussian Centers"}
            )
            geometries.append(tm_gs)
        
        scene = trimesh.Scene(geometries)
        scene.export(combined_path)

    # 7. Optional: Visualize
    if visualize or gs_path:
        print("Opening visualization window...")
        geometries = [pcd, mesh]
        
        if gs_pcd_data:
            gs_pcd = o3d.geometry.PointCloud()
            gs_pcd.points = o3d.utility.Vector3dVector(gs_pcd_data["points"])
            if gs_pcd_data["colors"] is not None:
                gs_pcd.colors = o3d.utility.Vector3dVector(gs_pcd_data["colors"])
            geometries.append(gs_pcd)
            print(f"Visualizing {len(gs_pcd_data['points'])} Gaussians.")

        print("Controls: [Left Click] Rotate, [Right Click] Pan, [Scroll] Zoom")
        o3d.visualization.draw_geometries(geometries, window_name="Point Cloud, Mesh & GS Comparison")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Point Cloud (.glb) to Mesh")
    parser.add_argument("input", help="Path to input .glb point cloud")
    parser.add_argument("--output", help="Path to output mesh (default: input_mesh.obj)")
    parser.add_argument("--depth", type=int, default=9, help="Poisson reconstruction depth (default: 9)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Density threshold for pruning (default: 0.01)")
    parser.add_argument("--visualize", action="store_true", help="Visualize point cloud and mesh together")
    parser.add_argument("--combined", action="store_true", help="Save both point cloud and mesh in one .glb file")
    parser.add_argument("--gs", help="Path to Gaussian Splatting .ply file to visualize")

    args = parser.parse_args()

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_mesh.obj"

    pc_to_mesh(args.input, args.output, depth=args.depth, density_threshold=args.threshold, 
               visualize=args.visualize, save_combined=args.combined, gs_path=args.gs)
