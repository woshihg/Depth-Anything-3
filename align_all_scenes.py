import os
import sys

# 将 src 目录添加到路径，以便导入
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from depth_anything_3.utils.colmap_change import recenter_colmap_model, write_points3D_binary

def batch_align(root_dir):
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Found {len(scenes)} scenes to align.")

    for scene in sorted(scenes):
        print(f"\n>>> Aligning scene: {scene}")
        # 根据 prepare_split_dataset.py 的逻辑，数据应该在 sparse/0 下
        sparse_path = os.path.join(root_dir, scene, "sparse", "0")
        
        if not os.path.exists(sparse_path):
            # 兼容性检查：如果在根目录
            sparse_path = os.path.join(root_dir, scene, "sparse")
            if not os.path.exists(os.path.join(sparse_path, "images.bin")):
                print(f"  [Skip] No COLMAP data found in {scene}/sparse/0")
                continue

        # 检查 points3D.bin 是否存在。
        # 如果不存在，为了让 colmap_change 运行，我们创建一个空的 points3D.bin
        points_bin = os.path.join(sparse_path, "points3D.bin")
        if not os.path.exists(points_bin):
            print(f"  [Note] No points3D.bin found. Creating a dummy empty points3D.bin to allow recentering.")
            write_points3D_binary({}, points_bin)

        try:
            recenter_colmap_model(sparse_path)
            print(f"  [Success] Finished aligning {scene}")
        except Exception as e:
            print(f"  [Error] Failed to align {scene}: {e}")

if __name__ == "__main__":
    TARGET_DATASET = "/home/woshihg/360_v2_split"
    batch_align(TARGET_DATASET)
