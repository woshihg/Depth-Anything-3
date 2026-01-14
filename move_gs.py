import os
import shutil

def copy_gs_files(src_root, dst_root):
    if not os.path.exists(src_root):
        print(f"Source root not found: {src_root}")
        return
    if not os.path.exists(dst_root):
        print(f"Destination root not found: {dst_root}")
        return

    scenes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    print(f"Found {len(scenes)} scenes in output directory.")

    for scene in sorted(scenes):
        src_file = os.path.join(src_root, scene, "gs_ply", "0000.ply")
        dst_scene_dir = os.path.join(dst_root, scene)
        dst_file = os.path.join(dst_scene_dir, "0000.ply")

        if os.path.exists(src_file):
            if os.path.exists(dst_scene_dir):
                shutil.copy2(src_file, dst_file)
                print(f"  [Success] Copied GS for {scene} -> {dst_file}")
            else:
                print(f"  [Skip] Destination scene directory {dst_scene_dir} does not exist.")
        else:
            print(f"  [Warning] GS file not found for {scene}: {src_file}")

if __name__ == "__main__":
    SRC_DIR = "/home/woshihg/PycharmProjects/Depth-Anything-3/output/360_v2_split"
    DST_DIR = "/home/woshihg/360_v2"
    
    copy_gs_files(SRC_DIR, DST_DIR)
    print("\nFile copying complete!")
