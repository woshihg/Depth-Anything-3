import os
import shutil
import random
import struct
import numpy as np
import collections

# 将360_v2 的数据抽取一部分用于da3 初始化
# Colmap image representation
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "num_points2D", "points_raw"])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """Reads COLMAP images.bin file"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # binary_image_properties: 
            # id (4), qvec (32), tvec (24), camera_id (4) = 64 bytes
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            points_raw = fid.read(num_points2D * 24) # Keep points2D (8*2) and point3D_ids (8) = 24 bytes per point
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None,
                                     num_points2D=num_points2D, points_raw=points_raw)
    return images

def write_images_binary(images_dict, path_to_output_file):
    """Writes filtered COLMAP images to binary format"""
    with open(path_to_output_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(images_dict)))
        for img_id in sorted(images_dict.keys()):
            img = images_dict[img_id]
            fid.write(struct.pack("<i", img.id))
            fid.write(struct.pack("<dddd", *img.qvec))
            fid.write(struct.pack("<ddd", *img.tvec))
            fid.write(struct.pack("<i", img.camera_id))
            fid.write(img.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", img.num_points2D if img.num_points2D is not None else 0))
            if img.points_raw is not None:
                fid.write(img.points_raw)

def read_images_text(path):
    """Reads COLMAP images.txt file"""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            items = line.split()
            image_id = int(items[0])
            qvec = np.array([float(x) for x in items[1:5]])
            tvec = np.array([float(x) for x in items[5:8]])
            camera_id = int(items[8])
            image_name = items[9]
            # Next line is points2D, skip it
            fid.readline()
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None,
                                     num_points2D=0, points_raw=None)
    return images

def create_split(src_root, dst_root, num_samples=20):
    """
    Creates a sampled dataset from 360_v2.
    - Excludes validation images (index % 8 == 0 in sorted COLMAP images)
    - Samples 'num_samples' images for each scene.
    - Uses original resolution images.
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
        print(f"Created destination directory: {dst_root}")

    # Identify scenes (subdirectories with sparse/0)
    all_entries = os.listdir(src_root)
    scenes = [d for d in all_entries if os.path.isdir(os.path.join(src_root, d))]
    scenes = [s for s in scenes if os.path.exists(os.path.join(src_root, s, "sparse", "0"))]
    
    print(f"Found {len(scenes)} scenes to process.")

    for scene in sorted(scenes):
        print(f"\n--- Processing Scene: {scene} ---")
        scene_path = os.path.join(src_root, scene)
        
        # 1. Load COLMAP extrinsics to get image list and order
        images_bin = os.path.join(scene_path, "sparse", "0", "images.bin")
        images_txt = os.path.join(scene_path, "sparse", "0", "images.txt")
        
        cam_extrinsics = None
        if os.path.exists(images_bin):
            try:
                cam_extrinsics = read_images_binary(images_bin)
            except Exception as e:
                print(f"  Warning: Failed to read binary images.bin: {e}")
        
        if cam_extrinsics is None and os.path.exists(images_txt):
            try:
                cam_extrinsics = read_images_text(images_txt)
            except Exception as e:
                print(f"  Warning: Failed to read text images.txt: {e}")
                
        if cam_extrinsics is None:
            print(f"  Skipping scene {scene}: No valid COLMAP data found.")
            continue

        # 2. Sort images by name (referencing the logic provided)
        # logic: ImageFrameListSorted = sorted(ImageFrameList.copy(), key = lambda x : x.name)
        sorted_images = sorted(cam_extrinsics.values(), key=lambda x: x.name)
        
        # 3. Filter Training images (index % 8 != 0)
        # User: index % 8 = 0 的数据是验证集
        train_images = [img.name for i, img in enumerate(sorted_images) if i % 8 != 0]
        
        print(f"  Total images in COLMAP: {len(sorted_images)}")
        print(f"  Training images available: {len(train_images)}")
        
        if not train_images:
            print(f"  Skipping scene {scene}: No training images found.")
            continue

        # 4. Sample 20 images
        if len(train_images) > num_samples:
            sampled_names = random.sample(train_images, num_samples)
        else:
            print(f"  Note: Only {len(train_images)} training images. Taking all.")
            sampled_names = train_images

        # 5. Filter extrinsics for sampled images
        sampled_extrinsics = {img_id: img_obj for img_id, img_obj in cam_extrinsics.items() if img_obj.name in sampled_names}
            
        # 6. Copy images
        dst_scene_images_dir = os.path.join(dst_root, scene, "images")
        os.makedirs(dst_scene_images_dir, exist_ok=True)
        
        src_images_dir = os.path.join(scene_path, "images")
        copy_count = 0
        for img_name in sampled_names:
            # Handle potential path in name
            fname = os.path.basename(img_name)
            src_file = os.path.join(src_images_dir, fname)
            
            # Fallback if the name in colmap includes a relative path
            if not os.path.exists(src_file):
                src_file = os.path.join(src_images_dir, img_name)
                
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dst_scene_images_dir, fname))
                copy_count += 1
            else:
                print(f"  Error: Could not find image file: {src_file}")

        print(f"  Successfully copied {copy_count} images to {dst_scene_images_dir}")

        # 7. Save filtered poses (sparse/0/images.bin)
        dst_sparse_0_dir = os.path.join(dst_root, scene, "sparse", "0")
        os.makedirs(dst_sparse_0_dir, exist_ok=True)
        
        write_images_binary(sampled_extrinsics, os.path.join(dst_sparse_0_dir, "images.bin"))
        
        # 8. Copy cameras.bin (intrinsics)
        src_cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
        if os.path.exists(src_cameras_bin):
            shutil.copy2(src_cameras_bin, os.path.join(dst_sparse_0_dir, "cameras.bin"))
        else:
            # Try cameras.txt if .bin doesn't exist
            src_cameras_txt = os.path.join(scene_path, "sparse", "0", "cameras.txt")
            if os.path.exists(src_cameras_txt):
                shutil.copy2(src_cameras_txt, os.path.join(dst_sparse_0_dir, "cameras.txt"))
            else:
                print(f"  Warning: No cameras.bin or cameras.txt found in {scene_path}/sparse/0")

if __name__ == "__main__":
    # Configure paths
    SOURCE_DATASET = "/home/woshihg/360_v2"
    TARGET_DATASET = "/home/woshihg/360_v2_split"
    NUM_TO_SAMPLE = 10
    
    if os.path.exists(SOURCE_DATASET):
        create_split(SOURCE_DATASET, TARGET_DATASET, NUM_TO_SAMPLE)
        print("\nDataset split creation finished.")
    else:
        print(f"Source directory not found: {SOURCE_DATASET}")
