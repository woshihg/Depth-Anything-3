import os
import subprocess
import sys

def process_all(root_dir, output_root_base, python_exe):
    if not os.path.exists(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    # 获取所有场景目录
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    scenes = sorted(scenes)
    
    print(f"Found {len(scenes)} scenes to process: {scenes}")

    for scene in scenes:
        print(f"\n" + "="*50)
        print(f">>> Processing Scene: {scene}")
        print("="*50)
        
        data_folder = os.path.join(root_dir, scene)
        output_dir = os.path.join(output_root_base, scene)
        
        # 构建命令
        cmd = [
            python_exe, "gausian.py",
            "--data_folder", data_folder,
            "--output_dir", output_dir,
            "--process_res", "504",
            "--mask", 
            # 如果需要 split 或 mask，可以在此添加 --split 或 --mask
        ]
        
        try:
            # 运行 gausian.py
            subprocess.run(cmd, check=True)
            print(f"\n[Success] Finished processing {scene}")
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] Failed to process {scene} with exit code {e.returncode}")
        except Exception as e:
            print(f"\n[Error] An unexpected error occurred for {scene}: {e}")

if __name__ == "__main__":
    # 配置
    TARGET_DATASET_ROOT = "/home/woshihg/graspnet_split"
    OUTPUT_ROOT_BASE = "output/graspnet_split_mask"
    PYTHON_EXECUTABLE = "/home/woshihg/miniconda3/envs/da3/bin/python"
    
    process_all(TARGET_DATASET_ROOT, OUTPUT_ROOT_BASE, PYTHON_EXECUTABLE)
    print("\nBatch processing complete!")
