import torch
import glob
import os
from depth_anything_3.api import DepthAnything3
import time
# 替换为你的代理地址，注意：
# 1. 即使是 https 协议，key 也建议全大写
# 2. 如果是本地代理，通常是 127.0.0.1:端口号
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
from depth_anything_3.utils.colmap_loader import load_colmap_data

def generate_3dgs_from_images(image_folder, output_dir, model_name="depth-anything/DA3-GIANT", process_res = 504, colmap_path=None, split=True, mask=False):
    """
    从图像文件夹生成 3D Gaussian Splatting (.glb) 文件。

    Args:
        image_folder (str): 包含输入图像序列的文件夹路径。
        output_dir (str): 保存输出文件的目录。
        model_name (str): 要使用的预训练模型名称。
    """
    # 检查 CUDA 是否可用，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Depth Anything 3 model...")
    # 从预训练权重加载模型
    model = DepthAnything3.from_pretrained(model_name).to(device)

    # 获取并排序图像文件列表
    # 1. 收集所有格式的路径
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]  # 建议加上大写支持
    image_paths = []
    image_test_paths = []
    image_train_paths = []

    if mask:
        image_mask_folder = os.path.join(image_folder, "masks")
        mask_paths = []
        for ext in extensions:
            mask_paths.extend(glob.glob(os.path.join(image_mask_folder, ext)))
        if not mask_paths:
            print(f"No mask images found in {image_mask_folder}")
            return
        mask_paths.sort()

    if split:
        image_train_folder = os.path.join(image_folder, "images", "train")
        image_test_folder = os.path.join(image_folder, "images", "test")
        for ext in extensions:
            image_test_paths.extend(glob.glob(os.path.join(image_test_folder, ext)))
            image_train_paths.extend(glob.glob(os.path.join(image_train_folder, ext)))
        if not image_test_paths or not image_train_paths:
            print(f"No train/test images found in {image_folder}/images/train or {image_folder}/images/test")
            return
        image_test_paths.sort()
        image_train_paths.sort()


        # 加载训练数据用于生成GS模型
        intrinsics, extrinsics = load_colmap_data(os.path.join(colmap_path), split='train')

        # 加载测试数据用于渲染
        test_intrinsics, test_extrinsics = load_colmap_data(os.path.join(colmap_path), split='test')


        prediction = model.inference(
            image=image_train_paths,
            export_dir=output_dir,
            process_res=process_res,
            export_format="npz-glb-gs_ply-gs_video-colmap-depth_uint16",
            align_to_input_ext_scale=True,
            infer_gs=True,  # Required for gs_ply and gs_video exports
            render_image=image_test_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            render_exts=test_extrinsics,
            render_ixts=test_intrinsics,
            export_kwargs={
                "mask_paths": mask_paths if mask else None,
            },
        )
    else:
        image_folder = os.path.join(image_folder, "images")
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        if not image_paths:
            print(f"No train images found in {image_folder}")
            return
        image_paths.sort()

        intrinsics, extrinsics = load_colmap_data(os.path.join(colmap_path))
        prediction = model.inference(
            image=image_paths,
            export_dir=output_dir,
            process_res=process_res,
            export_format="npz-glb-gs_ply-gs_video-colmap-depth_uint16",
            align_to_input_ext_scale=True,
            infer_gs=True,  # Required for gs_ply and gs_video exports
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_kwargs={
                "mask_paths": mask_paths if mask else None,
            },
        )


        if not image_paths:
            print(f"No images found in {image_folder}")
            return
        image_paths.sort()
        print(f"Found {len(image_paths)} images. Starting inference to generate 3DGS...")

    print("\nInference complete!")
    print(f"3DGS output has been saved to '{output_dir}'.")
    print("You should find a '.glb' file which can be viewed in a 3D viewer (e.g., Windows 3D Viewer, Blender).")
    print("You will also find:")
    print("- A '.ply' file containing the raw Gaussian Splatting data.")
    print("- A 'depth' subfolder with individual depth maps.")

    # 打印一些返回的预测信息
    if prediction.extrinsics is not None:
        print(f"\nEstimated extrinsics for {prediction.extrinsics.shape[0]} images.")

    # if colmap_path is not None:
    #     from depth_anything_3.utils.colmap_save import save_colmap_data
    #     save_colmap_data(colmap_path, prediction.extrinsics )
    #     print(f"Updated COLMAP extrinsics saved to '{colmap_path}/images.bin'.")

     # 如果有深度图，打印其形状
    if prediction.depth is not None:
        print(f"Generated depth maps with shape: {prediction.depth.shape}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate 3DGS from images.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the scene folder.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save output.')
    parser.add_argument('--process_res', type=int, default=504, help='Processing resolution.')
    parser.add_argument('--model_name', type=str, default="depth-anything/DA3-GIANT", help='Model name.')
    parser.add_argument('--split', action='store_true', help='Use train/test split logic.')
    parser.add_argument('--mask', action='store_true', help='Use masks.')

    args = parser.parse_args()

    time_start = time.time()
    generate_3dgs_from_images(
        image_folder=args.data_folder, 
        output_dir=args.output_dir, 
        model_name=args.model_name,
        process_res=args.process_res, 
        colmap_path=args.data_folder, 
        split=args.split, 
        mask=args.mask
    )
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start} seconds")
