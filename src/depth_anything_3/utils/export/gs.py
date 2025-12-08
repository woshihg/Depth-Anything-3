# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal, Optional
import moviepy.editor as mpy
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.visualize import vis_depth_map_tensor

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}


def calculate_psnr(img1, img2):
    """Calculates PSNR between two images. Images are tensors in [0, 1] range."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def export_to_gs_ply(
    prediction: Prediction,
    export_dir: str,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
):
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)  # v h w 1
    idx = 0
    os.makedirs(os.path.join(export_dir, "gs_ply"), exist_ok=True)
    save_path = os.path.join(export_dir, f"gs_ply/{idx:04d}.ply")
    if gs_views_interval is None:  # select around 12 views in total
        gs_views_interval = max(pred_depth.shape[0] // 12, 1)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=False,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def export_to_gs_video(
    prediction: Prediction,
    export_dir: str,
    render_extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    render_intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    out_image_hw: Optional[tuple[int, int]] = None,  # render views' resolution, (h, w)
    image: list[np.ndarray | Image.Image | str] = None,
    render_image: list[np.ndarray | Image.Image | str] = None,
    chunk_size: Optional[int] = 4,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "extend",
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    vis_depth: Optional[Literal["hcat", "vcat"]] = "hcat",
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    video_quality: Literal["low", "medium", "high"] = "high",
) -> None:
    gs_world = prediction.gaussians
    # if target poses are not provided, render the (smooth/interpolate) input poses
    if render_extrinsics is not None:
        tgt_extrs = render_extrinsics[..., :3, :].unsqueeze(0).to(gs_world.means)
    else:
        tgt_extrs = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs_world.means)
        if prediction.is_metric:
            scale_factor = prediction.scale_factor
            if scale_factor is not None:
                tgt_extrs[:, :, :3, 3] /= scale_factor
    # 确保 intrinsics 也是一个在正确设备上的张量
    if intrinsics is not None:
        tgt_intrs = intrinsics.unsqueeze(0).to(gs_world.means)
    else:
        tgt_intrs = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs_world.means)

    # if render resolution is not provided, render the input ones
    if out_image_hw is not None:
        H, W = out_image_hw
    else:
        H, W = prediction.depth.shape[-2:]

    # Render and save original input views before interpolation
    os.makedirs(os.path.join(export_dir, "gs_test_frames"), exist_ok=True)
    test_color, _ = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode="original",  # Use original poses without interpolation
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # 保存渲染的测试帧并计算 PSNR
    if test_color is not None:
        frames_rendered = (test_color[0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for f_idx, frame in enumerate(frames_rendered):
            # 按照原来的render_image路径中图像名称保存渲染图像
            save_path_base = os.path.join(export_dir, f"gs_test_frames/")
            if render_image is not None and f_idx < len(render_image):
                gt_image_path = render_image[f_idx]
                gt_image_name = os.path.basename(gt_image_path)
                save_path = os.path.join(save_path_base, gt_image_name)
            else:
                save_path = os.path.join(save_path_base, f"rendered_{f_idx:04d}.png")
            mpy.ImageClip(frame).save_frame(save_path)
        print(f"Saved {len(frames_rendered)} rendered test frames to 'gs_test_frames'.")

        # 计算 PSNR
        if render_image is not None and len(render_image) == len(frames_rendered):
            psnr_values = []
            for i, gt_image_path in enumerate(render_image):
                # 加载并预处理 GT 图像
                gt_image = Image.open(gt_image_path).convert("RGB")
                gt_image_resized = gt_image.resize((W, H), Image.LANCZOS)
                gt_tensor = to_tensor(gt_image_resized).to(gs_world.means.device) # C, H, W

                # 获取渲染图像
                rendered_tensor = test_color[0, i] # C, H, W

                # 计算 PSNR
                psnr = calculate_psnr(rendered_tensor, gt_tensor)
                psnr_values.append(psnr.item())
                print(f"PSNR for {os.path.basename(gt_image_path)}: {psnr.item():.2f} dB")

            avg_psnr = np.mean(psnr_values)
            print(f"-----------------------------------------")
            print(f"Average PSNR for test set: {avg_psnr:.2f} dB")
            print(f"-----------------------------------------")


    train_color, _ = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode="original",
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # 保存train views 图像并计算 PSNR
    if train_color is not None:
        frames_rendered = (train_color[0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for f_idx, frame in enumerate(frames_rendered):
            # 按照原来的image路径中图像名称保存渲染图像
            save_path_base = os.path.join(export_dir, f"gs_train_frames/")
            if image is not None and f_idx < len(image):
                gt_image_path = image[f_idx]
                gt_image_name = os.path.basename(gt_image_path)
                save_path = os.path.join(save_path_base, gt_image_name)
            else:
                save_path = os.path.join(save_path_base, f"rendered_{f_idx:04d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mpy.ImageClip(frame).save_frame(save_path)
        print(f"Saved {len(frames_rendered)} rendered train frames to 'gs_train_frames'.")

        # 计算 PSNR
        if image is not None and len(image) == len(frames_rendered):
            psnr_values = []
            for i, gt_image_path in enumerate(image):
                # 加载并预处理 GT 图像
                gt_image = Image.open(gt_image_path).convert("RGB")
                gt_image_resized = gt_image.resize((W, H), Image.LANCZOS)
                gt_tensor = to_tensor(gt_image_resized).to(gs_world.means.device) # C, H, W

                # 获取渲染图像
                rendered_tensor = train_color[0, i] # C, H, W

                # 计算 PSNR
                psnr = calculate_psnr(rendered_tensor, gt_tensor)
                psnr_values.append(psnr.item())
                print(f"PSNR for {os.path.basename(gt_image_path)}: {psnr.item():.2f} dB")

            avg_psnr = np.mean(psnr_values)
            print(f"-----------------------------------------")
            print(f"Average PSNR for train set: {avg_psnr:.2f} dB")
            print(f"-----------------------------------------")

    # if single views, render wander trj
    if tgt_extrs.shape[1] <= 1:
        trj_mode = "wander"
        # trj_mode = "dolly_zoom"

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # 保存color 图像
    os.makedirs(os.path.join(export_dir, "gs_video_frames"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()
        for f_idx, frame in enumerate(frames):
            save_path = os.path.join(export_dir, f"gs_video_frames/{idx:04d}_{f_idx:04d}.png")
            mpy.ImageClip(frame).save_frame(save_path)


    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[video_quality]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[video_quality]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join(export_dir, "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 24
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        output_name = f"{idx:04d}_{trj_mode}" if output_name is None else output_name
        save_path = os.path.join(export_dir, f"gs_video/{output_name}.mp4")
        # clip.write_videofile(save_path, codec="libx264", audio=False, bitrate="4000k")
        clip.write_videofile(
            save_path,
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=ffmpeg_params,
        )
    return
