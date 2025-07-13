import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def render_video_frames(scene, gaussians, pipe, background, render_func, output_dir, iteration, use_depth=False, use_colmap=False):
    """
    渲染视频帧并保存为图像序列，支持用深度图制作视频，创建视频后删除原始图像
    
    Args:
        scene: 场景对象
        gaussians: 高斯模型
        pipe: 渲染管道参数
        background: 背景颜色
        render_func: 渲染函数 (render 或 render2)
        output_dir: 输出目录
        iteration: 当前迭代次数
        use_depth: 是否用深度图制作视频
    """
    video_cameras = scene.getVideoCameras()
    if not video_cameras:
        print("No video cameras found, skipping video rendering")
        return
    
    # 创建统一的视频输出目录
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # 创建临时目录用于存放帧图像
    temp_frames_dir = os.path.join(output_dir, f"temp_frames_iter_{iteration}")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    print(f"Rendering {len(video_cameras)} video frames for iteration {iteration}...")
    
    frames = []
    for i, viewpoint_cam in enumerate(tqdm(video_cameras, desc="Rendering video frames")):
        render_pkg = render_func(viewpoint_cam, gaussians, pipe, background)
        if use_colmap:
            image = render_pkg["colmap"]
            if image.dim() == 3 and image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = torch.clamp(image, 0.0, 1.0)
            image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            frames.append(image_np)
        elif use_depth:
            # 用深度图
            depth = render_pkg["depthmap"]
            if depth.dim() == 3:
                depth = depth.squeeze(0)
            depth_np = depth.detach().cpu().numpy()
            depth_min, depth_max = depth_np.min(), depth_np.max()
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
            depth_img = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, depth_color)
            frames.append(depth_color)
            # 新增：保存深度直方图
            hist_path = os.path.join(temp_frames_dir, f"frame_{i:04d}_depth_hist.png")
            plot_depthmap_histogram(depth_np, save_path=hist_path)
        else:
            # 用RGB图
            image = render_pkg["render"]
            if image.dim() == 3 and image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = torch.clamp(image, 0.0, 1.0)
            image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            frames.append(image_np)
    
    # 创建视频并保存到统一目录
    render_func_name = render_func.__name__
    video_name = f"{render_func_name}_iter_{iteration}_depth.mp4" if use_depth else f"{render_func_name}_iter_{iteration}.mp4"
    create_video_from_frames(frames, temp_frames_dir, video_name)
    
    # 将视频移动到统一目录
    temp_video_path = os.path.join(temp_frames_dir, video_name)
    final_video_path = os.path.join(videos_dir, video_name)
    
    if os.path.exists(temp_video_path):
        import shutil
        shutil.move(temp_video_path, final_video_path)
        print(f"Video saved to {final_video_path}")
    
    # 删除临时目录和所有帧图像
    if os.path.exists(temp_frames_dir):
        import shutil
        shutil.rmtree(temp_frames_dir)
        print(f"Cleaned up temporary frames directory: {temp_frames_dir}")

def create_video_from_frames(frames, output_dir, video_name):
    """
    从帧序列创建视频
    
    Args:
        frames: 帧列表，每个帧是numpy数组 [H, W, 3]
        output_dir: 输出目录
        video_name: 视频文件名
    """
    if not frames:
        return
    
    # 获取帧的尺寸
    height, width = frames[0].shape[:2]
    
    # 创建视频写入器
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # 30帧每秒
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # 写入帧
    for frame in frames:
        # 转换为BGR格式
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    # 释放资源
    out.release()
    print(f"Video created: {video_path}")

def plot_depthmap_histogram(depthmap_np, save_path=None, bins=100):
    """
    绘制深度图的元素分布直方图
    Args:
        depthmap_np: numpy数组，深度图 (H, W) 或 (H, W, 1)
        save_path: 保存直方图的路径（可选）
        bins: 直方图分箱数
    """
    if depthmap_np.ndim == 3:
        depthmap_np = np.squeeze(depthmap_np)
    valid = np.isfinite(depthmap_np)
    data = depthmap_np[valid]
    plt.figure(figsize=(6,4))
    plt.hist(data.flatten(), bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Depth Value')
    plt.ylabel('Pixel Count')
    plt.title('Depthmap Value Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close() 