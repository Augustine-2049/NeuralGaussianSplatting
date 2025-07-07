import torch
import cv2
import numpy as np
import os
from tqdm import tqdm

def render_video_frames(scene, gaussians, pipe, background, render_func, output_dir, iteration):
    """
    渲染视频帧并保存为图像序列
    
    Args:
        scene: 场景对象
        gaussians: 高斯模型
        pipe: 渲染管道参数
        background: 背景颜色
        render_func: 渲染函数 (render 或 render2)
        output_dir: 输出目录
        iteration: 当前迭代次数
    """
    video_cameras = scene.getVideoCameras()
    if not video_cameras:
        print("No video cameras found, skipping video rendering")
        return
    
    # 创建输出目录
    video_output_dir = os.path.join(output_dir, f"video_iter_{iteration}")
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"Rendering {len(video_cameras)} video frames for iteration {iteration}...")
    
    frames = []
    for i, viewpoint_cam in enumerate(tqdm(video_cameras, desc="Rendering video frames")):
        # 渲染当前视角
        render_pkg = render_func(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        
        # 转换为numpy格式并保存
        if image.dim() == 3 and image.shape[0] == 3:  # [3, H, W]
            image = image.permute(1, 2, 0)  # [H, W, 3]
        
        # 确保值在[0, 1]范围内
        image = torch.clamp(image, 0.0, 1.0)
        
        # 转换为numpy并保存
        image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        frame_path = os.path.join(video_output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        frames.append(image_np)
    
    # 创建视频
    create_video_from_frames(frames, video_output_dir, f"video_iter_{iteration}.mp4")
    
    print(f"Video saved to {video_output_dir}/video_iter_{iteration}.mp4")

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