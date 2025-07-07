#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from typing import Union
import cv2


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def show_img(
    img: Union[np.ndarray, torch.Tensor],
    is_bgr: bool = False,
    window_name: str = "Image",
    wait_key: int = 1,
) -> None:
    if not isinstance(img, (np.ndarray, torch.Tensor)):
        raise ValueError("输入必须是 NumPy 数组或 PyTorch Tensor")

    # 转换为 NumPy 数组并确保在 CPU
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()  # 脱离计算图 + 转到 CPU
    else:
        img_np = img.copy()  # 避免修改原数组

    # 检查形状是否为 [3, H, W] 或 [H, W, 3]
    if img_np.ndim != 3:
        raise ValueError("输入必须是 3 维数组 (C, H, W) 或 (H, W, C)")

    # 调整通道顺序 [3, H, W] → [H, W, 3]
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)

    # 检查数据类型并缩放到 [0, 255]（如果是浮点数）
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.max() <= 1.0:  # 假设范围是 [0, 1]
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    # 转换颜色空间（如果需要）
    if not is_bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 显示图像
    cv2.imshow(window_name, img_np)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()


def show_img2(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    title1: str = "Rendered Image",
    title2: str = "Ground Truth",
    is_bgr: bool = False,
    wait_key: int = 1,
) -> None:
    """
    对比显示两张图像，用于训练中对比渲染图像和真实图像
    
    Args:
        img1: 第一张图像（通常是渲染图像）
        img2: 第二张图像（通常是真实图像）
        title1: 第一张图像的窗口标题
        title2: 第二张图像的窗口标题
        is_bgr: 是否为BGR格式
        wait_key: 等待时间（毫秒）
    """
    if not isinstance(img1, (np.ndarray, torch.Tensor)) or not isinstance(img2, (np.ndarray, torch.Tensor)):
        raise ValueError("输入必须是 NumPy 数组或 PyTorch Tensor")

    def process_img(img):
        # 转换为 NumPy 数组并确保在 CPU
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img.copy()

        # 检查形状是否为 [3, H, W] 或 [H, W, 3]
        if img_np.ndim != 3:
            raise ValueError("输入必须是 3 维数组 (C, H, W) 或 (H, W, C)")

        # 调整通道顺序 [3, H, W] → [H, W, 3]
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)

        # 检查数据类型并缩放到 [0, 255]（如果是浮点数）
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_np.max() <= 1.0:  # 假设范围是 [0, 1]
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # 转换颜色空间（如果需要）
        if not is_bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_np

    # 处理两张图像
    img1_np = process_img(img1)
    img2_np = process_img(img2)

    # 创建对比图像（水平拼接）
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    
    # 确保两张图像高度相同
    if h1 != h2:
        # 调整第二张图像的高度以匹配第一张
        img2_np = cv2.resize(img2_np, (w2, h1))
    
    # 水平拼接
    comparison_img = np.hstack([img1_np, img2_np])
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)  # 白色文字
    
    # 在图像上添加标签
    cv2.putText(comparison_img, title1, (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison_img, title2, (w1 + 10, 30), font, font_scale, color, thickness)
    
    # 显示对比图像
    cv2.imshow("Image Comparison", comparison_img)
    cv2.waitKey(wait_key)
    # 移除cv2.destroyAllWindows()，让窗口保持打开状态


def show_img_grid(
    images: list,
    titles: list = None,
    grid_size: tuple = None,
    is_bgr: bool = False,
    wait_key: int = 1,
) -> None:
    """
    以网格形式显示多张图像
    
    Args:
        images: 图像列表
        titles: 标题列表（可选）
        grid_size: 网格大小 (rows, cols)，如果为None则自动计算
        is_bgr: 是否为BGR格式
        wait_key: 等待时间（毫秒）
    """
    if not images:
        return
    
    def process_img(img):
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img.copy()

        if img_np.ndim != 3:
            raise ValueError("输入必须是 3 维数组 (C, H, W) 或 (H, W, C)")

        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)

        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        if not is_bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_np

    # 处理所有图像
    processed_images = [process_img(img) for img in images]
    
    # 确定网格大小
    n_images = len(processed_images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # 获取图像尺寸
    h, w = processed_images[0].shape[:2]
    
    # 创建网格画布
    grid_h = h * rows
    grid_w = w * cols
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 填充网格
    for i, img in enumerate(processed_images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # 调整图像大小以匹配网格
        img_resized = cv2.resize(img, (w, h))
        
        # 放置图像
        y_start = row * h
        y_end = (row + 1) * h
        x_start = col * w
        x_end = (col + 1) * w
        grid_img[y_start:y_end, x_start:x_end] = img_resized
        
        # 添加标题（如果提供）
        if titles and i < len(titles):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (255, 255, 255)
            cv2.putText(grid_img, titles[i], (x_start + 5, y_start + 20), 
                       font, font_scale, color, thickness)
    
    # 显示网格
    cv2.imshow("Image Grid", grid_img)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()
