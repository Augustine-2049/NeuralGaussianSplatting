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
