#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.dataset_readers import readCamerasFromTransforms

def test_video_camera_loading():
    """测试视频相机加载功能"""
    
    # 创建一个模拟的transforms_video.json文件
    test_data = {
        "camera_angle_x": 0.8575560541152954,
        "frames": [
            {
                "file_path": "video/frame_0000",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 4.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            },
            {
                "file_path": "video/frame_0001",
                "transform_matrix": [
                    [0.8660254037844387, 0.0, -0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.8660254037844387, 4.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
    }
    
    # 创建测试目录
    test_dir = "test_video_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # 写入测试文件
    import json
    with open(os.path.join(test_dir, "transforms_video.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    try:
        # 测试读取视频相机（没有对应图片的情况）
        print("Testing video camera loading without images...")
        video_cameras = readCamerasFromTransforms(
            test_dir, 
            "transforms_video.json", 
            white_background=True, 
            extension=".png",
            default_width=800,
            default_height=600
        )
        
        print(f"Successfully loaded {len(video_cameras)} video cameras")
        for i, cam in enumerate(video_cameras):
            print(f"Camera {i}: {cam.width}x{cam.height}, FovX={cam.FovX:.4f}, FovY={cam.FovY:.4f}")
            print(f"  R shape: {cam.R.shape}, T shape: {cam.T.shape}")
            print(f"  Image size: {cam.image.size}")
        
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_video_camera_loading() 