#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_analysis import analyze_gaussian_features, save_feature_history

def test_feature_analysis():
    """测试特征分析功能"""
    
    # 创建模拟的高斯模型数据
    class MockGaussians:
        def __init__(self):
            # 模拟1000个高斯点，每个点有64维神经网络特征
            self._neural_features = torch.randn(1000, 64) * 0.1  # 随机64维特征值
        
        @property
        def get_neural_features(self):
            return self._neural_features
    
    # 创建测试目录
    test_output_dir = "test_feature_analysis_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # 创建模拟的高斯模型
        mock_gaussians = MockGaussians()
        
        print("Testing feature analysis...")
        
        # 测试特征分析
        for iteration in [1000, 2000, 3000]:
            print(f"\nTesting iteration {iteration}...")
            feature_stats = analyze_gaussian_features(mock_gaussians, test_output_dir, iteration)
            save_feature_history(iteration, feature_stats, test_output_dir)
            
            print(f"Feature stats: {feature_stats}")
        
        print(f"\nTest completed! Check results in {test_output_dir}/feature_analysis/")
        
        # 列出生成的文件
        analysis_dir = os.path.join(test_output_dir, "feature_analysis")
        if os.path.exists(analysis_dir):
            print("\nGenerated files:")
            for file in os.listdir(analysis_dir):
                print(f"  - {file}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            print(f"\nCleaned up test directory: {test_output_dir}")

if __name__ == "__main__":
    test_feature_analysis() 