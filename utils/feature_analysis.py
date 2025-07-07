import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def analyze_gaussian_features(gaussians, output_dir, iteration):
    """
    分析高斯点的特征权值分布并生成可视化图表
    
    Args:
        gaussians: GaussianModel对象
        output_dir: 输出目录
        iteration: 当前迭代次数
    """
    print(f"Analyzing Gaussian features distribution for iteration {iteration}...")
    
    # 获取64维神经网络特征数据
    features = gaussians.get_neural_features[1:].detach().cpu().numpy()  # [N, 64]
    
    # 创建分析结果目录
    analysis_dir = os.path.join(output_dir, "feature_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. 基本统计信息
    feature_stats = {
        'mean': np.mean(features),
        'std': np.std(features),
        'min': np.min(features),
        'max': np.max(features),
        'median': np.median(features),
        'total_points': features.shape[0],
        'feature_dim': features.shape[1]  # 64维特征
    }
    
    # 保存统计信息到文本文件
    stats_file = os.path.join(analysis_dir, f"feature_stats_iter_{iteration}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Gaussian Features Analysis - Iteration {iteration}\n")
        f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Gaussian points: {feature_stats['total_points']}\n")
        f.write(f"Feature dimensions: {feature_stats['feature_dim']}\n")
        f.write(f"Mean: {feature_stats['mean']:.6f}\n")
        f.write(f"Std: {feature_stats['std']:.6f}\n")
        f.write(f"Min: {feature_stats['min']:.6f}\n")
        f.write(f"Max: {feature_stats['max']:.6f}\n")
        f.write(f"Median: {feature_stats['median']:.6f}\n")
    
    # 2. 创建可视化图表
    create_feature_visualizations(features, analysis_dir, iteration, feature_stats)
    
    print(f"Feature analysis saved to {analysis_dir}")
    return feature_stats

def create_feature_visualizations(features, analysis_dir, iteration, stats):
    """
    创建特征分布的可视化图表
    """
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 将特征展平为一维数组
    features_flat = features.flatten()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Gaussian Features Distribution - Iteration {iteration}', fontsize=16)
    
    # 1. 直方图
    axes[0, 0].hist(features_flat, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Feature Values Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加统计信息
    axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.4f}')
    axes[0, 0].axvline(stats['median'], color='green', linestyle='--', label=f'Median: {stats["median"]:.4f}')
    axes[0, 0].legend()
    
    # 2. 箱线图
    # 为了显示箱线图，我们取样本（如果数据太多）
    if len(features_flat) > 10000:
        sample_indices = np.random.choice(len(features_flat), 10000, replace=False)
        features_sample = features_flat[sample_indices]
    else:
        features_sample = features_flat
    
    axes[0, 1].boxplot(features_sample, vert=True)
    axes[0, 1].set_title('Feature Values Box Plot')
    axes[0, 1].set_ylabel('Feature Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 密度图
    axes[1, 0].hist(features_flat, bins=100, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Feature Values Density')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加正态分布拟合线
    from scipy import stats as scipy_stats
    mu, sigma = scipy_stats.norm.fit(features_flat)
    x = np.linspace(features_flat.min(), features_flat.max(), 100)
    y = scipy_stats.norm.pdf(x, mu, sigma)
    axes[1, 0].plot(x, y, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
    axes[1, 0].legend()
    
    # 4. 64维特征分析
    # 计算每个维度的统计信息
    dim_means = np.mean(features, axis=0)  # 对每个64维特征求平均
    dim_stds = np.std(features, axis=0)
    
    x_pos = np.arange(len(dim_means))
    axes[1, 1].bar(x_pos, dim_means, yerr=dim_stds, capsize=5, alpha=0.7, color='orange')
    axes[1, 1].set_title('Mean Feature Values by Dimension (64D)')
    axes[1, 1].set_xlabel('Feature Dimension (0-63)')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(analysis_dir, f"feature_distribution_iter_{iteration}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建额外的详细分析图表
    create_detailed_analysis(features, analysis_dir, iteration)
    
    # 创建64维特征的专门分析
    create_64d_feature_analysis(features, analysis_dir, iteration)

def create_detailed_analysis(features, analysis_dir, iteration):
    """
    创建更详细的分析图表
    """
    # 1. 64维特征相关性热力图
    # 由于64维特征矩阵太大，我们创建一个简化的相关性图
    # 计算特征之间的相关性
    corr_matrix = np.corrcoef(features.T)  # [64, 64]
    
    # 创建相关性热力图（只显示部分，避免过于密集）
    plt.figure(figsize=(12, 10))
    
    # 为了可读性，我们只显示每8个维度的相关性
    step = 8
    indices = np.arange(0, 64, step)
    corr_subset = corr_matrix[indices][:, indices]
    
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0,
               square=True, fmt='.2f', 
               xticklabels=[f'{i}' for i in indices],
               yticklabels=[f'{i}' for i in indices])
    plt.title(f'Feature Correlation Matrix (64D, sampled every {step}) - Iteration {iteration}')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Dimension')
    plt.tight_layout()
    
    corr_file = os.path.join(analysis_dir, f"feature_correlation_iter_{iteration}.png")
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 特征值随时间的变化趋势（如果有历史数据）
    # 这里可以扩展为保存历史数据并绘制趋势图
    
    # 3. 异常值检测
    features_flat = features.flatten()
    Q1 = np.percentile(features_flat, 25)
    Q3 = np.percentile(features_flat, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = features_flat[(features_flat < lower_bound) | (features_flat > upper_bound)]
    outlier_percentage = len(outliers) / len(features_flat) * 100
    
    # 保存异常值信息
    outlier_file = os.path.join(analysis_dir, f"outlier_analysis_iter_{iteration}.txt")
    with open(outlier_file, 'w') as f:
        f.write(f"Outlier Analysis - Iteration {iteration}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total data points: {len(features_flat)}\n")
        f.write(f"Outliers detected: {len(outliers)}\n")
        f.write(f"Outlier percentage: {outlier_percentage:.2f}%\n")
        f.write(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.6f}\n")
        f.write(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.6f}\n")
        if len(outliers) > 0:
            f.write(f"Outlier range: [{outliers.min():.6f}, {outliers.max():.6f}]\n")

def save_feature_history(iteration, stats, output_dir):
    """
    保存特征统计历史，用于绘制趋势图
    """
    history_file = os.path.join(output_dir, "feature_analysis", "feature_history.txt")
    
    # 读取现有历史数据
    history_data = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        history_data.append({
                            'iteration': int(parts[0]),
                            'mean': float(parts[1]),
                            'std': float(parts[2]),
                            'min': float(parts[3]),
                            'max': float(parts[4]),
                            'total_points': int(parts[5])
                        })
    
    # 添加新数据
    history_data.append({
        'iteration': iteration,
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max'],
        'total_points': stats['total_points']
    })
    
    # 按迭代次数排序
    history_data.sort(key=lambda x: x['iteration'])
    
    # 保存历史数据
    with open(history_file, 'w') as f:
        f.write("iteration,mean,std,min,max,total_points\n")
        for data in history_data:
            f.write(f"{data['iteration']},{data['mean']:.6f},{data['std']:.6f},"
                   f"{data['min']:.6f},{data['max']:.6f},{data['total_points']}\n")
    
    # 如果有多于1个数据点，创建趋势图
    if len(history_data) > 1:
        create_trend_plot(history_data, output_dir)

def create_trend_plot(history_data, output_dir):
    """
    创建特征统计趋势图
    """
    iterations = [d['iteration'] for d in history_data]
    means = [d['mean'] for d in history_data]
    stds = [d['std'] for d in history_data]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(iterations, means, 'b-o', linewidth=2, markersize=6)
    plt.title('Feature Mean Value Trend')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(iterations, stds, 'r-o', linewidth=2, markersize=6)
    plt.title('Feature Standard Deviation Trend')
    plt.xlabel('Iteration')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    trend_file = os.path.join(output_dir, "feature_analysis", "feature_trends.png")
    plt.savefig(trend_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_64d_feature_analysis(features, analysis_dir, iteration):
    """
    专门针对64维特征的分析
    """
    # 1. 64维特征的分布热力图
    plt.figure(figsize=(16, 8))
    
    # 为了可视化，我们取前1000个点（如果数据太多）
    if features.shape[0] > 1000:
        sample_indices = np.random.choice(features.shape[0], 1000, replace=False)
        features_sample = features[sample_indices]
    else:
        features_sample = features
    
    # 创建热力图
    plt.subplot(1, 2, 1)
    im = plt.imshow(features_sample.T, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.title(f'64D Feature Values Heatmap (Iteration {iteration})')
    plt.xlabel('Gaussian Point Index')
    plt.ylabel('Feature Dimension (0-63)')
    
    # 2. 每个维度的方差分布
    plt.subplot(1, 2, 2)
    feature_vars = np.var(features, axis=0)
    plt.bar(range(64), feature_vars, alpha=0.7, color='coral')
    plt.title(f'Feature Variance by Dimension (Iteration {iteration})')
    plt.xlabel('Feature Dimension (0-63)')
    plt.ylabel('Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    analysis_file = os.path.join(analysis_dir, f"64d_feature_analysis_iter_{iteration}.png")
    plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 保存64维特征的详细统计信息
    detailed_stats_file = os.path.join(analysis_dir, f"64d_detailed_stats_iter_{iteration}.txt")
    with open(detailed_stats_file, 'w') as f:
        f.write(f"64D Feature Detailed Statistics - Iteration {iteration}\n")
        f.write("=" * 60 + "\n")
        
        # 每个维度的统计信息
        for i in range(64):
            dim_data = features[:, i]
            f.write(f"Dimension {i:2d}: mean={dim_data.mean():.6f}, "
                   f"std={dim_data.std():.6f}, "
                   f"min={dim_data.min():.6f}, "
                   f"max={dim_data.max():.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"Most variable dimension: {np.argmax(np.var(features, axis=0))}\n")
        f.write(f"Least variable dimension: {np.argmin(np.var(features, axis=0))}\n")
        f.write(f"Average variance: {np.mean(np.var(features, axis=0)):.6f}\n")
        f.write(f"Variance of variances: {np.var(np.var(features, axis=0)):.6f}\n") 