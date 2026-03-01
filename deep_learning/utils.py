import numpy as np
import torch
import torch.nn as nn
from scipy import signal
import matplotlib.pyplot as plt

def preprocess_tactile_data(data, filter_type='median', window_size=3):
    """预处理触觉数据"""
    if filter_type == 'median':
        filtered = signal.medfilt(data, kernel_size=window_size)
    elif filter_type == 'gaussian':
        filtered = signal.gaussian_filter(data, sigma=1)
    else:
        filtered = data
    
    # 归一化
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    return filtered

def extract_spatial_features(tactile_matrix):
    """提取空间特征"""
    features = {}
    
    # 基本统计特征
    features['mean'] = np.mean(tactile_matrix)
    features['std'] = np.std(tactile_matrix)
    features['max'] = np.max(tactile_matrix)
    features['min'] = np.min(tactile_matrix)
    
    # 梯度特征
    grad_x, grad_y = np.gradient(tactile_matrix)
    features['grad_magnitude'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    features['grad_direction'] = np.arctan2(np.mean(grad_y), np.mean(grad_x))
    
    # 纹理特征
    from skimage.feature import greycomatrix, greycoprops
    if len(np.unique(tactile_matrix)) > 1:
        glcm = greycomatrix(
            (tactile_matrix * 255).astype(np.uint8),
            distances=[1],
            angles=[0],
            symmetric=True,
            normed=True
        )
        features['contrast'] = greycoprops(glcm, 'contrast')[0, 0]
        features['homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
    
    return features

def visualize_tactile_data(tactile_data, title="Tactile Sensor Data"):
    """可视化触觉数据"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3x3网格图
    im = axes[0].imshow(tactile_data, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Pressure Distribution")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im, ax=axes[0])
    
    # 添加数值标签
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{tactile_data[i, j]:.2f}",
                        ha="center", va="center", color="w")
    
    # 3D表面图
    x = np.arange(3)
    y = np.arange(3)
    X, Y = np.meshgrid(x, y)
    
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(X, Y, tactile_data, cmap='viridis')
    ax3d.set_title("3D Pressure Surface")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Pressure")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def calculate_grasp_metrics(pressure_data, target_force=None):
    """计算抓取指标"""
    metrics = {}
    
    # 总压力
    metrics['total_force'] = np.sum(pressure_data)
    
    # 压力均匀性
    normalized = pressure_data / (metrics['total_force'] + 1e-8)
    entropy = -np.sum(normalized * np.log(normalized + 1e-8))
    metrics['uniformity'] = entropy / np.log(pressure_data.size)
    
    # 压力中心
    height, width = pressure_data.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    metrics['cop_x'] = np.sum(x_coords * pressure_data) / metrics['total_force']
    metrics['cop_y'] = np.sum(y_coords * pressure_data) / metrics['total_force']
    
    # 相对于中心的偏差
    center_x, center_y = width/2, height/2
    metrics['cop_deviation'] = np.sqrt(
        (metrics['cop_x'] - center_x)**2 + 
        (metrics['cop_y'] - center_y)**2
    )
    
    # 抓取稳定性得分
    metrics['stability_score'] = (
        0.4 * metrics['uniformity'] +
        0.3 * (1.0 - min(metrics['cop_deviation'] / 2.0, 1.0)) +
        0.3 * min(metrics['total_force'] / (target_force + 1e-8 if target_force else 1.0), 1.0)
    )
    
    return metrics

def save_model_for_embedded(model, input_shape, save_path):
    """保存模型供嵌入式设备使用"""
    # 转换为ONNX格式
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=11
    )
    
    # 量化模型（可选）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    # 保存量化模型
    torch.save(quantized_model.state_dict(), save_path.replace('.onnx', '_quantized.pth'))
    
    print(f"模型已保存到: {save_path}")
    print(f"量化模型已保存到: {save_path.replace('.onnx', '_quantized.pth')}")