"""
深度学习配置模块
用于配置深度学习模型、训练参数和推理设置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
from pathlib import Path


@dataclass
class DeepLearningConfig:
    """深度学习配置类"""
    
    # 启用/禁用
    enabled: bool = False
    
    # 模型配置
    model_path: str = "models/best_grasp_model.pth"
    model_type: str = "GripControlNet"  # 或 "AdaptiveGraspingNetwork", "TactileCNN"
    
    # 推理配置
    inference_interval: int = 5  # 每多少帧进行一次推理
    buffer_size: int = 10
    use_gpu: bool = True
    max_inference_time_ms: float = 100.0  # 最大允许推理时间（毫秒）
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # 在线学习配置
    online_learning_enabled: bool = True
    online_learning_buffer_size: int = 1000
    online_batch_size: int = 16
    online_learning_rate: float = 1e-5
    online_learning_interval: int = 50  # 每多少次推理进行一次在线学习
    
    # PID控制配置
    adaptive_pid_enabled: bool = True
    base_pid_parameters: Dict[str, float] = field(default_factory=lambda: {
        'position_kp': 1.0,
        'position_ki': 0.1,
        'position_kd': 0.05,
        'force_kp': 2.0,
        'force_ki': 0.2,
        'force_kd': 0.1
    })
    
    # 物体类别配置
    object_classes: List[str] = field(default_factory=lambda: [
        'hard_plastic',
        'soft_rubber',
        'fragile_glass',
        'metal',
        'fabric'
    ])
    
    shape_classes: List[str] = field(default_factory=lambda: [
        'sphere',
        'cube',
        'cylinder',
        'irregular'
    ])
    
    # 特征提取配置
    tactile_feature_dim: int = 16
    use_manual_features: bool = True
    use_convolutional_features: bool = True
    
    # 性能监控
    log_inference_time: bool = True
    save_inference_history: bool = True
    inference_history_size: int = 100
    
    # 数据收集配置
    collect_training_data: bool = True
    training_data_dir: str = "data/training"
    auto_save_interval: int = 100  # 每收集多少样本自动保存一次
    
    # 阈值配置
    slip_detection_threshold: float = 0.3
    stable_grasp_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # 嵌入式配置
    use_quantized_model: bool = False
    quantized_model_path: str = "models/quantized_model.pth"
    max_memory_usage_mb: int = 50
    max_model_size_mb: int = 10
    
    # 安全限制
    max_force_adjustment: float = 0.3  # 最大力度调整比例
    min_position: float = 0.0
    max_position: float = 1.0

    def __post_init__(self):
        """初始化后处理"""
        # 确保 enabled 属性存在
        if not hasattr(self, 'enabled'):
            self.enabled = False
        
        # 确保其他必要属性存在
        if not hasattr(self, 'model_path'):
            self.model_path = "models/best_grasp_model.pth"
        
        if not hasattr(self, 'inference_interval'):
            self.inference_interval = 5
            
    def validate(self) -> List[str]:
        """验证配置有效性"""
        errors = []
        
        # 检查模型路径
        if self.enabled and self.model_path and not os.path.exists(self.model_path):
            errors.append(f"模型文件不存在: {self.model_path}")
        
        # 检查推理间隔
        if self.inference_interval <= 0:
            errors.append(f"推理间隔必须大于0: {self.inference_interval}")
        
        # 检查学习率
        if self.learning_rate <= 0:
            errors.append(f"学习率必须大于0: {self.learning_rate}")
        
        # 检查阈值
        if not (0 <= self.confidence_threshold <= 1):
            errors.append(f"置信度阈值必须在0-1之间: {self.confidence_threshold}")
        
        # 检查安全限制
        if self.max_force_adjustment < 0 or self.max_force_adjustment > 1:
            errors.append(f"最大力度调整比例必须在0-1之间: {self.max_force_adjustment}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'inference_interval': self.inference_interval,
            'buffer_size': self.buffer_size,
            'use_gpu': self.use_gpu,
            'max_inference_time_ms': self.max_inference_time_ms,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'validation_split': self.validation_split,
            'online_learning_enabled': self.online_learning_enabled,
            'online_learning_buffer_size': self.online_learning_buffer_size,
            'online_batch_size': self.online_batch_size,
            'online_learning_rate': self.online_learning_rate,
            'online_learning_interval': self.online_learning_interval,
            'adaptive_pid_enabled': self.adaptive_pid_enabled,
            'base_pid_parameters': self.base_pid_parameters,
            'object_classes': self.object_classes,
            'shape_classes': self.shape_classes,
            'tactile_feature_dim': self.tactile_feature_dim,
            'use_manual_features': self.use_manual_features,
            'use_convolutional_features': self.use_convolutional_features,
            'log_inference_time': self.log_inference_time,
            'save_inference_history': self.save_inference_history,
            'inference_history_size': self.inference_history_size,
            'collect_training_data': self.collect_training_data,
            'training_data_dir': self.training_data_dir,
            'auto_save_interval': self.auto_save_interval,
            'slip_detection_threshold': self.slip_detection_threshold,
            'stable_grasp_threshold': self.stable_grasp_threshold,
            'confidence_threshold': self.confidence_threshold,
            'use_quantized_model': self.use_quantized_model,
            'quantized_model_path': self.quantized_model_path,
            'max_memory_usage_mb': self.max_memory_usage_mb,
            'max_model_size_mb': self.max_model_size_mb,
            'max_force_adjustment': self.max_force_adjustment,
            'min_position': self.min_position,
            'max_position': self.max_position
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeepLearningConfig':
        """从字典创建配置"""
        # 处理嵌套字典
        base_pid_params = config_dict.pop('base_pid_parameters', {})
        obj_classes = config_dict.pop('object_classes', [])
        shape_classes = config_dict.pop('shape_classes', [])
        
        config = cls(**config_dict)
        config.base_pid_parameters = base_pid_params
        config.object_classes = obj_classes
        config.shape_classes = shape_classes
        
        return config
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        model_info = {
            'type': self.model_type,
            'path': self.model_path,
            'exists': os.path.exists(self.model_path) if self.model_path else False,
            'quantized': self.use_quantized_model,
            'quantized_path': self.quantized_model_path if self.use_quantized_model else None
        }
        
        # 尝试获取文件大小
        if self.model_path and os.path.exists(self.model_path):
            try:
                size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                model_info['size_mb'] = round(size_mb, 2)
            except:
                model_info['size_mb'] = None
        
        return model_info


def create_default_dl_config() -> DeepLearningConfig:
    """创建默认的深度学习配置"""
    return DeepLearningConfig()


def create_high_precision_dl_config() -> DeepLearningConfig:
    """创建高精度深度学习配置"""
    config = DeepLearningConfig()
    config.enabled = True
    config.inference_interval = 3  # 更频繁的推理
    config.buffer_size = 15  # 更大的缓冲区
    config.confidence_threshold = 0.8  # 更高的置信度阈值
    config.use_manual_features = True
    config.use_convolutional_features = True
    config.tactile_feature_dim = 32  # 更大的特征维度
    return config


def create_fast_inference_dl_config() -> DeepLearningConfig:
    """创建快速推理深度学习配置"""
    config = DeepLearningConfig()
    config.enabled = True
    config.inference_interval = 10  # 更少的推理
    config.buffer_size = 5  # 更小的缓冲区
    config.use_quantized_model = True  # 使用量化模型
    config.use_manual_features = True
    config.use_convolutional_features = False  # 不使用计算密集的卷积特征
    config.tactile_feature_dim = 8  # 更小的特征维度
    config.max_inference_time_ms = 20.0  # 更严格的推理时间限制
    return config


def create_embedded_dl_config() -> DeepLearningConfig:
    """创建嵌入式设备深度学习配置"""
    config = DeepLearningConfig()
    config.enabled = True
    config.model_type = "EdgeInference"  # 使用专门为嵌入式优化的模型
    config.use_quantized_model = True
    config.inference_interval = 20  # 非常少的推理
    config.buffer_size = 3  # 非常小的缓冲区
    config.use_gpu = False  # 嵌入式设备通常没有GPU
    config.use_manual_features = True
    config.use_convolutional_features = False
    config.tactile_feature_dim = 4
    config.max_memory_usage_mb = 10
    config.max_model_size_mb = 2
    config.max_inference_time_ms = 5.0  # 非常严格的推理时间限制
    return config


def merge_dl_configs(base_config: DeepLearningConfig, 
                    override_config: Dict[str, Any]) -> DeepLearningConfig:
    """
    合并深度学习配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置字典
        
    Returns:
        合并后的配置
    """
    import copy
    
    # 创建深度拷贝
    merged = copy.deepcopy(base_config)
    
    # 更新配置
    for key, value in override_config.items():
        if hasattr(merged, key):
            setattr(merged, key, value)
        else:
            print(f"警告: DeepLearningConfig没有属性 '{key}'，跳过")
    
    return merged


# 配置模板映射
DL_CONFIG_TEMPLATES = {
    'default': create_default_dl_config,
    'high_precision': create_high_precision_dl_config,
    'fast_inference': create_fast_inference_dl_config,
    'embedded': create_embedded_dl_config
}


def get_dl_config_template(template_name: str = 'default') -> DeepLearningConfig:
    """
    获取深度学习配置模板
    
    Args:
        template_name: 模板名称
        
    Returns:
        配置实例
        
    Raises:
        ValueError: 模板名称无效时抛出
    """
    if template_name not in DL_CONFIG_TEMPLATES:
        available = list(DL_CONFIG_TEMPLATES.keys())
        raise ValueError(f"未知的深度学习配置模板: {template_name}，可用模板: {available}")
    
    return DL_CONFIG_TEMPLATES[template_name]()