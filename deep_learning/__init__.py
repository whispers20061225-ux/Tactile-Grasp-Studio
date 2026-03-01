"""
深度学习模块入口
"""

from .models import *
from .grasp_predictor import GraspPredictor, GraspDataset
from .reinforcement_learning import (
    RLEnvWrapper, PPOAgent, RLTrainer, 
    ActorNetwork, CriticNetwork, ReplayBuffer
)
from .trainer import ModelTrainer, MultiModelTrainer, create_dataloader, get_default_config
from .inference import InferenceEngine
from .data_loader import (
    DataConfig, TactileDataset, ForceControlDataset, 
    MultiModalDataset, SlipDetectionDataset, DataLoaderFactory
)

__version__ = "1.0.0"
__author__ = "Robotics AI Team"

__all__ = [
    # 模型
    'GraspPredictor',
    'TactileCNN',
    'VisualEncoder',
    'ForceControlNN',
    'AttentionFusion',
    'SlipDetector',
    'ContactStateClassifier',
    'GraspStabilityPredictor',
    
    # 强化学习
    'RLEnvWrapper',
    'PPOAgent',
    'RLTrainer',
    'ActorNetwork',
    'CriticNetwork',
    'ReplayBuffer',
    
    # 训练
    'ModelTrainer',
    'MultiModelTrainer',
    'create_dataloader',
    'get_default_config',
    
    # 推理
    'InferenceEngine',
    
    # 数据
    'GraspDataset',
    'TactileDataset',
    'ForceControlDataset',
    'MultiModalDataset',
    'SlipDetectionDataset',
    'DataConfig',
    'DataLoaderFactory',
    
    # 工具
    'DataAugmentation'
]

# 配置默认设备
import torch
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device() -> torch.device:
    """获取默认设备"""
    return DEFAULT_DEVICE

def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False