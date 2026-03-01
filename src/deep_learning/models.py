"""
深度学习模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, List, Dict, Optional


class TactileCNN(nn.Module):
    """
    触觉数据处理CNN
    专为PAXINI Gen3 M2020传感器设计 (192像素点)
    """
    
    def __init__(self, input_channels: int = 192, output_dim: int = 128):
        super(TactileCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 输入: [B, 1, 192]
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出: [B, 32, 96]
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出: [B, 64, 48]
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 输出: [B, 128, 1]
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # [B, 128]
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入形状正确
        if x.dim() == 2:  # [B, 192]
            x = x.unsqueeze(1)  # [B, 1, 192]
        
        features = self.conv_layers(x)
        output = self.fc_layers(features)
        return output


class VisualEncoder(nn.Module):
    """
    视觉编码器 - 基于ResNet的特征提取
    """
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super(VisualEncoder, self).__init__()
        
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # 冻结早期层
        self._freeze_layers()
    
    def _freeze_layers(self):
        """冻结早期卷积层"""
        for name, param in self.feature_extractor.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        projected = self.projection(features)
        return projected


class ForceControlNN(nn.Module):
    """
    力控神经网络
    预测最优抓取力和刚度
    """
    
    def __init__(self, state_dim: int = 100, hidden_dims: List[int] = None):
        super(ForceControlNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # 编码器网络
        encoder_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 力预测头
        self.force_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Fx, Fy, Fz
        )
        
        # 刚度预测头
        self.stiffness_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 刚度矩阵的对角元素
        )
        
        # 自适应增益预测
        self.gain_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # P, I, D增益
        )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(state)
        
        # 力预测 (归一化到[-1, 1])
        force = torch.tanh(self.force_head(encoded))
        
        # 刚度预测 (正数)
        stiffness = F.softplus(self.stiffness_head(encoded))
        
        # 增益预测 (正数)
        gains = F.softplus(self.gain_head(encoded))
        
        return {
            'force': force,          # [B, 3]
            'stiffness': stiffness,  # [B, 6]
            'gains': gains           # [B, 3]
        }


class AttentionFusion(nn.Module):
    """
    注意力融合模块
    用于融合多模态特征
    """
    
    def __init__(self, visual_dim: int, tactile_dim: int, arm_dim: int, 
                 hidden_dim: int = 256):
        super(AttentionFusion, self).__init__()
        
        # 模态投影层
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.arm_proj = nn.Linear(arm_dim, hidden_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
    
    def forward(self, visual_feat: torch.Tensor, 
                tactile_feat: torch.Tensor, 
                arm_feat: torch.Tensor) -> torch.Tensor:
        
        batch_size = visual_feat.size(0)
        
        # 投影到相同维度
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1)  # [B, 1, D]
        tactile_proj = self.tactile_proj(tactile_feat).unsqueeze(1)  # [B, 1, D]
        arm_proj = self.arm_proj(arm_feat).unsqueeze(1)  # [B, 1, D]
        
        # 拼接模态特征
        combined = torch.cat([visual_proj, tactile_proj, arm_proj], dim=1)  # [B, 3, D]
        
        # 自注意力
        attended, _ = self.attention(combined, combined, combined)
        attended = self.norm1(combined + attended)  # 残差连接
        
        # 前馈网络
        ffn_out = self.ffn(attended)
        attended = self.norm2(attended + ffn_out)  # 残差连接
        
        # 全局平均池化
        attended_pooled = attended.mean(dim=1)  # [B, D]
        
        # 与原始特征拼接
        original_combined = torch.cat([visual_feat, tactile_feat, arm_feat], dim=1)
        
        # 最终融合
        output = self.output_proj(original_combined) + attended_pooled
        
        return output


class SlipDetector(nn.Module):
    """
    滑动检测网络
    基于触觉时序数据检测滑动
    """
    
    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, 
                 lstm_layers: int = 2):
        super(SlipDetector, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 滑动概率
            nn.Sigmoid()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 滑动方向向量
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 时序触觉数据 [B, T, 192]
        
        Returns:
            滑动预测结果
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim*2]
        
        # 注意力
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步
        last_step = attended[:, -1, :]
        
        # 分类和回归
        slip_prob = self.classifier(last_step)
        slip_direction = torch.tanh(self.regressor(last_step))
        
        return {
            'slip_probability': slip_prob,
            'slip_direction': slip_direction
        }


class ContactStateClassifier(nn.Module):
    """
    接触状态分类器
    识别抓取接触类型
    """
    
    def __init__(self, input_dim: int = 192, num_classes: int = 5):
        super(ContactStateClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # 接触力回归器
        self.force_regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 接触力估计
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 192]
        
        features = self.feature_extractor(x)
        
        # 分类
        logits = self.classifier(features)
        class_probs = F.softmax(logits, dim=1)
        
        # 力估计
        force_estimate = F.softplus(self.force_regressor(features))
        
        return {
            'logits': logits,
            'class_probabilities': class_probs,
            'force_estimate': force_estimate,
            'predicted_class': torch.argmax(class_probs, dim=1)
        }


class GraspStabilityPredictor(nn.Module):
    """
    抓取稳定性预测器
    预测抓取的稳定性和抗干扰能力
    """
    
    def __init__(self, feature_dim: int = 256):
        super(GraspStabilityPredictor, self).__init__()
        
        self.stability_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 3),  # 稳定性得分、抗扭得分、抗拉得分
            nn.Sigmoid()
        )
        
        self.disturbance_response = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 对6种扰动的响应
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        stability_scores = self.stability_predictor(features)
        disturbance_response = torch.tanh(self.disturbance_response(features))
        
        return {
            'stability_scores': stability_scores,
            'disturbance_response': disturbance_response
        }


# 导出所有模型类
__all__ = [
    'TactileCNN',
    'VisualEncoder',
    'ForceControlNN',
    'AttentionFusion',
    'SlipDetector',
    'ContactStateClassifier',
    'GraspStabilityPredictor'
]