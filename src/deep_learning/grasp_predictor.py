import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MultiModalAttention(nn.Module):
    """多模态注意力融合模块"""
    
    def __init__(self, visual_dim: int, tactile_dim: int, arm_dim: int, 
                 hidden_dim: int = 256):
        super(MultiModalAttention, self).__init__()
        
        # 模态特定的投影层
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.arm_proj = nn.Linear(arm_dim, hidden_dim)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, visual_feat: torch.Tensor, 
                tactile_feat: torch.Tensor, 
                arm_feat: torch.Tensor) -> torch.Tensor:
        # 投影到相同维度
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1)  # [B, 1, D]
        tactile_proj = self.tactile_proj(tactile_feat).unsqueeze(1)  # [B, 1, D]
        arm_proj = self.arm_proj(arm_feat).unsqueeze(1)  # [B, 1, D]
        
        # 拼接所有模态特征
        combined = torch.cat([visual_proj, tactile_proj, arm_proj], dim=1)  # [B, 3, D]
        
        # 自注意力
        attended, _ = self.attention(combined, combined, combined)
        attended = self.norm1(combined + attended)  # 残差连接
        
        # 前馈网络
        ffn_out = self.ffn(attended)
        output = self.norm2(attended + ffn_out)  # 残差连接
        
        # 全局平均池化
        output = output.mean(dim=1)  # [B, D]
        
        return output


class GraspPredictor(nn.Module):
    """
    基于多模态感知的抓取预测模型
    融合视觉、触觉和机械臂状态信息进行抓取预测
    """
    
    def __init__(self, config: Dict):
        super(GraspPredictor, self).__init__()
        
        # 配置参数
        self.config = config
        
        # 视觉特征提取器
        self.visual_encoder = self._build_visual_encoder()
        visual_dim = 2048  # ResNet50的特征维度
        
        # 触觉特征提取器
        self.tactile_encoder = self._build_tactile_encoder()
        tactile_dim = 128
        
        # 机械臂状态编码器
        self.arm_state_encoder = self._build_arm_state_encoder()
        arm_dim = 128
        
        # 多模态融合
        fusion_dim = config.get('fusion_dim', 512)
        self.attention_fusion = MultiModalAttention(
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            arm_dim=arm_dim,
            hidden_dim=fusion_dim
        )
        
        # 抓取位姿预测头
        self.grasp_pose_head = self._build_grasp_pose_head(fusion_dim)
        
        # 抓取质量评估头
        self.grasp_quality_head = self._build_grasp_quality_head(fusion_dim)
        
        # 力控预测头
        self.force_prediction_head = self._build_force_prediction_head(fusion_dim)
        
        # 不确定性估计
        self.uncertainty_estimator = self._build_uncertainty_estimator(fusion_dim)
    
    def _build_visual_encoder(self) -> nn.Module:
        """构建视觉特征提取网络"""
        # 使用预训练的ResNet作为骨干网络
        backbone = models.resnet50(pretrained=True)
        
        # 移除最后的全连接层和平均池化层
        modules = list(backbone.children())[:-2]
        visual_encoder = nn.Sequential(*modules)
        
        # 添加自适应池化层
        visual_encoder.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        # 冻结部分层进行微调
        for name, param in visual_encoder.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        return visual_encoder
    
    def _build_tactile_encoder(self) -> nn.Module:
        """构建触觉特征提取网络"""
        # PAXINI Gen3 M2020传感器: 12x16 = 192像素点
        input_channels = 192
        
        encoder = nn.Sequential(
            # 调整输入形状 [B, 192] -> [B, 1, 192]
            nn.Unflatten(1, (1, input_channels)),
            
            # 第一卷积层
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第二卷积层
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三卷积层
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 全局池化
            nn.AdaptiveAvgPool1d(1),
            
            # 展平
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(128)
        )
        
        return encoder
    
    def _build_arm_state_encoder(self) -> nn.Module:
        """构建机械臂状态编码网络"""
        # 机械臂状态: 关节角度(6), 末端位姿(6), 速度(6), 力/力矩(6) = 24
        input_dim = 24
        
        encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        return encoder
    
    def _build_grasp_pose_head(self, input_dim: int) -> nn.Module:
        """构建抓取位姿预测头"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 7)  # 位置(3) + 四元数(4) 或 欧拉角(3) + 开合度(1)
        )
    
    def _build_grasp_quality_head(self, input_dim: int) -> nn.Module:
        """构建抓取质量评估头"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            
            nn.Linear(64, 3),  # 成功概率, 稳定性, 安全评分
            
            # 输出层使用不同的激活函数
            nn.ModuleList([
                nn.Sigmoid(),  # 成功概率 [0, 1]
                nn.Sigmoid(),  # 稳定性评分 [0, 1]
                nn.Sigmoid()   # 安全评分 [0, 1]
            ])
        )
    
    def _build_force_prediction_head(self, input_dim: int) -> nn.Module:
        """构建力控预测头"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            
            nn.Linear(64, 6),  # 抓取力(3) + 滑动检测(1) + 接触分布(2)
            
            # 输出层
            nn.ModuleList([
                nn.Sigmoid(),  # 抓取力 [0, 1]
                nn.Sigmoid(),  # 抓取力 [0, 1]
                nn.Sigmoid(),  # 抓取力 [0, 1]
                nn.Sigmoid(),  # 滑动概率 [0, 1]
                nn.Sigmoid(),  # 接触平衡x [0, 1]
                nn.Sigmoid()   # 接触平衡y [0, 1]
            ])
        )
    
    def _build_uncertainty_estimator(self, input_dim: int) -> nn.Module:
        """构建不确定性估计模块"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # 输出正值的不确定性
        )
    
    def forward(self, visual_input: torch.Tensor, 
                tactile_input: torch.Tensor,
                arm_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_input: 视觉输入 [B, 3, H, W]
            tactile_input: 触觉输入 [B, 192]
            arm_state: 机械臂状态 [B, 24]
        
        Returns:
            包含预测结果的字典
        """
        batch_size = visual_input.size(0)
        
        # 视觉特征提取
        visual_features = self.visual_encoder(visual_input)
        visual_features = visual_features.view(batch_size, -1)  # [B, 2048]
        
        # 触觉特征提取
        tactile_features = self.tactile_encoder(tactile_input)  # [B, 128]
        
        # 机械臂状态编码
        arm_features = self.arm_state_encoder(arm_state)  # [B, 128]
        
        # 多模态融合
        fused_features = self.attention_fusion(
            visual_features, 
            tactile_features, 
            arm_features
        )  # [B, fusion_dim]
        
        # 抓取位姿预测
        grasp_pose_raw = self.grasp_pose_head(fused_features)
        
        # 分解位姿输出
        position = torch.tanh(grasp_pose_raw[:, :3])  # 归一化到[-1, 1]
        orientation = F.normalize(grasp_pose_raw[:, 3:7], dim=1)  # 四元数归一化
        gripper_width = torch.sigmoid(grasp_pose_raw[:, 7:])  # 开合度 [0, 1]
        grasp_pose = torch.cat([position, orientation, gripper_width], dim=1)
        
        # 抓取质量评估
        grasp_quality_raw = self.grasp_quality_head(fused_features)
        grasp_quality = torch.stack([
            grasp_quality_raw[0][0](grasp_quality_raw[0][-1]),  # 成功概率
            grasp_quality_raw[1][0](grasp_quality_raw[1][-1]),  # 稳定性
            grasp_quality_raw[2][0](grasp_quality_raw[2][-1])   # 安全性
        ], dim=1)
        
        # 力控预测
        force_pred_raw = self.force_prediction_head(fused_features)
        force_prediction = torch.stack([
            force_pred_raw[0][0](force_pred_raw[0][-1]),  # Fx
            force_pred_raw[1][0](force_pred_raw[1][-1]),  # Fy
            force_pred_raw[2][0](force_pred_raw[2][-1]),  # Fz
            force_pred_raw[3][0](force_pred_raw[3][-1]),  # 滑动概率
            force_pred_raw[4][0](force_pred_raw[4][-1]),  # 接触平衡x
            force_pred_raw[5][0](force_pred_raw[5][-1])   # 接触平衡y
        ], dim=1)
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator(fused_features)
        
        return {
            'grasp_pose': grasp_pose,        # [B, 8]
            'grasp_quality': grasp_quality,  # [B, 3]
            'force_prediction': force_prediction,  # [B, 6]
            'uncertainty': uncertainty,      # [B, 1]
            'fused_features': fused_features # [B, fusion_dim]
        }
    
    def predict(self, visual_input: np.ndarray, 
                tactile_input: np.ndarray,
                arm_state: np.ndarray,
                threshold: float = 0.7) -> Dict[str, Any]:
        """
        推理模式下的抓取预测
        
        Args:
            visual_input: RGB图像 [H, W, 3]，值范围[0, 255]
            tactile_input: 触觉数据 [192,]，值范围[0, 1]
            arm_state: 机械臂状态 [24,]，已归一化
            threshold: 抓取成功概率阈值
        
        Returns:
            抓取预测结果字典
        """
        self.eval()
        
        with torch.no_grad():
            # 转换为张量并归一化
            visual_tensor = torch.FloatTensor(visual_input).permute(2, 0, 1).unsqueeze(0) / 255.0
            tactile_tensor = torch.FloatTensor(tactile_input).unsqueeze(0)
            arm_tensor = torch.FloatTensor(arm_state).unsqueeze(0)
            
            # 移动到设备
            device = next(self.parameters()).device
            visual_tensor = visual_tensor.to(device)
            tactile_tensor = tactile_tensor.to(device)
            arm_tensor = arm_tensor.to(device)
            
            # 预测
            predictions = self(visual_tensor, tactile_tensor, arm_tensor)
            
            # 处理预测结果
            grasp_pose = predictions['grasp_pose'].cpu().numpy()[0]
            grasp_quality = predictions['grasp_quality'].cpu().numpy()[0]
            force_pred = predictions['force_prediction'].cpu().numpy()[0]
            uncertainty = predictions['uncertainty'].cpu().numpy()[0][0]
            
            # 判断是否执行抓取
            success_prob = grasp_quality[0]
            should_grasp = success_prob > threshold and uncertainty < 0.5
            
            result = {
                'grasp_pose': {
                    'position': grasp_pose[:3].tolist(),      # x, y, z [-1, 1]
                    'orientation': grasp_pose[3:7].tolist(),  # 四元数 [w, x, y, z]
                    'width': float(grasp_pose[7])            # 夹爪开合度 [0, 1]
                },
                'grasp_quality': {
                    'success_probability': float(grasp_quality[0]),
                    'stability_score': float(grasp_quality[1]),
                    'safety_score': float(grasp_quality[2])
                },
                'force_prediction': {
                    'grasp_force': force_pred[:3].tolist(),      # Fx, Fy, Fz [0, 1]
                    'slip_probability': float(force_pred[3]),
                    'contact_balance': force_pred[4:6].tolist()  # 接触分布
                },
                'uncertainty': float(uncertainty),
                'should_grasp': bool(should_grasp),
                'confidence': float(success_prob * (1 - uncertainty))
            }
            
            logger.info(f"抓取预测: 成功率={success_prob:.3f}, "
                       f"不确定性={uncertainty:.3f}, "
                       f"是否抓取={should_grasp}")
            
            return result
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到 {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info(f"模型已从 {path} 加载")
        return model


class GraspDataset(torch.utils.data.Dataset):
    """
    抓取数据集
    """
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 数据增强变换
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # 加载数据索引
        self.samples = self._load_samples()
        logger.info(f"加载 {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本索引"""
        samples = []
        
        # 检查数据目录
        if not self.data_dir.exists():
            logger.warning(f"数据目录不存在: {self.data_dir}")
            return samples
        
        # 查找所有样本文件
        sample_files = list(self.data_dir.glob("*.npz"))
        
        for file_path in sample_files:
            try:
                # 加载样本元数据
                with np.load(file_path, allow_pickle=True) as data:
                    sample_info = {
                        'file_path': str(file_path),
                        'visual_data': data['visual'],
                        'tactile_data': data['tactile'],
                        'arm_state': data['arm_state'],
                        'grasp_label': data['grasp_label'],
                        'success': bool(data['success']),
                        'timestamp': data.get('timestamp', 0)
                    }
                    samples.append(sample_info)
            except Exception as e:
                logger.error(f"加载样本 {file_path} 失败: {e}")
        
        # 按时间戳排序并划分数据集
        samples.sort(key=lambda x: x['timestamp'])
        total = len(samples)
        
        if self.split == 'train':
            samples = samples[:int(total * 0.7)]
        elif self.split == 'val':
            samples = samples[int(total * 0.7):int(total * 0.85)]
        elif self.split == 'test':
            samples = samples[int(total * 0.85):]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载数据
        with np.load(sample['file_path'], allow_pickle=True) as data:
            visual_data = data['visual'].astype(np.float32) / 255.0
            tactile_data = data['tactile'].astype(np.float32)
            arm_state = data['arm_state'].astype(np.float32)
            grasp_label = data['grasp_label'].astype(np.float32)
            success = data['success'].astype(np.float32)
        
        # 数据增强
        if self.transform and self.split == 'train':
            visual_data = self.transform(visual_data)
        
        # 转换为张量
        visual_tensor = torch.FloatTensor(visual_data).permute(2, 0, 1)  # [C, H, W]
        tactile_tensor = torch.FloatTensor(tactile_data)  # [192]
        arm_state_tensor = torch.FloatTensor(arm_state)  # [24]
        grasp_label_tensor = torch.FloatTensor(grasp_label)  # [8]
        success_tensor = torch.FloatTensor([success])  # [1]
        
        return {
            'visual': visual_tensor,
            'tactile': tactile_tensor,
            'arm_state': arm_state_tensor,
            'grasp_label': grasp_label_tensor,
            'success': success_tensor,
            'file_path': sample['file_path']
        }
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if not self.samples:
            return {}
        
        # 计算均值和标准差
        visual_data = []
        tactile_data = []
        arm_data = []
        
        for sample in self.samples[:100]:  # 使用前100个样本计算统计
            with np.load(sample['file_path'], allow_pickle=True) as data:
                visual_data.append(data['visual'].flatten())
                tactile_data.append(data['tactile'].flatten())
                arm_data.append(data['arm_state'].flatten())
        
        visual_mean = np.mean(np.concatenate(visual_data))
        visual_std = np.std(np.concatenate(visual_data))
        tactile_mean = np.mean(np.concatenate(tactile_data))
        tactile_std = np.std(np.concatenate(tactile_data))
        arm_mean = np.mean(np.concatenate(arm_data))
        arm_std = np.std(np.concatenate(arm_data))
        
        return {
            'visual': {'mean': visual_mean, 'std': visual_std},
            'tactile': {'mean': tactile_mean, 'std': tactile_std},
            'arm_state': {'mean': arm_mean, 'std': arm_std},
            'num_samples': len(self.samples),
            'success_rate': np.mean([s['success'] for s in self.samples])
        }