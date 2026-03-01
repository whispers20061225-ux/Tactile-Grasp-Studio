"""
数据加载器模块
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import h5py
import pickle
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .grasp_predictor import GraspDataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42


class BaseDataset(Dataset, ABC):
    """基础数据集类"""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self._load_samples()
    
    @abstractmethod
    def _load_samples(self):
        """加载数据样本"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.samples:
            return {}
        
        return {
            'num_samples': len(self.samples),
            'data_dir': str(self.data_dir)
        }


class TactileDataset(BaseDataset):
    """触觉数据集"""
    
    def __init__(self, data_dir: str, transform=None, seq_length: int = 10):
        self.seq_length = seq_length
        super().__init__(data_dir, transform)
    
    def _load_samples(self):
        """加载触觉数据样本"""
        # 查找所有数据文件
        data_files = list(self.data_dir.glob("*.npz")) + list(self.data_dir.glob("*.h5"))
        
        for file_path in data_files:
            try:
                if file_path.suffix == '.npz':
                    with np.load(file_path, allow_pickle=True) as data:
                        if 'tactile' in data:
                            self.samples.append({
                                'file_path': str(file_path),
                                'tactile_data': data['tactile'],
                                'label': data.get('label', 0),
                                'timestamp': data.get('timestamp', 0)
                            })
                elif file_path.suffix == '.h5':
                    with h5py.File(file_path, 'r') as f:
                        if 'tactile' in f:
                            tactile_data = f['tactile'][:]
                            self.samples.append({
                                'file_path': str(file_path),
                                'tactile_data': tactile_data,
                                'label': f.attrs.get('label', 0),
                                'timestamp': f.attrs.get('timestamp', 0)
                            })
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"触觉数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载数据
        if Path(sample['file_path']).suffix == '.npz':
            with np.load(sample['file_path'], allow_pickle=True) as data:
                tactile_data = data['tactile'].astype(np.float32)
        else:
            with h5py.File(sample['file_path'], 'r') as f:
                tactile_data = f['tactile'][:].astype(np.float32)
        
        # 预处理
        if len(tactile_data.shape) == 1:
            tactile_data = tactile_data.reshape(1, -1)
        
        # 转换为时序数据（如果需要）
        if self.seq_length > 1 and tactile_data.shape[0] >= self.seq_length:
            # 随机选取一个序列段
            start_idx = random.randint(0, tactile_data.shape[0] - self.seq_length)
            tactile_seq = tactile_data[start_idx:start_idx + self.seq_length]
        else:
            # 重复当前帧或填充
            tactile_seq = np.repeat(tactile_data[np.newaxis, :], self.seq_length, axis=0)
        
        # 转换为张量
        tactile_tensor = torch.FloatTensor(tactile_seq)
        label_tensor = torch.LongTensor([sample['label']])
        
        return {
            'tactile': tactile_tensor,
            'label': label_tensor
        }


class ForceControlDataset(BaseDataset):
    """力控数据集"""
    
    def __init__(self, data_dir: str, transform=None, state_dim: int = 100):
        self.state_dim = state_dim
        super().__init__(data_dir, transform)
    
    def _load_samples(self):
        """加载力控数据样本"""
        data_files = list(self.data_dir.glob("*.npz"))
        
        for file_path in data_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if 'state' in data and 'force' in data:
                        self.samples.append({
                            'file_path': str(file_path),
                            'state': data['state'],
                            'force': data['force'],
                            'stiffness': data.get('stiffness', np.zeros(6)),
                            'success': data.get('success', 1)
                        })
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"力控数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        with np.load(sample['file_path'], allow_pickle=True) as data:
            state = data['state'].astype(np.float32)
            force = data['force'].astype(np.float32)
            stiffness = data.get('stiffness', np.zeros(6)).astype(np.float32)
            success = data.get('success', 1).astype(np.float32)
        
        # 确保状态维度正确
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return {
            'state': torch.FloatTensor(state),
            'force_target': torch.FloatTensor(force),
            'stiffness_target': torch.FloatTensor(stiffness),
            'success': torch.FloatTensor([success])
        }


class MultiModalDataset(BaseDataset):
    """多模态数据集（视觉+触觉+机械臂状态）"""
    
    def __init__(self, data_dir: str, transform=None, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        super().__init__(data_dir, transform)
    
    def _load_samples(self):
        """加载多模态数据样本"""
        data_files = list(self.data_dir.glob("*.npz"))
        
        for file_path in data_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    required_fields = ['visual', 'tactile', 'arm_state', 'grasp_label']
                    if all(field in data for field in required_fields):
                        self.samples.append({
                            'file_path': str(file_path),
                            'visual_shape': data['visual'].shape,
                            'tactile_shape': data['tactile'].shape,
                            'timestamp': data.get('timestamp', 0),
                            'success': data.get('success', 1)
                        })
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"多模态数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        with np.load(sample['file_path'], allow_pickle=True) as data:
            # 视觉数据
            visual_data = data['visual'].astype(np.float32) / 255.0
            if visual_data.shape[-1] != 3:  # 确保RGB
                visual_data = np.transpose(visual_data, (1, 2, 0))
            
            # 调整图像大小
            if visual_data.shape[:2] != self.image_size:
                from PIL import Image
                pil_image = Image.fromarray((visual_data * 255).astype(np.uint8))
                pil_image = pil_image.resize(self.image_size[::-1])  # (W, H)
                visual_data = np.array(pil_image).astype(np.float32) / 255.0
            
            # 触觉数据
            tactile_data = data['tactile'].astype(np.float32)
            
            # 机械臂状态
            arm_state = data['arm_state'].astype(np.float32)
            
            # 抓取标签
            grasp_label = data['grasp_label'].astype(np.float32)
            
            # 成功标签
            success = data.get('success', 1).astype(np.float32)
        
        # 数据增强
        if self.transform:
            visual_data = self.transform(visual_data)
        
        # 转换为张量
        visual_tensor = torch.FloatTensor(visual_data).permute(2, 0, 1)  # [C, H, W]
        tactile_tensor = torch.FloatTensor(tactile_data)
        arm_state_tensor = torch.FloatTensor(arm_state)
        grasp_label_tensor = torch.FloatTensor(grasp_label)
        success_tensor = torch.FloatTensor([success])
        
        result = {
            'visual': visual_tensor,
            'tactile': tactile_tensor,
            'arm_state': arm_state_tensor,
            'grasp_label': grasp_label_tensor,
            'success': success_tensor
        }
        
        # 可选的力标签
        if 'force_label' in data:
            force_label = data['force_label'].astype(np.float32)
            result['force_label'] = torch.FloatTensor(force_label)
        
        return result


class SlipDetectionDataset(BaseDataset):
    """滑动检测数据集"""
    
    def __init__(self, data_dir: str, transform=None, seq_length: int = 20):
        self.seq_length = seq_length
        super().__init__(data_dir, transform)
    
    def _load_samples(self):
        """加载滑动检测数据"""
        data_files = list(self.data_dir.glob("*.npz"))
        
        for file_path in data_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if 'tactile_seq' in data and 'slip_label' in data:
                        seq_data = data['tactile_seq']
                        if len(seq_data) >= self.seq_length:
                            self.samples.append({
                                'file_path': str(file_path),
                                'seq_length': len(seq_data),
                                'slip_label': data['slip_label'],
                                'slip_direction': data.get('slip_direction', np.zeros(3))
                            })
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"滑动检测数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        with np.load(sample['file_path'], allow_pickle=True) as data:
            tactile_seq = data['tactile_seq'].astype(np.float32)
            slip_label = data['slip_label'].astype(np.float32)
            slip_direction = data.get('slip_direction', np.zeros(3)).astype(np.float32)
        
        # 确保序列长度
        if len(tactile_seq) > self.seq_length:
            # 随机裁剪
            start_idx = random.randint(0, len(tactile_seq) - self.seq_length)
            tactile_seq = tactile_seq[start_idx:start_idx + self.seq_length]
        elif len(tactile_seq) < self.seq_length:
            # 填充
            pad_len = self.seq_length - len(tactile_seq)
            tactile_seq = np.pad(tactile_seq, ((0, pad_len), (0, 0)), mode='constant')
        
        return {
            'tactile_seq': torch.FloatTensor(tactile_seq),
            'slip_label': torch.FloatTensor([slip_label]),
            'direction_label': torch.FloatTensor(slip_direction)
        }


class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def create_dataloader(dataset_type: str, data_dir: str, config: DataConfig, 
                         split: str = 'train') -> DataLoader:
        """创建数据加载器"""
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # 创建数据集
        if dataset_type == 'grasp':
            dataset = GraspDataset(data_dir, split=split)
        elif dataset_type == 'tactile':
            dataset = TactileDataset(data_dir, seq_length=10)
        elif dataset_type == 'force_control':
            dataset = ForceControlDataset(data_dir)
        elif dataset_type == 'multi_modal':
            dataset = MultiModalDataset(data_dir)
        elif dataset_type == 'slip_detection':
            dataset = SlipDetectionDataset(data_dir)
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")
        
        # 数据集划分
        if split == 'train':
            # 训练集使用完整数据
            pass
        elif split == 'val':
            # 验证集（如果需要从完整数据集中划分）
            pass
        elif split == 'test':
            # 测试集
            pass
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if split == 'train' else False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False
        )
        
        logger.info(f"创建数据加载器: 类型={dataset_type}, 划分={split}, "
                   f"批次大小={config.batch_size}, 样本数={len(dataset)}")
        
        return dataloader
    
    @staticmethod
    def split_dataset(dataset: Dataset, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15,
                      seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """划分数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        return random_split(dataset, [train_size, val_size, test_size], 
                           generator=torch.Generator().manual_seed(seed))


class DataAugmentation:
    """数据增强类"""
    
    @staticmethod
    def augment_visual(image: np.ndarray) -> np.ndarray:
        """视觉数据增强"""
        import cv2
        
        augmented = image.copy()
        
        # 随机水平翻转
        if random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # 随机亮度调整
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness, 0, 255)
        
        # 随机对比度调整
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * contrast + mean, 0, 255)
        
        return augmented
    
    @staticmethod
    def augment_tactile(tactile_data: np.ndarray) -> np.ndarray:
        """触觉数据增强"""
        augmented = tactile_data.copy()
        
        # 随机噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.05, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        # 随机缩放
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            augmented = np.clip(augmented * scale, 0, 1)
        
        # 随机偏移
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)
            augmented = np.clip(augmented + shift, 0, 1)
        
        return augmented
    
    @staticmethod
    def augment_pose(pose: np.ndarray) -> np.ndarray:
        """位姿数据增强"""
        augmented = pose.copy()
        
        # 位置噪声
        if len(augmented) >= 3:
            noise_pos = np.random.normal(0, 0.01, 3)
            augmented[:3] += noise_pos
        
        # 姿态噪声（四元数）
        if len(augmented) >= 7:
            noise_quat = np.random.normal(0, 0.05, 4)
            noise_quat = noise_quat / np.linalg.norm(noise_quat)
            # 四元数乘法（简化）
            w1, x1, y1, z1 = augmented[3:7]
            w2, x2, y2, z2 = noise_quat
            augmented[3:7] = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            # 归一化
            augmented[3:7] = augmented[3:7] / np.linalg.norm(augmented[3:7])
        
        return augmented