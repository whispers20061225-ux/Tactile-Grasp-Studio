"""
触觉数据处理模块 - 支持三维力数据 (Fx, Fy, Fz)
负责数据滤波、特征提取、接触检测、矢量场计算等
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import logging
import math
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessedTactileData:
    """处理后的触觉数据 - 支持三维力"""
    raw_data: Any  # 原始数据对象
    timestamp: float
    filtered_vectors: np.ndarray  # 滤波后的三维力向量 (N) - 形状: (num_taxels, 3)
    resultant_force: np.ndarray  # 合力向量 (N) - 形状: (3,)
    contact_mask: np.ndarray  # 接触掩码
    force_magnitudes: np.ndarray  # 每个触点的力大小 (N)
    force_directions: np.ndarray  # 每个触点的力方向 (单位向量)
    features: Dict[str, float]  # 提取的特征
    vector_field: np.ndarray  # 矢量场数据 (位置 + 力向量) - 用于可视化
    slip_detected: bool = False
    vibration_detected: bool = False
    contact_centroid: Optional[Tuple[float, float]] = None  # 接触质心
    shear_forces: np.ndarray = None  # 剪切力向量 (Fx, Fy)
    normal_forces: np.ndarray = None  # 法向力向量 (Fz)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.shear_forces is None and self.filtered_vectors is not None:
            self.shear_forces = self.filtered_vectors[:, :2]  # Fx, Fy
            self.normal_forces = self.filtered_vectors[:, 2]  # Fz
        
        if self.contact_centroid is None and np.any(self.contact_mask):
            self.contact_centroid = self._calculate_centroid()
    
    def _calculate_centroid(self) -> Tuple[float, float]:
        """计算接触质心"""
        if not np.any(self.contact_mask):
            return (0.0, 0.0)
        
        # 假设触点按3x3网格排列
        positions = []
        for i in range(3):
            for j in range(3):
                positions.append((j, i))  # (x, y) 坐标
        
        contact_positions = [positions[i] for i in range(len(self.contact_mask)) if self.contact_mask[i]]
        
        if not contact_positions:
            return (0.0, 0.0)
        
        centroid_x = sum(p[0] for p in contact_positions) / len(contact_positions)
        centroid_y = sum(p[1] for p in contact_positions) / len(contact_positions)
        
        return (centroid_x, centroid_y)


class DataProcessor:
    """
    三维触觉数据处理器
    负责数据滤波、特征提取、接触检测、矢量场计算等
    
    所有力值单位均为牛顿 (N)
    """
    
    def __init__(self, 
                 filter_cutoff: float = 20.0,  # 低通滤波截止频率 (Hz)
                 sampling_rate: float = 83.3,   # 采样率
                 contact_threshold: float = 0.5, # 接触检测阈值 (N) - 法向力
                 slip_threshold: float = 2.0,   # 滑移检测阈值 (N) - 剪切力变化
                 vibration_threshold: float = 0.3,  # 振动检测阈值
                 tactile_grid: Tuple[int, int] = (3, 3),  # 测点网格布局
                 max_force_range: float = 25.0,  # 最大力范围 (N) - 根据Paxini Gen3规格
                 shear_force_range: float = 10.0,  # 剪切力范围 (N)
                 enable_vector_field: bool = True):  # 启用矢量场计算
        
        # 滤波器参数
        self.filter_cutoff = filter_cutoff
        self.sampling_rate = sampling_rate
        
        # 检测阈值
        self.contact_threshold = contact_threshold
        self.slip_threshold = slip_threshold
        self.vibration_threshold = vibration_threshold
        
        # 传感器布局
        self.grid_shape = tactile_grid
        self.num_taxels = tactile_grid[0] * tactile_grid[1]
        
        # 力范围
        self.max_force_range = max_force_range
        self.shear_force_range = shear_force_range
        
        # 功能开关
        self.enable_vector_field = enable_vector_field
        
        # 滤波器设计
        self.b, self.a = self._design_filter()
        
        # 状态变量
        self.prev_filtered = None
        self.force_history: List[np.ndarray] = []  # 三维力历史
        self.resultant_history: List[np.ndarray] = []  # 合力历史
        self.history_length = 20
        
        # 矢量场相关
        self.taxel_positions = self._calculate_taxel_positions()
        self.vector_field_history: List[np.ndarray] = []
        
        # 统计信息
        self.processed_count = 0
        self.slip_count = 0
        self.vibration_count = 0
        
        logger.info(f"三维力数据处理器初始化完成: {filter_cutoff}Hz截止频率, {contact_threshold}N接触阈值")
    
    def _design_filter(self):
        """设计巴特沃斯低通滤波器"""
        nyquist = self.sampling_rate / 2
        normal_cutoff = self.filter_cutoff / nyquist
        
        # 二阶巴特沃斯滤波器
        b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def _calculate_taxel_positions(self) -> np.ndarray:
        """计算触点位置（归一化坐标）"""
        positions = []
        rows, cols = self.grid_shape
        
        for i in range(rows):
            for j in range(cols):
                # 归一化坐标 (0-1)
                x = j / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                positions.append([x, y])
        
        return np.array(positions)
    
    def process(self, tactile_data) -> ProcessedTactileData:
        """处理原始触觉数据 - 支持三维力"""
        try:
            # 1. 提取三维力数据
            force_vectors = self._extract_force_vectors(tactile_data)
            
            # 2. 数据滤波
            filtered_vectors = self._apply_filter(force_vectors)
            
            # 3. 计算合力和统计
            resultant_force = np.sum(filtered_vectors, axis=0)
            
            # 4. 接触检测（基于法向力）
            contact_mask = self._detect_contact(filtered_vectors)
            
            # 5. 计算力大小和方向
            force_magnitudes = np.linalg.norm(filtered_vectors, axis=1)
            force_directions = self._calculate_force_directions(filtered_vectors)
            
            # 6. 提取特征
            features = self._extract_features(filtered_vectors, contact_mask, resultant_force)
            
            # 7. 矢量场计算
            vector_field = self._calculate_vector_field(filtered_vectors) if self.enable_vector_field else None
            
            # 8. 滑移检测（基于剪切力变化）
            slip_detected = self._detect_slip(filtered_vectors)
            if slip_detected:
                self.slip_count += 1
            
            # 9. 振动检测
            vibration_detected = self._detect_vibration(filtered_vectors)
            if vibration_detected:
                self.vibration_count += 1
            
            # 10. 更新历史数据
            self._update_history(filtered_vectors, resultant_force)
            
            # 记录处理统计
            if self.processed_count % 100 == 0:
                avg_force = np.mean(force_magnitudes)
                max_force = np.max(force_magnitudes)
                logger.debug(f"处理数据 {self.processed_count}: 平均力={avg_force:.2f}N, 最大力={max_force:.2f}N, 接触点={np.sum(contact_mask)}")
            
            # 创建处理后的数据对象
            processed_data = ProcessedTactileData(
                raw_data=tactile_data,
                timestamp=getattr(tactile_data, 'timestamp', time.time()),
                filtered_vectors=filtered_vectors,
                resultant_force=resultant_force,
                contact_mask=contact_mask,
                force_magnitudes=force_magnitudes,
                force_directions=force_directions,
                features=features,
                vector_field=vector_field,
                slip_detected=slip_detected,
                vibration_detected=vibration_detected
            )
            
            self.processed_count += 1
            return processed_data
            
        except Exception as e:
            logger.error(f"处理触觉数据时出错: {e}")
            # 返回基本处理结果
            return ProcessedTactileData(
                raw_data=tactile_data,
                timestamp=getattr(tactile_data, 'timestamp', time.time()),
                filtered_vectors=np.zeros((self.num_taxels, 3)),
                resultant_force=np.zeros(3),
                contact_mask=np.zeros(self.num_taxels, dtype=bool),
                force_magnitudes=np.zeros(self.num_taxels),
                force_directions=np.zeros((self.num_taxels, 3)),
                features={'error': 1.0},
                vector_field=np.zeros((self.num_taxels, 5))  # x, y, fx, fy, fz
            )
    
    def _extract_force_vectors(self, tactile_data) -> np.ndarray:
        """从原始数据中提取三维力向量"""
        try:
            # 尝试从不同属性提取力向量
            if hasattr(tactile_data, 'force_vectors') and tactile_data.force_vectors is not None:
                # 直接获取力向量
                vectors = np.array(tactile_data.force_vectors, dtype=np.float32)
            elif hasattr(tactile_data, 'tactile_array'):
                # 从tactile_array提取
                tactile_array = tactile_data.tactile_array
                if tactile_array.shape[1] >= 3:
                    vectors = tactile_array[:, :3]  # 取前3列 (Fx, Fy, Fz)
                else:
                    # 只有Z方向力，填充X,Y为0
                    vectors = np.zeros((tactile_array.shape[0], 3), dtype=np.float32)
                    vectors[:, 2] = tactile_array.flatten()[:tactile_array.shape[0]]
            elif hasattr(tactile_data, 'force_data'):
                # 只有Z方向力
                force_data = tactile_data.force_data
                vectors = np.zeros((len(force_data), 3), dtype=np.float32)
                vectors[:, 2] = np.array(force_data)
            else:
                # 默认值
                vectors = np.zeros((self.num_taxels, 3), dtype=np.float32)
            
            # 确保维度正确
            if vectors.shape[0] != self.num_taxels:
                # 调整维度
                if vectors.shape[0] < self.num_taxels:
                    # 填充
                    padded = np.zeros((self.num_taxels, 3), dtype=np.float32)
                    padded[:vectors.shape[0], :] = vectors
                    vectors = padded
                else:
                    # 截断
                    vectors = vectors[:self.num_taxels, :]
            
            return vectors
            
        except Exception as e:
            logger.warning(f"提取力向量时出错: {e}")
            return np.zeros((self.num_taxels, 3), dtype=np.float32)
    
    def _apply_filter(self, force_vectors: np.ndarray) -> np.ndarray:
        """应用数字滤波器到三维力数据"""
        filtered = np.zeros_like(force_vectors)
        
        for i in range(3):  # 对每个力分量 (Fx, Fy, Fz)
            # 应用滤波器
            if self.prev_filtered is None:
                # 首次使用，初始化滤波器状态
                filtered[:, i] = signal.lfilter(self.b, self.a, force_vectors[:, i])
            else:
                # 使用上次的滤波器状态
                zi = signal.lfilter_zi(self.b, self.a)
                filtered_signal, _ = signal.lfilter(self.b, self.a, force_vectors[:, i], 
                                                   zi=zi * force_vectors[0, i])
                filtered[:, i] = filtered_signal
        
        # 更新上一次的滤波结果
        self.prev_filtered = filtered
        
        return filtered
    
    def _detect_contact(self, force_vectors: np.ndarray) -> np.ndarray:
        """检测接触状态 - 基于法向力 (Fz)"""
        # 使用Z方向力进行接触检测
        normal_forces = force_vectors[:, 2]  # Fz分量 (N)
        
        # 阈值检测
        contact_mask = normal_forces > self.contact_threshold
        
        # 形态学处理：去除孤立点
        contact_mask = self._apply_morphology(contact_mask)
        
        return contact_mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """应用形态学操作（开运算）"""
        # 简单的开运算：先腐蚀后膨胀
        window_size = 3
        
        # 腐蚀：所有邻居都为True时才为True
        eroded = np.zeros_like(mask, dtype=bool)
        for i in range(len(mask)):
            start = max(0, i - window_size // 2)
            end = min(len(mask), i + window_size // 2 + 1)
            eroded[i] = np.all(mask[start:end]) if any(mask[start:end]) else False
        
        # 膨胀：任意邻居为True就为True
        dilated = np.zeros_like(eroded, dtype=bool)
        for i in range(len(eroded)):
            start = max(0, i - window_size // 2)
            end = min(len(eroded), i + window_size // 2 + 1)
            dilated[i] = np.any(eroded[start:end])
        
        return dilated
    
    def _calculate_force_directions(self, force_vectors: np.ndarray) -> np.ndarray:
        """计算力方向（单位向量）"""
        magnitudes = np.linalg.norm(force_vectors, axis=1)
        
        # 避免除零错误
        with np.errstate(divide='ignore', invalid='ignore'):
            directions = force_vectors / magnitudes[:, np.newaxis]
            directions[magnitudes == 0] = [0, 0, 1]  # 默认向上
        
        return directions
    
    def _extract_features(self, force_vectors: np.ndarray, 
                          contact_mask: np.ndarray,
                          resultant_force: np.ndarray) -> Dict[str, float]:
        """提取三维力特征"""
        features = {}
        
        # 基础统计特征 (N)
        normal_forces = force_vectors[:, 2]  # Fz
        shear_forces = np.linalg.norm(force_vectors[:, :2], axis=1)  # Fx, Fy 的合成
        
        features['total_normal_force'] = np.sum(normal_forces)
        features['total_shear_force'] = np.sum(shear_forces)
        features['mean_normal_force'] = np.mean(normal_forces)
        features['mean_shear_force'] = np.mean(shear_forces)
        features['max_normal_force'] = np.max(normal_forces)
        features['max_shear_force'] = np.max(shear_forces)
        features['min_normal_force'] = np.min(normal_forces)
        features['min_shear_force'] = np.min(shear_forces)
        features['normal_force_std'] = np.std(normal_forces)
        features['shear_force_std'] = np.std(shear_forces)
        
        # 合力特征
        features['resultant_magnitude'] = np.linalg.norm(resultant_force)
        features['resultant_x'] = resultant_force[0]
        features['resultant_y'] = resultant_force[1]
        features['resultant_z'] = resultant_force[2]
        
        # 合力方向
        if features['resultant_magnitude'] > 0:
            features['resultant_direction_x'] = resultant_force[0] / features['resultant_magnitude']
            features['resultant_direction_y'] = resultant_force[1] / features['resultant_magnitude']
            features['resultant_direction_z'] = resultant_force[2] / features['resultant_magnitude']
        else:
            features['resultant_direction_x'] = 0
            features['resultant_direction_y'] = 0
            features['resultant_direction_z'] = 1
        
        # 接触相关特征
        contact_indices = np.where(contact_mask)[0]
        features['contact_count'] = len(contact_indices)
        features['contact_ratio'] = len(contact_indices) / len(contact_mask)
        
        if len(contact_indices) > 0:
            contact_normals = normal_forces[contact_indices]
            contact_shears = shear_forces[contact_indices]
            
            features['contact_mean_normal'] = np.mean(contact_normals)
            features['contact_max_normal'] = np.max(contact_normals)
            features['contact_mean_shear'] = np.mean(contact_shears)
            features['contact_max_shear'] = np.max(contact_shears)
            features['contact_normal_variance'] = np.var(contact_normals)
            features['contact_shear_variance'] = np.var(contact_shears)
            
            # 接触质心
            contact_positions = self.taxel_positions[contact_indices]
            features['contact_centroid_x'] = np.mean(contact_positions[:, 0])
            features['contact_centroid_y'] = np.mean(contact_positions[:, 1])
        else:
            features['contact_mean_normal'] = 0.0
            features['contact_max_normal'] = 0.0
            features['contact_mean_shear'] = 0.0
            features['contact_max_shear'] = 0.0
            features['contact_normal_variance'] = 0.0
            features['contact_shear_variance'] = 0.0
            features['contact_centroid_x'] = 0.0
            features['contact_centroid_y'] = 0.0
        
        # 力矩特征 (N·m) - 简化计算
        features['torque_x'] = self._calculate_torque(force_vectors, axis=0)
        features['torque_y'] = self._calculate_torque(force_vectors, axis=1)
        features['torque_z'] = self._calculate_torque(force_vectors, axis=2)
        
        # 力分布特征
        features['force_eccentricity'] = self._calculate_eccentricity(force_vectors)
        features['force_concentration'] = self._calculate_concentration(force_vectors)
        
        # 动态特征
        if len(self.force_history) > 1:
            features['normal_force_change_rate'] = self._calculate_change_rate(axis=2)
            features['shear_force_change_rate'] = self._calculate_shear_change_rate()
            features['force_variability'] = self._calculate_variability()
        
        # 摩擦特征
        if features['total_normal_force'] > 0:
            features['friction_coefficient_est'] = features['total_shear_force'] / features['total_normal_force']
        else:
            features['friction_coefficient_est'] = 0.0
        
        # 添加单位说明
        features['_force_unit'] = 'N'
        features['_torque_unit'] = 'N·m'
        
        return features
    
    def _calculate_torque(self, force_vectors: np.ndarray, axis: int = 0) -> float:
        """计算力矩"""
        # 计算每个触点的力矩贡献
        torque = 0.0
        
        if axis == 0:  # X轴力矩 (绕X轴旋转)
            # τ = r × F = (y*Fz - z*Fy)
            for i in range(len(force_vectors)):
                pos = self.taxel_positions[i]
                y = pos[1] - 0.5  # 中心化
                torque += y * force_vectors[i, 2]  # y * Fz
        elif axis == 1:  # Y轴力矩 (绕Y轴旋转)
            # τ = r × F = (z*Fx - x*Fz)
            for i in range(len(force_vectors)):
                pos = self.taxel_positions[i]
                x = pos[0] - 0.5  # 中心化
                torque += -x * force_vectors[i, 2]  # -x * Fz
        else:  # Z轴力矩 (绕Z轴旋转)
            # τ = r × F = (x*Fy - y*Fx)
            for i in range(len(force_vectors)):
                pos = self.taxel_positions[i]
                x = pos[0] - 0.5  # 中心化
                y = pos[1] - 0.5  # 中心化
                torque += x * force_vectors[i, 1] - y * force_vectors[i, 0]
        
        return float(torque)
    
    def _calculate_eccentricity(self, force_vectors: np.ndarray) -> float:
        """计算力分布的偏心率"""
        normal_forces = force_vectors[:, 2]
        
        if np.sum(normal_forces) == 0:
            return 0.0
        
        # 计算质心
        centroid_x = np.sum(self.taxel_positions[:, 0] * normal_forces) / np.sum(normal_forces)
        centroid_y = np.sum(self.taxel_positions[:, 1] * normal_forces) / np.sum(normal_forces)
        
        # 计算二阶矩
        Ixx = np.sum((self.taxel_positions[:, 0] - centroid_x) ** 2 * normal_forces)
        Iyy = np.sum((self.taxel_positions[:, 1] - centroid_y) ** 2 * normal_forces)
        Ixy = np.sum((self.taxel_positions[:, 0] - centroid_x) * 
                     (self.taxel_positions[:, 1] - centroid_y) * normal_forces)
        
        # 计算偏心率
        numerator = (Ixx - Iyy) ** 2 + 4 * Ixy ** 2
        denominator = (Ixx + Iyy) ** 2
        
        if denominator > 0:
            eccentricity = np.sqrt(numerator / denominator)
        else:
            eccentricity = 0.0
        
        return float(eccentricity)
    
    def _calculate_concentration(self, force_vectors: np.ndarray) -> float:
        """计算力集中度"""
        normal_forces = force_vectors[:, 2]
        
        if np.sum(normal_forces) == 0:
            return 0.0
        
        # 计算力分布的熵（越低表示越集中）
        probabilities = normal_forces / np.sum(normal_forces)
        probabilities = probabilities[probabilities > 0]  # 移除零概率
        
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(normal_forces))
        
        # 归一化到0-1，1表示最集中
        if max_entropy > 0:
            concentration = 1.0 - (entropy / max_entropy)
        else:
            concentration = 0.0
        
        return float(concentration)
    
    def _calculate_change_rate(self, axis: int = 2) -> float:
        """计算力变化率 (N/帧)"""
        if len(self.force_history) < 2:
            return 0.0
        
        recent_forces = [h[:, axis] for h in self.force_history[-2:]]
        change_rate = np.mean(np.abs(recent_forces[-1] - recent_forces[-2]))
        
        return float(change_rate)
    
    def _calculate_shear_change_rate(self) -> float:
        """计算剪切力变化率"""
        if len(self.force_history) < 2:
            return 0.0
        
        recent_shears = []
        for h in self.force_history[-2:]:
            shear = np.linalg.norm(h[:, :2], axis=1)  # Fx, Fy 的合成
            recent_shears.append(shear)
        
        change_rate = np.mean(np.abs(recent_shears[-1] - recent_shears[-2]))
        return float(change_rate)
    
    def _calculate_variability(self) -> float:
        """计算力变异性"""
        if len(self.force_history) < 2:
            return 0.0
        
        recent_magnitudes = []
        for h in self.force_history:
            magnitudes = np.linalg.norm(h, axis=1)
            recent_magnitudes.append(np.mean(magnitudes))
        
        variability = np.std(recent_magnitudes[-10:]) if len(recent_magnitudes) >= 10 else 0.0
        return float(variability)
    
    def _detect_slip(self, force_vectors: np.ndarray) -> bool:
        """检测滑移 - 基于剪切力变化"""
        if len(self.force_history) < 5:
            return False
        
        # 计算当前剪切力
        current_shear = np.linalg.norm(force_vectors[:, :2], axis=1)
        current_mean_shear = np.mean(current_shear)
        
        # 计算历史剪切力
        recent_shears = []
        for hist in self.force_history[-5:]:
            shear = np.linalg.norm(hist[:, :2], axis=1)
            recent_shears.append(np.mean(shear))
        
        # 计算变化率
        if recent_shears:
            prev_mean_shear = np.mean(recent_shears)
            shear_change = abs(current_mean_shear - prev_mean_shear)
            
            if shear_change > self.slip_threshold:
                logger.info(f"检测到滑移! 剪切力变化: {shear_change:.2f}N")
                return True
        
        # 检测剪切力方向变化
        if len(self.force_history) >= 3:
            current_shear_direction = np.mean(force_vectors[:, :2] / (np.linalg.norm(force_vectors[:, :2], axis=1, keepdims=True) + 1e-10), axis=0)
            prev_shear_direction = np.mean(self.force_history[-2][:, :2] / (np.linalg.norm(self.force_history[-2][:, :2], axis=1, keepdims=True) + 1e-10), axis=0)
            
            direction_change = np.linalg.norm(current_shear_direction - prev_shear_direction)
            if direction_change > 0.5:  # 方向变化阈值
                logger.info(f"检测到滑移方向变化: {direction_change:.2f}")
                return True
        
        return False
    
    def _detect_vibration(self, force_vectors: np.ndarray) -> bool:
        """检测振动"""
        if len(self.force_history) < 10:
            return False
        
        # 分析法向力的频率成分
        normal_forces = force_vectors[:, 2]
        
        # 计算功率谱密度
        frequencies, psd = signal.periodogram(normal_forces, fs=self.sampling_rate)
        
        # 检查高频成分
        high_freq_mask = frequencies > 30  # 30Hz以上
        high_freq_power = np.sum(psd[high_freq_mask])
        total_power = np.sum(psd)
        
        if total_power > 0:
            high_freq_ratio = high_freq_power / total_power
            if high_freq_ratio > self.vibration_threshold:
                logger.info(f"检测到振动! 高频能量占比: {high_freq_ratio:.2f}")
                return True
        
        return False
    
    def _calculate_vector_field(self, force_vectors: np.ndarray) -> np.ndarray:
        """计算矢量场数据用于可视化"""
        # 矢量场数据格式: [x, y, fx, fy, fz, magnitude, direction_x, direction_y, direction_z]
        vector_field = np.zeros((self.num_taxels, 9))
        
        for i in range(self.num_taxels):
            pos = self.taxel_positions[i]
            force = force_vectors[i]
            magnitude = np.linalg.norm(force)
            
            # 计算方向（单位向量）
            if magnitude > 0:
                direction = force / magnitude
            else:
                direction = np.array([0, 0, 1])  # 默认向上
            
            vector_field[i] = [
                pos[0], pos[1],           # 位置 (x, y)
                force[0], force[1], force[2],  # 力向量 (fx, fy, fz)
                magnitude,                # 力大小
                direction[0], direction[1], direction[2]  # 力方向
            ]
        
        return vector_field
    
    def _update_history(self, force_vectors: np.ndarray, resultant_force: np.ndarray):
        """更新历史数据"""
        self.force_history.append(force_vectors.copy())
        self.resultant_history.append(resultant_force.copy())
        
        # 保持历史数据长度
        if len(self.force_history) > self.history_length:
            self.force_history.pop(0)
        if len(self.resultant_history) > self.history_length:
            self.resultant_history.pop(0)
    
    def get_force_distribution(self) -> np.ndarray:
        """获取力分布矩阵"""
        if not self.force_history:
            return np.zeros(self.grid_shape)
        
        # 使用最近的数据
        latest_forces = self.force_history[-1]
        normal_forces = latest_forces[:, 2]  # 法向力
        
        # 重塑为网格
        force_grid = normal_forces.reshape(self.grid_shape)
        
        return force_grid
    
    def get_vector_field_data(self) -> Dict[str, np.ndarray]:
        """获取矢量场数据"""
        if not self.force_history:
            return {
                'positions': self.taxel_positions,
                'vectors': np.zeros((self.num_taxels, 3)),
                'magnitudes': np.zeros(self.num_taxels)
            }
        
        latest_vectors = self.force_history[-1]
        magnitudes = np.linalg.norm(latest_vectors, axis=1)
        
        return {
            'positions': self.taxel_positions,
            'vectors': latest_vectors,
            'magnitudes': magnitudes,
            'resultant': self.resultant_history[-1] if self.resultant_history else np.zeros(3)
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        return {
            "filter_cutoff_hz": self.filter_cutoff,
            "sampling_rate_hz": self.sampling_rate,
            "contact_threshold_n": self.contact_threshold,
            "slip_threshold_n": self.slip_threshold,
            "max_force_range_n": self.max_force_range,
            "shear_force_range_n": self.shear_force_range,
            "grid_shape": self.grid_shape,
            "processed_count": self.processed_count,
            "slip_detections": self.slip_count,
            "vibration_detections": self.vibration_count,
            "history_length": len(self.force_history),
            "force_dimensions": 3
        }
    
    def reset(self):
        """重置处理器状态"""
        self.prev_filtered = None
        self.force_history.clear()
        self.resultant_history.clear()
        self.vector_field_history.clear()
        self.slip_count = 0
        self.vibration_count = 0
        self.processed_count = 0
        logger.info("三维力数据处理器已重置")