# tactile_perception/tactile_analyzer.py
import numpy as np
from scipy import signal, stats
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)

class TactileAnalyzer:
    """
    高级触觉数据分析器
    提供物体识别、材质分类、接触分析等功能
    """
    
    def __init__(self, 
                 sampling_rate: float = 83.3,
                 tactile_grid: Tuple[int, int] = (3, 3)):
        
        self.sampling_rate = sampling_rate
        self.grid_shape = tactile_grid
        
        # 分析状态
        self.contact_history: List[np.ndarray] = []
        self.force_history: List[np.ndarray] = []
        self.feature_history: List[Dict] = []
        
        # 历史数据长度
        self.history_size = 100
        
        # 物体识别模型
        self.pca = PCA(n_components=3)
        self.clustering = DBSCAN(eps=0.5, min_samples=5)
        
        # 材质特征库
        self.material_features = {
            'hard': {'mean_force': 15.0, 'std_force': 2.0, 'hardness': 0.9},
            'soft': {'mean_force': 5.0, 'std_force': 1.0, 'hardness': 0.3},
            'elastic': {'mean_force': 8.0, 'std_force': 3.0, 'hardness': 0.6},
            'rigid': {'mean_force': 20.0, 'std_force': 1.5, 'hardness': 0.95}
        }
        
        logger.info("TactileAnalyzer initialized")
    
    def analyze_contact(self, tactile_data: np.ndarray) -> Dict:
        """分析接触模式"""
        try:
            # 提取Z方向力
            z_forces = tactile_data[:, 2]
            
            # 基础统计
            stats = {
                'mean_force': float(np.mean(z_forces)),
                'std_force': float(np.std(z_forces)),
                'max_force': float(np.max(z_forces)),
                'min_force': float(np.min(z_forces)),
                'total_force': float(np.sum(z_forces)),
                'force_range': float(np.ptp(z_forces))
            }
            
            # 接触区域分析
            contact_mask = z_forces > 0.1  # 简单阈值
            contact_indices = np.where(contact_mask)[0]
            
            if len(contact_indices) > 0:
                contact_forces = z_forces[contact_indices]
                
                # 接触区域统计
                stats.update({
                    'contact_area': len(contact_indices),
                    'contact_ratio': len(contact_indices) / len(z_forces),
                    'contact_mean_force': float(np.mean(contact_forces)),
                    'contact_std_force': float(np.std(contact_forces)),
                    'contact_force_skewness': float(stats.skew(contact_forces)),
                    'contact_force_kurtosis': float(stats.kurtosis(contact_forces))
                })
                
                # 接触中心
                if self.grid_shape[0] * self.grid_shape[1] == len(z_forces):
                    stats.update(self._calculate_contact_center(z_forces))
                else:
                    # 线性布局
                    weighted_pos = np.average(contact_indices, weights=contact_forces)
                    stats['contact_center'] = float(weighted_pos)
            else:
                stats.update({
                    'contact_area': 0,
                    'contact_ratio': 0.0,
                    'contact_mean_force': 0.0,
                    'contact_std_force': 0.0,
                    'contact_center_x': 0.0,
                    'contact_center_y': 0.0
                })
            
            # 力分布均匀性
            stats['force_uniformity'] = self._calculate_uniformity(z_forces)
            
            # 力梯度
            stats['force_gradient'] = self._calculate_force_gradient(z_forces)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in contact analysis: {e}")
            return {}
    
    def _calculate_contact_center(self, forces: np.ndarray) -> Dict:
        """计算接触中心"""
        # 将测点映射到网格
        if len(forces) == 9:  # 3x3网格
            # 网格坐标
            x_coords = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
            y_coords = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
            
            # 加权平均计算质心
            total_force = np.sum(forces)
            if total_force > 0:
                center_x = np.sum(x_coords * forces) / total_force
                center_y = np.sum(y_coords * forces) / total_force
            else:
                center_x = center_y = 0.0
            
            return {
                'contact_center_x': float(center_x),
                'contact_center_y': float(center_y)
            }
        else:
            return {
                'contact_center_x': 0.0,
                'contact_center_y': 0.0
            }
    
    def _calculate_uniformity(self, forces: np.ndarray) -> float:
        """计算力分布均匀性"""
        if np.sum(forces) == 0:
            return 1.0  # 完全均匀（无接触）
        
        # 使用变异系数 (CV) 的倒数作为均匀性度量
        mean_force = np.mean(forces)
        std_force = np.std(forces)
        
        if mean_force > 0:
            cv = std_force / mean_force
            uniformity = 1.0 / (1.0 + cv)
        else:
            uniformity = 1.0
        
        return float(uniformity)
    
    def _calculate_force_gradient(self, forces: np.ndarray) -> float:
        """计算力梯度"""
        if len(forces) < 2:
            return 0.0
        
        # 计算相邻测点间的力差
        if len(forces) == 9:  # 3x3网格
            # 计算行梯度
            row_gradients = []
            for i in range(3):
                row_forces = forces[i*3:(i+1)*3]
                if np.any(row_forces > 0):
                    row_gradient = np.max(row_forces) - np.min(row_forces)
                    row_gradients.append(row_gradient)
            
            # 计算列梯度
            col_gradients = []
            for j in range(3):
                col_forces = forces[j::3]
                if np.any(col_forces > 0):
                    col_gradient = np.max(col_forces) - np.min(col_forces)
                    col_gradients.append(col_gradient)
            
            # 平均梯度
            all_gradients = row_gradients + col_gradients
            if all_gradients:
                return float(np.mean(all_gradients))
            else:
                return 0.0
        else:
            # 线性梯度
            return float(np.max(forces) - np.min(forces))
    
    def analyze_material(self, tactile_data: np.ndarray, 
                         history: List[np.ndarray] = None) -> Dict:
        """分析材质特性"""
        try:
            z_forces = tactile_data[:, 2]
            
            # 计算材质特征
            features = {
                'hardness': self._estimate_hardness(z_forces),
                'elasticity': self._estimate_elasticity(z_forces, history),
                'roughness': self._estimate_roughness(z_forces),
                'compliance': self._estimate_compliance(z_forces)
            }
            
            # 材质分类
            material_type, confidence = self._classify_material(features)
            
            return {
                'material_features': features,
                'material_type': material_type,
                'confidence': confidence,
                'estimated_hardness': features['hardness'],
                'estimated_elasticity': features['elasticity']
            }
            
        except Exception as e:
            logger.error(f"Error in material analysis: {e}")
            return {
                'material_features': {},
                'material_type': 'unknown',
                'confidence': 0.0
            }
    
    def _estimate_hardness(self, forces: np.ndarray) -> float:
        """估计硬度"""
        # 基于力响应估计硬度
        max_force = np.max(forces)
        mean_force = np.mean(forces)
        
        # 简化的硬度估计
        if max_force > 0:
            # 硬度与最大力相关，但归一化到0-1范围
            hardness = min(max_force / 25.0, 1.0)
        else:
            hardness = 0.0
        
        return float(hardness)
    
    def _estimate_elasticity(self, current_forces: np.ndarray, 
                        history: List[np.ndarray] = None) -> float:
        """估计弹性"""
        if history is None or len(history) < 10:
            return 0.5  # 默认值
        
        try:
            # 分析力变化模式
            recent_forces = [np.mean(h[:, 2]) for h in history[-10:]]
            current_mean = np.mean(current_forces)
            
            # 计算力变化的标准差
            all_forces = recent_forces + [current_mean]
            force_std = np.std(all_forces)
            force_mean = np.mean(all_forces)
            
            if force_mean > 0:
                # 弹性与力变化的相对大小相关
                elasticity = min(force_std / force_mean, 1.0)
            else:
                elasticity = 0.0
            
            return float(elasticity)
            
        except Exception as e:
            logger.error(f"Error estimating elasticity: {e}")
            return 0.5

    
    def _estimate_roughness(self, forces: np.ndarray) -> float:
        """估计粗糙度"""
        # 粗糙度与力的局部变化相关
        if len(forces) < 2:
            return 0.0
        
        # 计算局部变化
        local_variations = []
        for i in range(len(forces) - 1):
            variation = abs(forces[i+1] - forces[i])
            local_variations.append(variation)
        
        if local_variations:
            roughness = np.mean(local_variations) / (np.mean(forces) + 1e-6)
            return float(min(roughness, 1.0))
        else:
            return 0.0
    
    def _estimate_compliance(self, forces: np.ndarray) -> float:
        """估计顺应性（柔顺度）"""
        # 顺应性与力的分布均匀性相关
        uniformity = self._calculate_uniformity(forces)
        mean_force = np.mean(forces)
        
        # 顺应性是均匀性和平均力的函数
        if mean_force > 0:
            compliance = uniformity * (1.0 - min(mean_force / 25.0, 1.0))
        else:
            compliance = 1.0
        
        return float(compliance)
    
    def _classify_material(self, features: Dict) -> Tuple[str, float]:
        """分类材质类型"""
        try:
            # 计算与每个材质原型的距离
            distances = {}
            for material_type, proto_features in self.material_features.items():
                # 计算特征距离
                dist = 0.0
                for key in ['hardness']:  # 只使用硬度特征
                    if key in features and key in proto_features:
                        dist += abs(features[key] - proto_features[key])
                
                distances[material_type] = dist
            
            # 找到最近的原型
            if distances:
                min_material = min(distances.items(), key=lambda x: x[1])
                material_type = min_material[0]
                
                # 计算置信度
                total_dist = sum(distances.values())
                if total_dist > 0:
                    confidence = 1.0 - (min_material[1] / total_dist)
                else:
                    confidence = 1.0
                
                return material_type, float(confidence)
            else:
                return 'unknown', 0.0
                
        except Exception as e:
            logger.error(f"Error in material classification: {e}")
            return 'unknown', 0.0
    
    def detect_slip(self, current_data: np.ndarray, 
                   history: List[np.ndarray] = None) -> Dict:
        """检测滑移事件"""
        try:
            if history is None or len(history) < 5:
                return {'slip_detected': False, 'confidence': 0.0, 'slip_direction': None}
            
            # 提取切向力历史
            tangential_history = []
            for data in history[-5:]:
                tangential = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
                tangential_history.append(np.mean(tangential))
            
            current_tangential = np.sqrt(current_data[:, 0]**2 + current_data[:, 1]**2)
            current_mean = np.mean(current_tangential)
            
            # 计算切向力变化
            if tangential_history:
                prev_mean = np.mean(tangential_history)
                change = abs(current_mean - prev_mean)
                
                # 滑移检测阈值
                slip_threshold = 0.5
                slip_detected = change > slip_threshold
                
                # 计算滑移方向
                slip_direction = None
                if slip_detected:
                    # 计算平均切向力向量
                    mean_tangential_x = np.mean(current_data[:, 0])
                    mean_tangential_y = np.mean(current_data[:, 1])
                    
                    # 计算方向角度
                    angle = np.arctan2(mean_tangential_y, mean_tangential_x)
                    slip_direction = float(np.degrees(angle))
                
                # 置信度基于变化大小
                confidence = min(change / slip_threshold, 1.0)
                
                return {
                    'slip_detected': slip_detected,
                    'confidence': float(confidence),
                    'slip_direction': slip_direction,
                    'tangential_change': float(change),
                    'current_tangential': float(current_mean)
                }
            else:
                return {'slip_detected': False, 'confidence': 0.0, 'slip_direction': None}
                
        except Exception as e:
            logger.error(f"Error in slip detection: {e}")
            return {'slip_detected': False, 'confidence': 0.0, 'slip_direction': None}
    
    def detect_vibration(self, forces: np.ndarray) -> Dict[str, float]:
        """检测振动"""
        try:
            # 分析力信号的频率成分
            z_forces = forces[:, 2]
            
            if len(z_forces) < 10:
                return {'vibration_detected': False, 'dominant_frequency': 0.0}
            
            # 计算功率谱密度
            frequencies, psd = signal.periodogram(z_forces, fs=self.sampling_rate)
            
            # 找到主要频率成分
            if len(psd) > 0:
                dominant_idx = np.argmax(psd)
                dominant_freq = frequencies[dominant_idx]
                
                # 检查是否有显著的高频成分
                high_freq_mask = frequencies > 30  # 30Hz以上
                if np.any(high_freq_mask):
                    high_freq_power = np.sum(psd[high_freq_mask])
                    total_power = np.sum(psd)
                    
                    if total_power > 0:
                        high_freq_ratio = high_freq_power / total_power
                        vibration_detected = high_freq_ratio > 0.1
                    else:
                        vibration_detected = False
                else:
                    vibration_detected = False
                
                result = {
                    'vibration_detected': vibration_detected,
                    'dominant_frequency': float(dominant_freq),
                    'total_power': float(total_power) if 'total_power' in locals() else 0.0
                }
                
                # 添加high_freq_ratio到结果中
                if 'high_freq_ratio' in locals():
                    result['high_freq_ratio'] = float(high_freq_ratio)
                else:
                    result['high_freq_ratio'] = 0.0
                    
                return result
            else:
                return {'vibration_detected': False, 'dominant_frequency': 0.0}
                
        except Exception as e:
            logger.error(f"Error in vibration detection: {e}")
            return {'vibration_detected': False, 'dominant_frequency': 0.0}
                 
    
    def analyze_object_shape(self, contact_pattern: np.ndarray) -> Dict:
        """分析物体形状"""
        try:
            # 将接触模式转换为网格
            if len(contact_pattern) == 9:
                grid = contact_pattern.reshape(self.grid_shape)
                
                # 计算形状特征
                features = {
                    'contact_area': np.sum(grid > 0),
                    'compactness': self._calculate_compactness(grid),
                    'aspect_ratio': self._calculate_aspect_ratio(grid),
                    'symmetry': self._calculate_symmetry(grid)
                }
                
                # 形状分类
                shape_type = self._classify_shape(features)
                
                return {
                    'shape_features': features,
                    'shape_type': shape_type,
                    'grid_pattern': grid.tolist()
                }
            else:
                return {
                    'shape_features': {},
                    'shape_type': 'unknown',
                    'grid_pattern': []
                }
                
        except Exception as e:
            logger.error(f"Error in shape analysis: {e}")
            return {
                'shape_features': {},
                'shape_type': 'unknown',
                'grid_pattern': []
            }
    
    def _calculate_compactness(self, grid: np.ndarray) -> float:
        """计算紧凑度"""
        # 紧凑度 = 接触面积 / 边界框面积
        contact_indices = np.where(grid > 0)
        
        if len(contact_indices[0]) == 0:
            return 0.0
        
        # 计算边界框
        min_row, max_row = np.min(contact_indices[0]), np.max(contact_indices[0])
        min_col, max_col = np.min(contact_indices[1]), np.max(contact_indices[1])
        
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        contact_area = len(contact_indices[0])
        
        if bbox_area > 0:
            return contact_area / bbox_area
        else:
            return 0.0
    
    def _calculate_aspect_ratio(self, grid: np.ndarray) -> float:
        """计算纵横比"""
        contact_indices = np.where(grid > 0)
        
        if len(contact_indices[0]) == 0:
            return 1.0
        
        # 计算边界框尺寸
        min_row, max_row = np.min(contact_indices[0]), np.max(contact_indices[0])
        min_col, max_col = np.min(contact_indices[1]), np.max(contact_indices[1])
        
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        
        if height > 0:
            return width / height
        else:
            return 1.0
    
    def _calculate_symmetry(self, grid: np.ndarray) -> float:
        """计算对称性"""
        # 检查水平和垂直对称
        horizontal_symmetry = 0.0
        vertical_symmetry = 0.0
        
        # 水平对称（左右对称）
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1] // 2):
                if grid[i, j] == grid[i, self.grid_shape[1] - 1 - j]:
                    horizontal_symmetry += 1
        
        # 垂直对称（上下对称）
        for i in range(self.grid_shape[0] // 2):
            for j in range(self.grid_shape[1]):
                if grid[i, j] == grid[self.grid_shape[0] - 1 - i, j]:
                    vertical_symmetry += 1
        
        total_cells = (self.grid_shape[0] * self.grid_shape[1] // 2)
        if total_cells > 0:
            symmetry = (horizontal_symmetry + vertical_symmetry) / (2 * total_cells)
            return float(symmetry)
        else:
            return 0.0
    
    def _classify_shape(self, features: Dict) -> str:
        """分类物体形状"""
        try:
            # 简单的基于规则的形状分类
            compactness = features.get('compactness', 0.0)
            aspect_ratio = features.get('aspect_ratio', 1.0)
            symmetry = features.get('symmetry', 0.0)
            
            if compactness > 0.8:
                return 'circular'
            elif aspect_ratio > 1.5:
                return 'rectangular'
            elif aspect_ratio < 0.67:
                return 'rectangular'  # 竖长方形
            elif symmetry > 0.7:
                return 'symmetric'
            else:
                return 'irregular'
                
        except:
            return 'unknown'
    
    def update_history(self, tactile_data: np.ndarray, features: Dict):
        """更新历史数据"""
        self.force_history.append(tactile_data.copy())
        self.feature_history.append(features.copy())
        
        # 保持历史数据长度
        if len(self.force_history) > self.history_size:
            self.force_history.pop(0)
            self.feature_history.pop(0)
    
    def get_temporal_features(self, window_size: int = 10) -> Dict:
        """提取时序特征"""
        if len(self.force_history) < window_size:
            return {}
        
        try:
            recent_forces = [h[:, 2] for h in self.force_history[-window_size:]]
            recent_features = self.feature_history[-window_size:]
            
            # 计算时序统计
            temporal_stats = {
                'force_mean_trend': float(np.mean([f.get('mean_force', 0) for f in recent_features])),
                'force_std_trend': float(np.mean([f.get('std_force', 0) for f in recent_features])),
                'force_change_rate': self._calculate_temporal_change(recent_forces),
                'force_autocorrelation': self._calculate_autocorrelation(recent_forces),
                'stability_index': self._calculate_stability_index(recent_forces)
            }
            
            return temporal_stats
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return {}
    
    def _calculate_temporal_change(self, force_history: List[np.ndarray]) -> float:
        """计算时序变化率"""
        if len(force_history) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(force_history)):
            change = np.mean(np.abs(force_history[i] - force_history[i-1]))
            changes.append(change)
        
        if changes:
            return float(np.mean(changes))
        else:
            return 0.0
    
    def _calculate_autocorrelation(self, force_history: List[np.ndarray], lag: int = 1) -> float:
        """计算自相关"""
        if len(force_history) < lag + 1:
            return 0.0
        
        try:
            # 计算每个测点的自相关，然后取平均
            autocorrs = []
            num_tactels = force_history[0].shape[0]
            
            for t in range(num_tactels):
                series = [h[t] for h in force_history]
                if len(series) > lag:
                    autocorr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrs.append(autocorr)
            
            if autocorrs:
                return float(np.mean(autocorrs))
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _calculate_stability_index(self, force_history: List[np.ndarray]) -> float:
        """计算稳定性指数"""
        if len(force_history) < 2:
            return 1.0  # 完全稳定
        
        # 计算力变化的标准差
        all_forces = np.concatenate([h.flatten() for h in force_history])
        force_std = np.std(all_forces)
        force_mean = np.mean(all_forces)
        
        if force_mean > 0:
            stability = 1.0 / (1.0 + force_std / force_mean)
            return float(stability)
        else:
            return 1.0
    
    def get_comprehensive_analysis(self, tactile_data: np.ndarray) -> Dict:
        """获取综合分析结果"""
        try:
            # 接触分析
            contact_analysis = self.analyze_contact(tactile_data)
            
            # 材质分析
            material_analysis = self.analyze_material(tactile_data, self.force_history)
            
            # 滑移检测
            slip_analysis = self.detect_slip(tactile_data, self.force_history)
            
            # 振动检测
            vibration_analysis = self.detect_vibration(tactile_data)
            
            # 形状分析
            shape_analysis = self.analyze_object_shape(tactile_data[:, 2] > 0.1)
            
            # 时序特征
            temporal_features = self.get_temporal_features()
            
            # 更新历史
            self.update_history(tactile_data, contact_analysis)
            
            # 综合结果
            comprehensive = {
                'contact_analysis': contact_analysis,
                'material_analysis': material_analysis,
                'slip_analysis': slip_analysis,
                'vibration_analysis': vibration_analysis,
                'shape_analysis': shape_analysis,
                'temporal_features': temporal_features,
                'timestamp': time.time() if 'time' in globals() else 0.0,
                'history_size': len(self.force_history)
            }
            
            # 计算整体置信度
            confidence_scores = []
            if material_analysis.get('confidence', 0) > 0:
                confidence_scores.append(material_analysis['confidence'])
            if slip_analysis.get('confidence', 0) > 0:
                confidence_scores.append(slip_analysis['confidence'])
            
            if confidence_scores:
                comprehensive['overall_confidence'] = float(np.mean(confidence_scores))
            else:
                comprehensive['overall_confidence'] = 0.0
            
            return comprehensive
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'error': str(e),
                'timestamp': time.time() if 'time' in globals() else 0.0
            }