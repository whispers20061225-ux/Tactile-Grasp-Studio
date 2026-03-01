import torch
import torch.nn as nn
import numpy as np

class GraspStateClassifier(nn.Module):
    """抓取状态分类器 - 实时判断抓取状态"""
    
    def __init__(self, input_dim=9, num_classes=4):
        super(GraspStateClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.class_names = ['slip', 'stable', 'tight', 'lost']
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
        # 状态转换检测
        self.state_history = []
        self.max_history = 20
        
        # 状态阈值
        self.slip_threshold = 0.3
        self.tight_threshold = 0.7
        
    def forward(self, tactile_data):
        # 提取特征
        features = self.feature_extractor(tactile_data)
        
        # 分类
        logits = self.classifier(features)
        
        # 获取概率
        probabilities = torch.softmax(logits, dim=1)
        
        # 获取预测类别
        _, predicted = torch.max(logits, 1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predicted': predicted
        }
    
    def detect_slip(self, tactile_data, previous_data=None):
        """检测滑动"""
        if previous_data is None:
            return False, 0.0
        
        # 计算压力中心移动
        current_cop = self._calculate_center_of_pressure(tactile_data)
        previous_cop = self._calculate_center_of_pressure(previous_data)
        
        # 计算移动距离
        movement = np.linalg.norm(current_cop - previous_cop)
        
        # 滑动判断
        is_slipping = movement > self.slip_threshold
        slip_confidence = min(movement / self.slip_threshold, 1.0)
        
        return is_slipping, slip_confidence
    
    def detect_grasp_quality(self, tactile_data):
        """检测抓取质量"""
        # 计算压力均匀性
        uniformity = self._calculate_pressure_uniformity(tactile_data)
        
        # 计算总压力
        total_pressure = np.sum(tactile_data)
        
        # 质量评分 (0-1)
        quality_score = uniformity * min(total_pressure, 1.0)
        
        if quality_score < 0.3:
            status = "poor"
        elif quality_score < 0.7:
            status = "fair"
        else:
            status = "good"
        
        return {
            'quality_score': quality_score,
            'status': status,
            'uniformity': uniformity,
            'total_pressure': total_pressure
        }
    
    def _calculate_center_of_pressure(self, tactile_data):
        """计算压力中心"""
        height, width = tactile_data.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        total_pressure = np.sum(tactile_data)
        if total_pressure > 0:
            cop_x = np.sum(x_coords * tactile_data) / total_pressure
            cop_y = np.sum(y_coords * tactile_data) / total_pressure
        else:
            cop_x, cop_y = width/2, height/2
        
        return np.array([cop_x, cop_y])
    
    def _calculate_pressure_uniformity(self, tactile_data):
        """计算压力均匀性"""
        # 使用熵来度量均匀性
        normalized = tactile_data / (np.sum(tactile_data) + 1e-8)
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        
        # 归一化到0-1
        max_entropy = np.log(tactile_data.size)
        uniformity = entropy / max_entropy
        
        return uniformity
    
    def get_control_suggestions(self, state, tactile_data):
        """根据状态提供控制建议"""
        suggestions = []
        
        if state == 'slip':
            # 滑动时增加抓取力
            suggestions.append({
                'action': 'increase_force',
                'amount': 0.2,
                'reason': 'detected slipping'
            })
            
        elif state == 'tight':
            # 过紧时减小抓取力
            suggestions.append({
                'action': 'decrease_force',
                'amount': 0.15,
                'reason': 'grip too tight'
            })
            
        elif state == 'lost':
            # 丢失抓取时重新接近
            suggestions.append({
                'action': 'reapproach',
                'params': {'speed': 0.1, 'distance': 0.02},
                'reason': 'lost grasp'
            })
        
        return suggestions