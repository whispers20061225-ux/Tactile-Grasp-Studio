import torch
import torch.nn as nn
import numpy as np

class TactileFeatureEncoder(nn.Module):
    """触觉特征编码器 - 提取有用的触觉特征"""
    
    def __init__(self, input_channels=9, feature_dim=16):
        super(TactileFeatureEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim)
        )
        
    def forward(self, x):
        # x: (batch, channels, height, width)
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        
        encoded_features = self.feature_extractor(conv_features)
        
        # 添加手工特征
        manual_features = self.extract_manual_features(x)
        
        # 组合特征
        combined_features = torch.cat([encoded_features, manual_features], dim=1)
        
        return combined_features
    
    def extract_manual_features(self, tactile_data):
        """提取手工设计的触觉特征"""
        batch_size = tactile_data.size(0)
        features_list = []
        
        for i in range(batch_size):
            data = tactile_data[i].cpu().numpy()
            
            # 1. 统计特征
            mean_val = np.mean(data)
            std_val = np.std(data)
            max_val = np.max(data)
            min_val = np.min(data)
            
            # 2. 梯度特征
            grad_x = np.gradient(data, axis=0)
            grad_y = np.gradient(data, axis=1)
            grad_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # 3. 对称性特征
            center_of_pressure = self.calculate_cop(data)
            symmetry_score = self.calculate_symmetry(data)
            
            # 4. 纹理特征 (通过快速傅里叶变换)
            texture_features = self.extract_texture_features(data)
            
            # 组合所有特征
            manual_feat = np.array([
                mean_val, std_val, max_val, min_val,
                grad_magnitude,
                center_of_pressure[0], center_of_pressure[1],
                symmetry_score,
                *texture_features
            ])
            
            features_list.append(manual_feat)
        
        return torch.FloatTensor(np.array(features_list)).to(tactile_data.device)
    
    def calculate_cop(self, data):
        """计算压力中心"""
        height, width = data.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        total_pressure = np.sum(data)
        if total_pressure > 0:
            cop_x = np.sum(x_coords * data) / total_pressure
            cop_y = np.sum(y_coords * data) / total_pressure
        else:
            cop_x, cop_y = width/2, height/2
        
        # 归一化到[0,1]
        cop_x = cop_x / width
        cop_y = cop_y / height
        
        return cop_x, cop_y
    
    def calculate_symmetry(self, data):
        """计算对称性得分"""
        # 水平对称
        horizontal_sym = np.mean(np.abs(data - np.flip(data, axis=1)))
        # 垂直对称
        vertical_sym = np.mean(np.abs(data - np.flip(data, axis=0)))
        
        symmetry_score = 1.0 / (1.0 + horizontal_sym + vertical_sym)
        return symmetry_score
    
    def extract_texture_features(self, data):
        """提取纹理特征"""
        # 简单的纹理特征：边缘检测响应
        from scipy import ndimage
        
        sobel_x = ndimage.sobel(data, axis=0, mode='constant')
        sobel_y = ndimage.sobel(data, axis=1, mode='constant')
        
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features = [
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.max(edge_magnitude)
        ]
        
        return features


class MultiModalEncoder(nn.Module):
    """多模态编码器 - 融合视觉和触觉"""
    
    def __init__(self, visual_dim=512, tactile_dim=16, latent_dim=64):
        super(MultiModalEncoder, self).__init__()
        
        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 触觉编码器
        self.tactile_encoder = TactileFeatureEncoder(feature_dim=tactile_dim)
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + tactile_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, visual_input, tactile_input):
        # 编码视觉特征
        visual_features = self.visual_encoder(visual_input)
        
        # 编码触觉特征
        tactile_features = self.tactile_encoder(tactile_input)
        
        # 融合特征
        combined = torch.cat([visual_features, tactile_features], dim=1)
        fused_features = self.fusion_net(combined)
        
        return fused_features