"""
图像处理模块 - 图像预处理、增强和特征提取
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image

from utils.logging_config import get_logger

logger = get_logger(__name__)

class ImageProcessor:
    """
    图像处理器
    提供各种图像处理和增强功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图像处理器
        
        Args:
            config: 图像处理配置
        """
        self.config = config
        
        # 预处理参数
        self.resize_width = config.get('resize_width', 640)
        self.resize_height = config.get('resize_height', 480)
        self.normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
        
        # 增强变换
        self.augmentations = self._create_augmentations()
        
        # 特征提取器
        self.feature_extractor = None
        if config.get('enable_feature_extraction', False):
            self._initialize_feature_extractor()
        
        logger.info("ImageProcessor initialized")
    
    def _create_augmentations(self) -> A.Compose:
        """创建数据增强管道"""
        train_transforms = A.Compose([
            A.Resize(self.resize_height, self.resize_width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                             rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                     contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ])
        
        return train_transforms
    
    def _initialize_feature_extractor(self):
        """初始化特征提取器"""
        try:
            # 使用预训练模型提取特征
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # 加载预训练模型
            model = models.resnet50(pretrained=True)
            # 移除最后的全连接层
            self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_extractor.eval()
            
            # 特征提取的预处理
            self.feature_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ])
            
            logger.info("Feature extractor initialized (ResNet50)")
            
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {str(e)}")
            self.feature_extractor = None
    
    def preprocess(self, image: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C)
            is_training: 是否为训练模式
            
        Returns:
            预处理后的张量
        """
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # 假设是BGR格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if is_training and self.augmentations:
            # 应用数据增强
            augmented = self.augmentations(image=image)
            return augmented['image']
        else:
            # 仅应用基本预处理
            transform = A.Compose([
                A.Resize(self.resize_height, self.resize_width),
                A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
                ToTensorV2(),
            ])
            augmented = transform(image=image)
            return augmented['image']
    
    def batch_preprocess(self, images: List[np.ndarray], 
                        is_training: bool = False) -> torch.Tensor:
        """
        批量预处理图像
        
        Args:
            images: 图像列表
            is_training: 是否为训练模式
            
        Returns:
            批处理张量
        """
        processed_images = []
        
        for image in images:
            processed = self.preprocess(image, is_training)
            processed_images.append(processed)
        
        return torch.stack(processed_images)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征向量
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized")
        
        # 预处理图像
        pil_image = Image.fromarray(image)
        input_tensor = self.feature_transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            features = self.feature_extractor(input_batch)
        
        # 展平特征
        features = features.squeeze().numpy()
        
        return features
    
    def segment_objects(self, image: np.ndarray, 
                       method: str = 'threshold') -> np.ndarray:
        """
        分割物体
        
        Args:
            image: 输入图像
            method: 分割方法 ('threshold', 'edge', 'contour', 'watershed')
            
        Returns:
            分割掩码
        """
        if method == 'threshold':
            return self._segment_by_threshold(image)
        elif method == 'edge':
            return self._segment_by_edge(image)
        elif method == 'contour':
            return self._segment_by_contour(image)
        elif method == 'watershed':
            return self._segment_by_watershed(image)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _segment_by_threshold(self, image: np.ndarray) -> np.ndarray:
        """基于阈值分割"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        mask = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return mask
    
    def _segment_by_edge(self, image: np.ndarray) -> np.ndarray:
        """基于边缘检测分割"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 填充边缘
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask
    
    def _segment_by_contour(self, image: np.ndarray) -> np.ndarray:
        """基于轮廓分割"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 根据颜色阈值创建掩码
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
        
        lower_bound = np.array([170, 50, 50])
        upper_bound = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound, upper_bound)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _segment_by_watershed(self, image: np.ndarray) -> np.ndarray:
        """基于分水岭算法分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 阈值处理
        _, thresh = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 
                                 255, 0)
        
        # 找到未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 创建掩码
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == 1] = 0
        mask[markers > 1] = 255
        
        return mask
    
    def detect_keypoints(self, image: np.ndarray, 
                        method: str = 'orb') -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        检测关键点
        
        Args:
            image: 输入图像
            method: 关键点检测方法 ('orb', 'sift', 'akaze', 'brisk')
            
        Returns:
            关键点列表和描述子
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == 'orb':
            detector = cv2.ORB_create(nfeatures=500)
        elif method == 'sift':
            detector = cv2.SIFT_create()
        elif method == 'akaze':
            detector = cv2.AKAZE_create()
        elif method == 'brisk':
            detector = cv2.BRISK_create()
        else:
            raise ValueError(f"Unknown keypoint method: {method}")
        
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_keypoints(self, descriptors1: np.ndarray, 
                       descriptors2: np.ndarray,
                       method: str = 'bruteforce') -> List[cv2.DMatch]:
        """
        匹配关键点
        
        Args:
            descriptors1: 第一个描述子集
            descriptors2: 第二个描述子集
            method: 匹配方法 ('bruteforce', 'flann')
            
        Returns:
            匹配列表
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if method == 'bruteforce':
            # 暴力匹配
            if descriptors1.dtype == np.uint8:
                # 对于二进制描述符（ORB, BRIEF, BRISK）
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                # 对于浮点描述符（SIFT, SURF）
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif method == 'flann':
            # FLANN匹配
            if descriptors1.dtype == np.uint8:
                # 对于二进制描述符
                index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            else:
                # 对于浮点描述符
                index_params = dict(algorithm=1,  # FLANN_INDEX_KDTREE
                                  trees=5)
            
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        
        matches = matcher.match(descriptors1, descriptors2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    def draw_keypoints(self, image: np.ndarray, 
                      keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """绘制关键点"""
        return cv2.drawKeypoints(image, keypoints, None, 
                               color=(0, 255, 0), flags=0)
    
    def draw_matches(self, image1: np.ndarray, keypoints1: List[cv2.KeyPoint],
                    image2: np.ndarray, keypoints2: List[cv2.KeyPoint],
                    matches: List[cv2.DMatch]) -> np.ndarray:
        """绘制匹配"""
        return cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                             matches[:50], None, flags=2)
    
    def get_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """获取图像统计信息"""
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'size_kb': image.nbytes / 1024
        }
        
        if len(image.shape) == 3:
            # 彩色图像
            for i, channel in enumerate(['R', 'G', 'B']):
                stats[f'{channel}_mean'] = float(image[:, :, i].mean())
                stats[f'{channel}_std'] = float(image[:, :, i].std())
        
        return stats