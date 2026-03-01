"""
姿态估计模块 - 6D姿态估计（位置和方向）
"""

import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

from utils.logging_config import get_logger
from utils.transformations import quaternion_from_matrix, rotation_matrix_to_euler

logger = get_logger(__name__)

@dataclass
class PoseEstimation:
    """姿态估计数据结构"""
    position: np.ndarray  # [x, y, z] 位置 (米)
    rotation: np.ndarray  # 旋转矩阵 (3x3) 或四元数 [x, y, z, w]
    confidence: float
    object_id: str
    object_class: str
    timestamp: float
    method: str  # 估计方法

class PoseEstimator:
    """
    6D姿态估计器
    支持多种姿态估计方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化姿态估计器
        
        Args:
            config: 姿态估计配置
        """
        self.config = config
        self.method = config.get('method', 'pnp')
        
        # 相机内参
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 3D模型数据库
        self.object_models: Dict[str, Dict] = {}
        
        # 特征检测器
        self.feature_detector = None
        self.descriptor_extractor = None
        
        # 深度学习模型
        self.pose_net = None
        self._missing_model_warned = set()
        self._simple_sizes = {
            "bottle": (0.065, 0.22, 0.065),
            "cup": (0.08, 0.10, 0.08),
            "mug": (0.09, 0.11, 0.09),
            "wine glass": (0.07, 0.16, 0.07),
            "bowl": (0.14, 0.06, 0.14),
            "can": (0.066, 0.12, 0.066),
            "laptop": (0.32, 0.22, 0.02),
            "keyboard": (0.30, 0.12, 0.03),
            "mouse": (0.10, 0.06, 0.04),
            "cell phone": (0.15, 0.07, 0.01),
            "book": (0.20, 0.13, 0.03),
            "remote": (0.18, 0.05, 0.02),
        }
        
        # 初始化
        self._initialize_camera_params()
        self._initialize_models()
        
        logger.info(f"PoseEstimator initialized with method: {self.method}")
    
    def _initialize_camera_params(self):
        """初始化相机参数"""
        camera_params = self.config.get('camera_params', {})
        
        if camera_params:
            fx = camera_params.get('fx', 500.0)
            fy = camera_params.get('fy', 500.0)
            cx = camera_params.get('cx', 320.0)
            cy = camera_params.get('cy', 240.0)
            
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.array(camera_params.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0]), 
                                      dtype=np.float32)
        else:
            logger.warning("Camera parameters not provided, using default")
            self.camera_matrix = np.array([
                [500.0, 0, 320.0],
                [0, 500.0, 240.0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.zeros((4,), dtype=np.float32)
    
    def _initialize_models(self):
        """初始化模型和检测器"""
        if self.method in ['pnp', 'epnp', 'solvepnp']:
            self._initialize_feature_detector()
        elif self.method == 'deep_learning':
            self._initialize_deep_learning_model()
        elif self.method == 'point_pair':
            self._initialize_point_pair_features()
    
    def _initialize_feature_detector(self):
        """初始化特征检测器"""
        detector_type = self.config.get('feature_detector', 'orb')
        
        if detector_type == 'orb':
            self.feature_detector = cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'sift':
            self.feature_detector = cv2.SIFT_create()
        elif detector_type == 'akaze':
            self.feature_detector = cv2.AKAZE_create()
        elif detector_type == 'brisk':
            self.feature_detector = cv2.BRISK_create()
        
        logger.info(f"Feature detector initialized: {detector_type}")
    
    def _initialize_deep_learning_model(self):
        """初始化深度学习模型"""
        model_path = self.config.get('model_path')
        
        if model_path:
            try:
                # 加载预训练的PoseNet或类似模型
                self.pose_net = torch.load(model_path, map_location='cpu')
                self.pose_net.eval()
                
                logger.info(f"Deep learning pose model loaded: {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to load deep learning model: {str(e)}")
                self.pose_net = None
        else:
            logger.warning("No model path provided for deep learning method")
            self.pose_net = None
    
    def _initialize_point_pair_features(self):
        """初始化点对特征"""
        # 用于PPF（Point Pair Features）方法
        self.ppf_detector = cv2.ppf_match_3d_PPF3DDetector()
        logger.info("PPF detector initialized")
    
    def register_object(self, object_id: str, 
                       model_points_3d: np.ndarray,
                       keypoints_3d: Optional[np.ndarray] = None,
                       descriptors: Optional[np.ndarray] = None,
                       mesh_file: Optional[str] = None):
        """
        注册物体模型
        
        Args:
            object_id: 物体ID
            model_points_3d: 3D模型点云 (N, 3)
            keypoints_3d: 3D关键点 (M, 3)
            descriptors: 关键点描述子 (M, D)
            mesh_file: 网格文件路径
        """
        object_model = {
            'id': object_id,
            'points_3d': model_points_3d,
            'keypoints_3d': keypoints_3d,
            'descriptors': descriptors,
            'mesh_file': mesh_file,
            'registered_at': time.time()
        }
        
        self.object_models[object_id] = object_model
        
        logger.info(f"Object registered: {object_id} with {len(model_points_3d)} points")

    def register_simple_box(self, object_id: str, size: Tuple[float, float, float]):
        """注册简易盒子模型（仅用于PnP快速估计）"""
        if object_id in self.object_models:
            return
        w, h, d = size
        # 使用矩形平面四角作为3D点（Z=0），与检测框角点对应
        points_3d = np.array([
            [0.0, 0.0, 0.0],
            [w, 0.0, 0.0],
            [w, h, 0.0],
            [0.0, h, 0.0]
        ], dtype=np.float32)
        self.register_object(object_id, points_3d)

    def ensure_simple_model(self, object_id: str):
        """确保存在简易模型，缺失时注册默认尺寸"""
        if object_id in self.object_models:
            return
        obj = (object_id or "object").lower()
        size = self._simple_sizes.get(obj, (0.10, 0.10, 0.02))
        self.register_simple_box(object_id, size)
    
    def estimate_pose(self, image: np.ndarray,
                     depth_image: Optional[np.ndarray] = None,
                     object_id: Optional[str] = None,
                     detection_bbox: Optional[List[float]] = None) -> Optional[PoseEstimation]:
        """
        估计物体姿态
        
        Args:
            image: 彩色图像
            depth_image: 深度图像（可选）
            object_id: 物体ID
            detection_bbox: 检测边界框 [x1, y1, x2, y2]
            
        Returns:
            姿态估计结果
        """
        start_time = time.time()
        
        obj_id = object_id or "object"

        if self.method in ('pnp', 'epnp', 'solvepnp'):
            if obj_id not in self.object_models:
                if obj_id not in self._missing_model_warned:
                    logger.warning(f"Object {obj_id} not registered, using stub pose")
                    self._missing_model_warned.add(obj_id)
                pose = self._estimate_pose_stub(image, detection_bbox, obj_id, depth_image)
            elif self.method == 'epnp':
                pose = self._estimate_pose_epnp(image, obj_id, detection_bbox)
            else:
                pose = self._estimate_pose_pnp(image, obj_id, detection_bbox)
        elif self.method == 'deep_learning':
            pose = self._estimate_pose_deep_learning(image, depth_image, detection_bbox)
        elif self.method == 'point_pair':
            pose = self._estimate_pose_ppf(image, depth_image, obj_id)
        elif self.method == 'icp':
            pose = self._estimate_pose_icp(image, depth_image, obj_id)
        elif self.method == 'stub':
            pose = self._estimate_pose_stub(image, detection_bbox, obj_id, depth_image)
        else:
            raise ValueError(f"Unknown pose estimation method: {self.method}")
        
        if pose:
            pose.timestamp = time.time()
            
            # 记录计算时间
            computation_time = time.time() - start_time
            logger.debug(f"Pose estimation time: {computation_time*1000:.1f}ms")
        
        return pose

    def _estimate_pose_stub(self, image: np.ndarray,
                            bbox: Optional[List[float]],
                            object_id: str,
                            depth_image: Optional[np.ndarray] = None) -> PoseEstimation:
        """生成占位姿态（无模型时），有深度则优先用深度补偿平移"""
        h, w = image.shape[:2]
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
        else:
            cx, cy = w / 2.0, h / 2.0

        # --- 深度辅助的位姿占位估计 ---
        # 若深度图可用，则在检测框中心附近取稳健深度，并用内参反投影得到相机坐标
        if depth_image is not None and self.camera_matrix is not None:
            depth = self._sample_depth(depth_image, int(cx), int(cy), window=5)
            if depth is not None:
                fx = float(self.camera_matrix[0, 0])
                fy = float(self.camera_matrix[1, 1])
                cxi = float(self.camera_matrix[0, 2])
                cyi = float(self.camera_matrix[1, 2])
                z = float(depth)
                tx = (cx - cxi) * z / fx if fx else 0.0
                ty = (cy - cyi) * z / fy if fy else 0.0
                tz = z
                return PoseEstimation(
                    position=np.array([tx, ty, tz], dtype=np.float32),
                    rotation=np.eye(3, dtype=np.float32),
                    confidence=0.2,
                    object_id=object_id,
                    object_class=object_id.split('_')[0] if '_' in object_id else object_id,
                    timestamp=time.time(),
                    method='stub_depth'
                )

        # --- 没有深度时使用归一化占位 ---
        # 保持原逻辑，避免完全失效
        tx = (cx - w / 2.0) / max(1.0, w)
        ty = (cy - h / 2.0) / max(1.0, h)
        tz = 0.5

        return PoseEstimation(
            position=np.array([tx, ty, tz], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
            confidence=0.0,
            object_id=object_id,
            object_class=object_id.split('_')[0] if '_' in object_id else object_id,
            timestamp=time.time(),
            method='stub'
        )

    def _sample_depth(self, depth_image: np.ndarray, x: int, y: int, window: int = 5) -> Optional[float]:
        """在像素邻域内取稳健深度值（忽略0和NaN），返回米制深度"""
        if depth_image is None:
            return None
        h, w = depth_image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        half = max(1, window // 2)
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)

        patch = depth_image[y1:y2, x1:x2].astype(np.float32)
        # 过滤掉无效值（0或NaN），避免错误深度
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        return float(np.median(patch))
    
    def _estimate_pose_pnp(self, image: np.ndarray,
                          object_id: str,
                          bbox: Optional[List[float]] = None) -> Optional[PoseEstimation]:
        """使用PnP方法估计姿态"""
        if object_id not in self.object_models:
            return None
        
        object_model = self.object_models[object_id]
        
        # 提取图像特征
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints_2d, descriptors_2d = self.feature_detector.detectAndCompute(gray, None)
        
        if keypoints_2d is None or len(keypoints_2d) < 4:
            logger.warning(f"Insufficient keypoints detected: {len(keypoints_2d)}")
            return None
        
        # 匹配特征
        if object_model['descriptors'] is not None:
            matches = self._match_features(descriptors_2d, object_model['descriptors'])
            
            if len(matches) < 4:
                logger.warning(f"Insufficient matches: {len(matches)}")
                return None
            
            # 获取匹配的2D-3D点对
            points_2d = np.array([keypoints_2d[m.queryIdx].pt for m in matches], dtype=np.float32)
            points_3d = object_model['keypoints_3d'][[m.trainIdx for m in matches]]
        else:
            # 使用检测边界框
            if bbox is None:
                logger.error("Bounding box required for PnP without keypoints")
                return None
            
            # 简化的假设：边界框角点对应3D模型角点
            points_2d = self._get_bbox_corners(bbox)
            points_3d = self._get_model_corners(object_model['points_3d'])
        
        # 使用PnP求解姿态
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            logger.warning("PnP failed")
            return None
        
        # 转换旋转向量到旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # 计算重投影误差作为置信度
        reprojected_points, _ = cv2.projectPoints(
            points_3d, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        reprojection_error = np.mean(np.linalg.norm(points_2d - reprojected_points.squeeze(), axis=1))
        confidence = max(0.0, 1.0 - reprojection_error / 10.0)  # 简单的置信度计算
        
        return PoseEstimation(
            position=tvec.squeeze(),
            rotation=rotation_matrix,
            confidence=confidence,
            object_id=object_id,
            object_class=object_id.split('_')[0] if '_' in object_id else object_id,
            timestamp=time.time(),
            method='pnp'
        )
    
    def _estimate_pose_epnp(self, image: np.ndarray,
                           object_id: str,
                           bbox: Optional[List[float]] = None) -> Optional[PoseEstimation]:
        """使用EPnP方法估计姿态"""
        # 实现类似于PnP，但使用EPnP算法
        if object_id not in self.object_models:
            return None
        
        object_model = self.object_models[object_id]
        
        # 提取特征和匹配
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints_2d, descriptors_2d = self.feature_detector.detectAndCompute(gray, None)
        
        if keypoints_2d is None or len(keypoints_2d) < 6:
            return None
        
        if object_model['descriptors'] is not None:
            matches = self._match_features(descriptors_2d, object_model['descriptors'])
            
            if len(matches) < 6:
                return None
            
            points_2d = np.array([keypoints_2d[m.queryIdx].pt for m in matches], dtype=np.float32)
            points_3d = object_model['keypoints_3d'][[m.trainIdx for m in matches]]
        else:
            if bbox is None:
                return None
            
            points_2d = self._get_bbox_corners(bbox)
            points_3d = self._get_model_corners(object_model['points_3d'])
        
        # 使用EPnP
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            return None
        
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        return PoseEstimation(
            position=tvec.squeeze(),
            rotation=rotation_matrix,
            confidence=0.8,  # 简化的置信度
            object_id=object_id,
            object_class=object_id.split('_')[0] if '_' in object_id else object_id,
            timestamp=time.time(),
            method='epnp'
        )
    
    def _estimate_pose_deep_learning(self, image: np.ndarray,
                                    depth_image: Optional[np.ndarray],
                                    bbox: Optional[List[float]] = None) -> Optional[PoseEstimation]:
        """使用深度学习估计姿态"""
        if self.pose_net is None:
            logger.error("Deep learning model not loaded")
            return None
        
        # 预处理图像
        input_tensor = self._preprocess_image_for_deep_learning(image, bbox)
        
        # 推理
        with torch.no_grad():
            predictions = self.pose_net(input_tensor)
        
        # 解析预测结果
        # 这里需要根据具体的模型架构进行解析
        # 简化实现
        position = predictions[0, :3].numpy()
        rotation_quat = predictions[0, 3:7].numpy()
        
        # 转换四元数到旋转矩阵
        rotation_matrix = self._quaternion_to_rotation_matrix(rotation_quat)
        
        return PoseEstimation(
            position=position,
            rotation=rotation_matrix,
            confidence=float(predictions[0, 7]),  # 假设第7个值是置信度
            object_id='unknown',
            object_class='unknown',
            timestamp=time.time(),
            method='deep_learning'
        )
    
    def _estimate_pose_ppf(self, image: np.ndarray,
                          depth_image: np.ndarray,
                          object_id: str) -> Optional[PoseEstimation]:
        """使用点对特征方法估计姿态"""
        if depth_image is None:
            logger.error("Depth image required for PPF method")
            return None
        
        if object_id not in self.object_models:
            return None
        
        # 创建点云
        scene_cloud = self._depth_to_pointcloud(depth_image)
        model_cloud = self.object_models[object_id]['points_3d']
        
        # 使用PPF匹配
        # 注意：这需要更复杂的实现
        # 简化实现
        logger.warning("PPF method not fully implemented")
        
        return None
    
    def _estimate_pose_icp(self, image: np.ndarray,
                          depth_image: np.ndarray,
                          object_id: str) -> Optional[PoseEstimation]:
        """使用ICP方法估计姿态"""
        if depth_image is None:
            logger.error("Depth image required for ICP method")
            return None
        
        if object_id not in self.object_models:
            return None
        
        # 创建点云
        scene_cloud = self._depth_to_pointcloud(depth_image)
        model_cloud = self.object_models[object_id]['points_3d']
        
        # 使用ICP对齐点云
        # 简化实现
        logger.warning("ICP method not fully implemented")
        
        return None
    
    def _match_features(self, descriptors1: np.ndarray, 
                       descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """匹配特征"""
        if descriptors1.dtype == np.uint8:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches[:50]  # 返回前50个最佳匹配
    
    def _get_bbox_corners(self, bbox: List[float]) -> np.ndarray:
        """获取边界框角点"""
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
    
    def _get_model_corners(self, points_3d: np.ndarray) -> np.ndarray:
        """获取3D模型角点"""
        if len(points_3d) <= 4:
            return points_3d[:4]
        
        # 计算边界框
        min_vals = np.min(points_3d, axis=0)
        max_vals = np.max(points_3d, axis=0)
        
        corners = np.array([
            [min_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], max_vals[1], max_vals[2]],
            [min_vals[0], max_vals[1], max_vals[2]]
        ], dtype=np.float32)
        
        return corners
    
    def _preprocess_image_for_deep_learning(self, image: np.ndarray,
                                           bbox: Optional[List[float]] = None) -> torch.Tensor:
        """为深度学习预处理图像"""
        # 如果有边界框，裁剪图像
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            image = image[y1:y2, x1:x2]
        
        # 调整大小
        image_resized = cv2.resize(image, (224, 224))
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
    
    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        x, y, z, w = quaternion
        
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)
        
        return rotation_matrix
    
    def _depth_to_pointcloud(self, depth_image: np.ndarray) -> np.ndarray:
        """深度图转点云"""
        height, width = depth_image.shape
        
        # 创建像素坐标网格
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # 转换为3D坐标
        z = depth_image.astype(np.float32) / 1000.0  # 假设深度单位为毫米
        
        # 使用相机内参
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 展平并组合
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # 移除无效点（深度为0）
        valid_mask = z.flatten() > 0
        points = points[valid_mask]
        
        return points
    
    def draw_pose_axes(self, image: np.ndarray,
                      pose: PoseEstimation,
                      length: float = 0.1) -> np.ndarray:
        """
        在图像上绘制姿态轴
        
        Args:
            image: 输入图像
            pose: 姿态估计结果
            length: 轴长度（米）
            
        Returns:
            绘制了姿态轴的图像
        """
        if pose.rotation.shape != (3, 3):
            rotation_matrix = self._quaternion_to_rotation_matrix(pose.rotation)
        else:
            rotation_matrix = pose.rotation
        
        # 轴端点
        axes_points_3d = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ], dtype=np.float32)
        
        # 变换到相机坐标系
        axes_points_camera = rotation_matrix @ axes_points_3d.T
        axes_points_camera = axes_points_camera.T + pose.position
        
        # 投影到图像平面
        axes_points_2d, _ = cv2.projectPoints(
            axes_points_camera,
            np.zeros((3, 1)),  # 旋转向量
            np.zeros((3, 1)),  # 平移向量
            self.camera_matrix,
            self.dist_coeffs
        )
        
        axes_points_2d = axes_points_2d.squeeze().astype(int)
        
        # 绘制轴
        origin = tuple(axes_points_2d[0])
        
        # X轴 (红色)
        cv2.line(image, origin, tuple(axes_points_2d[1]), (0, 0, 255), 2)
        
        # Y轴 (绿色)
        cv2.line(image, origin, tuple(axes_points_2d[2]), (0, 255, 0), 2)
        
        # Z轴 (蓝色)
        cv2.line(image, origin, tuple(axes_points_2d[3]), (255, 0, 0), 2)
        
        return image
    
    def get_pose_statistics(self, pose: PoseEstimation) -> Dict[str, Any]:
        """获取姿态统计信息"""
        if pose.rotation.shape == (3, 3):
            euler_angles = rotation_matrix_to_euler(pose.rotation)
            quaternion = quaternion_from_matrix(pose.rotation)
        else:
            # 假设已经是四元数
            quaternion = pose.rotation
            # 需要从四元数计算欧拉角
        
        return {
            'position': pose.position.tolist(),
            'position_norm': float(np.linalg.norm(pose.position)),
            'rotation_euler': euler_angles.tolist(),
            'rotation_quaternion': quaternion.tolist(),
            'confidence': pose.confidence,
            'object_id': pose.object_id,
            'method': pose.method,
            'timestamp': pose.timestamp
        }
