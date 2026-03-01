"""
相机配置文件
用于配置RGB相机、深度相机和视觉处理参数
"""

from typing import Any, Dict, Optional

class CameraConfig:
    """相机配置类"""
    
    # ========== 相机硬件配置 ==========
    HARDWARE = {
        'primary_camera': {
            'type': 'realsense',  # 'opencv', 'realsense', 'simulation'
            'camera_id': 0,  # 设备ID或序列号（RealSense可填序列号）
            'serial': '',  # RealSense序列号（可选，优先于camera_id）
            # D455/D455F: RGB 最大分辨率 1280x800 @30fps
            'resolution': [1280, 800],  # [宽度, 高度]
            'fps': 15,
            # D455 RGB FOV 约 90°x65°，用于 fx/fy 估算
            'fov': 90,
            'auto_exposure': True,
            'exposure_value': 100,
            'white_balance': 4600,
            'enable_depth': True,  # None=自动（RealSense默认开启）
            'color_format': 'rgb8',  # RealSense颜色格式：rgb8/bgr8/yuyv
            'max_failures': 5,
        },
        'secondary_camera': {
            'type': 'opencv',
            'camera_id': 1,
            'resolution': [640, 480],
            'fps': 30,
        },
        'depth_camera': {
            'enabled': None,  # None=自动（RealSense默认开启）
            'type': 'realsense',
            # D455 深度最大分辨率 1280x720（高帧率需更低分辨率）
            'depth_resolution': [1280, 720],
            'depth_units': 0.001,  # 米
            'depth_scale': 1000.0,
            # 深度最小距离：根据硬件规格（52cm）
            'depth_clipping': [0.52, 2.0],  # 米，深度范围
            'depth_format': 'z16',  # RealSense深度格式：z16/disparity16
        },
    }
    
    # ========== 相机内参（默认值，建议实际标定） ==========
    INTRINSICS = {
        'primary_camera': {
            'fx': 612.0,  # 焦距x
            'fy': 612.0,  # 焦距y
            'cx': 320.0,  # 主点x
            'cy': 240.0,  # 主点y
            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0],  # k1, k2, p1, p2, k3
            'camera_matrix': [
                [612.0, 0.0, 320.0],
                [0.0, 612.0, 240.0],
                [0.0, 0.0, 1.0]
            ],
        },
    }
    
    # ========== 相机外参（手眼标定结果） ==========
    EXTRINSICS = {
        'camera_to_robot': {
            'translation': [0.1, 0.0, 0.5],  # 米 [x, y, z]
            'rotation': [0.0, 0.0, 0.0, 1.0],  # 四元数 [x, y, z, w]
            'euler_angles': [0.0, 0.0, 0.0],  # 欧拉角（度） [roll, pitch, yaw]
            'matrix': [  # 4x4变换矩阵
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0]
            ],
        },
    }
    
    # ========== 图像处理配置 ==========
    PROCESSING = {
        'preprocessing': {
            'undistort': True,
            'crop_region': [100, 100, 1000, 620],  # [x, y, width, height]
            'resize': [640, 480],
            'normalization': 'zero_mean',  # 'zero_mean', 'minmax', 'none'
            'color_space': 'rgb',  # 'rgb', 'hsv', 'lab', 'grayscale'
            'histogram_equalization': False,
        },
        'filtering': {
            'gaussian_blur': {
                'enabled': True,
                'kernel_size': [3, 3],
                'sigma': 1.0,
            },
            'median_blur': {
                'enabled': False,
                'kernel_size': 3,
            },
            'bilateral_filter': {
                'enabled': False,
                'd': 9,
                'sigma_color': 75,
                'sigma_space': 75,
            },
        },
        'segmentation': {
            'threshold_method': 'adaptive',  # 'global', 'adaptive', 'otsu'
            'threshold_value': 127,
            'morphology_operations': {
                'erosion': True,
                'erosion_kernel': [3, 3],
                'dilation': True,
                'dilation_kernel': [3, 3],
            },
        },
    }
    
    # ========== 特征提取配置 ==========
    FEATURES = {
        'detector': 'orb',  # 'sift', 'surf', 'orb', 'akaze', 'brisk'
        'detector_params': {
            'n_features': 1000,
            'scale_factor': 1.2,
            'n_levels': 8,
        },
        'descriptor': 'orb',
        'matcher': 'bf',  # 'bf' (BruteForce), 'flann'
        'matcher_params': {
            'norm_type': 'hamming',
            'cross_check': True,
        },
    }
    
    # ========== 深度处理配置 ==========
    DEPTH = {
        'point_cloud': {
            'enabled': True,
            'voxel_size': 0.005,  # 米，体素下采样大小
            'normal_estimation_radius': 0.03,  # 米
            # 多视角融合参数
            'fusion_max_correspondence': 0.05,
            'fusion_icp_iterations': 30,
            'fusion_min_interval': 1.0 / 15.0,
            'fusion_max_points': 200000,
            'use_color': True,
            # ROI/检测框相关设置
            'roi_use_detection': True,
            'roi_min_confidence': 0.5,
            'roi_padding': 12,
            'roi_sample_step': 2,  # ROI 采样步长（步长越大点越稀疏）
            'roi_strategy': 'highest_confidence',  # highest_confidence / largest
            'roi_max_age': 0.5,  # 检测结果最大有效时间（秒）
            'render_max_points': 20000,  # UI 最大渲染点数
            'reset_on_request': True,  # 每次请求时重置点云缓存
            'statistical_outlier_removal': {
                'enabled': True,
                'nb_neighbors': 20,
                'std_ratio': 2.0,
            },
        },
        'alignment': {
            'align_depth_to_color': True,
            'temporal_filter': {
                'enabled': True,
                'persistency_index': 3,
            },
            'spatial_filter': {
                'enabled': True,
                'magnitude': 2,
                'smooth_alpha': 0.5,
            },
        },
    }
    
    # ========== 目标检测配置 ==========
    OBJECT_DETECTION = {
        'model_type': 'yolov8',  # 'yolov8', 'yolov5', 'ssd', 'faster_rcnn', 'custom'
        # 使用本地已存在的权重文件，默认放在 models 目录下
        'model_path': '',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'classes': [
            'cube', 'cylinder', 'sphere', 'box', 'bottle', 'cup'
        ],
        'input_size': [640, 640],
    }

    # ========== 材质识别配置 ==========
    MATERIAL_RECOGNITION = {
        'enabled': True,
        'model_type': 'resnet50',  # 预留
        # 若提供权重将使用模型推理；为空则使用启发式估计
        'model_path': '',
        # 自定义材质类别（可选）
        'classes': [
            'metal', 'plastic', 'glass', 'ceramic',
            'wood', 'fabric', 'paper', 'rubber'
        ],
        'use_object_prior': True,
    }
    
    # ========== 姿态估计配置 ==========
    POSE_ESTIMATION = {
        'method': 'pnp',  # 'pnp', 'deep_learning', 'template_matching'
        'pnp_method': 'iterative',  # 'iterative', 'epnp', 'sqpnp'
        'use_ransac': True,
        'ransac_threshold': 2.0,
        'iterations_count': 100,
        'reprojection_error': 5.0,
    }
    
    # ========== 标定配置 ==========
    CALIBRATION = {
        'chessboard_pattern': [9, 6],  # 内角点数量 [列, 行]
        'square_size': 0.024,  # 米
        'calibration_images': 20,
        'auto_capture': True,
        'save_path': 'calibration/camera_calibration.yaml',
    }

    def __init__(self):
        primary = self.HARDWARE['primary_camera']
        self.camera_type = primary.get('type', 'opencv')
        self.camera_index = primary.get('camera_id', 0)
        self.serial = primary.get('serial', None)
        self.width, self.height = primary.get('resolution', [640, 480])
        self.fps = primary.get('fps', 30)
        self.fov = primary.get('fov', None)
        self.auto_exposure = primary.get('auto_exposure', True)
        self.exposure_value = primary.get('exposure_value', 100)
        self.white_balance = primary.get('white_balance', 4600)
        self.color_format = primary.get('color_format', 'rgb8')
        self.max_failures = primary.get('max_failures', 5)

        # 允许使用 camera_id 字符串作为 RealSense 序列号
        if not self.serial and isinstance(self.camera_index, str):
            serial_str = self.camera_index.strip()
            if serial_str:
                self.serial = serial_str

        # 内参
        intr = self.INTRINSICS.get('primary_camera', {})
        self.fx = intr.get('fx')
        self.fy = intr.get('fy')
        self.cx = intr.get('cx')
        self.cy = intr.get('cy')
        self.distortion_coeffs = intr.get('distortion', [])

        # 深度相关
        depth_cfg = self.HARDWARE.get('depth_camera', {})
        depth_enabled = depth_cfg.get('enabled', None)
        if depth_enabled is None:
            # RealSense 默认开启深度，USB/仿真默认关闭
            depth_enabled = str(self.camera_type).lower() == 'realsense'
        self.enable_depth = depth_enabled
        self.depth_width, self.depth_height = depth_cfg.get('depth_resolution', [0, 0])
        self.depth_min, self.depth_max = depth_cfg.get('depth_clipping', [0.1, 3.0])
        self.depth_units = depth_cfg.get('depth_units', 0.001)
        self.depth_scale = depth_cfg.get('depth_scale', 1000.0)
        self.depth_format = depth_cfg.get('depth_format', 'z16')

        # 深度对齐设置来自 DEPTH 配置（用于RealSense对齐）
        align_cfg = self.DEPTH.get('alignment', {})
        self.align_depth_to_color = align_cfg.get('align_depth_to_color', True)

        # Object detection defaults for UI bindings
        det_cfg = self.OBJECT_DETECTION
        self.confidence_threshold = det_cfg.get('confidence_threshold', 0.5)
        self.nms_threshold = det_cfg.get('iou_threshold', 0.45)
        self.detection_model = self._map_detection_model(det_cfg.get('model_type', 'yolov5'))

        # 材质识别
        mat_cfg = self.MATERIAL_RECOGNITION
        self.material_enabled = mat_cfg.get('enabled', True)
        self.material_model_path = mat_cfg.get('model_path', '')
        self.material_model_type = mat_cfg.get('model_type', 'resnet50')
        self.material_classes = mat_cfg.get('classes', None)
        self.material_use_object_prior = mat_cfg.get('use_object_prior', True)

    def _map_detection_model(self, model_type: Optional[str]) -> str:
        model_key = str(model_type or "").lower()
        model_map = {
            "yolov5": "YOLOv5",
            "yolov8": "YOLOv8",
            "ssd": "SSD",
            "faster_rcnn": "Faster R-CNN",
            "custom": "自定义模型",
        }
        return model_map.get(model_key, "YOLOv5")

    @classmethod
    def from_dict(cls, camera_dict: Optional[Dict[str, Any]]):
        """
        从字典创建CameraConfig（用于读取DemoConfig中的camera字段）。

        注意：此方法只覆盖常用字段，未提供的字段保留默认值。
        """
        cfg = cls()
        cfg.update_from_dict(camera_dict or {})
        return cfg

    def update_from_dict(self, camera_dict: Dict[str, Any]):
        """
        根据相机配置字典更新当前实例。

        该方法用于兼容 JSON/YAML 中的 camera 配置结构，避免直接依赖类常量。
        """
        if not camera_dict:
            return

        hardware = camera_dict.get('HARDWARE', {})
        primary = hardware.get('primary_camera', {})
        depth_cfg = hardware.get('depth_camera', {})

        # 1) 主相机硬件字段
        depth_enabled_specified = False
        if 'type' in primary:
            self.camera_type = primary.get('type', self.camera_type)
        if 'camera_id' in primary:
            self.camera_index = primary.get('camera_id', self.camera_index)
        if 'serial' in primary:
            self.serial = primary.get('serial', self.serial)
        resolution = primary.get('resolution')
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            self.width, self.height = int(resolution[0]), int(resolution[1])
        if 'fps' in primary:
            self.fps = int(primary.get('fps', self.fps))
        if 'fov' in primary:
            self.fov = primary.get('fov', self.fov)
        if 'auto_exposure' in primary:
            self.auto_exposure = bool(primary.get('auto_exposure', self.auto_exposure))
        if 'exposure_value' in primary:
            self.exposure_value = primary.get('exposure_value', self.exposure_value)
        if 'white_balance' in primary:
            self.white_balance = primary.get('white_balance', self.white_balance)
        if 'enable_depth' in primary:
            self.enable_depth = primary.get('enable_depth')
            depth_enabled_specified = True
        if 'color_format' in primary:
            self.color_format = primary.get('color_format', self.color_format)
        if 'max_failures' in primary:
            self.max_failures = primary.get('max_failures', self.max_failures)

        # 允许使用 camera_id 字符串作为 RealSense 序列号
        if not self.serial and isinstance(self.camera_index, str):
            serial_str = self.camera_index.strip()
            if serial_str:
                self.serial = serial_str

        # 2) 深度相机字段
        if 'enabled' in depth_cfg:
            self.enable_depth = depth_cfg.get('enabled', self.enable_depth)
            depth_enabled_specified = True
        depth_resolution = depth_cfg.get('depth_resolution')
        if isinstance(depth_resolution, (list, tuple)) and len(depth_resolution) >= 2:
            self.depth_width, self.depth_height = int(depth_resolution[0]), int(depth_resolution[1])
        depth_clipping = depth_cfg.get('depth_clipping')
        if isinstance(depth_clipping, (list, tuple)) and len(depth_clipping) >= 2:
            self.depth_min, self.depth_max = float(depth_clipping[0]), float(depth_clipping[1])
        if 'depth_units' in depth_cfg:
            self.depth_units = depth_cfg.get('depth_units', self.depth_units)
        if 'depth_scale' in depth_cfg:
            self.depth_scale = depth_cfg.get('depth_scale', self.depth_scale)
        if 'depth_format' in depth_cfg:
            self.depth_format = depth_cfg.get('depth_format', self.depth_format)

        # 3) 内参字段
        intr = camera_dict.get('INTRINSICS', {}).get('primary_camera', {})
        if 'fx' in intr:
            self.fx = intr.get('fx', self.fx)
        if 'fy' in intr:
            self.fy = intr.get('fy', self.fy)
        if 'cx' in intr:
            self.cx = intr.get('cx', self.cx)
        if 'cy' in intr:
            self.cy = intr.get('cy', self.cy)
        if 'distortion' in intr:
            self.distortion_coeffs = intr.get('distortion', self.distortion_coeffs)

        # 4) 深度对齐设置
        align_cfg = camera_dict.get('DEPTH', {}).get('alignment', {})
        if 'align_depth_to_color' in align_cfg:
            self.align_depth_to_color = align_cfg.get('align_depth_to_color', self.align_depth_to_color)

        det_cfg = camera_dict.get('OBJECT_DETECTION', {})
        if det_cfg:
            self.confidence_threshold = det_cfg.get('confidence_threshold', self.confidence_threshold)
            self.nms_threshold = det_cfg.get('iou_threshold', self.nms_threshold)
            self.detection_model = self._map_detection_model(det_cfg.get('model_type', self.detection_model))

        # 如果未显式配置深度开关，则按相机类型自动补齐
        if not depth_enabled_specified:
            self.enable_depth = str(self.camera_type).lower() == 'realsense'

def create_default_camera_config():
    """创建默认的相机配置"""
    return CameraConfig()


# 测试代码
if __name__ == "__main__":
    config = create_default_camera_config()
    print("相机配置测试:")
    print(f"主相机类型: {config.HARDWARE['primary_camera']['type']}")
    print(f"分辨率: {config.HARDWARE['primary_camera']['resolution']}")
