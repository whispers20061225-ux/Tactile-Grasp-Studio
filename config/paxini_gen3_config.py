"""
Paxini Gen3 M2020 传感器配置文件 - 优化版
专门用于 3x3 触觉传感器阵列的配置，优化模拟数据平滑度
支持三维力数据生成 (Fx, Fy, Fz) 和机械臂集成
"""

import yaml
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import random
import time
import numpy as np
import math

# 动态添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入 DemoConfig 和相关类
try:
    # 尝试从 config 包导入
    from config.demo_config import DemoConfig, SensorConfig, ServoConfig, HardwareConfig, AlgorithmConfig, UIConfig
except ImportError as e:
    print(f"导入 config.demo_config 失败: {e}")
    # 尝试直接导入（如果文件在同一目录）
    try:
        from demo_config import DemoConfig, SensorConfig, ServoConfig, HardwareConfig, AlgorithmConfig, UIConfig
    except ImportError:
        # 如果都失败，使用占位符类
        print("警告: 无法导入 DemoConfig，将使用替代配置")
        # 定义替代的配置类
        @dataclass
        class DemoConfig:
            hardware: Any = None
            algorithm: Any = None
            ui: Any = None
            integration: Any = None
            def __init__(self, config_path=None):
                pass
        
        @dataclass
        class SensorConfig:
            type: str = "default"
            port: str = "COM3"
            baudrate: int = 115200
        
        @dataclass
        class ServoConfig:
            type: str = "st3215"
            port: str = "COM4"
            baudrate: int = 115200
            min_angle: int = 0
            max_angle: int = 180
            speed: int = 50
        
        @dataclass
        class HardwareConfig:
            sensor: SensorConfig = field(default_factory=SensorConfig)
            servo: ServoConfig = field(default_factory=ServoConfig)
        
        @dataclass
        class AlgorithmConfig:
            data_dimensions: tuple = (3, 3)
            feature_size: int = 9
        
        @dataclass
        class UIConfig:
            window_width: int = 1200
            window_height: int = 800
            control_panel_width: int = 300
            data_refresh_rate: int = 30
            sensor_grid_size: tuple = (3, 3)
            show_tactile_grid: bool = True


class PaxiniModel(Enum):
    """Paxini Gen3 传感器型号"""
    M2020_3x3 = "M2020_3x3"  # 3x3 触觉阵列
    M2020_6x6 = "M2020_6x6"  # 6x6 触觉阵列
    M2020_9x9 = "M2020_9x9"  # 9x9 触觉阵列


class ArmIntegrationMode(Enum):
    """机械臂集成模式"""
    DISABLED = "disabled"  # 不集成
    SIMULATION = "simulation"  # 仿真模式
    REAL = "real"  # 真实机械臂
    HYBRID = "hybrid"  # 混合模式


@dataclass
class PaxiniSensorConfig(SensorConfig):
    """Paxini Gen3 传感器专用配置"""
    
    # Paxini 特定参数
    model: PaxiniModel = PaxiniModel.M2020_3x3
    num_taxels: int = 9  # 3x3 = 9 个触觉单元
    rows: int = 3
    cols: int = 3
    pressure_range: tuple = (0, 100)  # 压力范围 (kPa)
    resolution: float = 0.1  # 压力分辨率 (kPa)
    sampling_rate: int = 100  # 采样率 (Hz)
    
    # 三维力模拟参数
    force_vector_range: tuple = (-10.0, 10.0)  # 切向力范围 (N)
    normal_force_range: tuple = (0.0, 25.0)    # 法向力范围 (N)
    friction_coefficient: float = 0.3          # 摩擦系数，用于计算切向力
    force_noise_level: float = 0.1             # 力噪声水平 (N)
    max_shear_force: float = 5.0               # 最大剪切力 (N)
    
    # 机械臂集成参数
    arm_integration: ArmIntegrationMode = ArmIntegrationMode.DISABLED
    tool_center_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.1])  # TCP偏移
    sensor_to_tcp_transform: List[List[float]] = field(default_factory=lambda: 
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.1],
         [0.0, 0.0, 0.0, 1.0]])  # 4x4变换矩阵
    
    # 校准参数
    calibration_enabled: bool = True
    calibration_samples: int = 100
    auto_calibration: bool = True
    
    # 滤波参数
    filter_enabled: bool = True
    filter_type: str = "median"  # median, gaussian, kalman
    filter_window_size: int = 5
    
    # 数据处理参数
    normalize_data: bool = True
    remove_offset: bool = True
    dynamic_range_adjustment: bool = True
    
    # 新增：sensor_reader.py 需要的参数
    force_scale: float = 1.0  # 力值缩放因子，恢复为1.0
    max_pressure_range: float = 100.0  # 最大压力范围 (kPa)
    timeout: float = 0.1  # 串口超时
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保 Paxini 特定参数正确
        if self.model == PaxiniModel.M2020_3x3:
            self.num_taxels = 9
            self.rows = 3
            self.cols = 3
        elif self.model == PaxiniModel.M2020_6x6:
            self.num_taxels = 36
            self.rows = 6
            self.cols = 6
        elif self.model == PaxiniModel.M2020_9x9:
            self.num_taxels = 81
            self.rows = 9
            self.cols = 9


@dataclass
class SimulatedTactileData:
    """模拟触觉数据类，与 sensor_reader.py 中的 TactileData 兼容
    支持三维力数据 (Fx, Fy, Fz) 和机械臂集成
    """
    def __init__(self, force_vectors, timestamp, sequence=0, arm_state=None):
        """
        初始化模拟触觉数据
        
        Args:
            force_vectors: 每个触点的三维力向量列表，每个向量为 [Fx, Fy, Fz] (N)
            timestamp: 时间戳
            sequence: 序列号
            arm_state: 机械臂状态字典
        """
        self.timestamp = timestamp
        self.sequence = sequence
        
        # 将力向量转换为numpy数组，形状为 (num_taxels, 3)
        self.tactile_array = np.array(force_vectors, dtype=np.float32)
        
        # 计算合力向量
        self.resultant_force = np.sum(self.tactile_array, axis=0)
        
        # 接触状态：如果法向力大于阈值则认为接触
        self.contact_state = np.array([1 if f[2] > 0.5 else 0 for f in force_vectors], dtype=np.uint8)
        self.temperature = 25.0
        self.data_valid = True
        
        # 机械臂状态（如果提供）
        self.arm_state = arm_state
        
    @property
    def total_normal_force(self) -> float:
        """总法向力 (N)"""
        return np.sum(self.tactile_array[:, 2])
    
    @property
    def total_shear_force(self) -> float:
        """总剪切力大小 (N)"""
        shear_forces = self.tactile_array[:, :2]  # Fx, Fy
        return np.linalg.norm(np.sum(shear_forces, axis=0))
    
    @property
    def average_pressure(self) -> float:
        """平均压力 (kPa) - 转换为压力单位"""
        return np.mean(self.tactile_array[:, 2]) * 0.1  # 假设 1N = 0.1 kPa
    
    @property
    def max_normal_force(self) -> float:
        """最大法向力 (N)"""
        return np.max(self.tactile_array[:, 2])
    
    @property
    def min_normal_force(self) -> float:
        """最小法向力 (N)"""
        return np.min(self.tactile_array[:, 2])
    
    @property
    def force_magnitudes(self) -> np.ndarray:
        """每个触点的力大小 (N)"""
        return np.linalg.norm(self.tactile_array, axis=1)
    
    @property
    def force_directions(self) -> np.ndarray:
        """每个触点的力方向 (单位向量)"""
        magnitudes = self.force_magnitudes
        with np.errstate(divide='ignore', invalid='ignore'):
            directions = self.tactile_array / magnitudes[:, np.newaxis]
            directions[np.isnan(directions)] = 0
        return directions
    
    def transform_to_world_frame(self, tcp_to_world_transform):
        """
        将力数据转换到世界坐标系
        
        Args:
            tcp_to_world_transform: 4x4变换矩阵，从TCP到世界坐标系
            
        Returns:
            世界坐标系中的力数据
        """
        # 将合力向量转换到世界坐标系
        force_vector_world = np.dot(tcp_to_world_transform[:3, :3], self.resultant_force)
        return force_vector_world


@dataclass
class PaxiniGen3Config(DemoConfig):
    """Paxini Gen3 传感器完整配置 - 支持三维力数据模拟和机械臂集成"""
    
    def __init__(self, config_path: Optional[str] = None, 
                 model: PaxiniModel = PaxiniModel.M2020_3x3,
                 arm_integration: bool = False):
        """
        初始化 Paxini Gen3 配置
        
        Args:
            config_path: 配置文件路径
            model: 传感器型号
            arm_integration: 是否启用机械臂集成
        """
        # 初始化日志器
        import logging
        self.logger = logging.getLogger(__name__)
        
        # 三维力模拟参数
        self._smooth_factor = 0.8  # 平滑因子，0-1之间，越大越平滑
        self._prev_force_data = None  # 上一帧的力数据
        self._noise_level = 0.5  # 噪声水平
        self._sim_update_rate = 0.033  # 更新间隔（秒），30Hz
        
        # 三维力模拟状态
        self._grasp_force = 0.0  # 当前抓取力
        self._shear_angle = 0.0  # 剪切力角度
        self._object_weight = 1.0  # 物体重量 (kg)
        self._friction_coeff = 0.3  # 摩擦系数
        
        # 机械臂集成状态
        self._arm_integration = arm_integration
        self._arm_state = None
        self._tcp_to_world_transform = np.eye(4)
        
        if config_path:
            # 从文件加载配置
            super().__init__(config_path)
        else:
            # 创建默认配置，然后更新为 Paxini 专用配置
            super().__init__()
            
            # 更新传感器配置
            self.hardware = HardwareConfig()
            self.algorithm = AlgorithmConfig()
            self.ui = UIConfig()
            
            # 创建传感器配置 - 使用修正的参数
            self.hardware.sensor = PaxiniSensorConfig(
                type="paxini_gen3",
                model=model,
                port="COM3",
                baudrate=115200,
                num_taxels=9 if model == PaxiniModel.M2020_3x3 else 36,
                rows=3 if model == PaxiniModel.M2020_3x3 else 6,
                cols=3 if model == PaxiniModel.M2020_3x3 else 6,
                sampling_rate=100,
                filter_enabled=True,
                filter_type="median",
                force_scale=1.0,
                max_pressure_range=100.0,
                friction_coefficient=0.3,
                force_noise_level=0.1,
                max_shear_force=5.0,
                arm_integration=ArmIntegrationMode.REAL if arm_integration else ArmIntegrationMode.DISABLED
            )
            
            # 创建舵机配置
            self.hardware.servo = ServoConfig(
                type="st3215",
                port="COM4",
                baudrate=115200,
                min_angle=0,
                max_angle=180,
                speed=50
            )
            
            # 更新算法配置以适应 3x3 传感器
            self.algorithm.data_dimensions = (3, 3)
            self.algorithm.feature_size = 9
            self.algorithm.use_spatial_features = True
            self.algorithm.use_temporal_features = True
            self.algorithm.force_dimensions = 3  # 三维力数据
            
            # 如果启用机械臂集成，添加相关配置
            if arm_integration:
                # 创建简单的机械臂配置字典，而不是导入
                self.learm_arm = {
                    'CONNECTION': {'type': 'simulation'},
                    'PHYSICAL': {'dof': 6},
                    'SAFETY': {'workspace_limits': {'x': [-0.9, 0.9], 'y': [-0.9, 0.9], 'z': [0, 0.9]}}
                }
                
                # 创建简单的相机配置字典
                self.camera = {
                    'HARDWARE': {
                        'primary_camera': {
                            'type': 'realsense',
                            'resolution': [1280, 720]
                        }
                    }
                }
                
                # 更新演示模式 - 确保 demo_modes 是字典
                if not hasattr(self, 'demo_modes'):
                    self.demo_modes = {}
                
                self.demo_modes["arm_grasping"] = {
                    "enabled": True,
                    "grasp_strategy": "force_closure",
                    "visual_feedback": True,
                    "force_feedback": True,
                    "adaptive_control": True
                }
            
            # 现在安全地添加 tactile_mapping
            if not hasattr(self, 'demo_modes'):
                self.demo_modes = {}
            
            self.demo_modes["tactile_mapping"] = {
                "enabled": True,
                "grid_size": (3, 3),
                "visualization": "heatmap",
                "vector_field": True  # 新增：启用矢量场显示
            }
            
            # 更新界面配置以显示 3x3 网格和矢量图
            self.ui.sensor_grid_size = (3, 3)
            self.ui.show_tactile_grid = True
            self.ui.show_vector_field = True  # 新增：显示矢量图
            
            # 如果启用机械臂集成，更新UI配置
            if arm_integration:
                self.ui.arm_ui = {
                    'show_arm_status': True,
                    'show_joint_states': True,
                    'show_trajectory': True,
                    'show_workspace': True,
                }
                self.ui.vision_ui = {
                    'show_camera_view': True,
                    'show_point_cloud': False,
                    'show_detection_results': True,
                    'camera_view_size': [640, 480],
                }
        
        # 标记为 Paxini 配置
        self.sensor_type = "paxini_gen3"
        self.sensor_model = model
        self._last_sim_time = 0
        self._sim_position = 90
        self._sim_target = 90
        self._sim_speed = 0.5
        self._sim_mode = "holding"
        self._sim_mode_timer = 0
        self._grasp_cycle = 0  # 抓取周期计数器
        
        # 三维力模拟参数
        self._base_normal_pattern = self._generate_base_normal_pattern()  # 基础法向力模式
        self._contact_points = []  # 接触点索引列表
        self._object_pose = np.array([0.0, 0.0, 0.0])  # 物体姿态 [x, y, rotation]
        
        # 机械臂模拟状态
        self._arm_joint_positions = [0.0] * 6  # 6轴机械臂
        self._arm_cartesian_pose = np.eye(4)  # 4x4位姿矩阵
        self._grasp_object_id = None  # 当前抓取的物体ID

    def _generate_base_normal_pattern(self):
        """生成基础法向力模式"""
        rows = self.hardware.sensor.rows
        cols = self.hardware.sensor.cols
        num_taxels = self.hardware.sensor.num_taxels
        
        base_pattern = np.zeros((num_taxels, 3))  # 三维力
        
        # 创建更自然的基础模式
        for i in range(num_taxels):
            row_idx = i // cols
            col_idx = i % cols
            
            # 中心区域压力较高，边缘较低
            center_row = (rows - 1) / 2.0
            center_col = (cols - 1) / 2.0
            distance_to_center = ((row_idx - center_row) ** 2 + (col_idx - center_col) ** 2) ** 0.5
            max_distance = ((rows-1) ** 2 + (cols-1) ** 2) ** 0.5 / 2.0
            
            # 根据距离设置基础法向力
            if distance_to_center <= max_distance / 3:
                normal_force = 8.0  # 中心区域，单位N
            elif distance_to_center <= max_distance * 2/3:
                normal_force = 5.0  # 中间区域
            else:
                normal_force = 2.0  # 边缘区域
            
            # 初始只有法向力（Z方向）
            base_pattern[i] = [0.0, 0.0, normal_force]
        
        return base_pattern

    def _calculate_contact_forces(self, position, object_pose, arm_pose=None):
        """
        根据物理模型计算接触力，支持机械臂集成
        
        Args:
            position: 夹爪位置 (角度)
            object_pose: 物体姿态 [x, y, rotation]
            arm_pose: 机械臂末端位姿 (4x4矩阵)，可选
            
        Returns:
            每个触点的三维力向量列表
        """
        rows = self.hardware.sensor.rows
        cols = self.hardware.sensor.cols
        num_taxels = self.hardware.sensor.num_taxels
        
        # 根据夹爪位置计算抓取力
        # 假设夹爪位置与抓取力成正比（0-180度对应0-20N）
        grip_force = position / 180.0 * 20.0
        
        # 更新抓取力状态
        self._grasp_force = grip_force
        
        # 计算物体重量产生的力分布
        object_weight_force = self._object_weight * 9.8  # N
        
        # 初始化力向量
        force_vectors = np.zeros((num_taxels, 3))
        
        # 如果提供机械臂位姿，考虑重力方向
        gravity_direction = np.array([0.0, 0.0, -1.0])  # 默认Z向下
        if arm_pose is not None:
            # 将重力方向转换到传感器坐标系
            gravity_sensor = np.dot(arm_pose[:3, :3].T, gravity_direction)
        else:
            gravity_sensor = gravity_direction
        
        # 计算接触点（假设物体在中心区域）
        contact_radius = min(rows, cols) / 3.0
        
        for i in range(num_taxels):
            row_idx = i // cols
            col_idx = i % cols
            
            # 触点相对于中心的坐标
            x_pos = (col_idx - (cols-1)/2.0) / (cols-1) if cols > 1 else 0
            y_pos = (row_idx - (rows-1)/2.0) / (rows-1) if rows > 1 else 0
            
            # 判断是否接触物体
            distance_to_center = math.sqrt((x_pos - object_pose[0])**2 + (y_pos - object_pose[1])**2)
            
            if distance_to_center < contact_radius:
                # 触点接触物体，计算法向力
                # 法向力 = 基础法向力 + 抓取力分量
                base_normal = self._base_normal_pattern[i, 2]
                
                # 根据距离衰减
                distance_factor = 1.0 - (distance_to_center / contact_radius)
                
                # 法向力（Z方向） - 与抓取力和基础压力相关
                normal_force = (base_normal + grip_force * 0.5) * distance_factor
                
                # 剪切力计算（X,Y方向）
                # 1. 物体重量引起的剪切力（考虑重力方向）
                weight_shear = object_weight_force * self._friction_coeff * distance_factor
                
                # 2. 物体姿态引起的剪切力（如果物体倾斜）
                rotation = object_pose[2]
                shear_x = weight_shear * math.sin(rotation)
                shear_y = weight_shear * math.cos(rotation)
                
                # 3. 添加重力分量引起的剪切力
                if arm_pose is not None:
                    gravity_component = gravity_sensor[:2] * normal_force * 0.3
                    shear_x += gravity_component[0]
                    shear_y += gravity_component[1]
                
                # 4. 添加滑动趋势（如果夹爪正在移动）
                if self._sim_mode in ["opening", "closing"]:
                    movement_direction = 1.0 if self._sim_mode == "opening" else -1.0
                    movement_shear = normal_force * self._friction_coeff * 0.3 * movement_direction
                    shear_x += movement_shear
                
                # 限制剪切力不超过最大值
                max_shear = normal_force * self._friction_coeff
                total_shear = math.sqrt(shear_x**2 + shear_y**2)
                if total_shear > max_shear:
                    scale = max_shear / total_shear
                    shear_x *= scale
                    shear_y *= scale
                
                # 添加噪声
                noise_level = self.hardware.sensor.force_noise_level
                normal_force += random.uniform(-noise_level, noise_level)
                shear_x += random.uniform(-noise_level, noise_level)
                shear_y += random.uniform(-noise_level, noise_level)
                
                # 确保力值为正（法向力不能为负）
                normal_force = max(0.1, normal_force)
                
                force_vectors[i] = [shear_x, shear_y, normal_force]
                
                # 记录接触点
                if i not in self._contact_points:
                    self._contact_points.append(i)
            else:
                # 未接触物体，只有基础法向力
                force_vectors[i] = self._base_normal_pattern[i].copy()
                # 添加轻微噪声
                noise = random.uniform(-0.1, 0.1)
                force_vectors[i, 2] += noise
                force_vectors[i, 2] = max(0.0, force_vectors[i, 2])
                
                # 从接触点移除
                if i in self._contact_points:
                    self._contact_points.remove(i)
        
        return force_vectors

    def update_arm_state(self, joint_positions=None, cartesian_pose=None):
        """
        更新机械臂状态
        
        Args:
            joint_positions: 关节位置列表
            cartesian_pose: 笛卡尔位姿 (4x4矩阵)
        """
        if joint_positions is not None:
            self._arm_joint_positions = joint_positions
            
        if cartesian_pose is not None:
            self._arm_cartesian_pose = cartesian_pose
            # 更新TCP到世界的变换矩阵
            self._tcp_to_world_transform = cartesian_pose
            
        # 更新机械臂状态字典
        self._arm_state = {
            'joint_positions': self._arm_joint_positions,
            'cartesian_pose': self._arm_cartesian_pose.tolist(),
            'timestamp': time.time()
        }

    def create_sensor_pipeline(self):
        """
        创建 Paxini Gen3 传感器处理管道
        
        Returns:
            传感器处理管道字典
        """
        # 创建模拟管道
        pipeline = self._create_simulated_pipeline()
        
        # 添加 Paxini 特定处理
        pipeline['paxini_specific'] = {
            'model': self.hardware.sensor.model.value,
            'calibration': self.hardware.sensor.calibration_enabled,
            'normalize': self.hardware.sensor.normalize_data,
            'dynamic_range': self.hardware.sensor.dynamic_range_adjustment,
            'force_scale': self.hardware.sensor.force_scale,
            'force_dimensions': 3,  # 三维力数据
            'friction_coefficient': self.hardware.sensor.friction_coefficient,
            'arm_integration': self.hardware.sensor.arm_integration.value
        }
        
        # 如果启用机械臂集成，添加相关配置
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            pipeline['arm_integration'] = {
                'enabled': True,
                'mode': self.hardware.sensor.arm_integration.value,
                'tcp_offset': self.hardware.sensor.tool_center_point,
                'sensor_to_tcp_transform': self.hardware.sensor.sensor_to_tcp_transform
            }
        
        return pipeline
    

    def _create_simulated_pipeline(self):
        """创建模拟管道 - 修正：确保与 sensor_reader.py 兼容"""
        pipeline = {
            'reader': {
                'port': self.hardware.sensor.port,
                'baudrate': self.hardware.sensor.baudrate,
                'simulated': True,
                'num_tactels': self.hardware.sensor.num_taxels,
                'rows': self.hardware.sensor.rows,
                'cols': self.hardware.sensor.cols,
                'sampling_rate': self.hardware.sensor.sampling_rate,
                'force_scale': self.hardware.sensor.force_scale,
                'max_pressure_range': self.hardware.sensor.max_pressure_range,
                'timeout': self.hardware.sensor.timeout,
                'force_dimensions': 3  # 三维力数据
            },
            'processor': {
                'filter_enabled': self.hardware.sensor.filter_enabled,
                'filter_type': self.hardware.sensor.filter_type,
                'filter_window': self.hardware.sensor.filter_window_size,
                'force_dimensions': 3
            },
            'mapper': {
                'rows': self.hardware.sensor.rows,
                'cols': self.hardware.sensor.cols,
                'vector_field': True  # 支持矢量场
            },
            'config': {
                'port': self.hardware.sensor.port,
                'baudrate': self.hardware.sensor.baudrate,
                'num_tactels': self.hardware.sensor.num_taxels,
                'rows': self.hardware.sensor.rows,
                'cols': self.hardware.sensor.cols,
                'sampling_rate': self.hardware.sensor.sampling_rate,
                'filter_type': self.hardware.sensor.filter_type,
                'force_scale': self.hardware.sensor.force_scale,
                'max_pressure_range': self.hardware.sensor.max_pressure_range,
                'force_dimensions': 3
            },
            'warning': '使用模拟管道进行开发和测试',
            'simulated': True,
            'generate_data_func': self._generate_simulated_data
        }
        
        # 如果启用机械臂集成，添加相关配置
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            pipeline['arm_integration'] = {
                'enabled': True,
                'mode': self.hardware.sensor.arm_integration.value,
                'update_arm_state_func': self.update_arm_state
            }
        
        return pipeline
    

    def _generate_simulated_data(self):
        """
        生成模拟传感器数据 - 支持三维力数据和机械臂集成
        使用物理模型模拟真实夹取情况
        """
        # 控制生成频率
        current_time = time.time()
        elapsed = current_time - self._last_sim_time
        
        if elapsed < self._sim_update_rate:
            time.sleep(self._sim_update_rate - elapsed)
        
        self._last_sim_time = current_time
        
        # 更新模拟模式计时器
        self._sim_mode_timer += self._sim_update_rate
        
        # 模拟物体姿态变化
        # 物体可能在夹爪中轻微移动
        if random.random() < 0.05:  # 5%概率改变物体姿态
            self._object_pose[0] += random.uniform(-0.1, 0.1)  # X方向偏移
            self._object_pose[1] += random.uniform(-0.1, 0.1)  # Y方向偏移
            self._object_pose[2] += random.uniform(-0.05, 0.05)  # 旋转
            
            # 限制偏移范围
            self._object_pose[0] = max(-0.5, min(0.5, self._object_pose[0]))
            self._object_pose[1] = max(-0.5, min(0.5, self._object_pose[1]))
        
        # 如果启用机械臂集成，模拟机械臂运动
        arm_pose_for_calculation = None
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            # 模拟机械臂轻微运动
            if random.random() < 0.1:  # 10%概率轻微调整
                for i in range(len(self._arm_joint_positions)):
                    self._arm_joint_positions[i] += random.uniform(-0.01, 0.01)
            
            # 更新机械臂位姿（简化的正向运动学）
            # 这里使用一个简化的变换矩阵，实际应用中应该使用机械臂的FK
            self._arm_cartesian_pose = np.array([
                [1.0, 0.0, 0.0, 0.5 + 0.1 * math.sin(current_time)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.3 + 0.05 * math.cos(current_time)],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            arm_pose_for_calculation = self._arm_cartesian_pose
            
            # 更新机械臂状态
            self.update_arm_state()
        
        # 每5-8秒改变一次模式
        if self._sim_mode_timer > random.uniform(5, 8):
            if self._sim_mode == "holding":
                # 从保持切换到打开或关闭
                self._sim_mode = random.choice(["opening", "closing"])
                self._sim_target = random.uniform(30, 150)
                self._sim_speed = random.uniform(0.3, 1.0)
            else:
                # 从运动切换到保持
                self._sim_mode = "holding"
                self._sim_target = self._sim_position
            self._sim_mode_timer = 0
        
        # 根据模式更新位置
        if self._sim_mode == "opening":
            if self._sim_position < self._sim_target:
                self._sim_position += self._sim_speed
            else:
                self._sim_position = self._sim_target
        elif self._sim_mode == "closing":
            if self._sim_position > self._sim_target:
                self._sim_position -= self._sim_speed
            else:
                self._sim_position = self._sim_target
        else:
            # 保持位置，有轻微抖动
            self._sim_position += random.uniform(-0.05, 0.05)
        
        # 限制位置范围
        self._sim_position = max(0, min(180, self._sim_position))
        
        # 根据物理模型计算三维力，考虑机械臂位姿
        force_vectors = self._calculate_contact_forces(
            self._sim_position, 
            self._object_pose,
            arm_pose_for_calculation
        )
        
        # 应用平滑滤波
        if self._prev_force_data is not None:
            alpha = self._smooth_factor
            force_vectors = alpha * self._prev_force_data + (1 - alpha) * force_vectors
        
        # 保存当前帧数据用于下一帧平滑
        self._prev_force_data = force_vectors.copy()
        
        # 创建模拟数据对象，包含机械臂状态
        sim_data = SimulatedTactileData(
            force_vectors=force_vectors,
            timestamp=current_time,
            sequence=int(current_time * 1000),
            arm_state=self._arm_state
        )
        
        # 添加位置信息
        sim_data.servo_position = self._sim_position
        sim_data.position = self._sim_position
        sim_data.servo_target = self._sim_target
        sim_data.target_position = self._sim_target
        sim_data.servo_mode = self._sim_mode
        sim_data.object_pose = self._object_pose.copy()  # 物体姿态
        sim_data.contact_points = self._contact_points.copy()  # 接触点列表
        
        # 添加机械臂信息（如果启用）
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            sim_data.arm_integration_enabled = True
            sim_data.arm_joint_positions = self._arm_joint_positions.copy()
            sim_data.arm_cartesian_pose = self._arm_cartesian_pose.copy()
            sim_data.tcp_to_world_transform = self._tcp_to_world_transform.copy()
            
            # 计算世界坐标系中的力
            sim_data.world_force = sim_data.transform_to_world_frame(
                self._tcp_to_world_transform
            )
        else:
            sim_data.arm_integration_enabled = False
        
        return sim_data
    

    def get_sensor_layout(self):
        """
        获取传感器布局信息
        
        Returns:
            传感器布局字典
        """
        layout = {
            "type": "grid",
            "rows": self.hardware.sensor.rows,
            "cols": self.hardware.sensor.cols,
            "spacing_mm": 5.0,
            "active_area_mm": (15.0, 15.0),
            "taxel_positions": self._calculate_taxel_positions(),
            "force_dimensions": 3  # 三维力
        }
        
        # 如果启用机械臂集成，添加相关布局信息
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            layout["arm_integration"] = {
                "enabled": True,
                "tcp_offset": self.hardware.sensor.tool_center_point,
                "coordinate_frame": "tcp"  # 相对于工具中心点
            }
        
        return layout
    
    def _calculate_taxel_positions(self):
        """
        计算触觉单元位置
        
        Returns:
            触觉单元位置列表
        """
        positions = []
        rows = self.hardware.sensor.rows
        cols = self.hardware.sensor.cols
        
        for i in range(rows):
            for j in range(cols):
                # 计算归一化位置 (0-1)
                x = j / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                positions.append((x, y))
        
        return positions
    
    def validate_sensor_config(self) -> list:
        """
        验证传感器配置
        
        Returns:
            错误消息列表
        """
        errors = []
        
        # 检查基本参数
        if not self.hardware.sensor.port:
            errors.append("传感器端口不能为空")
        
        if self.hardware.sensor.baudrate not in [9600, 19200, 38400, 57600, 115200]:
            errors.append(f"不支持的波特率: {self.hardware.sensor.baudrate}")
        
        # 检查 Paxini 特定参数
        if self.hardware.sensor.rows * self.hardware.sensor.cols != self.hardware.sensor.num_taxels:
            errors.append(f"行列数 ({self.hardware.sensor.rows}x{self.hardware.sensor.cols}) "
                         f"与触觉单元数 ({self.hardware.sensor.num_taxels}) 不匹配")
        
        if self.hardware.sensor.sampling_rate > 200:
            errors.append(f"采样率过高: {self.hardware.sensor.sampling_rate} Hz (最大 200 Hz)")
        
        # 检查机械臂集成参数
        if self.hardware.sensor.arm_integration != ArmIntegrationMode.DISABLED:
            if len(self.hardware.sensor.tool_center_point) != 3:
                errors.append("工具中心点必须是三维坐标")
            if len(self.hardware.sensor.sensor_to_tcp_transform) != 4:
                errors.append("传感器到TCP变换矩阵必须是4x4")
        
        return errors


def create_3x3_config_with_arm(arm_integration: bool = True) -> PaxiniGen3Config:
    """
    创建带机械臂集成的 3x3 传感器配置
    
    Args:
        arm_integration: 是否启用机械臂集成
        
    Returns:
        Paxini Gen3 3x3 配置实例
    """
    return PaxiniGen3Config(model=PaxiniModel.M2020_3x3, arm_integration=arm_integration)


def create_3x3_config() -> PaxiniGen3Config:
    """
    创建 3x3 传感器配置
    
    Returns:
        Paxini Gen3 3x3 配置实例
    """
    return PaxiniGen3Config(model=PaxiniModel.M2020_3x3, arm_integration=False)


def create_6x6_config() -> PaxiniGen3Config:
    """
    创建 6x6 传感器配置
    
    Returns:
        Paxini Gen3 6x6 配置实例
    """
    return PaxiniGen3Config(model=PaxiniModel.M2020_6x6)


def save_paxini_config(config: PaxiniGen3Config, filepath: str):
    """
    保存 Paxini 配置到文件
    
    Args:
        config: Paxini 配置实例
        filepath: 文件路径
    """
    config_dict = {
        "sensor_type": config.sensor_type,
        "sensor_model": config.sensor_model.value,
        "hardware": {
            "sensor": {
                "type": config.hardware.sensor.type,
                "model": config.hardware.sensor.model.value,
                "port": config.hardware.sensor.port,
                "baudrate": config.hardware.sensor.baudrate,
                "num_taxels": config.hardware.sensor.num_taxels,
                "rows": config.hardware.sensor.rows,
                "cols": config.hardware.sensor.cols,
                "sampling_rate": config.hardware.sensor.sampling_rate,
                "filter_enabled": config.hardware.sensor.filter_enabled,
                "filter_type": config.hardware.sensor.filter_type,
                "calibration_enabled": config.hardware.sensor.calibration_enabled,
                "normalize_data": config.hardware.sensor.normalize_data,
                "force_scale": config.hardware.sensor.force_scale,
                "max_pressure_range": config.hardware.sensor.max_pressure_range,
                "friction_coefficient": config.hardware.sensor.friction_coefficient,
                "force_noise_level": config.hardware.sensor.force_noise_level,
                "max_shear_force": config.hardware.sensor.max_shear_force,
                "arm_integration": config.hardware.sensor.arm_integration.value,
                "tool_center_point": config.hardware.sensor.tool_center_point,
                "sensor_to_tcp_transform": config.hardware.sensor.sensor_to_tcp_transform
            },
            "servo": {
                "type": config.hardware.servo.type,
                "port": config.hardware.servo.port,
                "baudrate": config.hardware.servo.baudrate,
                "min_angle": config.hardware.servo.min_angle,
                "max_angle": config.hardware.servo.max_angle,
                "speed": config.hardware.servo.speed
            }
        },
        "algorithm": {
            "data_dimensions": config.algorithm.data_dimensions,
            "feature_size": config.algorithm.feature_size,
            "use_spatial_features": config.algorithm.use_spatial_features,
            "use_temporal_features": config.algorithm.use_temporal_features,
            "force_dimensions": getattr(config.algorithm, 'force_dimensions', 3)
        },
        "ui": {
            "sensor_grid_size": config.ui.sensor_grid_size,
            "show_tactile_grid": config.ui.show_tactile_grid,
            "show_vector_field": getattr(config.ui, 'show_vector_field', True)
        },
        "integration": {
            "arm_integration_enabled": config.integration.arm_integration_enabled,
            "vision_integration_enabled": config.integration.vision_integration_enabled
        }
    }
    
    # 添加机械臂配置（如果存在）
    if hasattr(config, 'learm_arm') and config.learm_arm is not None:
        config_dict["learm_arm"] = config.learm_arm
    
    # 添加相机配置（如果存在）
    if hasattr(config, 'camera') and config.camera is not None:
        config_dict["camera"] = config.camera
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_paxini_config(filepath: str) -> PaxiniGen3Config:
    """
    从文件加载 Paxini 配置
    
    Args:
        filepath: 文件路径
        
    Returns:
        Paxini Gen3 配置实例
    """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 确定模型
    model_str = config_dict.get("sensor_model", "M2020_3x3")
    model = PaxiniModel[model_str]
    
    # 检查是否启用机械臂集成
    arm_integration = False
    if "integration" in config_dict:
        arm_integration = config_dict["integration"].get("arm_integration_enabled", False)
    
    # 创建配置
    config = PaxiniGen3Config(model=model, arm_integration=arm_integration)
    
    # 更新配置
    if "hardware" in config_dict:
        hw_dict = config_dict["hardware"]
        if "sensor" in hw_dict:
            sensor_dict = hw_dict["sensor"]
            for key, value in sensor_dict.items():
                if hasattr(config.hardware.sensor, key):
                    # 处理枚举类型
                    if key == "arm_integration":
                        value = ArmIntegrationMode(value)
                    setattr(config.hardware.sensor, key, value)
        
        if "servo" in hw_dict:
            servo_dict = hw_dict["servo"]
            for key, value in servo_dict.items():
                if hasattr(config.hardware.servo, key):
                    setattr(config.hardware.servo, key, value)
    
    # 更新集成配置
    if "integration" in config_dict:
        integration_dict = config_dict["integration"]
        for key, value in integration_dict.items():
            if hasattr(config.integration, key):
                setattr(config.integration, key, value)
    
    # 更新其他配置
    if "learm_arm" in config_dict:
        config.learm_arm = config_dict["learm_arm"]
    
    if "camera" in config_dict:
        config.camera = config_dict["camera"]
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("测试 Paxini Gen3 三维力配置与机械臂集成...")
    
    # 创建带机械臂集成的 3x3 配置
    config_3x3_arm = create_3x3_config_with_arm(arm_integration=True)
    print(f"带机械臂集成的 3x3 配置创建成功")
    print(f"传感器型号: {config_3x3_arm.hardware.sensor.model.value}")
    print(f"触觉单元数: {config_3x3_arm.hardware.sensor.num_taxels}")
    print(f"力数据维度: 3 (Fx, Fy, Fz)")
    print(f"机械臂集成: {config_3x3_arm.hardware.sensor.arm_integration.value}")
    print(f"TCP偏移: {config_3x3_arm.hardware.sensor.tool_center_point}")
    
    # 测试模拟数据生成
    print("\n测试带机械臂集成的三维力数据生成...")
    for i in range(3):
        sim_data = config_3x3_arm._generate_simulated_data()
        print(f"第{i+1}次模拟:")
        print(f"  力数据形状: {sim_data.tactile_array.shape}")
        print(f"  合力向量: [{sim_data.resultant_force[0]:.2f}, {sim_data.resultant_force[1]:.2f}, {sim_data.resultant_force[2]:.2f}] N")
        print(f"  总法向力: {sim_data.total_normal_force:.2f} N")
        print(f"  总剪切力: {sim_data.total_shear_force:.2f} N")
        print(f"  接触点数量: {len(sim_data.contact_points)}")
        print(f"  夹爪位置: {sim_data.servo_position:.1f} 度")
        if sim_data.arm_integration_enabled:
            print(f"  机械臂关节位置: {sim_data.arm_joint_positions[:3]}...")
            print(f"  世界坐标系中的力: {sim_data.world_force}")
        print("  " + "-"*40)