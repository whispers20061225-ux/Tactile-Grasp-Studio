"""
触觉夹爪演示系统 - 主配置文件
定义系统配置的数据结构和默认值。
支持机械臂、视觉、仿真和路径规划集成。
"""

import yaml
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

# 添加 YAML 构造器来处理元组
def tuple_constructor(loader, node):
    """YAML 构造器：将列表转换为元组"""
    value = loader.construct_sequence(node)
    return tuple(value)

def tuple_representer(dumper, data):
    """YAML 表示器：将元组表示为列表"""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', list(data))

# 注册自定义构造器和表示器
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
yaml.add_representer(tuple, tuple_representer)


class ConfigEncoder:
    """配置编码器：处理 YAML 序列化问题"""
    
    @staticmethod
    def encode(obj):
        """将对象转换为 YAML 安全的格式"""
        if isinstance(obj, dict):
            return {k: ConfigEncoder.encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigEncoder.encode(item) for item in obj]
        elif isinstance(obj, tuple):
            # 将元组转换为列表（YAML 可以处理列表）
            return [ConfigEncoder.encode(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj
    
    @staticmethod
    def decode(obj, target_type=None):
        """将 YAML 加载的对象转换回原始格式"""
        if isinstance(obj, dict):
            if target_type and hasattr(target_type, '__dataclass_fields__'):
                # 如果是数据类，递归处理
                result = {}
                for field_name, field_type in target_type.__dataclass_fields__.items():
                    if field_name in obj:
                        result[field_name] = ConfigEncoder.decode(obj[field_name], field_type.type)
                return target_type(**result)
            else:
                return {k: ConfigEncoder.decode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 检查是否需要转换为元组
            if target_type and hasattr(target_type, '__origin__') and target_type.__origin__ == tuple:
                return tuple(ConfigEncoder.decode(item) for item in obj)
            else:
                return [ConfigEncoder.decode(item) for item in obj]
        else:
            return obj


class SensorType(Enum):
    """传感器类型枚举"""
    DEFAULT = "default"
    PAXINI_GEN3 = "paxini_gen3"
    CUSTOM = "custom"


class ServoType(Enum):
    """舵机类型枚举"""
    ST3215 = "st3215"
    MG996R = "mg996r"
    CUSTOM = "custom"


@dataclass
class SensorConfig:
    """传感器配置"""
    type: str = "default"
    port: str = "COM3"
    baudrate: int = 115200
    timeout: float = 0.1  # 修改：从1.0改为0.1，匹配sensor_reader.py
    retry_count: int = 3
    sampling_rate: int = 100
    data_format: str = "raw"
    calibration_enabled: bool = True
    
    # Paxini Gen3 特定参数
    model: str = "M2020_3x3"
    num_taxels: int = 9
    rows: int = 3
    cols: int = 3
    pressure_range: List[float] = field(default_factory=lambda: [0, 100])
    resolution: float = 0.1
    sensitivity: float = 1.0
    offset: List[float] = field(default_factory=lambda: [0.0] * 9)
    gain: List[float] = field(default_factory=lambda: [1.0] * 9)
    
    # 滤波参数
    filter_enabled: bool = True
    filter_type: str = "median"
    filter_window: int = 5
    filter_window_size: int = 5  # 添加 filter_window_size
    low_pass_cutoff: float = 50.0
    high_pass_cutoff: float = 0.1
    moving_average_window: int = 10
    
    # 新增可能需要的属性
    normalize_data: bool = True
    remove_offset: bool = True
    dynamic_range_adjustment: bool = True
    
    # 新增：sensor_reader.py 需要的参数
    force_scale: float = 10.0  # 关键：从1.0改为10.0
    max_pressure_range: float = 100.0  # 最大压力范围 (kPa)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保数组长度匹配
        if len(self.offset) != self.num_taxels:
            self.offset = [0.0] * self.num_taxels
        if len(self.gain) != self.num_taxels:
            self.gain = [1.0] * self.num_taxels
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        result = asdict(self)
        # 处理枚举类型
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        return result
    
    def validate(self) -> list:
        """验证配置"""
        errors = []
        
        if not self.port:
            errors.append("传感器端口不能为空")
        
        if self.baudrate not in [9600, 19200, 38400, 57600, 115200]:
            errors.append(f"不支持的波特率: {self.baudrate}")
        
        if self.sampling_rate <= 0 or self.sampling_rate > 1000:
            errors.append(f"采样率超出范围: {self.sampling_rate} Hz")
        
        if self.rows * self.cols != self.num_taxels:
            errors.append(f"行列数 ({self.rows}x{self.cols}) 与触觉单元数 ({self.num_taxels}) 不匹配")
        
        if self.force_scale <= 0:
            errors.append(f"力值缩放因子必须大于0: {self.force_scale}")
        
        return errors


@dataclass
class ServoConfig:
    """舵机配置"""
    type: str = "st3215"
    port: str = "COM4"
    baudrate: int = 115200
    timeout: float = 1.0
    retry_count: int = 3
    min_angle: int = 0
    max_angle: int = 180
    home_position: int = 90
    speed: int = 50
    acceleration: int = 50
    deceleration: int = 50
    torque: float = 1.0  # 添加 torque 属性，单位：牛·米 (Nm)
    torque_limit: float = 1.0
    max_torque: float = 2.0
    current_limit: float = 1.0
    max_force: float = 10.0
    position_kp: float = 1.0
    position_ki: float = 0.1
    position_kd: float = 0.05
    velocity_kp: float = 0.5
    velocity_ki: float = 0.01
    velocity_kd: float = 0.02
    current_kp: float = 0.3
    current_ki: float = 0.01
    current_kd: float = 0.001
    feedforward_velocity: float = 0.0
    feedforward_acceleration: float = 0.0
    deadband: float = 1.0
    smooth_factor: float = 0.1
    backlash_compensation: float = 0.0
    
    def __post_init__(self):
        """初始化后处理，确保属性值在有效范围内"""
        # 确保角度在合理范围内
        self.min_angle = max(-360, min(360, self.min_angle))
        self.max_angle = max(-360, min(360, self.max_angle))
        self.home_position = max(self.min_angle, min(self.max_angle, self.home_position))
        
        # 确保速度、加速度在合理范围内
        self.speed = max(0, min(100, self.speed))
        self.acceleration = max(0, min(100, self.acceleration))
        self.deceleration = max(0, min(100, self.deceleration))
        
        # 确保扭矩相关参数在合理范围内
        self.torque = max(0.0, min(5.0, self.torque))
        self.torque_limit = max(0.0, min(1.0, self.torque_limit))
        self.max_torque = max(0.0, min(5.0, self.max_torque))
        self.max_force = max(0.0, min(100.0, self.max_force))
        
        # 如果 torque_limit 大于 max_torque，进行调整
        if self.torque_limit * self.max_torque > self.max_torque:
            self.torque_limit = 1.0
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        result = asdict(self)
        # 处理枚举类型
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        return result
    
    def validate(self) -> list:
        """验证配置"""
        errors = []
        
        if self.min_angle >= self.max_angle:
            errors.append(f"最小角度 ({self.min_angle}) 必须小于最大角度 ({self.max_angle})")
        
        if not (self.min_angle <= self.home_position <= self.max_angle):
            errors.append(f"零点位置 ({self.home_position}) 必须在最小角度和最大角度之间")
        
        if self.speed < 0 or self.speed > 100:
            errors.append(f"速度 ({self.speed}) 必须在0-100之间")
        
        if self.torque < 0:
            errors.append(f"扭矩 ({self.torque}) 必须大于0")
        
        if self.torque_limit < 0 or self.torque_limit > 1.0:
            errors.append(f"扭矩限制 ({self.torque_limit}) 必须在0-1之间")
        
        if self.max_force < 0:
            errors.append(f"最大力 ({self.max_force}) 必须大于0")
        
        return errors


@dataclass
class HardwareConfig:
    """硬件配置"""
    sensor: SensorConfig = field(default_factory=SensorConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    auto_connect: bool = True
    connection_timeout: float = 5.0
    
    # 添加可能缺失的属性
    emergency_timeout: float = 5.0
    max_force_limit: float = 50.0
    safety_enabled: bool = True
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        return {
            'sensor': self.sensor.to_dict(),
            'servo': self.servo.to_dict(),
            'auto_connect': self.auto_connect,
            'connection_timeout': self.connection_timeout,
            'emergency_timeout': self.emergency_timeout,
            'max_force_limit': self.max_force_limit,
            'safety_enabled': self.safety_enabled,
        }


@dataclass
class AlgorithmConfig:
    """算法配置"""
    data_dimensions: List[int] = field(default_factory=lambda: [3, 3])  # 改为列表
    feature_size: int = 9
    use_spatial_features: bool = True
    use_temporal_features: bool = True
    temporal_window: int = 10
    feature_extraction_method: str = "pca"
    clustering_enabled: bool = True
    clustering_method: str = "kmeans"
    num_clusters: int = 3
    
    # 新增：视觉算法配置
    vision_algorithm: Dict[str, Any] = field(default_factory=lambda: {
        'object_detection': True,
        'pose_estimation': True,
        'segmentation': False,
        'tracking': True,
    })
    
    # 新增：运动规划配置
    motion_planning: Dict[str, Any] = field(default_factory=lambda: {
        'use_moveit': False,
        'planner_type': 'rrt_connect',
        'collision_checking': True,
        'optimization_enabled': True,
    })
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        return asdict(self)


@dataclass
class UIConfig:
    """用户界面配置"""
    window_width: int = 1200
    window_height: int = 800
    control_panel_width: int = 300
    data_refresh_rate: int = 30
    theme: str = "light"
    language: str = "zh_CN"
    show_grid: bool = True
    auto_save_data: bool = False
    save_interval: int = 60
    sensor_grid_size: List[int] = field(default_factory=lambda: [3, 3])  # 改为列表
    show_tactile_grid: bool = True
    
    # 新增：机械臂界面配置
    arm_ui: Dict[str, Any] = field(default_factory=lambda: {
        'show_arm_status': True,
        'show_joint_states': True,
        'show_trajectory': True,
        'show_workspace': False,
    })
    
    # 新增：视觉界面配置
    vision_ui: Dict[str, Any] = field(default_factory=lambda: {
        'show_camera_view': True,
        'show_point_cloud': False,
        'show_detection_results': True,
        'camera_view_size': [640, 480],
    })
    
    # 新增：仿真界面配置
    simulation_ui: Dict[str, Any] = field(default_factory=lambda: {
        'show_simulation_view': False,
        'physics_debug': False,
        'show_collision_boxes': True,
    })
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        return asdict(self)


@dataclass
class SystemIntegrationConfig:
    """系统集成配置"""
    # 机械臂集成
    arm_integration_enabled: bool = False
    arm_config_path: Optional[str] = None
    
    # 视觉集成
    vision_integration_enabled: bool = False
    camera_config_path: Optional[str] = None
    
    # 仿真集成
    simulation_integration_enabled: bool = False
    simulation_config_path: Optional[str] = None
    
    # 路径规划集成
    planning_integration_enabled: bool = False
    planning_config_path: Optional[str] = None
    
    # 任务集成
    task_integration_enabled: bool = False
    task_config_path: Optional[str] = None
    
    # 通信设置
    communication: Dict[str, Any] = field(default_factory=lambda: {
        'protocol': 'direct',  # direct, ros, mqtt
        'sync_rate': 100,  # Hz
        'timeout': 5.0,  # seconds
    })
    
    # 坐标系设置
    coordinate_systems: Dict[str, Any] = field(default_factory=lambda: {
        'world_frame': 'base_link',
        'tool_frame': 'gripper_tcp',
        'camera_frame': 'camera_color_optical_frame',
        'hand_eye_calibrated': False,
    })
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        return asdict(self)


@dataclass
class DemoConfig:
    """演示系统主配置"""
    
    # 配置版本
    version: str = "3.0.0"
    
    # 硬件配置
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # 算法配置
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    
    # 界面配置
    ui: UIConfig = field(default_factory=UIConfig)
    
    # 系统集成配置
    integration: SystemIntegrationConfig = field(default_factory=SystemIntegrationConfig)
    
    # 演示模式
    demo_modes: Dict[str, Any] = field(default_factory=dict)
    
    # 传感器类型
    sensor_type: str = "default"
    
    # 深度学习配置
    deep_learning: Optional[Dict[str, Any]] = None
    
    # 新增：机械臂配置
    learm_arm: Optional[Dict[str, Any]] = None
    
    # 新增：相机配置
    camera: Optional[Dict[str, Any]] = None
    
    # 新增：仿真配置
    simulation: Optional[Dict[str, Any]] = None
    
    # 新增：路径规划配置
    path_planning: Optional[Dict[str, Any]] = None
    
    # 新增：任务配置
    task: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保配置版本
        self.version = "3.0.0"
        
        # 确保硬件配置存在
        if not hasattr(self, 'hardware') or self.hardware is None:
            self.hardware = HardwareConfig()
        
        # 确保算法配置存在
        if not hasattr(self, 'algorithm') or self.algorithm is None:
            self.algorithm = AlgorithmConfig()
        
        # 确保界面配置存在
        if not hasattr(self, 'ui') or self.ui is None:
            self.ui = UIConfig()
        
        # 确保集成配置存在
        if not hasattr(self, 'integration') or self.integration is None:
            self.integration = SystemIntegrationConfig()
        
        # 确保演示模式存在
        if not hasattr(self, 'demo_modes') or self.demo_modes is None:
            self.demo_modes = {}
        
        # 确保传感器类型存在
        if not hasattr(self, 'sensor_type'):
            self.sensor_type = "default"
        
        # 确保传感器配置中的 force_scale 设置正确
        if hasattr(self.hardware.sensor, 'force_scale'):
            if self.hardware.sensor.force_scale < 5.0:  # 如果太小，调整为合理值
                self.hardware.sensor.force_scale = 10.0
        
        # 初始化新增配置（如果不存在）
        if not hasattr(self, 'learm_arm'):
            self.learm_arm = None
        if not hasattr(self, 'camera'):
            self.camera = None
        if not hasattr(self, 'simulation'):
            self.simulation = None
        if not hasattr(self, 'path_planning'):
            self.path_planning = None
        if not hasattr(self, 'task'):
            self.task = None
        
        # 更新集成配置标志
        self._update_integration_flags()
    
    def _update_integration_flags(self):
        """更新集成配置标志"""
        if self.learm_arm is not None:
            self.integration.arm_integration_enabled = True
        if self.camera is not None:
            self.integration.vision_integration_enabled = True
        if self.simulation is not None:
            self.integration.simulation_integration_enabled = True
        if self.path_planning is not None:
            self.integration.planning_integration_enabled = True
        if self.task is not None:
            self.integration.task_integration_enabled = True
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误消息列表
        """
        errors = []
        
        # 检查硬件配置
        if not self.hardware:
            errors.append("硬件配置不能为空")
        else:
            # 检查传感器配置
            sensor_errors = self.hardware.sensor.validate()
            errors.extend(sensor_errors)
            
            # 检查舵机配置
            servo_errors = self.hardware.servo.validate()
            errors.extend(servo_errors)
        
        # 检查界面配置
        if not self.ui:
            errors.append("界面配置不能为空")
        else:
            if self.ui.window_width <= 0 or self.ui.window_height <= 0:
                errors.append("窗口尺寸必须大于0")
            if self.ui.data_refresh_rate <= 0:
                errors.append("数据刷新率必须大于0")
        
        # 检查集成配置
        if self.integration.arm_integration_enabled and not self.learm_arm:
            errors.append("机械臂集成已启用，但缺少机械臂配置")
        
        if self.integration.vision_integration_enabled and not self.camera:
            errors.append("视觉集成已启用，但缺少相机配置")
        
        return errors
    
    def save(self, filepath: str, format: str = 'yaml'):
        """
        保存配置到文件
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('yaml' 或 'json')
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换为可序列化的字典
        config_dict = self._to_dict()
        
        # 根据格式保存
        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:  # yaml
            # 使用自定义编码器处理元组
            safe_dict = ConfigEncoder.encode(config_dict)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(safe_dict, f, default_flow_style=False, allow_unicode=True)
    
    def _to_dict(self):
        """转换为字典（用于序列化）"""
        config_dict = {
            'version': self.version,
            'hardware': self.hardware.to_dict(),
            'algorithm': self.algorithm.to_dict(),
            'ui': self.ui.to_dict(),
            'integration': self.integration.to_dict(),
            'demo_modes': self.demo_modes,
            'sensor_type': self.sensor_type,
            'deep_learning': self.deep_learning,
        }
        
        # 添加新增配置（如果存在）
        if self.learm_arm is not None:
            config_dict['learm_arm'] = self.learm_arm
        if self.camera is not None:
            config_dict['camera'] = self.camera
        if self.simulation is not None:
            config_dict['simulation'] = self.simulation
        if self.path_planning is not None:
            config_dict['path_planning'] = self.path_planning
        if self.task is not None:
            config_dict['task'] = self.task
        
        return config_dict
    
    @classmethod
    def load(cls, filepath: str, format: str = 'auto'):
        """
        从文件加载配置
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('auto', 'yaml' 或 'json')
            
        Returns:
            配置实例
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        # 自动检测格式
        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            else:
                format = 'yaml'
        
        # 加载配置
        with open(filepath, 'r', encoding='utf-8') as f:
            if format.lower() == 'json':
                config_dict = json.load(f)
            else:  # yaml
                # 使用自定义解码器处理
                raw_dict = yaml.safe_load(f)
                config_dict = ConfigEncoder.decode(raw_dict)
        
        # 创建配置实例
        config = cls()
        
        # 更新配置
        if 'version' in config_dict:
            config.version = config_dict['version']
        
        # 加载硬件配置
        if 'hardware' in config_dict:
            hw_dict = config_dict['hardware']
            
            # 加载传感器配置
            if 'sensor' in hw_dict:
                sensor_dict = hw_dict['sensor']
                for key, value in sensor_dict.items():
                    if hasattr(config.hardware.sensor, key):
                        # 处理列表到元组的转换
                        if key in ['pressure_range', 'data_dimensions'] and isinstance(value, list):
                            value = tuple(value)
                        setattr(config.hardware.sensor, key, value)
            
            # 加载舵机配置
            if 'servo' in hw_dict:
                servo_dict = hw_dict['servo']
                for key, value in servo_dict.items():
                    if hasattr(config.hardware.servo, key):
                        setattr(config.hardware.servo, key, value)
            
            # 加载其他硬件配置
            for key in ['auto_connect', 'connection_timeout', 'emergency_timeout', 
                       'max_force_limit', 'safety_enabled']:
                if key in hw_dict:
                    setattr(config.hardware, key, hw_dict[key])
        
        # 加载算法配置
        if 'algorithm' in config_dict:
            algo_dict = config_dict['algorithm']
            for key, value in algo_dict.items():
                if hasattr(config.algorithm, key):
                    # 处理列表到元组的转换
                    if key in ['data_dimensions'] and isinstance(value, list):
                        value = tuple(value)
                    setattr(config.algorithm, key, value)
        
        # 加载界面配置
        if 'ui' in config_dict:
            ui_dict = config_dict['ui']
            for key, value in ui_dict.items():
                if hasattr(config.ui, key):
                    # 处理列表到元组的转换
                    if key in ['sensor_grid_size'] and isinstance(value, list):
                        value = tuple(value)
                    setattr(config.ui, key, value)
        
        # 加载集成配置
        if 'integration' in config_dict:
            integration_dict = config_dict['integration']
            for key, value in integration_dict.items():
                if hasattr(config.integration, key):
                    setattr(config.integration, key, value)
        
        # 加载新增配置
        for key in ['learm_arm', 'camera', 'simulation', 'path_planning', 'task']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # 加载其他配置
        for key in ['demo_modes', 'sensor_type', 'deep_learning']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # 更新集成标志
        config._update_integration_flags()
        
        return config
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        # 这里可以添加更新逻辑
        # 目前使用加载逻辑
        pass
    
    def copy(self):
        """
        创建配置的副本
        
        Returns:
            配置副本
        """
        import copy
        return copy.deepcopy(self)
    
    def enable_arm_integration(self, arm_config=None):
        """
        启用机械臂集成
        
        Args:
            arm_config: 机械臂配置字典，如果为None则使用默认配置
        """
        try:
            # 尝试导入机械臂配置类
            from config.learm_arm_config import LearmArmConfig
            if arm_config is None:
                # 创建默认配置实例
                default_config = LearmArmConfig()
                self.learm_arm = {
                    'CONNECTION': default_config.CONNECTION,
                    'PHYSICAL': default_config.PHYSICAL,
                    'MOTION': default_config.MOTION,
                    'TOOL': default_config.TOOL,
                    'USER_FRAMES': default_config.USER_FRAMES,
                    'SAFETY': default_config.SAFETY,
                    'PROTOCOL': default_config.PROTOCOL,
                    'ROS': default_config.ROS,
                    'SIMULATION': default_config.SIMULATION,
                    'CALIBRATION': default_config.CALIBRATION
                }
            else:
                self.learm_arm = arm_config
        except ImportError:
            # 如果导入失败，使用简化配置
            print("注意: 无法导入 LearmArmConfig，使用简化机械臂配置")
            self.learm_arm = {
                'CONNECTION': {'type': 'simulation'},
                'PHYSICAL': {'dof': 6},
                'SAFETY': {'workspace_limits': {'x': [-0.9, 0.9], 'y': [-0.9, 0.9], 'z': [0, 0.9]}}
            }
        
        self.integration.arm_integration_enabled = True
    
    def enable_vision_integration(self, camera_config=None):
        """
        启用视觉集成
        
        Args:
            camera_config: 相机配置字典，如果为None则使用默认配置
        """
        try:
            # 尝试导入相机配置类
            from config.camera_config import CameraConfig
            if camera_config is None:
                # 创建默认配置实例
                default_config = CameraConfig()
                self.camera = {
                    'HARDWARE': default_config.HARDWARE,
                    'INTRINSICS': default_config.INTRINSICS,
                    'EXTRINSICS': default_config.EXTRINSICS,
                    'PROCESSING': default_config.PROCESSING,
                    'FEATURES': default_config.FEATURES,
                    'DEPTH': default_config.DEPTH,
                    'OBJECT_DETECTION': default_config.OBJECT_DETECTION,
                    'MATERIAL_RECOGNITION': default_config.MATERIAL_RECOGNITION,
                    'POSE_ESTIMATION': default_config.POSE_ESTIMATION,
                    'CALIBRATION': default_config.CALIBRATION
                }
            else:
                self.camera = camera_config
        except ImportError:
            # 如果导入失败，使用简化配置
            print("注意: 无法导入 CameraConfig，使用简化相机配置")
            self.camera = {
                'HARDWARE': {
                    'primary_camera': {
                        'type': 'realsense',
                        'resolution': [1280, 720],
                        'fps': 30
                    }
                }
            }
        
        self.integration.vision_integration_enabled = True
    
    def enable_simulation_integration(self, sim_config=None):
        """
        启用仿真集成
        
        Args:
            sim_config: 仿真配置字典，如果为None则使用默认配置
        """
        try:
            # 尝试导入仿真配置类
            from config.simulation_config import SimulationConfig
            if sim_config is None:
                # 创建默认配置实例
                default_config = SimulationConfig()
                self.simulation = {
                    'ENGINE': default_config.ENGINE,
                    'SCENE': default_config.SCENE,
                    'ARM_SIMULATION': default_config.ARM_SIMULATION,
                    'GRIPPER_SIMULATION': default_config.GRIPPER_SIMULATION,
                    'TACTILE_SIMULATION': default_config.TACTILE_SIMULATION,
                    'OBJECT_SIMULATION': default_config.OBJECT_SIMULATION,
                    'PHYSICS_ADVANCED': default_config.PHYSICS_ADVANCED,
                    'VISUALIZATION': default_config.VISUALIZATION,
                    'DATA_RECORDING': default_config.DATA_RECORDING,
                    'PERFORMANCE': default_config.PERFORMANCE
                }
            else:
                self.simulation = sim_config
        except ImportError:
            # 如果导入失败，使用简化配置
            print("注意: 无法导入 SimulationConfig，使用简化仿真配置")
            self.simulation = {
                'ENGINE': {'type': 'pybullet', 'mode': 'gui'},
                'GRIPPER_SIMULATION': {'type': 'pneumatic'}
            }
        
        self.integration.simulation_integration_enabled = True
    
    def enable_planning_integration(self, planning_config=None):
        """
        启用路径规划集成
        
        Args:
            planning_config: 路径规划配置字典，如果为None则使用默认配置
        """
        try:
            # 尝试导入路径规划配置类
            from config.path_planning_config import PathPlanningConfig
            if planning_config is None:
                # 创建默认配置实例
                default_config = PathPlanningConfig()
                self.path_planning = {
                    'PLANNER_TYPE': default_config.PLANNER_TYPE,
                    'RRT_CONFIG': default_config.RRT_CONFIG,
                    'PRM_CONFIG': default_config.PRM_CONFIG,
                    'TRAJECTORY_OPTIMIZATION': default_config.TRAJECTORY_OPTIMIZATION,
                    'CARTESIAN_PLANNING': default_config.CARTESIAN_PLANNING,
                    'COLLISION_DETECTION': default_config.COLLISION_DETECTION,
                    'WORKSPACE_CONSTRAINTS': default_config.WORKSPACE_CONSTRAINTS,
                    'GRASP_PLANNING': default_config.GRASP_PLANNING,
                    'OBSTACLE_AVOIDANCE': default_config.OBSTACLE_AVOIDANCE,
                    'TRAJECTORY_GENERATION': default_config.TRAJECTORY_GENERATION,
                    'KINEMATICS': default_config.KINEMATICS,
                    'REALTIME': default_config.REALTIME,
                    'VISUALIZATION': default_config.VISUALIZATION
                }
            else:
                self.path_planning = planning_config
        except ImportError:
            # 如果导入失败，使用简化配置
            print("注意: 无法导入 PathPlanningConfig，使用简化路径规划配置")
            self.path_planning = {
                'PLANNER_TYPE': {'global_planner': 'rrt_connect'},
                'COLLISION_DETECTION': {'enabled': True}
            }
        
        self.integration.planning_integration_enabled = True
    
    def enable_task_integration(self, task_config=None):
        """
        启用任务集成
        
        Args:
            task_config: 任务配置字典，如果为None则使用默认配置
        """
        try:
            # 尝试导入任务配置类
            from config.task_config import TaskConfig
            if task_config is None:
                # 创建默认配置实例
                default_config = TaskConfig()
                self.task = {
                    'BASIC_GRASP_TASK': default_config.BASIC_GRASP_TASK,
                    'ASSEMBLY_TASK': default_config.ASSEMBLY_TASK,
                    'EXPERIMENT_RECORDING': default_config.EXPERIMENT_RECORDING
                }
            else:
                self.task = task_config
        except ImportError:
            # 如果导入失败，使用简化配置
            print("注意: 无法导入 TaskConfig，使用简化任务配置")
            self.task = {
                'BASIC_GRASP_TASK': {
                    'name': 'basic_grasp',
                    'description': '基本物体抓取任务',
                    'steps': [
                        {'action': 'move_to_ready', 'params': {'speed': 30.0}},
                        {'action': 'detect_object', 'params': {'object_class': 'cube'}},
                        {'action': 'execute_grasp', 'params': {'force': 10.0}}
                    ]
                }
            }
        
        self.integration.task_integration_enabled = True


# 工具函数
def create_default_config() -> DemoConfig:
    """
    创建默认配置
    
    Returns:
        默认配置实例
    """
    config = DemoConfig()
    # 确保 force_scale 设置正确
    config.hardware.sensor.force_scale = 10.0
    return config


def save_config(config: DemoConfig, filepath: str, format: str = 'yaml'):
    """
    保存配置到文件
    
    Args:
        config: 配置实例
        filepath: 文件路径
        format: 文件格式
    """
    config.save(filepath, format)


def load_config(filepath: str, format: str = 'auto') -> DemoConfig:
    """
    从文件加载配置
    
    Args:
        filepath: 文件路径
        format: 文件格式
        
    Returns:
        配置实例
    """
    return DemoConfig.load(filepath, format)


# 测试函数
def test_config():
    """测试配置功能"""
    print("测试配置系统...")
    
    # 创建默认配置
    config = create_default_config()
    print(f"默认配置创建成功 (版本: {config.version})")
    print(f"传感器 force_scale: {config.hardware.sensor.force_scale}")
    
    # 启用各种集成
    config.enable_arm_integration()
    config.enable_vision_integration()
    config.enable_simulation_integration()
    config.enable_planning_integration()
    config.enable_task_integration()
    
    print(f"\n集成状态:")
    print(f"  机械臂集成: {config.integration.arm_integration_enabled}")
    print(f"  视觉集成: {config.integration.vision_integration_enabled}")
    print(f"  仿真集成: {config.integration.simulation_integration_enabled}")
    print(f"  路径规划集成: {config.integration.planning_integration_enabled}")
    print(f"  任务集成: {config.integration.task_integration_enabled}")
    
    # 验证配置
    errors = config.validate()
    if errors:
        print(f"配置验证失败: {errors}")
    else:
        print("配置验证通过")
    
    # 保存配置
    try:
        # 确保目录存在
        os.makedirs("config/test", exist_ok=True)
        
        # 保存为 YAML
        config.save("config/test/full_integration_config.yaml")
        print("\n配置已保存到: config/test/full_integration_config.yaml")
        
        # 保存为 JSON
        config.save("config/test/full_integration_config.json", format='json')
        print("配置已保存到: config/test/full_integration_config.json")
        
    except Exception as e:
        print(f"保存配置失败: {e}")
    
    # 加载配置
    try:
        # 加载 YAML 配置
        loaded_config = load_config("config/test/full_integration_config.yaml")
        print(f"\n配置加载成功 (版本: {loaded_config.version})")
        
        # 验证加载的配置
        errors = loaded_config.validate()
        if errors:
            print(f"加载的配置验证失败: {errors}")
        else:
            print("加载的配置验证通过")
            
    except Exception as e:
        print(f"加载配置失败: {e}")
    
    print("\n配置测试完成")


# 主程序
if __name__ == "__main__":
    test_config()
