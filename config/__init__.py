"""
触觉夹爪演示系统 - 配置模块
配置模块提供系统配置管理功能，包括硬件参数、算法参数、界面设置等。
"""

from .demo_config import (
    # 配置类
    DemoConfig,
    SensorConfig,
    ServoConfig,
    HardwareConfig,
    AlgorithmConfig,
    UIConfig,
    
    # 枚举类
    SensorType,
    ServoType,
    
    # 工具函数
    create_default_config,
    load_config,
    save_config,
)

from .paxini_gen3_config import (
    PaxiniGen3Config,
    create_3x3_config,
)

from .deep_learning_config import (
    DeepLearningConfig,
    create_default_dl_config,
)
from .learm_arm_config import (
    LearmArmConfig,
    create_default_learm_config,
)
from .camera_config import (
    CameraConfig,
    create_default_camera_config,
)
from .simulation_config import (
    SimulationConfig,
    create_default_sim_config,
)
from .path_planning_config import (
    PathPlanningConfig,
    create_default_path_planning_config,
)
from .task_config import (
    TaskConfig,
    create_default_task_config,
)

# 版本信息
__version__ = "3.0.0"  # 更新版本号
__author__ = "Tactile Gripper Team"
__description__ = "触觉夹爪演示系统配置模块 v3.0.0 - 包含机械臂与视觉集成"

# 导出列表
__all__ = [
    # 配置类
    "DemoConfig",
    "SensorConfig",
    "ServoConfig",
    "HardwareConfig",
    "AlgorithmConfig",
    "UIConfig",
    "PaxiniGen3Config",
    "DeepLearningConfig",
    "LearmArmConfig",
    "CameraConfig",
    "SimulationConfig",
    "PathPlanningConfig",
    "TaskConfig",
    
    # 枚举类
    "SensorType",
    "ServoType",
    
    # 工具函数
    "create_default_config",
    "load_config",
    "save_config",
    "create_3x3_config",
    "create_default_dl_config",
    "create_default_learm_config",
    "create_default_camera_config",
    "create_default_sim_config",
    "create_default_path_planning_config",
    "create_default_task_config",
    
    # 元信息
    "__version__",
    "__author__",
    "__description__",
]


def get_config_info() -> dict:
    """
    获取配置模块信息
    
    Returns:
        包含配置模块信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_classes": [
            "DemoConfig",
            "SensorConfig",
            "ServoConfig",
            "HardwareConfig",
            "AlgorithmConfig",
            "UIConfig",
            "PaxiniGen3Config",
            "DeepLearningConfig",
            "LearmArmConfig",
            "CameraConfig",
            "SimulationConfig",
            "PathPlanningConfig",
            "TaskConfig",
        ],
        "available_enums": [
            "SensorType",
            "ServoType",
        ],
        "available_functions": [
            "create_default_config",
            "load_config",
            "save_config",
            "create_3x3_config",
            "create_default_dl_config",
            "create_default_learm_config",
            "create_default_camera_config",
            "create_default_sim_config",
            "create_default_path_planning_config",
            "create_default_task_config",
        ]
    }


def list_config_templates() -> list:
    """
    列出可用的配置模板
    
    Returns:
        配置模板列表
    """
    templates = [
        {
            "name": "default",
            "description": "默认配置（室内环境）",
            "filename": "default_config.yaml"
        },
        {
            "name": "high_precision",
            "description": "高精度模式（实验室环境）",
            "filename": "high_precision_config.yaml"
        },
        {
            "name": "fast_response",
            "description": "快速响应模式（实时控制）",
            "filename": "fast_response_config.yaml"
        },
        {
            "name": "safe_mode",
            "description": "安全模式（教学演示）",
            "filename": "safe_mode_config.yaml"
        },
        {
            "name": "deep_learning",
            "description": "深度学习模式（自适应抓取）",
            "filename": "deep_learning_config.yaml"
        },
        {
            "name": "arm_integration",
            "description": "机械臂集成模式（完整系统）",
            "filename": "arm_integration_config.yaml"
        }
    ]
    return templates


def create_config_template(template_name: str = "default") -> DemoConfig:
    """
    创建配置模板
    
    Args:
        template_name: 模板名称
        
    Returns:
        配置模板实例
        
    Raises:
        ValueError: 模板名称无效时抛出
    """
    config = DemoConfig()
    
    if template_name == "default":
        # 默认配置已创建
        pass
        
    elif template_name == "high_precision":
        # 高精度模式
        config.hardware.sensor.sampling_rate = 200  # Hz
        config.hardware.sensor.filter_enabled = True
        config.hardware.sensor.filter_type = "kalman"
        config.algorithm.inference_interval = 0.05  # 更快的推理
        config.algorithm.confidence_threshold = 0.9  # 更高的置信度阈值
        
    elif template_name == "fast_response":
        # 快速响应模式
        config.hardware.servo.speed = 100  # 最大速度
        config.algorithm.slip_detection_enabled = True
        config.algorithm.slip_window_size = 5  # 更小的窗口
        config.ui.plot_refresh_rate = 60  # 更高的刷新率
        config.emergency_timeout = 2.0  # 更短的紧急超时
        
    elif template_name == "safe_mode":
        # 安全模式
        config.hardware.servo.max_force = 20.0  # 限制最大力
        config.hardware.servo.speed = 50  # 中等速度
        config.max_force_limit = 30.0  # 系统力限制
        config.emergency_timeout = 10.0  # 更长的紧急超时
        config.ui.show_debug_info = False  # 简化界面
        
    elif template_name == "deep_learning":
        # 深度学习模式
        config.deep_learning = DeepLearningConfig()
        config.deep_learning.enabled = True
        config.deep_learning.inference_interval = 5  # 每5帧推理一次
        config.deep_learning.use_gpu = True
        config.deep_learning.adaptive_pid_enabled = True
        config.algorithm.slip_detection_enabled = True
        config.algorithm.adaptive_control_enabled = True
        
    elif template_name == "arm_integration":
        # 机械臂集成模式
        config.learm_arm = create_default_learm_config()
        config.camera = create_default_camera_config()
        config.simulation = create_default_sim_config()
        config.path_planning = create_default_path_planning_config()
        config.task = create_default_task_config()
        
    else:
        raise ValueError(f"未知的配置模板: {template_name}")
    
    return config


def validate_config_file(filepath: str) -> tuple:
    """
    验证配置文件
    
    Args:
        filepath: 配置文件路径
        
    Returns:
        (is_valid, errors, config_instance) 元组
        is_valid: 配置是否有效
        errors: 错误消息列表
        config_instance: 配置实例（如果有效）
    """
    try:
        config = load_config(filepath)
        errors = config.validate()
        
        if errors:
            return False, errors, None
        else:
            return True, [], config
            
    except Exception as e:
        return False, [f"配置文件加载失败: {str(e)}"], None


def merge_configs(base_config: DemoConfig, override_config: dict) -> DemoConfig:
    """
    合并配置（基础配置 + 覆盖配置）
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置字典
        
    Returns:
        合并后的配置实例
    """
    import copy
    
    # 创建配置副本
    merged = copy.deepcopy(base_config)
    
    def update_config(config_obj, updates):
        """递归更新配置对象"""
        for key, value in updates.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                
                # 如果是数据类且有子字段
                if hasattr(current_value, '__dataclass_fields__'):
                    if isinstance(value, dict):
                        update_config(current_value, value)
                    else:
                        setattr(config_obj, key, value)
                else:
                    setattr(config_obj, key, value)
            else:
                print(f"警告: 配置对象没有属性 '{key}'，跳过")
    
    # 更新配置
    update_config(merged, override_config)
    
    return merged


def create_arm_integration_config():
    """
    创建机械臂集成配置
    
    Returns:
        完整的集成配置字典
    """
    return {
        "system": {
            "name": "TactileGripper-ArmSystem",
            "version": "3.0.0",
            "mode": "real",  # real, simulation, hybrid
            "components": {
                "gripper": True,
                "arm": True,
                "vision": True,
                "planning": True,
                "simulation": False,
                "learning": True
            }
        },
        "integration": {
            "coordinate_system": "world",
            "hand_eye_calibrated": False,
            "communication": {
                "protocol": "ros",  # ros, direct, hybrid
                "sync_frequency": 100,  # Hz
                "timeout": 5.0  # seconds
            }
        }
    }


# 初始化代码
if __name__ == "__main__":
    import os
    
    print(f"触觉夹爪配置模块 v{__version__}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")
    
    # 显示可用配置模板
    templates = list_config_templates()
    print("\n可用配置模板:")
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")
    
    # 创建默认配置
    default_config = create_default_config()
    print(f"\n默认配置创建成功")
    print(f"传感器端口: {default_config.hardware.sensor.port}")
    print(f"舵机端口: {default_config.hardware.servo.port}")
    
    # 创建机械臂集成配置
    arm_config = create_default_learm_config()
    print(f"\n机械臂配置创建成功")
    print(f"机械臂类型: {arm_config.CONNECTION['type']}")
    print(f"关节数量: {arm_config.PHYSICAL['dof']}")