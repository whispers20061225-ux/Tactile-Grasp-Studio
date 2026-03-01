"""
触觉夹爪演示系统 - 舵机控制模块
提供高级的舵机控制接口，包括位置控制、速度控制、力控制和轨迹规划。
"""

from .gripper_controller import (
    # 主控制器
    GripperController,
    
    # 数据类
    GripperState,
    ControlMode,
    ControlStatus,
    
    # 控制算法
    TrajectoryPlanner,
    ForceController,
    PositionController,
    VelocityController,
    ImpedanceController,
    
    # 滤波器
    KalmanFilter,
    LowPassFilter,
    MovingAverageFilter,
    
    # 工具函数
    #angle_to_pulse,
    #pulse_to_angle,
    #position_to_pulse,
    #pulse_to_position,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Tactile Gripper Team"
__description__ = "触觉夹爪演示系统舵机控制模块"

# 导出列表
__all__ = [
    # 主控制器
    "GripperController",
    
    # 数据类
    "GripperState",
    "ControlMode",
    "ControlStatus",
    
    # 控制算法
    "TrajectoryPlanner",
    "ForceController",
    "PositionController",
    "VelocityController",
    "ImpedanceController",
    
    # 滤波器
    "KalmanFilter",
    "LowPassFilter",
    "MovingAverageFilter",
    
    # 工具函数
    "angle_to_pulse",
    "pulse_to_angle",
    "position_to_pulse",
    "pulse_to_position",
    
    # 元信息
    "__version__",
    "__author__",
    "__description__",
]


def get_servo_control_info() -> dict:
    """
    获取舵机控制模块信息
    
    Returns:
        包含模块信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_classes": [
            "GripperController",
            "GripperState",
            "TrajectoryPlanner",
            "ForceController",
            "PositionController",
            "VelocityController",
            "ImpedanceController",
            "KalmanFilter",
            "LowPassFilter",
            "MovingAverageFilter",
        ],
        "available_functions": [
            "angle_to_pulse",
            "pulse_to_angle",
            "position_to_pulse",
            "pulse_to_position",
        ]
    }


# 转换函数
def angle_to_pulse(angle: float, min_angle: float = 0.0, max_angle: float = 180.0,
                  min_pulse: int = 500, max_pulse: int = 2500) -> int:
    """
    角度转换为脉冲宽度
    
    Args:
        angle: 角度值 (度)
        min_angle: 最小角度
        max_angle: 最大角度
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        脉冲宽度 (μs)
    """
    # 限制角度在范围内
    angle = max(min_angle, min(max_angle, angle))
    
    # 线性映射
    pulse = int((angle - min_angle) * (max_pulse - min_pulse) / 
                (max_angle - min_angle) + min_pulse)
    
    return pulse


def pulse_to_angle(pulse: int, min_angle: float = 0.0, max_angle: float = 180.0,
                  min_pulse: int = 500, max_pulse: int = 2500) -> float:
    """
    脉冲宽度转换为角度
    
    Args:
        pulse: 脉冲宽度 (μs)
        min_angle: 最小角度
        max_angle: 最大角度
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        角度值 (度)
    """
    # 限制脉冲在范围内
    pulse = max(min_pulse, min(max_pulse, pulse))
    
    # 线性映射
    angle = (pulse - min_pulse) * (max_angle - min_angle) / \
            (max_pulse - min_pulse) + min_angle
    
    return angle


def position_to_pulse(position: float, min_position: float = 0.0, max_position: float = 100.0,
                     min_pulse: int = 500, max_pulse: int = 2500) -> int:
    """
    位置转换为脉冲宽度
    
    Args:
        position: 位置值 (mm 或 %)
        min_position: 最小位置
        max_position: 最大位置
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        脉冲宽度 (μs)
    """
    # 限制位置在范围内
    position = max(min_position, min(max_position, position))
    
    # 线性映射
    pulse = int((position - min_position) * (max_pulse - min_pulse) / 
                (max_position - min_position) + min_pulse)
    
    return pulse


def pulse_to_position(pulse: int, min_position: float = 0.0, max_position: float = 100.0,
                     min_pulse: int = 500, max_pulse: int = 2500) -> float:
    """
    脉冲宽度转换为位置
    
    Args:
        pulse: 脉冲宽度 (μs)
        min_position: 最小位置
        max_position: 最大位置
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        位置值 (mm 或 %)
    """
    # 限制脉冲在范围内
    pulse = max(min_pulse, min(max_pulse, pulse))
    
    # 线性映射
    position = (pulse - min_pulse) * (max_position - min_position) / \
               (max_pulse - min_pulse) + min_position
    
    return position

# 转换函数
def angle_to_pulse(angle: float, min_angle: float = 0.0, max_angle: float = 180.0,
                  min_pulse: int = 500, max_pulse: int = 2500) -> int:
    """
    角度转换为脉冲宽度
    
    Args:
        angle: 角度值 (度)
        min_angle: 最小角度
        max_angle: 最大角度
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        脉冲宽度 (μs)
    """
    # 限制角度在范围内
    angle = max(min_angle, min(max_angle, angle))
    
    # 线性映射
    pulse = int((angle - min_angle) * (max_pulse - min_pulse) / 
                (max_angle - min_angle) + min_pulse)
    
    return pulse


def pulse_to_angle(pulse: int, min_angle: float = 0.0, max_angle: float = 180.0,
                  min_pulse: int = 500, max_pulse: int = 2500) -> float:
    """
    脉冲宽度转换为角度
    
    Args:
        pulse: 脉冲宽度 (μs)
        min_angle: 最小角度
        max_angle: 最大角度
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        角度值 (度)
    """
    # 限制脉冲在范围内
    pulse = max(min_pulse, min(max_pulse, pulse))
    
    # 线性映射
    angle = (pulse - min_pulse) * (max_angle - min_angle) / \
            (max_pulse - min_pulse) + min_angle
    
    return angle


def position_to_pulse(position: float, min_position: float = 0.0, max_position: float = 100.0,
                     min_pulse: int = 500, max_pulse: int = 2500) -> int:
    """
    位置转换为脉冲宽度
    
    Args:
        position: 位置值 (mm 或 %)
        min_position: 最小位置
        max_position: 最大位置
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        脉冲宽度 (μs)
    """
    # 限制位置在范围内
    position = max(min_position, min(max_position, position))
    
    # 线性映射
    pulse = int((position - min_position) * (max_pulse - min_pulse) / 
                (max_position - min_position) + min_pulse)
    
    return pulse


def pulse_to_position(pulse: int, min_position: float = 0.0, max_position: float = 100.0,
                     min_pulse: int = 500, max_pulse: int = 2500) -> float:
    """
    脉冲宽度转换为位置
    
    Args:
        pulse: 脉冲宽度 (μs)
        min_position: 最小位置
        max_position: 最大位置
        min_pulse: 最小脉冲宽度 (μs)
        max_pulse: 最大脉冲宽度 (μs)
    
    Returns:
        位置值 (mm 或 %)
    """
    # 限制脉冲在范围内
    pulse = max(min_pulse, min(max_pulse, pulse))
    
    # 线性映射
    position = (pulse - min_pulse) * (max_position - min_position) / \
               (max_pulse - min_pulse) + min_position
    
    return position


# 初始化代码
if __name__ == "__main__":
    print(f"触觉夹爪舵机控制模块 v{__version__}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")
    
    # 显示模块信息
    module_info = get_servo_control_info()
    print(f"\n可用类: {', '.join(module_info['available_classes'])}")
    print(f"可用函数: {', '.join(module_info['available_functions'])}")
    
    # 测试转换函数
    print("\n转换函数测试:")
    
    # 角度转脉冲测试
    test_angle = 90.0
    test_pulse = angle_to_pulse(test_angle, 0, 180, 500, 2500)
    recovered_angle = pulse_to_angle(test_pulse, 0, 180, 500, 2500)
    print(f"角度 {test_angle}° -> 脉冲 {test_pulse}μs -> 角度 {recovered_angle}°")
    
    # 位置转脉冲测试
    test_position = 50.0
    test_pulse = position_to_pulse(test_position, 0, 100, 500, 2500)
    recovered_position = pulse_to_position(test_pulse, 0, 100, 500, 2500)
    print(f"位置 {test_position}% -> 脉冲 {test_pulse}μs -> 位置 {recovered_position}%")