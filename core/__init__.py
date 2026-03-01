"""
触觉夹爪演示系统 - 核心逻辑模块
提供系统的核心功能，包括硬件接口、数据采集、控制逻辑和演示管理。
"""

from .hardware_interface import HardwareInterface, ArmState
from .data_acquisition import DataAcquisitionThread, SensorData
from .control_thread import ControlThread, ControlCommand, ControlStatus
from .demo_manager import DemoManager, DemoMode, DemoStatus
from .system_controller import SystemController
from .safety_monitor import SafetyMonitor

# 版本信息
__version__ = "3.0.0"
__author__ = "Tactile Gripper Team"
__description__ = "触觉夹爪演示系统核心逻辑模块（含机械臂/安全监控集成）"

# 导出列表
__all__ = [
    # 硬件接口
    "HardwareInterface",
    "ArmState",
    
    # 数据采集
    "DataAcquisitionThread",
    "SensorData",
    
    # 控制线程
    "ControlThread",
    "ControlCommand",
    "ControlStatus",
    
    # 演示管理
    "DemoManager",
    "DemoMode",
    "DemoStatus",

    # 系统协调/安全
    "SystemController",
    "SafetyMonitor",
    
    # 元信息
    "__version__",
    "__author__",
    "__description__",
]


def get_core_info() -> dict:
    """
    获取核心模块信息
    
    Returns:
        包含核心模块信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": [
            "HardwareInterface",
            "DataAcquisitionThread",
            "ControlThread",
            "DemoManager",
            "SystemController",
            "SafetyMonitor"
        ],
        "data_structures": [
            "SensorData",
            "ControlCommand",
            "ControlStatus",
            "DemoMode",
            "DemoStatus",
            "ArmState"
        ]
    }


# 系统状态枚举
class SystemState:
    """系统状态常量"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CALIBRATING = "calibrating"
    RUNNING_DEMO = "running_demo"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


# 系统事件常量
class SystemEvent:
    """系统事件常量"""
    INITIALIZE = "initialize"
    CONNECT_HARDWARE = "connect_hardware"
    DISCONNECT_HARDWARE = "disconnect_hardware"
    CALIBRATE = "calibrate"
    START_DEMO = "start_demo"
    STOP_DEMO = "stop_demo"
    PAUSE_DEMO = "pause_demo"
    RESUME_DEMO = "resume_demo"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


# 错误代码
class ErrorCode:
    """错误代码常量"""
    NO_ERROR = 0
    HARDWARE_NOT_FOUND = 1
    SENSOR_CONNECTION_FAILED = 2
    SERVO_CONNECTION_FAILED = 3
    CALIBRATION_FAILED = 4
    DEMO_NOT_FOUND = 5
    DEMO_ALREADY_RUNNING = 6
    DEMO_NOT_RUNNING = 7
    INVALID_PARAMETERS = 8
    SYSTEM_BUSY = 9
    UNKNOWN_ERROR = 99


class CoreException(Exception):
    """核心模块异常基类"""
    
    def __init__(self, message: str, error_code: int = ErrorCode.UNKNOWN_ERROR):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        return f"[Error {self.error_code}] {self.message}"


class HardwareException(CoreException):
    """硬件异常"""
    pass


class SensorException(HardwareException):
    """传感器异常"""
    pass


class ServoException(HardwareException):
    """舵机异常"""
    pass


class DemoException(CoreException):
    """演示异常"""
    pass


# 系统配置验证器
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_hardware_config(config: dict) -> list:
        """
        验证硬件配置
        
        Args:
            config: 硬件配置字典
            
        Returns:
            错误消息列表
        """
        errors = []
        
        # 检查传感器配置
        if 'sensor' not in config:
            errors.append("缺少传感器配置")
        else:
            sensor_config = config['sensor']
            if 'port' not in sensor_config or not sensor_config['port']:
                errors.append("传感器端口未配置")
            if 'baudrate' not in sensor_config or sensor_config['baudrate'] <= 0:
                errors.append("传感器波特率无效")
        
        # 检查舵机配置
        if 'servo' not in config:
            errors.append("缺少舵机配置")
        else:
            servo_config = config['servo']
            if 'port' not in servo_config or not servo_config['port']:
                errors.append("舵机端口未配置")
            if 'min_angle' not in servo_config or 'max_angle' not in servo_config:
                errors.append("舵机角度范围未配置")
            elif servo_config['min_angle'] >= servo_config['max_angle']:
                errors.append("舵机最小角度必须小于最大角度")
        
        return errors
    
    @staticmethod
    def validate_system_state(current_state: str, target_event: str) -> bool:
        """
        验证系统状态转换是否合法
        
        Args:
            current_state: 当前系统状态
            target_event: 目标事件
            
        Returns:
            状态转换是否合法
        """
        # 状态转换表
        state_transitions = {
            SystemState.UNINITIALIZED: [SystemEvent.INITIALIZE],
            SystemState.INITIALIZING: [SystemEvent.CONNECT_HARDWARE, SystemEvent.SHUTDOWN],
            SystemState.READY: [
                SystemEvent.CONNECT_HARDWARE,
                SystemEvent.SHUTDOWN
            ],
            SystemState.CONNECTING: [SystemEvent.DISCONNECT_HARDWARE, SystemEvent.SHUTDOWN],
            SystemState.CONNECTED: [
                SystemEvent.DISCONNECT_HARDWARE,
                SystemEvent.CALIBRATE,
                SystemEvent.START_DEMO,
                SystemEvent.EMERGENCY_STOP,
                SystemEvent.SHUTDOWN
            ],
            SystemState.CALIBRATING: [
                SystemEvent.STOP_DEMO,
                SystemEvent.EMERGENCY_STOP,
                SystemEvent.SHUTDOWN
            ],
            SystemState.RUNNING_DEMO: [
                SystemEvent.STOP_DEMO,
                SystemEvent.PAUSE_DEMO,
                SystemEvent.EMERGENCY_STOP,
                SystemEvent.SHUTDOWN
            ],
            SystemState.PAUSED: [
                SystemEvent.RESUME_DEMO,
                SystemEvent.STOP_DEMO,
                SystemEvent.EMERGENCY_STOP,
                SystemEvent.SHUTDOWN
            ],
            SystemState.ERROR: [
                SystemEvent.EMERGENCY_STOP,
                SystemEvent.SHUTDOWN
            ]
        }
        
        if current_state not in state_transitions:
            return False
        
        return target_event in state_transitions[current_state]


# 系统工具函数
class CoreUtils:
    """核心工具函数"""
    
    @staticmethod
    def create_timestamp() -> float:
        """
        创建时间戳
        
        Returns:
            当前时间戳（秒）
        """
        import time
        return time.time()
    
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """
        格式化时间戳
        
        Args:
            timestamp: 时间戳
            
        Returns:
            格式化后的时间字符串
        """
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    @staticmethod
    def calculate_data_rate(timestamps: list) -> float:
        """
        计算数据率
        
        Args:
            timestamps: 时间戳列表
            
        Returns:
            数据率（Hz）
        """
        if len(timestamps) < 2:
            return 0.0
        
        time_span = timestamps[-1] - timestamps[0]
        if time_span > 0:
            return len(timestamps) / time_span
        else:
            return 0.0
    
    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """
        限制值在范围内
        
        Args:
            value: 原始值
            min_value: 最小值
            max_value: 最大值
            
        Returns:
            限制后的值
        """
        return max(min_value, min(max_value, value))
    
    @staticmethod
    def map_value(value: float, 
                  in_min: float, in_max: float, 
                  out_min: float, out_max: float) -> float:
        """
        映射值到新范围
        
        Args:
            value: 原始值
            in_min: 输入范围最小值
            in_max: 输入范围最大值
            out_min: 输出范围最小值
            out_max: 输出范围最大值
            
        Returns:
            映射后的值
        """
        # 限制输入值在范围内
        value = CoreUtils.clamp(value, in_min, in_max)
        
        # 线性映射
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    @staticmethod
    def calculate_average(data: list, window_size: int = 10) -> list:
        """
        计算移动平均
        
        Args:
            data: 原始数据列表
            window_size: 窗口大小
            
        Returns:
            移动平均后的数据
        """
        if not data:
            return []
        
        if window_size <= 1:
            return data
        
        averaged_data = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window = data[start_idx:i+1]
            averaged_data.append(sum(window) / len(window))
        
        return averaged_data
    
    @staticmethod
    def detect_slips(force_data: list, threshold: float = 5.0) -> list:
        """
        检测滑动
        
        Args:
            force_data: 力数据列表
            threshold: 滑动阈值
            
        Returns:
            滑动检测结果列表
        """
        if len(force_data) < 2:
            return []
        
        slips = []
        for i in range(1, len(force_data)):
            delta = abs(force_data[i] - force_data[i-1])
            slips.append(delta > threshold)
        
        return slips


# 初始化代码
if __name__ == "__main__":
    print(f"触觉夹爪核心模块 v{__version__}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")
    
    # 显示核心模块信息
    core_info = get_core_info()
    print(f"\n可用模块: {', '.join(core_info['modules'])}")
    print(f"数据结构: {', '.join(core_info['data_structures'])}")
    
    # 测试工具函数
    print("\n工具函数测试:")
    timestamp = CoreUtils.create_timestamp()
    print(f"时间戳: {timestamp}")
    print(f"格式化时间: {CoreUtils.format_timestamp(timestamp)}")
    
    # 测试值限制和映射
    test_value = 150
    clamped = CoreUtils.clamp(test_value, 0, 100)
    mapped = CoreUtils.map_value(test_value, 0, 200, 0, 1)
    print(f"原始值: {test_value}, 限制后: {clamped}, 映射后: {mapped}")
    
    # 测试数据率计算
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    data_rate = CoreUtils.calculate_data_rate(timestamps)
    print(f"数据率: {data_rate:.2f} Hz")
    
    # 测试滑动检测
    force_data = [10.0, 10.5, 11.0, 16.0, 15.5, 15.0]
    slips = CoreUtils.detect_slips(force_data, threshold=4.0)
    print(f"力数据: {force_data}")
    print(f"滑动检测: {slips}")
