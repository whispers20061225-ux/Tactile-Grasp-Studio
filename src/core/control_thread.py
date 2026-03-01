"""
触觉夹爪演示系统 - 控制线程模块
负责舵机控制和系统状态管理。
"""

import threading
import time
import queue
import logging
import math
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

try:
    # 方案1：先尝试绝对导入
    from config import DemoConfig
except ImportError:
    try:
        # 方案2：如果是作为模块导入，尝试相对导入
        from ..config import DemoConfig
    except ImportError:
        try:
            # 方案3：直接导入 config 模块
            import sys
            import os
            # 添加父目录到路径
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import DemoConfig
        except ImportError as e:
            raise ImportError(f"无法导入 DemoConfig: {e}")

# 导入其他模块
from .hardware_interface import HardwareInterface, ServoState

try:
    from arm_control.cartesian_controller import CartesianController, Pose
except Exception:
    CartesianController = None
    Pose = None

# 可选：PyBullet 仿真进程客户端（用于“未连接真机时自动启仿真”）
try:
    from simulation.pybullet_process_client import PyBulletProcessClient
except Exception:
    PyBulletProcessClient = None

# 导入深度学习模型 - 使用绝对导入
try:
    from deep_learning.models import GripNet
    GRIPNET_AVAILABLE = True
except ImportError:
    try:
        from ..deep_learning.models import GripNet
        GRIPNET_AVAILABLE = True
    except ImportError:
        GRIPNET_AVAILABLE = False
        print("警告: GripNet 不可用，深度学习功能将被禁用")


class ControlCommand:
    """控制命令类"""
    
    def __init__(self, command_type: str, parameters: Dict[str, Any] = None):
        """
        初始化控制命令
        
        Args:
            command_type: 命令类型
            parameters: 命令参数
        """
        self.command_type = command_type
        self.parameters = parameters or {}
        self.timestamp = time.time()
        self.id = id(self)
    
    def __str__(self):
        return f"ControlCommand({self.command_type}, params={self.parameters})"


class ControlStatus(Enum):
    """控制状态枚举"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CALIBRATING = "calibrating"
    MOVING = "moving"
    HOLDING = "holding"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ControlState:
    """控制状态"""
    status: ControlStatus = ControlStatus.IDLE
    current_position: float = 0.0
    target_position: float = 0.0
    current_force: float = 0.0
    target_force: float = 0.0
    speed: float = 0.0
    torque_limit: float = 0.0
    temperature: float = 25.0
    voltage: float = 12.0
    error_code: int = 0
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': self.status.value,
            'current_position': self.current_position,
            'target_position': self.target_position,
            'current_force': self.current_force,
            'target_force': self.target_force,
            'speed': self.speed,
            'torque_limit': self.torque_limit,
            'temperature': self.temperature,
            'voltage': self.voltage,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }


class ControlThread(QThread):
    """控制线程"""
    
    # 信号定义
    status_updated = pyqtSignal(str, dict)  # 状态更新 (status, info)
    command_completed = pyqtSignal(str, dict)  # 命令完成 (command_type, result)
    error_occurred = pyqtSignal(str, dict)  # 错误发生 (error_type, info)
    
    def __init__(self, hardware_interface: HardwareInterface, config: DemoConfig,
                 arm_interface=None, joint_controller=None):
        """
        初始化控制线程
        
        Args:
            hardware_interface: 硬件接口
            config: 系统配置
            arm_interface: 可选机械臂接口
        """
        super().__init__()
        
        self.hardware_interface = hardware_interface
        self.arm_interface = arm_interface
        self.joint_controller = joint_controller
        self.config = config

        # 笛卡尔控制器（用于自动抓取与位姿控制）
        self.cartesian_controller = None
        if self.joint_controller is not None and CartesianController is not None:
            try:
                self.cartesian_controller = CartesianController(self.joint_controller)
            except Exception as e:
                self.logger = logging.getLogger(__name__)
                self.logger.warning(f"笛卡尔控制器初始化失败: {e}")

        # 仿真控制相关（当真机未连接时使用 PyBullet 作为替代）
        self.sim_client = None
        self._use_simulation_arm = False
        self._sim_joint_count = 6  # 默认 6 轴（dofbot: 5 关节 + 夹爪）
        self._sim_joint_targets = [0.0] * self._sim_joint_count
        self._sim_joint_zero_offsets = [0.0] * self._sim_joint_count
        self._arm_joint_targets = [0.0] * self._sim_joint_count  # 真机关节目标缓存
        self._arm_has_target = False  # 是否已下发过关节目标（用于区分“未设置”与“全为0”的零位）
        
        # 线程控制
        self.running = False
        self.paused = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 命令队列
        self.command_queue = queue.Queue(maxsize=100)
        
        # 控制状态
        self.control_state = ControlState()
        self.state_lock = threading.Lock()
        
        # PID控制器（用于力控制）
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.pid_last_time = 0.0
        
        # 深度学习模型
        self.grip_net = None
        self.model_loaded = False
        self.gripnet_available = GRIPNET_AVAILABLE
        
        # 日志
        self.logger = logging.getLogger(__name__)

        # 机械臂状态缓存
        self._arm_enabled = False
        self._arm_homed = False
        self._arm_status_last_update = 0.0
        self._arm_status_interval = 0.1
        
        # 初始化
        self._init_pid_controller()
        self._init_model()
    
    def _init_pid_controller(self):
        """初始化PID控制器"""
        # 使用配置中的PID参数，如果不存在则使用默认值
        if hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'pid_kp'):
            self.pid_kp = self.config.algorithm.pid_kp
            self.pid_ki = self.config.algorithm.pid_ki
            self.pid_kd = self.config.algorithm.pid_kd
        else:
            # 默认PID参数
            self.pid_kp = 1.0
            self.pid_ki = 0.1
            self.pid_kd = 0.05
        
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.pid_last_time = time.time()
    
    def _init_model(self):
        """初始化深度学习模型"""
        try:
            if not self.gripnet_available:
                self.logger.warning("GripNet不可用，跳过模型初始化")
                self.model_loaded = False
                return
            
            # 尝试从配置获取模型路径
            model_path = None
            if hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'model_path'):
                model_path = self.config.algorithm.model_path
            
            # 尝试从深度学习配置获取
            elif hasattr(self.config, 'deep_learning') and hasattr(self.config.deep_learning, 'model_path'):
                model_path = self.config.deep_learning.model_path
            
            if model_path:
                # 获取输入输出大小
                input_size = 10
                output_size = 3
                num_taxels = 9  # Paxini Gen3 M2020 有9个taxel
                
                if hasattr(self.config, 'sensor') and hasattr(self.config.sensor, 'num_taxels'):
                    num_taxels = self.config.sensor.num_taxels
                
                # 创建GripNet实例
                self.grip_net = GripNet(
                    input_size=input_size,
                    output_size=output_size,
                    num_taxels=num_taxels
                )
                
                # 这里应该加载预训练权重
                # self.grip_net.load_model(model_path)
                self.model_loaded = True
                self.logger.info("深度学习模型初始化成功")
        except Exception as e:
            self.logger.warning(f"深度学习模型初始化失败: {e}")
            self.model_loaded = False

    def _sync_arm_joint_target_size(self, joint_count: int) -> None:
        """同步真机关节目标缓存长度，避免UI目标长度不匹配。"""
        if not joint_count or joint_count <= 0:
            return
        if len(self._arm_joint_targets) == joint_count:
            return
        if len(self._arm_joint_targets) < joint_count:
            # 目标不足时补零，保持长度与关节数一致
            self._arm_joint_targets = (
                self._arm_joint_targets + [0.0] * (joint_count - len(self._arm_joint_targets))
            )
        else:
            # 目标过长时截断，防止越界显示
            self._arm_joint_targets = self._arm_joint_targets[:joint_count]
    
    def run(self):
        """线程主循环"""
        self.logger.info("控制线程启动")
        self.running = True
        
        # 发送状态信号
        self.status_updated.emit("starting", {"message": "控制线程启动"})
        
        while self.running:
            try:
                # 检查暂停状态
                self.mutex.lock()
                if self.paused:
                    self.condition.wait(self.mutex)
                self.mutex.unlock()
                
                # 处理命令队列
                self._process_commands()
                
                # 更新硬件状态
                self._update_hardware_status()
                
                # 控制循环（如果处于控制模式）
                self._control_loop()
                
                # 控制线程频率
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                self.logger.error(f"控制线程错误: {e}")
                self.error_occurred.emit("thread_error", {"error": str(e)})
                time.sleep(0.1)
        
        # 线程结束
        self.logger.info("控制线程停止")
        self.status_updated.emit("stopped", {"message": "控制线程停止"})
    
    def _process_commands(self):
        """处理命令队列"""
        try:
            # 非阻塞获取命令
            command = self.command_queue.get_nowait()
            
            # 处理命令
            result = self._execute_command(command)
            
            # 发送完成信号
            self.command_completed.emit(command.command_type, result)
            
            # 标记任务完成
            self.command_queue.task_done()
            
        except queue.Empty:
            # 队列为空，正常情况
            pass
        except Exception as e:
            self.logger.error(f"处理命令时出错: {e}")
            self.error_occurred.emit("command_error", {
                "error": str(e),
                "command": str(command) if 'command' in locals() else "unknown"
            })
    
    def _execute_command(self, command: ControlCommand) -> Dict[str, Any]:
        """
        执行控制命令
        
        Args:
            command: 控制命令
            
        Returns:
            执行结果
        """
        command_type = command.command_type
        params = command.parameters
        
        self.logger.info(f"执行命令: {command_type}, 参数: {params}")
        
        try:
            if command_type == "connect_hardware":
                return self._cmd_connect_hardware(params)
            elif command_type == "disconnect_hardware":
                return self._cmd_disconnect_hardware(params)
            elif command_type == "connect_stm32":
                return self._cmd_connect_stm32(params)
            elif command_type == "disconnect_stm32":
                return self._cmd_disconnect_stm32(params)
            elif command_type == "connect_tactile":
                return self._cmd_connect_tactile(params)
            elif command_type == "disconnect_tactile":
                return self._cmd_disconnect_tactile(params)
            elif command_type == "calibrate_hardware":
                return self._cmd_calibrate_hardware(params)
            elif command_type == "move_gripper":
                return self._cmd_move_gripper(params)
            elif command_type == "set_servo_position":
                return self._cmd_set_servo_position(params)
            elif command_type == "set_servo_speed":
                return self._cmd_set_servo_speed(params)
            elif command_type == "set_servo_force":
                return self._cmd_set_servo_force(params)
            elif command_type == "emergency_stop":
                return self._cmd_emergency_stop(params)
            elif command_type == "start_force_control":
                return self._cmd_start_force_control(params)
            elif command_type == "stop_force_control":
                return self._cmd_stop_force_control(params)
            elif command_type == "start_learning":
                return self._cmd_start_learning(params)
            elif command_type == "stop_learning":
                return self._cmd_stop_learning(params)
            elif command_type == "connect_arm":
                return self._cmd_connect_arm(params)
            elif command_type == "disconnect_arm":
                return self._cmd_disconnect_arm(params)
            elif command_type == "move_arm_pose":
                return self._cmd_move_arm_pose(params)
            elif command_type == "move_arm_joint":
                return self._cmd_move_arm_joint(params)
            elif command_type == "move_arm_joints":
                return self._cmd_move_arm_joints(params)
            elif command_type == "arm_home":
                return self._cmd_arm_home(params)
            elif command_type == "arm_enable":
                return self._cmd_arm_enable(params)
            elif command_type == "arm_disable":
                return self._cmd_arm_disable(params)
            elif command_type == "auto_grasp":
                return self._cmd_auto_grasp(params)
            else:
                raise ValueError(f"未知命令: {command_type}")
                
        except Exception as e:
            self.logger.error(f"执行命令失败: {command_type}, 错误: {e}")
            
            # 更新错误状态
            with self.state_lock:
                self.control_state.status = ControlStatus.ERROR
                self.control_state.error_code = 1
                self.control_state.error_message = str(e)
            
            return {"success": False, "error": str(e)}
    
    def _cmd_connect_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """连接硬件"""
        self.logger.info("开始连接硬件")

        # 更新连接状态
        with self.state_lock:
            self.control_state.status = ControlStatus.CONNECTING

        self.status_updated.emit("connecting", {"message": "正在连接硬件..."})

        # 先连接触觉传感器，再连接 STM32 舵机
        sensor_success = self.hardware_interface.connect_sensor()
        sensor_status = self.hardware_interface.get_status().get("sensor", {})
        sensor_message = "触觉已连接"
        if sensor_success and sensor_status.get("simulation"):
            sensor_message = "触觉已连接(模拟)"
        if not sensor_success:
            sensor_message = "触觉连接失败"
        self.status_updated.emit("tactile_connect_result", {
            "success": bool(sensor_success),
            "message": sensor_message,
            **sensor_status,
        })

        stm32_success = self.hardware_interface.connect_servo()
        stm32_status = self.hardware_interface.get_status().get("servo", {})
        stm32_message = "STM32 已连接"
        if stm32_success and stm32_status.get("simulation"):
            stm32_message = "STM32 已连接(模拟)"
        if not stm32_success:
            stm32_message = "STM32 连接失败"
        self.status_updated.emit("stm32_connect_result", {
            "success": bool(stm32_success),
            "message": stm32_message,
            **stm32_status,
        })

        success = sensor_success and stm32_success
        if success:
            with self.state_lock:
                self.control_state.status = ControlStatus.CONNECTED
            self.control_state.error_code = 0
            self.control_state.error_message = ""

            self.status_updated.emit("connected", {"message": "硬件连接成功"})
            return {"success": True, "message": "硬件连接成功"}

        with self.state_lock:
            self.control_state.status = ControlStatus.ERROR
            self.control_state.error_code = 2
            self.control_state.error_message = "硬件连接失败"

        self.status_updated.emit("error", {"message": "硬件连接失败"})
        return {"success": False, "error": "硬件连接失败"}

    def _cmd_disconnect_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """断开硬件"""
        self.logger.info("断开硬件连接")

        # 断开触觉与 STM32 连接
        self.hardware_interface.disconnect_sensor()
        self.hardware_interface.disconnect_servo()

        with self.state_lock:
            self.control_state.status = ControlStatus.IDLE

        self.status_updated.emit("disconnected", {"message": "硬件已断开"})
        return {"success": True, "message": "硬件已断开"}

    def _cmd_connect_stm32(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """连接 STM32"""
        success = self.hardware_interface.connect_servo()
        status = self.hardware_interface.get_status().get("servo", {})
        message = "STM32 已连接"
        if success and status.get("simulation"):
            message = "STM32 已连接(模拟)"
        if not success:
            message = "STM32 连接失败"
        self.status_updated.emit("stm32_connect_result", {
            "success": bool(success),
            "message": message,
            **status,
        })
        return {"success": bool(success), **status}

    def _cmd_disconnect_stm32(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """断开 STM32"""
        self.hardware_interface.disconnect_servo()
        self.status_updated.emit("stm32_disconnect_result", {
            "success": True,
            "message": "STM32 已断开",
        })
        return {"success": True}

    def _cmd_connect_tactile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """连接触觉传感器"""
        success = self.hardware_interface.connect_sensor()
        status = self.hardware_interface.get_status().get("sensor", {})
        message = "触觉已连接"
        if success and status.get("simulation"):
            message = "触觉已连接(模拟)"
        if not success:
            message = "触觉连接失败"
        self.status_updated.emit("tactile_connect_result", {
            "success": bool(success),
            "message": message,
            **status,
        })
        return {"success": bool(success), **status}

    def _cmd_disconnect_tactile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """断开触觉传感器"""
        self.hardware_interface.disconnect_sensor()
        self.status_updated.emit("tactile_disconnect_result", {
            "success": True,
            "message": "触觉已断开",
        })
        return {"success": True}

    def _cmd_calibrate_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """校准硬件命令"""
        calibration_type = params.get("type", "all")
        self.logger.info(f"执行校准硬件命令，类型: {calibration_type}")
        
        with self.state_lock:
            self.control_state.status = ControlStatus.CALIBRATING
        
        self.status_updated.emit("calibrating", {"message": "正在校准硬件..."})
        
        try:
            if calibration_type in ["sensor", "all"]:
                # 校准传感器
                success = self.hardware_interface.calibrate_sensor("zero")
                if not success:
                    raise Exception("传感器零点校准失败")
                
                success = self.hardware_interface.calibrate_sensor("scale")
                if not success:
                    raise Exception("传感器量程校准失败")
            
            if calibration_type in ["servo", "all"]:
                # 校准舵机
                success = self.hardware_interface.calibrate_servo("limits")
                if not success:
                    raise Exception("舵机极限位置校准失败")
            
            # 保存校准数据
            self.hardware_interface.save_calibration("calibration/latest.json")
            
            with self.state_lock:
                self.control_state.status = ControlStatus.CONNECTED
            
            self.status_updated.emit("calibrated", {"message": "硬件校准完成"})
            return {"success": True, "message": "硬件校准完成"}
            
        except Exception as e:
            with self.state_lock:
                self.control_state.status = ControlStatus.ERROR
                self.control_state.error_code = 3
                self.control_state.error_message = str(e)
            
            self.status_updated.emit("calibration_error", {"error": str(e)})
            return {"success": False, "error": str(e)}
    
    def _cmd_move_gripper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移动夹爪命令"""
        position = params.get("position", 50.0)
        speed = params.get("speed", 50.0)  # 默认速度
        
        # 从配置获取默认速度
        if hasattr(self.config, 'hardware') and hasattr(self.config.hardware, 'servo'):
            if not speed:  # 如果参数中没有指定速度
                speed = getattr(self.config.hardware.servo, 'speed', 50.0)
        
        self.logger.info(f"移动夹爪到位置: {position}, 速度: {speed}")
        
        with self.state_lock:
            self.control_state.status = ControlStatus.MOVING
            self.control_state.target_position = position
            self.control_state.speed = speed
        
        # 发送移动命令
        success = self.hardware_interface.set_servo_position(position, speed)
        
        if success:
            # 等待移动完成
            time.sleep(0.5)  # 简化处理，实际应该轮询状态
            
            with self.state_lock:
                self.control_state.status = ControlStatus.HOLDING
                self.control_state.current_position = position
            
            self.status_updated.emit("moved", {"position": position})
            return {"success": True, "position": position}
        else:
            with self.state_lock:
                self.control_state.status = ControlStatus.ERROR
                self.control_state.error_code = 4
            
            self.status_updated.emit("move_error", {"error": "移动失败"})
            return {"success": False, "error": "移动失败"}
    
    def _cmd_set_servo_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """设置舵机位置命令"""
        return self._cmd_move_gripper(params)
    
    def _cmd_set_servo_speed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """设置舵机速度命令"""
        speed = params.get("speed", 50.0)
        
        # 从配置获取默认速度
        if hasattr(self.config, 'hardware') and hasattr(self.config.hardware, 'servo'):
            if not speed:  # 如果参数中没有指定速度
                speed = getattr(self.config.hardware.servo, 'speed', 50.0)
        
        self.logger.info(f"设置舵机速度: {speed}")
        
        # 更新配置
        if hasattr(self.config, 'hardware') and hasattr(self.config.hardware, 'servo'):
            self.config.hardware.servo.speed = speed
        
        with self.state_lock:
            self.control_state.speed = speed
        
        self.status_updated.emit("speed_set", {"speed": speed})
        return {"success": True, "speed": speed}
    
    def _cmd_set_servo_force(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """设置舵机力限制命令"""
        force_limit = params.get("force", 30.0)
        
        # 从配置获取默认扭矩
        if hasattr(self.config, 'hardware') and hasattr(self.config.hardware, 'servo'):
            if not force_limit:  # 如果参数中没有指定力
                force_limit = getattr(self.config.hardware.servo, 'torque', 30.0)
        
        self.logger.info(f"设置舵机力限制: {force_limit}")
        
        # 更新配置
        if hasattr(self.config, 'hardware') and hasattr(self.config.hardware, 'servo'):
            self.config.hardware.servo.torque = force_limit
        
        with self.state_lock:
            self.control_state.torque_limit = force_limit
        
        self.status_updated.emit("force_limit_set", {"force_limit": force_limit})
        return {"success": True, "force_limit": force_limit}
    
    def _cmd_emergency_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """紧急停止命令"""
        self.logger.warning("执行紧急停止命令")
        
        # 执行紧急停止
        self.hardware_interface.emergency_stop()
        
        with self.state_lock:
            self.control_state.status = ControlStatus.EMERGENCY_STOP
        
        self.status_updated.emit("emergency_stop", {"message": "紧急停止已执行"})
        return {"success": True, "message": "紧急停止已执行"}
    
    def _cmd_start_force_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """开始力控制命令"""
        target_force = params.get("target_force", 30.0)
        
        self.logger.info(f"开始力控制，目标力: {target_force}N")
        
        # 设置目标力
        with self.state_lock:
            self.control_state.target_force = target_force
        
        # 这里应该启动力控制循环
        # 简化实现：只是设置状态
        
        self.status_updated.emit("force_control_started", {"target_force": target_force})
        return {"success": True, "target_force": target_force}
    
    def _cmd_stop_force_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """停止力控制命令"""
        self.logger.info("停止力控制")
        
        with self.state_lock:
            self.control_state.target_force = 0.0
        
        self.status_updated.emit("force_control_stopped", {})
        return {"success": True}
    
    def _cmd_start_learning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """开始学习命令"""
        self.logger.info("开始在线学习")
        
        # 这里应该启动学习循环
        # 简化实现：只是设置状态
        
        self.status_updated.emit("learning_started", {})
        return {"success": True}
    
    def _cmd_stop_learning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """停止学习命令"""
        self.logger.info("停止在线学习")
        
        self.status_updated.emit("learning_stopped", {})
        return {"success": True}

    # === 仿真机械臂辅助方法 ===
    def _build_simulation_config(self) -> Any:
        """构建仿真配置（优先使用 DemoConfig.simulation）"""
        try:
            if hasattr(self.config, "simulation") and self.config.simulation:
                return self.config.simulation
            # 没有显式仿真配置时，用默认 SimulationConfig
            from config.simulation_config import SimulationConfig
            return SimulationConfig()
        except Exception as exc:
            # 最简配置兜底，保证能启动 GUI
            self.logger.warning(f"读取仿真配置失败，使用简化配置: {exc}")
            return {"ENGINE": {"type": "pybullet", "mode": "gui"}}

    def _get_sim_section(self, sim_config: Any, name: str, default: Any) -> Any:
        if sim_config is None:
            return default
        if isinstance(sim_config, dict):
            return sim_config.get(name, default) or default
        return getattr(sim_config, name, default) or default

    def _resolve_sim_joint_zero_offsets(self, sim_config: Any) -> List[float]:
        arm_cfg = self._get_sim_section(sim_config, "ARM_SIMULATION", {}) or {}
        count = self._sim_joint_count

        offsets_deg = arm_cfg.get("joint_zero_offsets_deg")
        if offsets_deg is None:
            offsets_deg = arm_cfg.get("initial_joint_positions_deg")
        if isinstance(offsets_deg, (list, tuple)) and offsets_deg:
            offsets = [float(v) for v in offsets_deg]
            return (offsets + [0.0] * count)[:count]

        offsets_rad = arm_cfg.get("initial_joint_positions")
        if isinstance(offsets_rad, (list, tuple)) and offsets_rad:
            offsets = [math.degrees(float(v)) for v in offsets_rad]
            return (offsets + [0.0] * count)[:count]

        urdf_path = str(arm_cfg.get("urdf_path", ""))
        if os.path.basename(urdf_path) == "dofbot.urdf" and count >= 6:
            default_deg = [-90.0, 0.0, 0.0, 0.0, -90.0, 0.0]
            return (default_deg + [0.0] * count)[:count]

        return [0.0] * count

    def _ensure_simulation_arm(self) -> Tuple[bool, str]:
        """确保仿真机械臂已启动，用于替代真实硬件"""
        if PyBulletProcessClient is None:
            return False, "PyBullet 进程客户端不可用"
        try:
            # 如果已有仿真进程，先做一次健康检查
            if self.sim_client and self.sim_client.running:
                try:
                    # 读一次状态，失败说明进程不健康（比如窗口被关掉）
                    if not self.sim_client.get_state():
                        raise RuntimeError("simulator not responding")
                except Exception:
                    # 旧进程异常时强制重启，避免 UI 继续超时
                    self._shutdown_simulation_arm()

            if self.sim_client is None:
                sim_config = self._build_simulation_config()
                self.sim_client = PyBulletProcessClient(sim_config)
            else:
                sim_config = self._build_simulation_config()
            self._sim_joint_zero_offsets = self._resolve_sim_joint_zero_offsets(sim_config)
            if not self.sim_client.running:
                self.sim_client.start()
            # 切换到仿真模式
            self._use_simulation_arm = True
            self._arm_enabled = True
            self._arm_homed = True
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _shutdown_simulation_arm(self) -> None:
        """关闭仿真机械臂进程"""
        if self.sim_client:
            try:
                self.sim_client.shutdown()
            except Exception:
                pass
        self.sim_client = None
        self._use_simulation_arm = False

    def _sim_set_joint_targets(self, targets_deg: List[float]) -> bool:
        """将 UI 的角度（度）发送到 PyBullet（弧度）"""
        if not self.sim_client:
            return False
        # 保存一份“度”值，便于 UI 回显
        self._sim_joint_targets = list(targets_deg)
        # 角度→弧度（PyBullet 使用弧度制）
        offsets = self._sim_joint_zero_offsets
        actual_deg = [
            float(v) + (offsets[i] if i < len(offsets) else 0.0)
            for i, v in enumerate(self._sim_joint_targets)
        ]
        targets_rad = [math.radians(v) for v in actual_deg]
        self.sim_client.set_joint_targets(targets_rad)
        return True

    def _resolve_arm_connection_config(self) -> Tuple[str, str]:
        """解析机械臂连接配置（连接类型 + 串口），用于决定是否跳过真机连接"""
        connection_type = ""
        serial_port = ""

        if self.arm_interface and hasattr(self.arm_interface, "arm_config"):
            # arm_config 是 LearmInterface 解析后的连接配置
            cfg = self.arm_interface.arm_config or {}
            connection_type = str(cfg.get("type", "")).lower()
            serial_port = str(cfg.get("serial_port", "")).strip()
            return connection_type, serial_port

        # 兜底：从主配置里解析
        if hasattr(self.config, "learm_arm") and self.config.learm_arm:
            arm_profile = self.config.learm_arm
            if isinstance(arm_profile, dict):
                cfg = arm_profile.get("CONNECTION", arm_profile)
            else:
                cfg = getattr(arm_profile, "CONNECTION", {}) or {}
            connection_type = str(cfg.get("type", "")).lower()
            serial_port = str(cfg.get("serial_port", "")).strip()

        return connection_type, serial_port

    def _cmd_connect_arm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """连接机械臂"""
        error_message = None
        # 先判断配置是否明确要求仿真，或串口未配置（避免无意义的真机连接）
        conn_type, serial_port = self._resolve_arm_connection_config()
        shared_serial = None
        shared_lock = None
        if self.hardware_interface:
            protocol = getattr(self.hardware_interface, "servo_protocol", "")
            shared_serial = getattr(self.hardware_interface, "servo_serial", None)
            shared_lock = getattr(self.hardware_interface, "_stm32_lock", None)
            if protocol != "stm32":
                shared_serial = None
                shared_lock = None

        if conn_type == "simulation":
            error_message = "当前机械臂配置为仿真模式，已禁用自动切换"
            self.status_updated.emit("arm_connect_result", {
                "success": False,
                "message": error_message
            })
            return {"success": False, "error": error_message}
        if not serial_port and not (shared_serial and getattr(shared_serial, "is_open", False)):
            error_message = "机械臂串口未配置且STM32未连接，无法连接真机"
            self.status_updated.emit("arm_connect_result", {
                "success": False,
                "message": error_message
            })
            return {"success": False, "error": error_message}

        # 先尝试真机连接，失败后再回退到仿真
        if self.arm_interface and hasattr(self.arm_interface, "connect"):
            # 若 STM32 已连接，优先复用同一 CDC 串口
            if shared_serial and getattr(shared_serial, "is_open", False):
                success = self.arm_interface.connect(shared_serial=shared_serial, shared_lock=shared_lock)
            else:
                success = self.arm_interface.connect()
            if not success and hasattr(self.arm_interface, "status"):
                error_message = getattr(self.arm_interface.status, "error_msg", None)
        elif hasattr(self.hardware_interface, "connect_arm"):
            success = self.hardware_interface.connect_arm()
        else:
            success = False
        if success:
            # 真机连接成功时，优先走硬件控制
            self._arm_enabled = False
            self._arm_homed = False
            # 重新连接后清空目标缓存，避免UI显示旧目标
            self._arm_has_target = False
            self._arm_joint_targets = [0.0] * len(self._arm_joint_targets)
            if self._use_simulation_arm:
                self._shutdown_simulation_arm()
            self.status_updated.emit("arm_connect_result", {
                "success": True,
                "message": "机械臂连接成功"
            })
            return {"success": True, "mode": "hardware"}

        if not error_message:
            error_message = "机械臂连接失败"
        self.status_updated.emit("arm_connect_result", {
            "success": False,
            "message": error_message
        })
        return {"success": False, "error": error_message}

    def _cmd_disconnect_arm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """断开机械臂"""
        if self._use_simulation_arm:
            self._shutdown_simulation_arm()
        elif self.arm_interface and hasattr(self.arm_interface, "disconnect"):
            self.arm_interface.disconnect()
        elif hasattr(self.hardware_interface, "disconnect_arm"):
            self.hardware_interface.disconnect_arm()
        self._arm_enabled = False
        self._arm_homed = False
        # 断开后清空目标状态，避免下次连接残留
        self._arm_has_target = False
        self._arm_joint_targets = [0.0] * len(self._arm_joint_targets)
        return {"success": True}

    def _cmd_move_arm_pose(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移动机械臂到姿态（占位）"""
        pose = params.get("pose")
        if self.arm_interface and hasattr(self.arm_interface, "send_joint_command"):
            result = self.arm_interface.send_joint_command(pose)
        elif hasattr(self.hardware_interface, "get_arm_state"):
            # 确保标记为已连接
            if hasattr(self.hardware_interface, "arm_state"):
                self.hardware_interface.arm_state.connected = True
            result = {"status": "not_implemented", "pose": pose}
        else:
            result = {"status": "no_arm_interface", "pose": pose}
        return {"success": True, "result": result}

    def _cmd_move_arm_joint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移动单个关节角度"""
        joint_index = int(params.get("joint_index", -1))
        angle = params.get("angle", None)
        speed = params.get("speed", None)
        wait = bool(params.get("wait", False))

        if joint_index < 0:
            return {"success": False, "error": "invalid_joint_index"}
        if angle is None:
            return {"success": False, "error": "missing_angle"}

        # 仿真模式：直接将角度写入 PyBullet
        if self._use_simulation_arm:
            if joint_index >= self._sim_joint_count:
                return {"success": False, "error": "invalid_joint_index"}
            # 更新单个关节目标角（单位：度）
            self._sim_joint_targets[joint_index] = float(angle)
            success = self._sim_set_joint_targets(self._sim_joint_targets)
            return {"success": bool(success), "joint_index": joint_index, "angle": angle, "mode": "simulation"}

        joint_count = None
        if self.joint_controller:
            joint_count = self.joint_controller.num_joints
        elif self.arm_interface and hasattr(self.arm_interface, "num_joints"):
            joint_count = getattr(self.arm_interface, "num_joints", None)
        if joint_count:
            # 同步目标缓存长度，避免后续写入越界
            self._sync_arm_joint_target_size(joint_count)

        if not self.joint_controller and not self.arm_interface:
            return {"success": False, "error": "no_arm_interface"}
        if self.joint_controller and joint_index >= self.joint_controller.num_joints:
            return {"success": False, "error": "invalid_joint_index"}

        if self.joint_controller:
            success = self.joint_controller.move_single_joint(joint_index, float(angle), speed, wait)
        else:
            success = self.arm_interface.move_joint(joint_index + 1, float(angle), wait=wait)

        if success and 0 <= joint_index < len(self._arm_joint_targets):
            # 记录真机关节目标，供UI显示目标/误差
            self._arm_joint_targets[joint_index] = float(angle)
            self._arm_has_target = True

        return {"success": bool(success), "joint_index": joint_index, "angle": angle}

    def _cmd_move_arm_joints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移动多个关节角度"""
        angles = params.get("angles", None)
        speed = params.get("speed", None)
        wait = bool(params.get("wait", False))

        if not angles or not isinstance(angles, (list, tuple)):
            return {"success": False, "error": "invalid_angles"}

        # 仿真模式：批量设置目标角度
        if self._use_simulation_arm:
            angle_list = list(angles)
            if len(angle_list) < self._sim_joint_count:
                angle_list += [0.0] * (self._sim_joint_count - len(angle_list))
            if len(angle_list) > self._sim_joint_count:
                angle_list = angle_list[:self._sim_joint_count]
            success = self._sim_set_joint_targets(angle_list)
            return {"success": bool(success), "angles": angle_list, "mode": "simulation"}

        if not self.joint_controller:
            return {"success": False, "error": "no_joint_controller"}

        angle_list = list(angles)
        success = self.joint_controller.move_to_angles(angle_list, speed, wait)
        if success:
            joint_count = self.joint_controller.num_joints
            self._sync_arm_joint_target_size(joint_count)
            # 用本次指令刷新目标缓存，便于UI显示目标与误差
            targets = list(self._arm_joint_targets)
            if len(targets) < joint_count:
                targets += [0.0] * (joint_count - len(targets))
            for idx in range(min(len(angle_list), joint_count)):
                targets[idx] = float(angle_list[idx])
            self._arm_joint_targets = targets[:joint_count]
            self._arm_has_target = True
        return {"success": bool(success), "angles": angles}

    def _cmd_arm_home(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """机械臂回零"""
        speed = params.get("speed", None)
        wait = bool(params.get("wait", True))
        home_angles = None

        # 仿真模式：直接将关节置零
        if self._use_simulation_arm:
            home_angles = [0.0] * self._sim_joint_count
            success = self._sim_set_joint_targets(home_angles)
            if success:
                self._arm_homed = True
            return {"success": bool(success), "home_angles": home_angles, "mode": "simulation"}

        if not self.joint_controller:
            return {"success": False, "error": "no_joint_controller"}

        try:
            if hasattr(self.config, "learm_arm") and self.config.learm_arm is not None:
                motion = getattr(self.config.learm_arm, "MOTION", {})
                home_angles = motion.get("home_position")
        except Exception:
            home_angles = None

        if not home_angles:
            home_angles = [0.0] * self.joint_controller.num_joints

        # 回零角度仍为 0°，但在底层会叠加固定零点偏移
        success = self.joint_controller.move_to_angles(home_angles, speed, wait)
        if success:
            self._arm_homed = True
            joint_count = self.joint_controller.num_joints
            self._sync_arm_joint_target_size(joint_count)
            # 回零时把零位目标写入缓存（允许全为0的合法目标）
            targets = list(self._arm_joint_targets)
            if len(targets) < joint_count:
                targets += [0.0] * (joint_count - len(targets))
            for idx in range(min(len(home_angles), joint_count)):
                targets[idx] = float(home_angles[idx])
            self._arm_joint_targets = targets[:joint_count]
            self._arm_has_target = True

        return {"success": bool(success), "home_angles": home_angles}

    def _cmd_arm_enable(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """使能机械臂"""
        if self._use_simulation_arm:
            self._arm_enabled = True
            return {"success": True, "mode": "simulation"}
        if self.arm_interface and hasattr(self.arm_interface, "servo_on"):
            self.arm_interface.servo_on()
        self._arm_enabled = True
        return {"success": True}

    def _cmd_arm_disable(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """禁用机械臂"""
        if self._use_simulation_arm:
            self._arm_enabled = False
            return {"success": True, "mode": "simulation"}
        if self.arm_interface and hasattr(self.arm_interface, "servo_off"):
            self.arm_interface.servo_off()
        self._arm_enabled = False
        return {"success": True}


    def _get_hand_eye_config(self) -> Dict[str, Any]:
        """读取相机-夹爪的简化标定配置"""
        cfg = {}
        if self.arm_interface and hasattr(self.arm_interface, "arm_config"):
            cfg = self.arm_interface.arm_config or {}
        elif hasattr(self.config, "learm_arm") and self.config.learm_arm:
            arm_profile = self.config.learm_arm
            if isinstance(arm_profile, dict):
                cfg = arm_profile.get("CONNECTION", arm_profile) or {}
            else:
                cfg = getattr(arm_profile, "CONNECTION", {}) or {}

        return {
            "camera_to_gripper_offset_mm": list(cfg.get("camera_to_gripper_offset_mm", [0.0, 0.0, 0.0])),
            "camera_rotation_offset_rpy_deg": list(cfg.get("camera_rotation_offset_rpy_deg", [0.0, 0.0, 0.0])),
            "camera_joint_axis": list(cfg.get("camera_joint_axis", [0.0, 0.0, 1.0])),
            "camera_translation_rotate": bool(cfg.get("camera_translation_rotate", True)),
            "camera_use_joint5_rotation": bool(cfg.get("camera_use_joint5_rotation", False)),
            "camera_offset_is_gripper_to_camera": bool(cfg.get("camera_offset_is_gripper_to_camera", False)),
        }

    @staticmethod
    def _rotation_matrix_axis(axis: List[float], angle_deg: float) -> np.ndarray:
        """轴角 -> 旋转矩阵"""
        axis_v = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis_v)
        if norm <= 1e-9:
            raise ValueError("旋转轴长度为0")
        axis_v = axis_v / norm
        x, y, z = axis_v.tolist()
        theta = math.radians(angle_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        t = 1.0 - c
        return np.array(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_matrix_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """ZYX欧拉角 -> 旋转矩阵"""
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    @staticmethod
    def _pose_to_mat4(pose) -> np.ndarray:
        """末端位姿(Pose) -> 4x4齐次矩阵"""
        roll = math.radians(float(pose.roll))
        pitch = math.radians(float(pose.pitch))
        yaw = math.radians(float(pose.yaw))
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        r = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )
        t = np.array([float(pose.x), float(pose.y), float(pose.z)], dtype=float)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = r
        mat[:3, 3] = t
        return mat

    def _cmd_auto_grasp(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动夹取流程：
        1) 使用深度解算得到的相机坐标
        2) 结合关节5角度与手眼简化参数换算到基座坐标
        3) 规划预抓取 -> 抓取 -> 抬升
        """
        if Pose is None:
            return {"success": False, "error": "pose_class_missing"}
        if not self.cartesian_controller or not self.joint_controller:
            return {"success": False, "error": "no_arm_controller"}

        object_cam = params.get("object_cam_mm") or params.get("object_cam")
        if not object_cam or len(object_cam) != 3:
            return {"success": False, "error": "invalid_object_cam"}

        joint_index = int(params.get("joint5_index", 4))
        joint_angles = self.joint_controller.get_current_angles(update=True)
        if joint_index < 0 or joint_index >= len(joint_angles):
            return {"success": False, "error": "joint5_index_out_of_range"}
        joint5_angle = float(joint_angles[joint_index])

        hand_eye_cfg = self._get_hand_eye_config()
        t_cam_to_gripper = np.array(hand_eye_cfg["camera_to_gripper_offset_mm"], dtype=float)
        rpy_offset = hand_eye_cfg["camera_rotation_offset_rpy_deg"]
        axis = hand_eye_cfg["camera_joint_axis"]
        rotate_translation = bool(hand_eye_cfg["camera_translation_rotate"])
        use_joint_rotation = bool(hand_eye_cfg["camera_use_joint5_rotation"])
        offset_is_gripper = bool(hand_eye_cfg["camera_offset_is_gripper_to_camera"])

        # ???rpy_offset ?????5=0 ????? -> ????????
        r_offset = self._rotation_matrix_zyx(float(rpy_offset[0]), float(rpy_offset[1]), float(rpy_offset[2]))
        r_joint = self._rotation_matrix_axis(axis, joint5_angle) if use_joint_rotation else np.eye(3, dtype=float)

        # ??????????????????????5?????????????
        r_cam_gripper = (r_joint @ r_offset) if use_joint_rotation else r_offset

        # 计算相机->夹爪的平移向量（转换到夹爪坐标系）
        if offset_is_gripper:
            # 偏移量为“夹爪->相机”（夹爪坐标系测量）
            t_gripper_cam_vec = t_cam_to_gripper.copy()
        else:
            # 偏移量为“相机->夹爪”（相机坐标系测量）
            # 需要转换到夹爪坐标系：t_GC = -R_GC * t_CG
            t_gripper_cam_vec = -r_cam_gripper @ t_cam_to_gripper

        t_gripper_cam = np.eye(4, dtype=float)
        t_gripper_cam[:3, :3] = r_cam_gripper
        t_gripper_cam[:3, 3] = t_gripper_cam_vec

        # ?????????????->??
        current_pose = self.cartesian_controller.get_current_pose(update=True)
        t_base_gripper = self._pose_to_mat4(current_pose)
        t_base_cam = t_base_gripper @ t_gripper_cam

        obj_cam = np.array([float(object_cam[0]), float(object_cam[1]), float(object_cam[2]), 1.0], dtype=float)
        obj_base = t_base_cam @ obj_cam
        target_x, target_y, target_z = obj_base[:3].tolist()

        approach = float(params.get("approach_offset_mm", 50.0))
        speed = float(params.get("speed", 0.3))
        close_gripper = bool(params.get("close_gripper", True))
        close_pos = float(params.get("gripper_close_position", 40.0))
        open_pos = params.get("gripper_open_position")

        # 生成预抓取/抓取/抬升位姿（保持当前姿态）
        pre_pose = Pose(target_x, target_y, target_z + approach, current_pose.roll, current_pose.pitch, current_pose.yaw)
        grasp_pose = Pose(target_x, target_y, target_z, current_pose.roll, current_pose.pitch, current_pose.yaw)
        lift_pose = Pose(target_x, target_y, target_z + approach, current_pose.roll, current_pose.pitch, current_pose.yaw)

        with self.state_lock:
            self.control_state.status = ControlStatus.MOVING

        if open_pos is not None:
            self._cmd_move_gripper({"position": float(open_pos)})

        if not self.cartesian_controller.move_to_pose(pre_pose, speed=speed, wait=True):
            return {"success": False, "error": "pre_grasp_failed"}
        if not self.cartesian_controller.move_to_pose(grasp_pose, speed=speed, wait=True):
            return {"success": False, "error": "grasp_failed"}

        if close_gripper:
            self._cmd_move_gripper({"position": close_pos})

        if not self.cartesian_controller.move_to_pose(lift_pose, speed=speed, wait=True):
            return {"success": False, "error": "lift_failed"}

        with self.state_lock:
            self.control_state.status = ControlStatus.HOLDING

        self.status_updated.emit("auto_grasp_result", {
            "success": True,
            "target_base": [target_x, target_y, target_z],
        })
        return {"success": True, "target_base": [target_x, target_y, target_z]}
    
    def _update_hardware_status(self):
        """更新硬件状态"""
        try:
            now = time.time()

            # 获取舵机状态
            poll_servo_state = True
            # 当机械臂与舵机共用 STM32 CDC 串口且舵机 ID 与关节 ID 冲突时，
            # 频繁轮询会抢占响应，导致 J1 等关节“读数很慢/回零”的假象
            try:
                protocol = getattr(self.hardware_interface, "servo_protocol", "")
                servo_id = getattr(self.hardware_interface, "servo_id", None)
                if protocol == "stm32" and self.arm_interface:
                    arm_connected = False
                    if hasattr(self.arm_interface, "is_connected"):
                        arm_connected = self.arm_interface.is_connected()
                    if arm_connected and servo_id is not None:
                        joint_count = getattr(self.arm_interface, "num_joints", 6)
                        if 1 <= int(servo_id) <= int(joint_count):
                            poll_servo_state = False
            except Exception:
                poll_servo_state = True

            servo_state = self.hardware_interface.get_servo_state() if poll_servo_state else None
            
            if servo_state:
                with self.state_lock:
                    self.control_state.current_position = servo_state.position
                    self.control_state.temperature = servo_state.temperature
                    self.control_state.voltage = servo_state.voltage

            # 获取机械臂状态
            if (now - self._arm_status_last_update) >= self._arm_status_interval:
                self._arm_status_last_update = now
                if self._use_simulation_arm and self.sim_client:
                    # 仿真模式：从 PyBullet 读取关节角度（弧度→度）
                    state = self.sim_client.get_state()
                    if state:
                        control_names = state.get("control_joint_names", [])
                        if control_names:
                            # 同步仿真关节数量，避免长度不一致
                            self._sim_joint_count = len(control_names)
                            if len(self._sim_joint_targets) != self._sim_joint_count:
                                self._sim_joint_targets = (
                                    self._sim_joint_targets + [0.0] * self._sim_joint_count
                                )[:self._sim_joint_count]
                            if len(self._sim_joint_zero_offsets) != self._sim_joint_count:
                                self._sim_joint_zero_offsets = (
                                    self._sim_joint_zero_offsets + [0.0] * self._sim_joint_count
                                )[:self._sim_joint_count]
                        angles_rad = state.get("joint_angles", [])
                        vels_rad = state.get("joint_velocities", [])
                        torques = state.get("joint_torques", [])
                        targets_rad = state.get("joint_targets", [])
                        angles_deg = [math.degrees(v) for v in angles_rad]
                        vels_deg = [math.degrees(v) for v in vels_rad]
                        targets_deg = [math.degrees(v) for v in targets_rad]
                        if self._sim_joint_zero_offsets:
                            offsets = self._sim_joint_zero_offsets
                            angles_deg = [
                                v - (offsets[i] if i < len(offsets) else 0.0)
                                for i, v in enumerate(angles_deg)
                            ]
                            targets_deg = [
                                v - (offsets[i] if i < len(offsets) else 0.0)
                                for i, v in enumerate(targets_deg)
                            ]
                        if not targets_deg:
                            targets_deg = list(self._sim_joint_targets)
                        self.status_updated.emit("arm_state", {
                            "connected": True,
                            "enabled": bool(self._arm_enabled),
                            "homed": bool(self._arm_homed),
                            "joint_angles": angles_deg,
                            "joint_positions": angles_deg,
                            "joint_velocities": vels_deg,
                            "joint_torques": torques,
                            "joint_targets": targets_deg,
                            "battery_voltage": 0.0,
                            "safety": "normal",
                            "control_mode": "joint",
                            "connection_type": "simulation",
                        })
                elif self.arm_interface:
                    arm_status = self.arm_interface.update_status()
                    if arm_status:
                        joint_angles = list(arm_status.joint_angles or [])
                        joint_positions = list(arm_status.joint_positions or [])
                        joint_count = len(joint_angles) or len(joint_positions)
                        if joint_count:
                            self._sync_arm_joint_target_size(joint_count)
                        # 未下发过目标时，用当前角度作为显示目标，避免误差假阳性
                        if self._arm_has_target and joint_count:
                            joint_targets = list(self._arm_joint_targets[:joint_count])
                        elif joint_angles:
                            joint_targets = list(joint_angles)
                        elif joint_positions:
                            joint_targets = list(joint_positions)
                        else:
                            joint_targets = []

                        self.status_updated.emit("arm_state", {
                            "connected": bool(arm_status.connected),
                            "enabled": bool(self._arm_enabled),
                            "homed": bool(self._arm_homed),
                            "joint_angles": joint_angles,
                            "joint_positions": joint_positions,
                            "joint_targets": joint_targets,
                            "battery_voltage": getattr(arm_status, "battery_voltage", 0.0),
                            "safety": "normal",
                            "control_mode": "joint",
                            "connection_type": "hardware",
                        })
            
            # 定期发送状态更新
            if now - self.control_state.timestamp > 0.5:  # 每0.5秒更新一次
                with self.state_lock:
                    self.control_state.timestamp = now
                    state_dict = self.control_state.to_dict()
                
                self.status_updated.emit("status_update", state_dict)
                
        except Exception as e:
            self.logger.warning(f"更新硬件状态失败: {e}")
    
    def _control_loop(self):
        """控制循环"""
        # 根据当前状态执行控制逻辑
        with self.state_lock:
            current_status = self.control_state.status
        
        if current_status == ControlStatus.MOVING:
            # 移动控制逻辑
            self._position_control_loop()
        elif current_status == ControlStatus.HOLDING and self.control_state.target_force > 0:
            # 力控制逻辑
            self._force_control_loop()
    
    def _position_control_loop(self):
        """位置控制循环"""
        # 简化的位置控制
        # 实际实现应该包含PID控制和位置反馈
        
        target_pos = self.control_state.target_position
        current_pos = self.control_state.current_position
        
        # 检查是否到达目标位置
        position_tolerance = 1.0  # 度
        if abs(target_pos - current_pos) < position_tolerance:
            with self.state_lock:
                self.control_state.status = ControlStatus.HOLDING
    
    def _force_control_loop(self):
        """力控制循环"""
        # 简化的力控制
        # 实际实现应该包含PID控制和力反馈
        
        target_force = self.control_state.target_force
        current_force = self.control_state.current_force
        
        # 使用PID计算位置调整
        error = target_force - current_force
        current_time = time.time()
        dt = current_time - self.pid_last_time if self.pid_last_time > 0 else 0.01
        
        # PID计算
        self.pid_integral += error * dt
        derivative = (error - self.pid_last_error) / dt if dt > 0 else 0
        
        output = (self.pid_kp * error + 
                 self.pid_ki * self.pid_integral + 
                 self.pid_kd * derivative)
        
        # 限制输出
        output = max(-10, min(10, output))  # 限制调整范围
        
        # 保存状态
        self.pid_last_error = error
        self.pid_last_time = current_time
        
        # 调整位置
        if abs(output) > 0.5:  # 死区
            new_position = self.control_state.current_position + output
            self.hardware_interface.set_servo_position(new_position)
    
    def send_command(self, command_type: str, parameters: Dict[str, Any] = None) -> bool:
        """
        发送控制命令
        
        Args:
            command_type: 命令类型
            parameters: 命令参数
            
        Returns:
            是否成功添加到队列
        """
        try:
            command = ControlCommand(command_type, parameters)
            self.command_queue.put(command, timeout=0.1)
            return True
        except queue.Full:
            self.logger.warning("命令队列已满")
            return False
        except Exception as e:
            self.logger.error(f"发送命令失败: {e}")
            return False
    
    def get_state(self) -> ControlState:
        """
        获取当前控制状态
        
        Returns:
            控制状态
        """
        with self.state_lock:
            return self.control_state
    
    def start_control(self):
        """开始控制线程"""
        if not self.running:
            self.start()
        elif self.paused:
            self.resume_control()
    
    def stop_control(self):
        """停止控制线程"""
        self.running = False
        self.resume_control()  # 确保线程不会在暂停状态下卡住
        self.wait()
    
    def pause_control(self):
        """暂停控制线程"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
        self.status_updated.emit("paused", {"message": "控制线程暂停"})
        self.logger.info("控制线程暂停")
    
    def resume_control(self):
        """恢复控制线程"""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
        self.status_updated.emit("resumed", {"message": "控制线程恢复"})
        self.logger.info("控制线程恢复")
    
    def clear_command_queue(self):
        """清空命令队列"""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
            except queue.Empty:
                break
        
        self.logger.info("命令队列已清空")
    
    def is_running(self) -> bool:
        """检查是否在运行"""
        return self.running and not self.paused
    
    def is_paused(self) -> bool:
        """检查是否暂停"""
        return self.paused
