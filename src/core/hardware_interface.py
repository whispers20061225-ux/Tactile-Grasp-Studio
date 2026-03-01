"""
触觉夹爪演示系统 - 硬件接口模块
负责与传感器和舵机的底层通信。
支持三维力数据 (Fx, Fy, Fz)
"""

import serial
import time
import logging
import numpy as np
import sys
import os
import struct
import random
import math
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用绝对导入
try:
    from config import HardwareConfig, SensorConfig, ServoConfig
except ImportError as e:
    print(f"警告: 无法导入配置类: {e}")
    # 定义备用类
    @dataclass
    class HardwareConfig:
        sensor: Any = None
        servo: Any = None
    
    @dataclass
    class SensorConfig:
        port: str = "COM3"
        baudrate: int = 115200
        num_taxels: int = 9
        timeout: float = 1.0
        # Paxini Gen3 M2020 特定配置
        data_format: str = "binary"  # 或 "ascii"
        pressure_range: float = 100.0  # kPa，传感器最大量程
        sample_rate: int = 100  # Hz
        calibration_factor: float = 1.0  # 校准因子
        # 三维力参数
        friction_coefficient: float = 0.3
        force_noise_level: float = 0.1
        max_shear_force: float = 5.0
    
    @dataclass 
    class ServoConfig:
        port: str = "COM4"
        baudrate: int = 115200
        min_angle: float = 0
        max_angle: float = 180
        speed: int = 50
        timeout: float = 1.0
        home_position: float = 90
        min_pulse: int = 500
        max_pulse: int = 2500
        torque: float = 100


class HardwareStatus(Enum):
    """硬件状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class SensorReading:
    """传感器读数 - 支持三维力数据"""
    timestamp: float
    force_data: List[float]  # Z方向力值 (kPa) - 向后兼容
    force_vectors: Optional[List[List[float]]] = None  # 三维力向量 [Fx, Fy, Fz] (N)
    temperature: Optional[float] = None
    status: int = 0
    raw_data: Optional[bytes] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.force_data:
            self.force_data = []
        
        # 如果 force_vectors 未提供，从 force_data 生成
        if self.force_vectors is None:
            self.force_vectors = [[0.0, 0.0, f] for f in self.force_data]
    
    @property
    def total_force(self) -> float:
        """总法向力 (kPa) - 向后兼容"""
        return sum(self.force_data) if self.force_data else 0.0
    
    @property
    def resultant_force(self) -> List[float]:
        """合力向量 [Fx, Fy, Fz] (N)"""
        if self.force_vectors:
            result = [0.0, 0.0, 0.0]
            for vec in self.force_vectors:
                result[0] += vec[0]
                result[1] += vec[1]
                result[2] += vec[2]
            return result
        else:
            return [0.0, 0.0, self.total_force]
    
    @property
    def total_normal_force(self) -> float:
        """总法向力 (N)"""
        if self.force_vectors:
            return sum(vec[2] for vec in self.force_vectors)
        return self.total_force
    
    @property
    def total_shear_force(self) -> float:
        """总剪切力大小 (N)"""
        if self.force_vectors:
            shear_x = sum(vec[0] for vec in self.force_vectors)
            shear_y = sum(vec[1] for vec in self.force_vectors)
            return math.sqrt(shear_x**2 + shear_y**2)
        return 0.0
    
    @property
    def average_force(self) -> float:
        """平均力 (kPa) - 向后兼容"""
        return self.total_force / len(self.force_data) if self.force_data else 0.0
    
    @property
    def max_force(self) -> float:
        """最大力 (kPa) - 向后兼容"""
        return max(self.force_data) if self.force_data else 0.0
    
    @property
    def min_force(self) -> float:
        """最小力 (kPa) - 向后兼容"""
        return min(self.force_data) if self.force_data else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'force_data': self.force_data,
            'force_vectors': self.force_vectors,
            'temperature': self.temperature,
            'status': self.status,
            'total_force': self.total_force,
            'total_normal_force': self.total_normal_force,
            'total_shear_force': self.total_shear_force,
            'resultant_force': self.resultant_force,
            'average_force': self.average_force,
            'max_force': self.max_force,
            'min_force': self.min_force
        }
    
    def get_force_magnitudes(self) -> List[float]:
        """获取每个触点的力大小"""
        if self.force_vectors:
            return [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in self.force_vectors]
        return self.force_data
    
    def get_force_directions(self) -> List[List[float]]:
        """获取每个触点的力方向（单位向量）"""
        if self.force_vectors:
            directions = []
            for vec in self.force_vectors:
                magnitude = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                if magnitude > 0:
                    directions.append([vec[0]/magnitude, vec[1]/magnitude, vec[2]/magnitude])
                else:
                    directions.append([0.0, 0.0, 0.0])
            return directions
        return [[0.0, 0.0, 1.0] for _ in range(len(self.force_data))]


@dataclass
class ArmState:
    """机械臂状态（占位实现）"""
    timestamp: float
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    end_effector_pose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz] 或齐次矩阵展开
    connected: bool = False


@dataclass
class ServoState:
    """舵机状态"""
    timestamp: float
    position: float  # 位置（度或mm）
    velocity: float = 0.0  # 速度
    current: float = 0.0  # 电流
    temperature: float = 25.0  # 温度
    load: float = 0.0  # 负载百分比
    voltage: float = 12.0  # 电压
    moving: bool = False  # 是否在运动
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'position': self.position,
            'velocity': self.velocity,
            'current': self.current,
            'temperature': self.temperature,
            'load': self.load,
            'voltage': self.voltage,
            'moving': self.moving
        }


class HardwareInterface:
    """硬件接口类 - 支持三维力数据"""
    
    def __init__(self, config: HardwareConfig = None):
        """
        初始化硬件接口
        
        Args:
            config: 硬件配置，如果为None则创建默认配置
        """
        self.logger = logging.getLogger(__name__)
        
        # 如果未提供配置，创建默认配置
        if config is None:
            self.logger.warning("未提供硬件配置，使用默认配置")
            self.config = HardwareConfig(
                sensor=SensorConfig(),
                servo=ServoConfig()
            )
        else:
            self.config = config
        
        # 硬件状态
        self.sensor_status = HardwareStatus.DISCONNECTED
        self.servo_status = HardwareStatus.DISCONNECTED
        self.arm_status = HardwareStatus.DISCONNECTED
        self.arm_state = ArmState(timestamp=time.time(), connected=False)
        
        # 硬件连接
        self.sensor_serial = None
        self.servo_serial = None
        self.servo_protocol = "st3215"  # st3215/stm32/simulation，用于区分指令协议
        self.servo_id = getattr(self.config.servo, "id", 1)  # 默认舵机ID=1（STM32指令需要）
        self._stm32_lock = threading.RLock()  # STM32串口读写锁，避免并发冲突

        # 未连接告警节流（避免高频打印导致卡顿）
        self._last_warning_time: Dict[str, float] = {}
        self._warning_interval_sec = 3.0
        
        # 数据缓冲区
        self.sensor_buffer = bytearray()
        self.servo_buffer = bytearray()
        
        # 校准数据初始化
        num_taxels = self.config.sensor.num_taxels
        self.sensor_calibration = {
            'zero_offset': [0.0] * num_taxels,
            'scale_factor': [1.0] * num_taxels,
            'pressure_range': self.config.sensor.pressure_range if hasattr(self.config.sensor, 'pressure_range') else 100.0,
            'force_dimensions': 3  # 三维力校准
        }
        
        self.servo_calibration = {
            'min_position': self.config.servo.min_angle,
            'max_position': self.config.servo.max_angle,
            'home_position': self.config.servo.home_position
        }
        
        # 三维力模拟状态
        self._sim_force_state = {
            'grasp_force': 0.0,
            'object_weight': 1.0,
            'friction_coeff': getattr(self.config.sensor, 'friction_coefficient', 0.3),
            'object_pose': np.array([0.0, 0.0, 0.0]),  # [x, y, rotation]
            'contact_points': [],
            'prev_forces': None
        }
        
        # 统计信息
        self.sensor_read_count = 0
        self.servo_command_count = 0
        
        # 模拟数据（用于测试）
        self.sensor_simulation = False
        self.servo_simulation = False
        # 兼容旧逻辑：任一设备进入模拟模式即为 True
        self.simulation_mode = False
        self.simulated_servo_position = self.config.servo.home_position
        
        self.logger.info("三维力硬件接口初始化完成")
    
    def connect_sensor(self) -> bool:
        """
        连接传感器
        
        Returns:
            连接是否成功
        """
        try:
            self.logger.info(f"连接传感器: {self.config.sensor.port}")
            self.sensor_status = HardwareStatus.CONNECTING
            self.sensor_simulation = False
            self._refresh_simulation_mode()
            
            # 检查是否为模拟模式
            if self.config.sensor.port.startswith("SIM"):
                self.sensor_status = HardwareStatus.CONNECTED
                self.sensor_simulation = True
                self._refresh_simulation_mode()
                self.logger.info("进入传感器模拟模式（三维力）")
                return True
            
            # 创建串口连接
            self.sensor_serial = serial.Serial(
                port=self.config.sensor.port,
                baudrate=self.config.sensor.baudrate,
                timeout=self.config.sensor.timeout
            )
            
            # 等待连接稳定
            time.sleep(0.5)
            
            # Paxini Gen3 M2020 特定初始化命令
            init_commands = [
                b'\x55\xAA\x01\x00\x01',  # 启动数据流命令
                b'\x55\xAA\x02\x00\x02',  # 设置采样率
            ]
            
            for cmd in init_commands:
                self.sensor_serial.write(cmd)
                time.sleep(0.1)
            
            # 读取初始响应
            response = self._read_sensor_response(timeout=1.0)
            if response:
                self.sensor_status = HardwareStatus.CONNECTED
                self.logger.info("Paxini Gen3 M2020 传感器连接成功（三维力）")
                return True
            
            # 连接失败
            self.sensor_status = HardwareStatus.ERROR
            self.logger.error("传感器连接失败")
            return False
            
        except serial.SerialException as e:
            self.sensor_status = HardwareStatus.ERROR
            self.logger.error(f"传感器串口错误: {e}")
            # 进入模拟模式
            self.sensor_simulation = True
            self._refresh_simulation_mode()
            self.sensor_status = HardwareStatus.CONNECTED
            self.logger.info("由于串口错误，进入传感器模拟模式（三维力）")
            return True
        except Exception as e:
            self.sensor_status = HardwareStatus.ERROR
            self.logger.error(f"传感器连接异常: {e}")
            return False
    
    def disconnect_sensor(self):
        """断开传感器连接"""
        if self.sensor_serial and self.sensor_serial.is_open:
            try:
                # 发送停止数据流命令
                stop_cmd = b'\x55\xAA\x01\x00\x00'
                self.sensor_serial.write(stop_cmd)
                time.sleep(0.1)
                
                self.sensor_serial.close()
                self.logger.info("传感器断开连接")
            except Exception as e:
                self.logger.error(f"断开传感器时出错: {e}")
        
        self.sensor_serial = None
        self.sensor_status = HardwareStatus.DISCONNECTED
        self.sensor_simulation = False
        self._refresh_simulation_mode()
    
    def connect_servo(self) -> bool:
        """
        连接舵机
        
        Returns:
            连接是否成功
        """
        try:
            # 如果串口已打开，直接复用，避免重复打开导致 PermissionError
            if self.servo_status == HardwareStatus.CONNECTED:
                if self.servo_simulation:
                    self.logger.info("舵机模拟已连接，跳过重复连接")
                    return True
                if self.servo_serial and self.servo_serial.is_open:
                    self.logger.info("舵机串口已打开，跳过重复连接")
                    return True
            if self.servo_serial and self.servo_serial.is_open:
                self.logger.info("检测到舵机串口已打开，恢复连接状态")
                self.servo_status = HardwareStatus.CONNECTED
                return True

            self.logger.info(f"连接舵机: {self.config.servo.port}")
            self.servo_status = HardwareStatus.CONNECTING
            self.servo_simulation = False
            self.servo_protocol = "st3215"
            self._refresh_simulation_mode()
            
            # 检查是否为模拟模式
            if self.config.servo.port.startswith("SIM"):
                self.servo_status = HardwareStatus.CONNECTED
                self.servo_simulation = True
                self.servo_protocol = "simulation"
                self._refresh_simulation_mode()
                self.logger.info("进入舵机模拟模式")
                return True
            
            # 创建串口连接
            self.servo_serial = serial.Serial(
                port=self.config.servo.port,
                baudrate=self.config.servo.baudrate,
                timeout=self.config.servo.timeout
            )
            
            # 等待连接稳定
            time.sleep(0.5)

            # 先尝试 STM32 CDC 指令（PING -> PONG），避免误把 ERR 当作舵机响应
            pong = self._send_stm32_command("PING", timeout=1.0)
            if pong and "PONG" in pong:
                self.servo_status = HardwareStatus.CONNECTED
                self.servo_protocol = "stm32"
                self.logger.info("STM32 连接成功（CDC 行命令）")
                return True

            # 再尝试传统舵机协议（#1P...）
            if self._send_servo_command(b'#1P1500T100\r\n'):
                response = self._read_servo_response(timeout=1.0)
                if response:
                    self.servo_status = HardwareStatus.CONNECTED
                    self.servo_protocol = "st3215"
                    self.logger.info("舵机连接成功（ST3215 协议）")

                    # 初始化舵机参数
                    self._initialize_servo()
                    return True

            # 连接失败
            self.servo_status = HardwareStatus.ERROR
            self.logger.error("舵机/STM32 连接失败（未收到有效响应）")
            return False
            
        except serial.SerialException as e:
            self.servo_status = HardwareStatus.ERROR
            self.logger.error(f"舵机串口错误: {e}")
            # 进入模拟模式
            self.servo_simulation = True
            self._refresh_simulation_mode()
            self.servo_status = HardwareStatus.CONNECTED
            self.logger.info("由于串口错误，进入舵机模拟模式")
            return True
        except Exception as e:
            self.servo_status = HardwareStatus.ERROR
            self.logger.error(f"舵机连接异常: {e}")
            return False
    
    def disconnect_servo(self):
        """断开舵机连接"""
        if self.servo_serial and self.servo_serial.is_open:
            try:
                # 仅在传统舵机协议下发送停止命令，STM32 由上位机策略处理
                if self.servo_protocol == "st3215":
                    self._stop_servo()
                
                # 关闭连接
                self.servo_serial.close()
                self.logger.info("舵机断开连接")
            except Exception as e:
                self.logger.error(f"断开舵机时出错: {e}")
        
        self.servo_serial = None
        self.servo_status = HardwareStatus.DISCONNECTED
        self.servo_simulation = False
        self.servo_protocol = "st3215"
        self._refresh_simulation_mode()

    # 机械臂接口（占位实现，可对接 arm_control.LearmInterface）
    def connect_arm(self) -> bool:
        """连接机械臂（占位逻辑）"""
        try:
            self.arm_status = HardwareStatus.CONNECTED
            self.arm_state.connected = True
            self.arm_state.timestamp = time.time()
            self.logger.info("机械臂已标记为连接状态（占位）")
            return True
        except Exception as e:
            self.logger.error(f"机械臂连接失败: {e}")
            self.arm_status = HardwareStatus.ERROR
            return False

    def disconnect_arm(self):
        """断开机械臂"""
        self.arm_status = HardwareStatus.DISCONNECTED
        self.arm_state.connected = False
        self.logger.info("机械臂已断开（占位）")

    def is_arm_connected(self) -> bool:
        """机械臂是否已连接"""
        return self.arm_status == HardwareStatus.CONNECTED

    def get_arm_state(self) -> ArmState:
        """获取机械臂状态（占位数据）"""
        self.arm_state.timestamp = time.time()
        return self.arm_state
    
    def connect_all(self) -> bool:
        """
        连接所有硬件
        
        Returns:
            所有硬件连接是否成功
        """
        sensor_success = self.connect_sensor()
        servo_success = self.connect_servo()
        arm_success = True
        if hasattr(self.config, "arm") or hasattr(self.config, "learm"):
            arm_success = self.connect_arm()
        
        return sensor_success and servo_success and arm_success
    
    def disconnect_all(self):
        """断开所有硬件连接"""
        self.disconnect_sensor()
        self.disconnect_servo()
        self.disconnect_arm()
        self.logger.info("所有硬件已断开连接")
    
    def read_sensor(self) -> Optional[SensorReading]:
        """
        读取传感器数据
        
        Returns:
            传感器读数，失败返回None
        """
        if self.sensor_status != HardwareStatus.CONNECTED:
            self._throttled_warning("tactile", "触觉传感器未连接")
            return None
        
        try:
            if self.sensor_simulation:
                # 生成三维力模拟数据
                return self._generate_simulated_sensor_data()
            
            # 读取真实传感器数据
            raw_data = self._read_sensor_response(timeout=0.01)
            if not raw_data:
                return None
            
            # 解析数据
            reading = self._parse_sensor_data(raw_data)
            if reading:
                self.sensor_read_count += 1
                
                # 应用校准
                reading = self._apply_sensor_calibration(reading)
                
                if self.sensor_read_count % 100 == 0:
                    self.logger.debug(f"传感器读数: 合力向量={reading.resultant_force}")
                
                return reading
            
            return None
            
        except Exception as e:
            self.logger.error(f"读取传感器数据时出错: {e}")
            return None
    
    def set_servo_position(self, position: float, speed: Optional[float] = None) -> bool:
        """
        设置舵机位置
        
        Args:
            position: 目标位置（百分比或角度）
            speed: 运动速度（百分比）
            
        Returns:
            命令是否成功发送
        """
        if self.servo_status != HardwareStatus.CONNECTED:
            self._throttled_warning("stm32", "STM32/舵机未连接")
            return False
        
        try:
            # 限制位置范围
            position = max(self.config.servo.min_angle, 
                          min(self.config.servo.max_angle, position))
            
            if self.servo_simulation:
                # 模拟舵机运动
                self.simulated_servo_position = position
                self.servo_command_count += 1
                self.logger.debug(f"模拟舵机移动到位置: {position}")
                return True

            # STM32 行命令协议（SMOVE）
            if self.servo_protocol == "stm32":
                if speed is None:
                    speed = self.config.servo.speed
                # 将速度映射为运动时长，避免过快导致机械抖动
                move_ms = int(max(50, 1000 - float(speed) * 9))
                cmd = f"SMOVE {int(self.servo_id)} {int(position)} {move_ms}"
                response = self._send_stm32_command(cmd, timeout=1.0)
                if response and response.strip().startswith("OK"):
                    self.servo_command_count += 1
                    return True
                return False
            
            # 转换位置到脉冲宽度
            pulse_width = self._position_to_pulse(position)
            
            # 设置速度
            if speed is None:
                speed = self.config.servo.speed
            
            speed_time = int(1000 - (speed * 10))
            command = f"#1P{pulse_width}T{speed_time}\r\n".encode()
            
            # 发送命令
            if self._send_servo_command(command):
                self.servo_command_count += 1
                time.sleep(0.01)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"设置舵机位置时出错: {e}")
            return False
    
    def get_servo_state(self) -> Optional[ServoState]:
        """
        获取舵机状态
        
        Returns:
            舵机状态，失败返回None
        """
        if self.servo_status != HardwareStatus.CONNECTED:
            self._throttled_warning("stm32", "STM32/舵机未连接")
            return None
        
        try:
            if self.servo_simulation:
                return ServoState(
                    timestamp=time.time(),
                    position=self.simulated_servo_position,
                    moving=False
                )

            # STM32 行命令协议（SREADPOS）
            if self.servo_protocol == "stm32":
                cmd = f"SREADPOS {int(self.servo_id)}"
                response = self._send_stm32_command(cmd, timeout=1.0)
                if response and response.startswith("POS"):
                    try:
                        _, sid, pos = response.split()
                        return ServoState(
                            timestamp=time.time(),
                            position=float(pos),
                            moving=False
                        )
                    except Exception:
                        return None
                return None
            
            # 发送状态查询命令
            command = b"#1PRAD\r\n"
            if not self._send_servo_command(command):
                return None
            
            # 读取响应
            response = self._read_servo_response(timeout=0.1)
            if not response:
                return None
            
            # 解析状态
            state = self._parse_servo_state(response)
            return state
            
        except Exception as e:
            self.logger.error(f"获取舵机状态时出错: {e}")
            return None
    
    def emergency_stop(self):
        """紧急停止"""
        self.logger.warning("执行紧急停止")
        
        try:
            # 停止舵机运动
            if self.servo_serial and self.servo_serial.is_open:
                stop_command = b"#1STOP\r\n"
                self.servo_serial.write(stop_command)
            
            # 停止传感器数据流
            if self.sensor_serial and self.sensor_serial.is_open:
                stop_command = b'\x55\xAA\x01\x00\x00'
                self.sensor_serial.write(stop_command)
            
            self.logger.info("紧急停止完成")
            
        except Exception as e:
            self.logger.error(f"紧急停止时出错: {e}")
    
    def calibrate_sensor(self, calibration_type: str = "zero") -> bool:
        """
        校准传感器
        
        Args:
            calibration_type: 校准类型（"zero"或"scale"）
            
        Returns:
            校准是否成功
        """
        self.logger.info(f"开始传感器校准: {calibration_type}")
        
        try:
            if calibration_type == "zero":
                return self._calibrate_sensor_zero()
            elif calibration_type == "scale":
                return self._calibrate_sensor_scale()
            elif calibration_type == "vector":
                # 新增：三维力校准
                return self._calibrate_sensor_vector()
            else:
                self.logger.error(f"未知的校准类型: {calibration_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"传感器校准失败: {e}")
            return False
    
    def calibrate_servo(self, calibration_type: str = "limits") -> bool:
        """
        校准舵机
        
        Args:
            calibration_type: 校准类型（"limits"或"home"）
            
        Returns:
            校准是否成功
        """
        self.logger.info(f"开始舵机校准: {calibration_type}")
        
        try:
            if calibration_type == "limits":
                return self._calibrate_servo_limits()
            elif calibration_type == "home":
                return self._calibrate_servo_home()
            else:
                self.logger.error(f"未知的校准类型: {calibration_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"舵机校准失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取硬件状态
        
        Returns:
            状态字典
        """
        return {
            "sensor": {
                "status": self.sensor_status.value,
                "connected": self.sensor_status == HardwareStatus.CONNECTED,
                "read_count": self.sensor_read_count,
                "simulation": self.sensor_simulation,
                "mode": "simulation" if self.sensor_simulation else "hardware",
                "force_dimensions": 3
            },
            "servo": {
                "status": self.servo_status.value,
                "connected": self.servo_status == HardwareStatus.CONNECTED,
                "command_count": self.servo_command_count,
                "simulation": self.servo_simulation,
                "mode": "simulation" if self.servo_simulation else "hardware",
            },
            "arm": {
                "status": self.arm_status.value,
                "connected": self.arm_status == HardwareStatus.CONNECTED,
                "simulation": False,
                "mode": "hardware",
            }
        }
    
    def _refresh_simulation_mode(self):
        """根据各设备模拟标记刷新总模拟状态"""
        self.simulation_mode = bool(self.sensor_simulation or self.servo_simulation)

    def _throttled_warning(self, key: str, message: str, interval: Optional[float] = None):
        """
        低频告警输出，避免高频日志导致UI卡顿

        Args:
            key: 告警键（用于节流）
            message: 告警内容
            interval: 节流间隔（秒），为空则使用默认值
        """
        now = time.time()
        gap = interval if interval is not None else self._warning_interval_sec
        last_time = self._last_warning_time.get(key, 0.0)
        if now - last_time >= gap:
            self._last_warning_time[key] = now
            self.logger.warning(message)

    def _read_line_from_servo(self, timeout: float = 1.0) -> Optional[str]:
        """
        从舵机串口读取一行（以 \\n 结束），用于 STM32 CDC 文本协议
        """
        if not self.servo_serial or not self.servo_serial.is_open:
            return None

        end_time = time.time() + timeout
        buffer = bytearray()
        while time.time() < end_time:
            try:
                if self.servo_serial.in_waiting > 0:
                    buffer.extend(self.servo_serial.read(self.servo_serial.in_waiting))
                    if b"\n" in buffer:
                        line, _, _ = buffer.partition(b"\n")
                        return line.strip(b"\r").decode(errors="ignore")
            except Exception:
                break
            time.sleep(0.01)

        if buffer:
            return buffer.strip().decode(errors="ignore")
        return None

    def _send_stm32_command(self, command: str, timeout: float = 1.0) -> Optional[str]:
        """
        发送 STM32 CDC 行命令，并读取返回行

        Args:
            command: 指令文本（不含换行）
            timeout: 等待响应超时（秒）
        """
        if not self.servo_serial or not self.servo_serial.is_open:
            return None

        # 串口共享时需要加锁，避免与机械臂/夹爪并发读写冲突
        with self._stm32_lock:
            try:
                cmd = command.strip()
                if not cmd.endswith("\n"):
                    cmd = f"{cmd}\n"
                # 清空残留输入，避免干扰本次解析
                try:
                    self.servo_serial.reset_input_buffer()
                except Exception:
                    pass
                self.servo_serial.write(cmd.encode())
                self.servo_serial.flush()
                return self._read_line_from_servo(timeout=timeout)
            except Exception:
                return None

    def save_calibration(self, filepath: str):
        """
        保存校准数据
        
        Args:
            filepath: 文件路径
        """
        import json
        
        calibration_data = {
            "sensor": self.sensor_calibration,
            "servo": self.servo_calibration,
            "timestamp": time.time(),
            "simulation_mode": self.simulation_mode,
            "force_dimensions": 3
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info(f"校准数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存校准数据失败: {e}")
    
    def load_calibration(self, filepath: str) -> bool:
        """
        加载校准数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载是否成功
        """
        import json
        
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            if "sensor" in calibration_data:
                self.sensor_calibration.update(calibration_data["sensor"])
            
            if "servo" in calibration_data:
                self.servo_calibration.update(calibration_data["servo"])
            
            if "simulation_mode" in calibration_data:
                self.simulation_mode = calibration_data["simulation_mode"]
            
            self.logger.info(f"校准数据已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载校准数据失败: {e}")
            return False
    
    # 私有方法
    def _send_sensor_command(self, command: bytes) -> bool:
        """发送传感器命令"""
        try:
            if self.sensor_serial and self.sensor_serial.is_open:
                self.sensor_serial.write(command)
                self.sensor_serial.flush()
                return True
            return False
        except Exception as e:
            self.logger.error(f"发送传感器命令失败: {e}")
            return False
    
    def _read_sensor_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """读取传感器响应"""
        try:
            if not self.sensor_serial or not self.sensor_serial.is_open:
                return None
            
            original_timeout = self.sensor_serial.timeout
            self.sensor_serial.timeout = timeout
            
            response = self.sensor_serial.readline()
            
            self.sensor_serial.timeout = original_timeout
            
            return response if response else None
            
        except Exception as e:
            self.logger.error(f"读取传感器响应失败: {e}")
            return None
    
    def _send_servo_command(self, command: bytes) -> bool:
        """发送舵机命令"""
        try:
            if self.servo_serial and self.servo_serial.is_open:
                self.servo_serial.write(command)
                self.servo_serial.flush()
                return True
            return False
        except Exception as e:
            self.logger.error(f"发送舵机命令失败: {e}")
            return False
    
    def _read_servo_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """读取舵机响应"""
        try:
            if not self.servo_serial or not self.servo_serial.is_open:
                return None
            
            original_timeout = self.servo_serial.timeout
            self.servo_serial.timeout = timeout
            
            start_time = time.time()
            response = bytearray()
            
            while time.time() - start_time < timeout:
                if self.servo_serial.in_waiting > 0:
                    response.extend(self.servo_serial.read(self.servo_serial.in_waiting))
                    if b'\r\n' in response:
                        break
                time.sleep(0.01)
            
            self.servo_serial.timeout = original_timeout
            
            return bytes(response) if response else None
            
        except Exception as e:
            self.logger.error(f"读取舵机响应失败: {e}")
            return None
    
    def _parse_sensor_data(self, raw_data: bytes) -> Optional[SensorReading]:
        """解析传感器数据 - 支持三维力格式"""
        try:
            # 尝试解析三维力数据格式
            # 假设格式: 头(2字节) + 数据类型(1字节) + 数据长度(1字节) + 三维力数据 + 校验和(1字节)
            if len(raw_data) >= 6:
                try:
                    # 检查是否为三维力格式
                    if raw_data[0] == 0x56 and raw_data[1] == 0xAB:  # 三维力数据头
                        data_type = raw_data[2]
                        data_len = raw_data[3]
                        
                        if data_type == 0x03 and data_len >= 4:  # 三维力数据
                            num_taxels = self.config.sensor.num_taxels
                            force_vectors = []
                            
                            data_start = 4
                            for i in range(num_taxels):
                                if data_start + 12 <= len(raw_data) - 1:  # 每个向量4个浮点数 (12字节)
                                    # 读取三维力向量 (4字节浮点数，小端序)
                                    fx = struct.unpack('<f', raw_data[data_start:data_start+4])[0]
                                    fy = struct.unpack('<f', raw_data[data_start+4:data_start+8])[0]
                                    fz = struct.unpack('<f', raw_data[data_start+8:data_start+12])[0]
                                    force_vectors.append([fx, fy, fz])
                                    data_start += 12
                            
                            # 提取Z方向力用于向后兼容
                            force_data = [vec[2] for vec in force_vectors]
                            
                            # 读取温度
                            temperature = None
                            if data_start + 4 <= len(raw_data) - 1:
                                temperature = struct.unpack('<f', raw_data[data_start:data_start+4])[0]
                            
                            return SensorReading(
                                timestamp=time.time(),
                                force_data=force_data,
                                force_vectors=force_vectors,
                                temperature=temperature,
                                raw_data=raw_data
                            )
                except Exception as e:
                    self.logger.warning(f"三维力解析失败: {e}")
            
            # 尝试解析标准Paxini帧 (0x55 0xAA) 并支持三轴
            if len(raw_data) >= 8 and raw_data[0] == 0x55 and raw_data[1] == 0xAA:
                data_length = int.from_bytes(raw_data[2:4], byteorder='little', signed=False)
                # 预期最小长度: 头2 + 长度2 + 地址1 + 预留1 + 功能1 + 起始4 + 字节数2 + 状态1 + LRC1
                if len(raw_data) >= data_length + 4 and len(raw_data) >= 15:
                    num_taxels = self.config.sensor.num_taxels
                    payload_start = 14  # 状态后的数据起点 (见协议图)
                    payload_end = len(raw_data) - 1  # 去掉LRC
                    payload = raw_data[payload_start:payload_end] if payload_end > payload_start else b''
                    payload_len = len(payload)
                    
                    force_vectors = []
                    force_data = []
                    temperature = None
                    
                    # 优先尝试三轴解析: 每触点6字节（Fx,Fy,Fz int16小端）
                    if payload_len >= num_taxels * 6:
                        force_scale = getattr(self.config.sensor, 'force_scale', 0.01)  # 单位换算系数
                        for i in range(num_taxels):
                            base = i * 6
                            fx_raw = struct.unpack_from('<h', payload, base)[0]
                            fy_raw = struct.unpack_from('<h', payload, base + 2)[0]
                            fz_raw = struct.unpack_from('<h', payload, base + 4)[0]
                            fx = fx_raw * force_scale
                            fy = fy_raw * force_scale
                            fz = fz_raw * force_scale
                            force_vectors.append([fx, fy, fz])
                            force_data.append(fz)
                    elif payload_len >= num_taxels * 2:
                        # 仅有法向力，按原逻辑推导剪切
                        force_data_raw = []
                        data_start = 0
                        for i in range(num_taxels):
                            if data_start + 2 <= payload_len:
                                pressure_raw = int.from_bytes(payload[data_start:data_start+2], byteorder='little', signed=False)
                                pressure_kpa = (pressure_raw / 65535.0) * 100.0
                                force_data_raw.append(pressure_kpa)
                                data_start += 2
                        normal_scale = getattr(self.config.sensor, 'normal_scale', 0.1)  # kPa -> N
                        shear_scale = getattr(self.config.sensor, 'shear_scale', 0.05)   # 放大剪切推导
                        rows = getattr(self.config.sensor, 'rows', int(math.sqrt(num_taxels)) if num_taxels > 0 else 3)
                        if rows <= 0:
                            rows = 3
                        cols = getattr(self.config.sensor, 'cols', num_taxels // rows if rows > 0 else num_taxels)
                        if cols <= 0:
                            cols = max(1, num_taxels)
                        for idx, f in enumerate(force_data_raw):
                            row_idx = idx // cols if cols > 0 else 0
                            col_idx = idx % cols if cols > 0 else 0
                            x_off = (col_idx - (cols - 1) / 2.0) if cols > 1 else 0.0
                            y_off = (row_idx - (rows - 1) / 2.0) if rows > 1 else 0.0
                            fx = x_off * f * shear_scale
                            fy = y_off * f * shear_scale
                            fz = f * normal_scale
                            force_vectors.append([fx, fy, fz])
                            force_data.append(fz)
                    else:
                        # 数据长度不足，回退模拟
                        raise ValueError("Paxini帧长度不足以解析力数据")
                    
                    return SensorReading(
                        timestamp=time.time(),
                        force_data=force_data,
                        force_vectors=force_vectors,
                        temperature=temperature,
                        raw_data=raw_data
                    )
            
            # 如果以上都失败，生成模拟数据
            self.logger.warning("无法解析传感器数据，使用模拟数据")
            return self._generate_simulated_sensor_data()
            
        except Exception as e:
            self.logger.error(f"解析传感器数据失败: {e}")
            return self._generate_simulated_sensor_data()
    
    def _parse_servo_state(self, raw_data: bytes) -> Optional[ServoState]:
        """解析舵机状态"""
        try:
            data_str = raw_data.decode('utf-8').strip()
            parts = data_str.split()
            
            state_dict = {}
            for part in parts:
                if len(part) >= 2:
                    key = part[0]
                    try:
                        value = float(part[1:])
                        state_dict[key] = value
                    except ValueError:
                        continue
            
            timestamp = time.time()
            position = state_dict.get('P', 0.0)
            
            if 'P' in state_dict:
                position = self._pulse_to_position(state_dict['P'])
            
            return ServoState(
                timestamp=timestamp,
                position=position,
                velocity=state_dict.get('S', 0.0),
                current=state_dict.get('C', 0.0),
                temperature=state_dict.get('T', 25.0),
                load=state_dict.get('L', 0.0),
                voltage=state_dict.get('V', 12.0),
                moving=state_dict.get('M', 0) == 1
            )
            
        except Exception as e:
            self.logger.error(f"解析舵机状态失败: {e}")
            return None
    
    def _apply_sensor_calibration(self, reading: SensorReading) -> SensorReading:
        """应用传感器校准"""
        if len(reading.force_data) != len(self.sensor_calibration['zero_offset']):
            self.logger.warning("传感器数据长度与校准参数不匹配")
            return reading
        
        # 校准Z方向力
        calibrated_forces = []
        for i, force in enumerate(reading.force_data):
            try:
                calibrated = (force - self.sensor_calibration['zero_offset'][i]) * \
                            self.sensor_calibration['scale_factor'][i]
                max_pressure = self.sensor_calibration.get('pressure_range', 100.0)
                calibrated = max(0.0, min(max_pressure, calibrated))
                calibrated_forces.append(calibrated)
            except Exception as e:
                self.logger.warning(f"校准传感器数据{i}时出错: {e}")
                calibrated_forces.append(force)
        
        reading.force_data = calibrated_forces
        
        # 如果存在三维力向量，也需要校准
        if reading.force_vectors:
            calibrated_vectors = []
            for i, vec in enumerate(reading.force_vectors):
                try:
                    # 只校准Z方向，X和Y方向使用相同的缩放因子
                    calibrated_z = (vec[2] - self.sensor_calibration['zero_offset'][i]) * \
                                  self.sensor_calibration['scale_factor'][i]
                    calibrated_z = max(0.0, calibrated_z)
                    calibrated_vectors.append([vec[0], vec[1], calibrated_z])
                except Exception as e:
                    calibrated_vectors.append(vec)
            reading.force_vectors = calibrated_vectors
        
        return reading
    
    def _position_to_pulse(self, position: float) -> int:
        """转换位置到脉冲宽度"""
        pulse = int((position - self.config.servo.min_angle) * 
                   (self.config.servo.max_pulse - self.config.servo.min_pulse) /
                   (self.config.servo.max_angle - self.config.servo.min_angle) +
                   self.config.servo.min_pulse)
        
        return max(self.config.servo.min_pulse, 
                  min(self.config.servo.max_pulse, pulse))
    
    def _pulse_to_position(self, pulse: float) -> float:
        """转换脉冲宽度到位置"""
        position = (pulse - self.config.servo.min_pulse) * \
                  (self.config.servo.max_angle - self.config.servo.min_angle) / \
                  (self.config.servo.max_pulse - self.config.servo.min_pulse) + \
                  self.config.servo.min_angle
        
        return max(self.config.servo.min_angle,
                  min(self.config.servo.max_angle, position))
    
    def _reset_sensor(self):
        """重置传感器"""
        self._send_sensor_command(b'\x55\xAA\x03\x00\x03')
        time.sleep(0.1)
    
    def _initialize_servo(self):
        """初始化舵机"""
        speed_cmd = f"#1S{self.config.servo.speed}\r\n".encode()
        self._send_servo_command(speed_cmd)
        
        torque_cmd = f"#1L{int(self.config.servo.torque)}\r\n".encode()
        self._send_servo_command(torque_cmd)
        
        time.sleep(0.1)
    
    def _stop_servo(self):
        """停止舵机"""
        stop_cmd = b"#1STOP\r\n"
        self._send_servo_command(stop_cmd)
        time.sleep(0.1)
    
    def _calibrate_sensor_zero(self) -> bool:
        """传感器零点校准"""
        self.logger.info("开始传感器零点校准...")
        
        try:
            if not self._send_sensor_command(b'\x55\xAA\x04\x00\x04'):
                return False
            
            time.sleep(2.0)
            
            response = self._read_sensor_response(timeout=1.0)
            if response:
                try:
                    if len(response) >= 3 and response[0] == 0x55 and response[1] == 0xAA:
                        self.logger.info("传感器零点校准完成")
                        return True
                except Exception as e:
                    self.logger.warning(f"解析校准响应时出错: {e}")
            
            self.logger.error("传感器零点校准失败")
            return False
            
        except Exception as e:
            self.logger.error(f"传感器零点校准异常: {e}")
            return False
    
    def _calibrate_sensor_scale(self) -> bool:
        """传感器量程校准"""
        self.logger.info("开始传感器量程校准...")
        
        try:
            if not self._send_sensor_command(b'\x55\xAA\x05\x00\x05'):
                return False
            
            time.sleep(2.0)
            
            response = self._read_sensor_response(timeout=1.0)
            if response:
                try:
                    if len(response) >= 3 and response[0] == 0x55 and response[1] == 0xAA:
                        self.logger.info("传感器量程校准完成")
                        return True
                except Exception as e:
                    self.logger.warning(f"解析校准响应时出错: {e}")
            
            self.logger.error("传感器量程校准失败")
            return False
            
        except Exception as e:
            self.logger.error(f"传感器量程校准异常: {e}")
            return False
    
    def _calibrate_sensor_vector(self) -> bool:
        """传感器三维力校准"""
        self.logger.info("开始传感器三维力校准...")
        
        try:
            # 发送三维力校准命令
            if not self._send_sensor_command(b'\x55\xAA\x06\x00\x06'):
                return False
            
            time.sleep(2.0)
            
            response = self._read_sensor_response(timeout=1.0)
            if response:
                try:
                    if len(response) >= 3 and response[0] == 0x56 and response[1] == 0xAB:
                        self.logger.info("传感器三维力校准完成")
                        return True
                except Exception as e:
                    self.logger.warning(f"解析校准响应时出错: {e}")
            
            self.logger.error("传感器三维力校准失败")
            return False
            
        except Exception as e:
            self.logger.error(f"传感器三维力校准异常: {e}")
            return False
    
    def _calibrate_servo_limits(self) -> bool:
        """舵机极限位置校准"""
        self.logger.info("开始舵机极限位置校准...")
        
        try:
            current_state = self.get_servo_state()
            if current_state:
                self.servo_calibration['min_position'] = current_state.position
                self.logger.info(f"最小位置设置为: {current_state.position}")
            
            self.set_servo_position(self.config.servo.max_angle)
            time.sleep(2.0)
            
            current_state = self.get_servo_state()
            if current_state:
                self.servo_calibration['max_position'] = current_state.position
                self.logger.info(f"最大位置设置为: {current_state.position}")
            
            home_pos = (self.servo_calibration['min_position'] + 
                       self.servo_calibration['max_position']) / 2
            self.set_servo_position(home_pos)
            
            self.logger.info("舵机极限位置校准完成")
            return True
            
        except Exception as e:
            self.logger.error(f"舵机极限位置校准异常: {e}")
            return False
    
    def _calibrate_servo_home(self) -> bool:
        """舵机零点校准"""
        self.logger.info("开始舵机零点校准...")
        
        try:
            home_cmd = b"#1CAL_HOME\r\n"
            if not self._send_servo_command(home_cmd):
                return False
            
            time.sleep(3.0)
            
            current_state = self.get_servo_state()
            if current_state and abs(current_state.position) < 1.0:
                self.servo_calibration['home_position'] = current_state.position
                self.logger.info("舵机零点校准完成")
                return True
            
            self.logger.error("舵机零点校准失败")
            return False
            
        except Exception as e:
            self.logger.error(f"舵机零点校准异常: {e}")
            return False
    
    def _generate_simulated_sensor_data(self) -> SensorReading:
        """生成模拟传感器数据 - 更贴近真实夹取受力"""
        timestamp = time.time()
        num_taxels = self.config.sensor.num_taxels
        rows = getattr(self.config.sensor, 'rows', int(math.sqrt(num_taxels)) if num_taxels > 0 else 3)
        if rows <= 0:
            rows = 3
        cols = getattr(self.config.sensor, 'cols', num_taxels // rows if rows > 0 else num_taxels)
        if cols <= 0:
            cols = max(1, num_taxels)
        
        # 根据舵机位置计算抓取力（目标法向分配）
        grip_force = self.simulated_servo_position / 180.0 * 18.0  # 0-18N
        sim_scale = getattr(self.config.sensor, 'simulation_force_scale', 4.0)  # 默认更高的放大系数
        object_weight = self._sim_force_state['object_weight'] * 9.8
        friction_coeff = self._sim_force_state['friction_coeff']
        
        # 目标法向力（中间大，两端小），加入姿态微偏
        obj_x, obj_y, obj_rot = self._sim_force_state['object_pose']
        contact_radius = getattr(self.config.sensor, 'contact_radius', 0.35)
        max_normal = getattr(self.config.sensor, 'sim_max_normal', 80.0)  # N，抬高上限
        
        # 轻微随机姿态变化
        if random.random() < 0.04:
            self._sim_force_state['object_pose'][0] = max(-0.35, min(0.35, obj_x + random.uniform(-0.05, 0.05)))
            self._sim_force_state['object_pose'][1] = max(-0.35, min(0.35, obj_y + random.uniform(-0.05, 0.05)))
            self._sim_force_state['object_pose'][2] = max(-0.2, min(0.2, obj_rot + random.uniform(-0.04, 0.04)))
            obj_x, obj_y, obj_rot = self._sim_force_state['object_pose']
        
        force_vectors = []
        force_data = []
        
        # slip 方向基于姿态旋转
        slip_dir = np.array([math.cos(obj_rot + math.pi / 2), math.sin(obj_rot + math.pi / 2)])
        
        for i in range(num_taxels):
            row_idx = i // cols
            col_idx = i % cols
            x_pos = (col_idx - (cols - 1) / 2.0) / (cols - 1) if cols > 1 else 0.0
            y_pos = (row_idx - (rows - 1) / 2.0) / (rows - 1) if rows > 1 else 0.0
            
            # 与物体中心距离衰减
            dist = math.sqrt((x_pos - obj_x) ** 2 + (y_pos - obj_y) ** 2)
            # 接触增益：拉平中心/边缘差异，保留小幅衰减
            contact_gain = max(0.15, 1.0 - dist / contact_radius)
            contact_gain = contact_gain ** 1.0  # 减少指数收缩，中心与边缘差异更小
            
            # 法向力分配：握力 + 重力分担 + 姿态偏置
            base_normal = (grip_force * 0.6 + object_weight * 0.15) * (0.6 + 0.4 * contact_gain)
            tilt_bias = 1.0 + 0.12 * (x_pos * math.sin(obj_rot) + y_pos * math.cos(obj_rot))
            normal_force = base_normal * tilt_bias * sim_scale
            
            # 剪切力：来自重力+姿态，沿 slip_dir，受摩擦限制
            # 剪切力基准：重力+握力分担，略提高系数，让XY更明显
            shear_target = (object_weight * friction_coeff * 0.18 + grip_force * 0.05) * contact_gain
            shear_vec = shear_target * slip_dir
            
            # 摩擦极限
            max_shear = max(0.01, normal_force * friction_coeff)
            shear_mag = math.sqrt(shear_vec[0] ** 2 + shear_vec[1] ** 2)
            slip_ratio = 0.0
            if shear_mag > max_shear:
                slip_ratio = min(1.0, (shear_mag - max_shear) / (max_shear + 1e-6))
                scale = max_shear / shear_mag
                shear_vec *= scale
                # 滑移导致法向略下降
                normal_force *= (1.0 - 0.15 * slip_ratio)
            
            # 小幅随机扰动
            noise_level = getattr(self.config.sensor, 'force_noise_level', 0.1)
            normal_force += random.uniform(-noise_level, noise_level) * sim_scale
            shear_vec[0] += random.uniform(-noise_level, noise_level) * sim_scale * 0.5
            shear_vec[1] += random.uniform(-noise_level, noise_level) * sim_scale * 0.5
            
            # 限幅与下限
            normal_force = min(max_normal, max(0.05, normal_force))
            
            # 非接触区域保持极小背景力
            if contact_gain <= 0.15:
                normal_force = max(normal_force, random.uniform(0.2, 0.5))
                shear_vec *= 0.6
            
            fx, fy = shear_vec.tolist()
            fz = normal_force
            
            force_vectors.append([fx, fy, fz])
            force_data.append(fz)
            
            # 记录接触点
            if contact_gain > 0.05:
                if i not in self._sim_force_state['contact_points']:
                    self._sim_force_state['contact_points'].append(i)
            else:
                if i in self._sim_force_state['contact_points']:
                    self._sim_force_state['contact_points'].remove(i)
        
        # 应用平滑滤波，模拟软组织缓冲
        if self._sim_force_state['prev_forces'] is not None:
            alpha = 0.6
            for i in range(num_taxels):
                force_vectors[i][0] = alpha * self._sim_force_state['prev_forces'][i][0] + (1 - alpha) * force_vectors[i][0]
                force_vectors[i][1] = alpha * self._sim_force_state['prev_forces'][i][1] + (1 - alpha) * force_vectors[i][1]
                force_vectors[i][2] = alpha * self._sim_force_state['prev_forces'][i][2] + (1 - alpha) * force_vectors[i][2]
                force_data[i] = force_vectors[i][2]
        
        self._sim_force_state['prev_forces'] = force_vectors.copy()
        
        temperature = random.uniform(22.0, 28.0)
        status = 0
        
        return SensorReading(
            timestamp=timestamp,
            force_data=force_data,
            force_vectors=force_vectors,
            temperature=temperature,
            status=status
        )
    
    def close(self):
        """关闭硬件接口"""
        self.disconnect_all()
        self.logger.info("硬件接口已关闭")

    # 运行时更新仿真/传感参数
    def set_simulation_params(self, params: Dict[str, Any]):
        """更新仿真参数，如放大倍数、剪切系数、噪声等"""
        try:
            if not params:
                return
            if hasattr(self.config, "sensor"):
                for k, v in params.items():
                    setattr(self.config.sensor, k, v)
            # 同时更新内部使用的模拟字段
            if "friction_coeff" in params:
                self._sim_force_state["friction_coeff"] = params["friction_coeff"]
            if "object_weight" in params:
                self._sim_force_state["object_weight"] = params["object_weight"]
            if "contact_radius" in params:
                self.config.sensor.contact_radius = params["contact_radius"]
            self.logger.info(f"更新仿真参数: {params}")
        except Exception as e:
            self.logger.error(f"更新仿真参数失败: {e}")
