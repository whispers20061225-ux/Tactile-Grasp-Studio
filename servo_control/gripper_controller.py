"""
触觉夹爪演示系统 - 夹爪控制器
提供高级的夹爪控制功能，包括位置控制、速度控制、力控制和轨迹规划。
"""

import time
import threading
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# 修改导入方式 - 使用绝对导入
from config.demo_config import ServoConfig, DemoConfig
from core.hardware_interface import HardwareInterface, ServoState


class ControlMode(Enum):
    """控制模式枚举"""
    POSITION = "position"      # 位置控制模式
    VELOCITY = "velocity"      # 速度控制模式
    FORCE = "force"            # 力控制模式
    IMPEDANCE = "impedance"    # 阻抗控制模式
    TORQUE = "torque"          # 扭矩控制模式
    IDLE = "idle"              # 空闲模式


class ControlStatus(Enum):
    """控制状态枚举"""
    IDLE = "idle"              # 空闲
    CONNECTING = "connecting"  # 连接中
    CONNECTED = "connected"    # 已连接
    CALIBRATING = "calibrating" # 校准中
    MOVING = "moving"          # 运动中
    HOLDING = "holding"        # 保持中
    ERROR = "error"            # 错误
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


@dataclass
class GripperState:
    """夹爪状态"""
    timestamp: float           # 时间戳
    position: float            # 当前位置 (度)
    velocity: float            # 当前速度 (度/秒)
    acceleration: float        # 当前加速度 (度/秒²)
    force: float               # 当前力 (N)
    current: float             # 当前电流 (A)
    temperature: float         # 当前温度 (°C)
    voltage: float             # 当前电压 (V)
    load: float                # 当前负载 (%)
    moving: bool               # 是否在运动
    mode: ControlMode          # 当前控制模式
    status: ControlStatus      # 当前控制状态
    target_position: float = 0.0      # 目标位置
    target_velocity: float = 0.0      # 目标速度
    target_force: float = 0.0         # 目标力
    error_code: int = 0                # 错误代码
    error_message: str = ""            # 错误消息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'force': self.force,
            'current': self.current,
            'temperature': self.temperature,
            'voltage': self.voltage,
            'load': self.load,
            'moving': self.moving,
            'mode': self.mode.value,
            'status': self.status.value,
            'target_position': self.target_position,
            'target_velocity': self.target_velocity,
            'target_force': self.target_force,
            'error_code': self.error_code,
            'error_message': self.error_message
        }


class TrajectoryPlanner:
    """轨迹规划器"""
    
    def __init__(self, config: ServoConfig):
        """
        初始化轨迹规划器
        
        Args:
            config: 舵机配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 规划参数
        self.max_velocity = 180.0  # 度/秒
        self.max_acceleration = 360.0  # 度/秒²
        self.max_jerk = 720.0  # 度/秒³
        
        # 当前状态
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        
        # 轨迹缓冲区
        self.trajectory_buffer = []
        self.current_trajectory_index = 0
        
        # 滤波器
        self.position_filter = LowPassFilter(cutoff_frequency=10.0, sampling_rate=100.0)
        self.velocity_filter = LowPassFilter(cutoff_frequency=5.0, sampling_rate=100.0)
        
    def plan_trapezoidal_trajectory(self, start_pos: float, target_pos: float,
                                   max_vel: Optional[float] = None,
                                   max_acc: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """
        规划梯形速度轨迹
        
        Args:
            start_pos: 起始位置 (度)
            target_pos: 目标位置 (度)
            max_vel: 最大速度 (度/秒)
            max_acc: 最大加速度 (度/秒²)
            
        Returns:
            轨迹列表，每个元素为(位置, 速度, 加速度)
        """
        max_vel = max_vel or self.max_velocity
        max_acc = max_acc or self.max_acceleration
        
        # 计算距离
        distance = target_pos - start_pos
        direction = 1.0 if distance >= 0 else -1.0
        distance = abs(distance)
        
        # 计算能达到的最大速度
        acc_distance = max_vel**2 / (2 * max_acc)  # 加速到最大速度所需的距离
        dec_distance = max_vel**2 / (2 * max_acc)  # 从最大速度减速到0所需的距离
        
        if acc_distance + dec_distance <= distance:
            # 梯形轨迹：加速 -> 匀速 -> 减速
            const_distance = distance - acc_distance - dec_distance
            acc_time = max_vel / max_acc
            const_time = const_distance / max_vel
            dec_time = max_vel / max_acc
            total_time = acc_time + const_time + dec_time
            
            # 生成轨迹
            trajectory = []
            dt = 0.01  # 10ms时间间隔
            
            # 加速段
            for t in np.arange(0, acc_time, dt):
                vel = max_acc * t
                pos = start_pos + direction * (0.5 * max_acc * t**2)
                acc = max_acc
                trajectory.append((pos, direction * vel, direction * acc))
            
            # 匀速段
            for t in np.arange(0, const_time, dt):
                vel = max_vel
                pos = start_pos + direction * (acc_distance + max_vel * t)
                acc = 0.0
                trajectory.append((pos, direction * vel, direction * acc))
            
            # 减速段
            for t in np.arange(0, dec_time, dt):
                vel = max_vel - max_acc * t
                pos = start_pos + direction * (acc_distance + const_distance + 
                                              max_vel * t - 0.5 * max_acc * t**2)
                acc = -max_acc
                trajectory.append((pos, direction * vel, direction * acc))
            
            # 添加终点
            trajectory.append((target_pos, 0.0, 0.0))
            
        else:
            # 三角形轨迹：加速 -> 减速（没有匀速段）
            # 计算实际能达到的最大速度
            actual_max_vel = np.sqrt(max_acc * distance)
            acc_distance = actual_max_vel**2 / (2 * max_acc)
            dec_distance = acc_distance
            
            acc_time = actual_max_vel / max_acc
            dec_time = actual_max_vel / max_acc
            total_time = acc_time + dec_time
            
            # 生成轨迹
            trajectory = []
            dt = 0.01
            
            # 加速段
            for t in np.arange(0, acc_time, dt):
                vel = max_acc * t
                pos = start_pos + direction * (0.5 * max_acc * t**2)
                acc = max_acc
                trajectory.append((pos, direction * vel, direction * acc))
            
            # 减速段
            for t in np.arange(0, dec_time, dt):
                vel = actual_max_vel - max_acc * t
                pos = start_pos + direction * (acc_distance + 
                                              actual_max_vel * t - 0.5 * max_acc * t**2)
                acc = -max_acc
                trajectory.append((pos, direction * vel, direction * acc))
            
            # 添加终点
            trajectory.append((target_pos, 0.0, 0.0))
        
        return trajectory
    
    def plan_s_curve_trajectory(self, start_pos: float, target_pos: float,
                               max_vel: Optional[float] = None,
                               max_acc: Optional[float] = None,
                               max_jerk: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """
        规划S曲线轨迹（七段轨迹）
        
        Args:
            start_pos: 起始位置
            target_pos: 目标位置
            max_vel: 最大速度
            max_acc: 最大加速度
            max_jerk: 最大加加速度
            
        Returns:
            轨迹列表
        """
        max_vel = max_vel or self.max_velocity
        max_acc = max_acc or self.max_acceleration
        max_jerk = max_jerk or self.max_jerk
        
        # 这里实现S曲线轨迹规划
        # 简化实现：使用梯形轨迹
        return self.plan_trapezoidal_trajectory(start_pos, target_pos, max_vel, max_acc)
    
    def set_current_state(self, position: float, velocity: float = 0.0,
                         acceleration: float = 0.0):
        """
        设置当前状态
        
        Args:
            position: 当前位置
            velocity: 当前速度
            acceleration: 当前加速度
        """
        self.current_position = position
        self.current_velocity = velocity
        self.current_acceleration = acceleration
        
        # 更新滤波器状态
        self.position_filter.reset(position)
        self.velocity_filter.reset(velocity)
    
    def get_next_point(self, target_pos: float, dt: float = 0.01) -> Tuple[float, float, float]:
        """
        获取下一个轨迹点（实时规划）
        
        Args:
            target_pos: 目标位置
            dt: 时间步长
            
        Returns:
            (位置, 速度, 加速度)
        """
        # 使用PD控制器生成平滑轨迹
        kp = 100.0
        kv = 20.0
        ka = 5.0
        
        # 位置误差
        pos_error = target_pos - self.current_position
        
        # 计算目标速度（限制最大速度）
        target_velocity = kp * pos_error
        if abs(target_velocity) > self.max_velocity:
            target_velocity = np.sign(target_velocity) * self.max_velocity
        
        # 速度误差
        vel_error = target_velocity - self.current_velocity
        
        # 计算目标加速度
        target_acceleration = kv * vel_error
        if abs(target_acceleration) > self.max_acceleration:
            target_acceleration = np.sign(target_acceleration) * self.max_acceleration
        
        # 加速度误差
        acc_error = target_acceleration - self.current_acceleration
        
        # 计算加加速度（限制最大加加速度）
        jerk = ka * acc_error
        if abs(jerk) > self.max_jerk:
            jerk = np.sign(jerk) * self.max_jerk
        
        # 更新状态
        self.current_acceleration += jerk * dt
        self.current_velocity += self.current_acceleration * dt
        self.current_position += self.current_velocity * dt
        
        # 滤波
        filtered_position = self.position_filter.update(self.current_position)
        filtered_velocity = self.velocity_filter.update(self.current_velocity)
        
        return filtered_position, filtered_velocity, self.current_acceleration
    
    def start_trajectory(self, trajectory: List[Tuple[float, float, float]]):
        """
        开始执行轨迹
        
        Args:
            trajectory: 轨迹列表
        """
        self.trajectory_buffer = trajectory
        self.current_trajectory_index = 0
    
    def get_trajectory_point(self) -> Optional[Tuple[float, float, float]]:
        """
        获取当前轨迹点
        
        Returns:
            当前轨迹点 (位置, 速度, 加速度)
        """
        if self.current_trajectory_index < len(self.trajectory_buffer):
            point = self.trajectory_buffer[self.current_trajectory_index]
            self.current_trajectory_index += 1
            return point
        else:
            return None
    
    def is_trajectory_completed(self) -> bool:
        """检查轨迹是否完成"""
        return self.current_trajectory_index >= len(self.trajectory_buffer)
    
    def stop_trajectory(self):
        """停止轨迹执行"""
        self.trajectory_buffer = []
        self.current_trajectory_index = 0


class ForceController:
    """力控制器"""
    
    def __init__(self, config: ServoConfig):
        """
        初始化力控制器
        
        Args:
            config: 舵机配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PID参数
        self.kp = 2.0
        self.ki = 0.5
        self.kd = 0.1
        
        # 状态
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # 限制
        self.max_force = config.max_force
        self.min_force = 0.1
        self.max_output = 50.0  # 最大位置调整量 (度)
        
        # 抗饱和限制
        self.integral_limit = 10.0
        
        # 滤波器
        self.force_filter = LowPassFilter(cutoff_frequency=5.0, sampling_rate=100.0)
        self.output_filter = LowPassFilter(cutoff_frequency=10.0, sampling_rate=100.0)
        
        # 死区
        self.dead_zone = 0.5  # 力误差死区 (N)
        
    def update(self, current_force: float, target_force: float,
               dt: Optional[float] = None) -> float:
        """
        更新力控制器
        
        Args:
            current_force: 当前力测量值
            target_force: 目标力
            dt: 时间步长
            
        Returns:
            位置调整量 (度)
        """
        # 滤波当前力
        filtered_force = self.force_filter.update(current_force)
        
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            if self.last_time > 0:
                dt = current_time - self.last_time
            else:
                dt = 0.01
        self.last_time = current_time
        
        # 计算误差
        error = target_force - filtered_force
        
        # 死区处理
        if abs(error) < self.dead_zone:
            error = 0.0
            self.integral = 0.0  # 重置积分项
        else:
            # 更新积分项（带抗饱和）
            self.integral += error * dt
            if abs(self.integral) > self.integral_limit:
                self.integral = np.sign(self.integral) * self.integral_limit
        
        # 计算微分项
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.last_error) / dt
        
        # 保存误差
        self.last_error = error
        
        # 计算PID输出
        output = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)
        
        # 输出限幅
        output = max(-self.max_output, min(self.max_output, output))
        
        # 滤波输出
        filtered_output = self.output_filter.update(output)
        
        return filtered_output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        self.force_filter.reset()
        self.output_filter.reset()
    
    def set_pid_params(self, kp: float, ki: float, kd: float):
        """
        设置PID参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def set_limits(self, max_force: float, min_force: float = 0.1):
        """
        设置力限制
        
        Args:
            max_force: 最大力
            min_force: 最小力
        """
        self.max_force = max_force
        self.min_force = min_force


class PositionController:
    """位置控制器"""
    
    def __init__(self, config: ServoConfig):
        """
        初始化位置控制器
        
        Args:
            config: 舵机配置
        """
        self.config = config
        
        # PID参数
        self.kp = 10.0
        self.ki = 0.1
        self.kd = 0.5
        
        # 状态
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # 限制
        self.max_output = 100.0  # 最大输出 (脉冲宽度调整量)
        
        # 滤波器
        self.position_filter = KalmanFilter(
            initial_state=np.array([0.0, 0.0]),  # [位置, 速度]
            initial_covariance=np.eye(2) * 0.1,
            process_noise=np.diag([0.01, 0.1]),
            measurement_noise=0.1
        )
        
    def update(self, current_position: float, target_position: float,
               dt: Optional[float] = None) -> float:
        """
        更新位置控制器
        
        Args:
            current_position: 当前位置
            target_position: 目标位置
            dt: 时间步长
            
        Returns:
            输出值 (脉冲宽度或角度)
        """
        # 使用卡尔曼滤波估计位置和速度
        self.position_filter.predict(dt=dt or 0.01)
        measurement = np.array([current_position])
        estimated_state = self.position_filter.update(measurement)
        
        # 使用估计的位置
        filtered_position = estimated_state[0, 0]
        
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            if self.last_time > 0:
                dt = current_time - self.last_time
            else:
                dt = 0.01
        self.last_time = current_time
        
        # 计算误差
        error = target_position - filtered_position
        
        # 更新积分项
        self.integral += error * dt
        if abs(self.integral) > 100.0:
            self.integral = np.sign(self.integral) * 100.0
        
        # 计算微分项
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.last_error) / dt
        
        # 保存误差
        self.last_error = error
        
        # 计算PID输出
        output = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)
        
        # 输出限幅
        output = max(-self.max_output, min(self.max_output, output))
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        self.position_filter.reset()
    
    def set_pid_params(self, kp: float, ki: float, kd: float):
        """
        设置PID参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd


class VelocityController:
    """速度控制器"""
    
    def __init__(self, config: ServoConfig):
        """
        初始化速度控制器
        
        Args:
            config: 舵机配置
        """
        self.config = config
        
        # PI参数（速度控制通常不需要微分项）
        self.kp = 5.0
        self.ki = 0.5
        
        # 状态
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # 限制
        self.max_output = 100.0
        
    def update(self, current_velocity: float, target_velocity: float,
               dt: Optional[float] = None) -> float:
        """
        更新速度控制器
        
        Args:
            current_velocity: 当前速度
            target_velocity: 目标速度
            dt: 时间步长
            
        Returns:
            输出值
        """
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            if self.last_time > 0:
                dt = current_time - self.last_time
            else:
                dt = 0.01
        self.last_time = current_time
        
        # 计算误差
        error = target_velocity - current_velocity
        
        # 更新积分项
        self.integral += error * dt
        if abs(self.integral) > 50.0:
            self.integral = np.sign(self.integral) * 50.0
        
        # 保存误差
        self.last_error = error
        
        # 计算PI输出
        output = self.kp * error + self.ki * self.integral
        
        # 输出限幅
        output = max(-self.max_output, min(self.max_output, output))
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
    
    def set_pi_params(self, kp: float, ki: float):
        """
        设置PI参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
        """
        self.kp = kp
        self.ki = ki


class ImpedanceController:
    """阻抗控制器"""
    
    def __init__(self, config: ServoConfig):
        """
        初始化阻抗控制器
        
        Args:
            config: 舵机配置
        """
        self.config = config
        
        # 阻抗参数
        self.mass = 0.1  # 质量 (kg)
        self.damping = 10.0  # 阻尼 (N·s/m)
        self.stiffness = 100.0  # 刚度 (N/m)
        
        # 状态
        self.position = 0.0
        self.velocity = 0.0
        self.last_time = 0.0
        
    def update(self, external_force: float, target_position: float,
               dt: Optional[float] = None) -> Tuple[float, float]:
        """
        更新阻抗控制器
        
        Args:
            external_force: 外力
            target_position: 目标位置
            dt: 时间步长
            
        Returns:
            (目标位置, 目标速度)
        """
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            if self.last_time > 0:
                dt = current_time - self.last_time
            else:
                dt = 0.01
        self.last_time = current_time
        
        # 计算位置误差
        position_error = target_position - self.position
        
        # 计算阻抗力
        impedance_force = (self.stiffness * position_error -
                          self.damping * self.velocity)
        
        # 计算加速度 (F = ma)
        acceleration = (external_force + impedance_force) / self.mass
        
        # 更新速度和位置
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        return self.position, self.velocity
    
    def reset(self):
        """重置控制器状态"""
        self.position = 0.0
        self.velocity = 0.0
        self.last_time = 0.0
    
    def set_impedance_params(self, mass: float, damping: float, stiffness: float):
        """
        设置阻抗参数
        
        Args:
            mass: 质量
            damping: 阻尼
            stiffness: 刚度
        """
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness


class KalmanFilter:
    """卡尔曼滤波器"""
    
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray,
                 process_noise: np.ndarray, measurement_noise: float):
        """
        初始化卡尔曼滤波器
        
        Args:
            initial_state: 初始状态向量
            initial_covariance: 初始协方差矩阵
            process_noise: 过程噪声协方差矩阵
            measurement_noise: 测量噪声方差
        """
        self.state = initial_state.copy()
        self.covariance = initial_covariance.copy()
        self.process_noise = process_noise.copy()
        self.measurement_noise = measurement_noise
        
        # 状态转移矩阵（假设为恒速模型）
        self.state_dim = initial_state.shape[0]
        self.F = np.eye(self.state_dim)
        
        # 测量矩阵（假设只能测量位置）
        self.H = np.array([[1.0, 0.0]])
        
        # 测量噪声协方差矩阵
        self.R = np.array([[measurement_noise]])
        
    def predict(self, dt: float = 0.01):
        """
        预测步骤
        
        Args:
            dt: 时间步长
        """
        # 更新状态转移矩阵（恒速模型）
        if self.state_dim >= 2:
            self.F[0, 1] = dt
        
        # 状态预测
        self.state = self.F @ self.state
        
        # 协方差预测
        self.covariance = self.F @ self.covariance @ self.F.T + self.process_noise
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新步骤
        
        Args:
            measurement: 测量值
            
        Returns:
            更新后的状态估计
        """
        # 计算卡尔曼增益
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        
        # 协方差更新
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ self.H) @ self.covariance
        
        return self.state.copy()
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """
        重置滤波器
        
        Args:
            initial_state: 新的初始状态
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = np.zeros(self.state_dim)
        
        self.covariance = np.eye(self.state_dim) * 0.1


class LowPassFilter:
    """低通滤波器"""
    
    def __init__(self, cutoff_frequency: float, sampling_rate: float):
        """
        初始化低通滤波器
        
        Args:
            cutoff_frequency: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
        """
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        
        # 计算滤波器系数
        dt = 1.0 / sampling_rate
        rc = 1.0 / (2 * np.pi * cutoff_frequency)
        self.alpha = dt / (rc + dt)
        
        # 状态
        self.filtered_value = 0.0
        
    def update(self, new_value: float) -> float:
        """
        更新滤波器
        
        Args:
            new_value: 新测量值
            
        Returns:
            滤波后的值
        """
        self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
    def reset(self, initial_value: float = 0.0):
        """重置滤波器"""
        self.filtered_value = initial_value


class MovingAverageFilter:
    """移动平均滤波器"""
    
    def __init__(self, window_size: int = 5):
        """
        初始化移动平均滤波器
        
        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        self.buffer = []
    
    def update(self, new_value: float) -> float:
        """
        更新滤波器
        
        Args:
            new_value: 新测量值
            
        Returns:
            滤波后的值
        """
        self.buffer.append(new_value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        return sum(self.buffer) / len(self.buffer)
    
    def reset(self):
        """重置滤波器"""
        self.buffer = []


class GripperController:
    """夹爪控制器"""
    
    def __init__(self, config: ServoConfig, hardware_interface: HardwareInterface):
        """
        初始化夹爪控制器
        
        Args:
            config: 舵机配置
            hardware_interface: 硬件接口
        """
        self.config = config
        self.hardware_interface = hardware_interface
        
        # 控制模块
        self.trajectory_planner = TrajectoryPlanner(config)
        self.force_controller = ForceController(config)
        self.position_controller = PositionController(config)
        self.velocity_controller = VelocityController(config)
        self.impedance_controller = ImpedanceController(config)
        
        # 控制状态
        self.control_mode = ControlMode.POSITION
        self.control_status = ControlStatus.IDLE
        
        self.current_state = GripperState(
            timestamp=time.time(),
            position=config.home_position,
            velocity=0.0,
            acceleration=0.0,
            force=0.0,
            current=0.0,
            temperature=25.0,
            voltage=12.0,
            load=0.0,
            moving=False,
            mode=ControlMode.POSITION,
            status=ControlStatus.IDLE
        )
        
        # 控制目标
        self.target_position = config.home_position
        self.target_velocity = 0.0
        self.target_force = 0.0
        
        # 控制线程
        self.control_thread = None
        self.running = False
        self.paused = False
        self.control_rate = 100  # Hz
        
        # 线程同步
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        
        # 回调函数
        self.state_callbacks = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化
        self._init_controller()
        
        self.logger.info("夹爪控制器初始化完成")
    
    def _init_controller(self):
        """初始化控制器"""
        # 设置轨迹规划器初始状态
        self.trajectory_planner.set_current_state(
            position=self.current_state.position,
            velocity=self.current_state.velocity,
            acceleration=self.current_state.acceleration
        )
    
    def connect(self) -> bool:
        """
        连接夹爪
        
        Returns:
            连接是否成功
        """
        self.logger.info("连接夹爪")
        
        with self.lock:
            self.control_status = ControlStatus.CONNECTING
        
        # 通过硬件接口连接舵机
        success = self.hardware_interface.connect_servo()
        
        if success:
            # 获取初始状态
            self.update_state()
            
            # 设置初始位置
            self.move_to(self.config.home_position, block=False)
            
            with self.lock:
                self.control_status = ControlStatus.CONNECTED
            
            self.logger.info("夹爪连接成功")
        else:
            with self.lock:
                self.control_status = ControlStatus.ERROR
                self.current_state.error_code = 1
                self.current_state.error_message = "连接失败"
            
            self.logger.error("夹爪连接失败")
        
        return success
    
    def disconnect(self):
        """断开夹爪连接"""
        self.logger.info("断开夹爪连接")
        
        self.stop_control()
        self.hardware_interface.disconnect_servo()
        
        with self.lock:
            self.control_status = ControlStatus.IDLE
        
        self.logger.info("夹爪断开连接")
    
    def update_state(self) -> Optional[GripperState]:
        """
        更新夹爪状态
        
        Returns:
            更新后的状态，失败返回None
        """
        try:
            # 从硬件接口获取状态
            servo_state = self.hardware_interface.get_servo_state()
            
            if servo_state:
                with self.lock:
                    # 更新当前状态
                    self.current_state.timestamp = servo_state.timestamp
                    self.current_state.position = servo_state.position
                    self.current_state.velocity = servo_state.velocity
                    self.current_state.current = servo_state.current
                    self.current_state.temperature = servo_state.temperature
                    self.current_state.voltage = servo_state.voltage
                    self.current_state.load = servo_state.load
                    self.current_state.moving = servo_state.moving
                    self.current_state.mode = self.control_mode
                    self.current_state.status = self.control_status
                    self.current_state.target_position = self.target_position
                    self.current_state.target_velocity = self.target_velocity
                    self.current_state.target_force = self.target_force
                    
                    # 清空错误状态
                    self.current_state.error_code = 0
                    self.current_state.error_message = ""
                
                # 通知回调函数
                self._notify_state_callbacks()
                
                return self.current_state
            
            return None
            
        except Exception as e:
            self.logger.error(f"更新夹爪状态失败: {e}")
            
            with self.lock:
                self.current_state.error_code = 2
                self.current_state.error_message = f"更新状态失败: {str(e)}"
            
            return None
    
    def move_to(self, position: float, speed: Optional[float] = None,
                block: bool = True, wait_timeout: float = 5.0) -> bool:
        """
        移动到指定位置
        
        Args:
            position: 目标位置 (度)
            speed: 运动速度 (None使用默认速度)
            block: 是否阻塞直到完成
            wait_timeout: 等待超时时间 (秒)
            
        Returns:
            移动是否成功
        """
        self.logger.info(f"移动到位置: {position}, 速度: {speed}")
        
        # 限制位置范围
        position = max(self.config.min_angle,
                      min(self.config.max_angle, position))
        
        with self.lock:
            # 设置控制模式
            self.control_mode = ControlMode.POSITION
            self.target_position = position
            self.control_status = ControlStatus.MOVING
        
        # 发送移动命令
        success = self.hardware_interface.set_servo_position(
            position=position,
            speed=speed
        )
        
        if success and block:
            # 等待移动完成
            return self.wait_for_move(wait_timeout)
        
        return success
    
    def move_with_trajectory(self, target_position: float,
                            max_velocity: Optional[float] = None,
                            max_acceleration: Optional[float] = None,
                            block: bool = True) -> bool:
        """
        使用轨迹规划移动到指定位置
        
        Args:
            target_position: 目标位置
            max_velocity: 最大速度
            max_acceleration: 最大加速度
            block: 是否阻塞直到完成
            
        Returns:
            移动是否成功
        """
        self.logger.info(f"使用轨迹规划移动到位置: {target_position}")
        
        # 规划轨迹
        trajectory = self.trajectory_planner.plan_trapezoidal_trajectory(
            start_pos=self.current_state.position,
            target_pos=target_position,
            max_vel=max_velocity,
            max_acc=max_acceleration
        )
        
        # 开始执行轨迹
        self.trajectory_planner.start_trajectory(trajectory)
        
        with self.lock:
            self.control_mode = ControlMode.POSITION
            self.target_position = target_position
            self.control_status = ControlStatus.MOVING
        
        # 启动控制线程（如果未运行）
        self.start_control()
        
        if block:
            # 等待轨迹完成
            return self.wait_for_trajectory(timeout=10.0)
        
        return True
    
    def set_velocity(self, velocity: float, block: bool = False) -> bool:
        """
        设置速度
        
        Args:
            velocity: 目标速度 (度/秒)
            block: 是否阻塞
            
        Returns:
            设置是否成功
        """
        self.logger.info(f"设置速度: {velocity} 度/秒")
        
        with self.lock:
            self.control_mode = ControlMode.VELOCITY
            self.target_velocity = velocity
            self.control_status = ControlStatus.MOVING
        
        # 启动控制线程（如果未运行）
        self.start_control()
        
        if block:
            # 等待一段时间
            time.sleep(1.0)
        
        return True
    
    def set_force(self, force: float, block: bool = True,
                  wait_timeout: float = 5.0) -> bool:
        """
        设置夹持力
        
        Args:
            force: 目标力 (N)
            block: 是否阻塞直到完成
            wait_timeout: 等待超时时间 (秒)
            
        Returns:
            设置是否成功
        """
        self.logger.info(f"设置夹持力: {force}N")
        
        # 限制力范围
        force = max(self.force_controller.min_force,
                   min(self.force_controller.max_force, force))
        
        with self.lock:
            # 设置控制模式
            self.control_mode = ControlMode.FORCE
            self.target_force = force
            self.control_status = ControlStatus.HOLDING
        
        # 启动力控制
        self.start_control()
        
        if block:
            # 等待力稳定
            return self.wait_for_force(target_force=force, timeout=wait_timeout)
        
        return True
    
    def open(self, speed: Optional[float] = None, block: bool = True) -> bool:
        """
        打开夹爪
        
        Args:
            speed: 运动速度
            block: 是否阻塞直到完成
            
        Returns:
            操作是否成功
        """
        self.logger.info("打开夹爪")
        return self.move_to(self.config.max_angle, speed, block)
    
    def close(self, speed: Optional[float] = None, block: bool = True) -> bool:
        """
        关闭夹爪
        
        Args:
            speed: 运动速度
            block: 是否阻塞直到完成
            
        Returns:
            操作是否成功
        """
        self.logger.info("关闭夹爪")
        return self.move_to(self.config.min_angle, speed, block)
    
    def home(self, speed: Optional[float] = None, block: bool = True) -> bool:
        """
        回零（回到初始位置）
        
        Args:
            speed: 运动速度
            block: 是否阻塞直到完成
            
        Returns:
            操作是否成功
        """
        self.logger.info("夹爪回零")
        return self.move_to(self.config.home_position, speed, block)
    
    def wait_for_move(self, timeout: float = 5.0) -> bool:
        """
        等待移动完成
        
        Args:
            timeout: 超时时间 (秒)
            
        Returns:
            是否成功完成移动
        """
        start_time = time.time()
        moving_check_count = 0
        
        while time.time() - start_time < timeout:
            # 更新状态
            self.update_state()
            
            if not self.current_state.moving:
                moving_check_count += 1
                if moving_check_count >= 3:  # 连续3次检查都不在运动
                    with self.lock:
                        self.control_status = ControlStatus.HOLDING
                    
                    self.logger.info("移动完成")
                    return True
            else:
                moving_check_count = 0
            
            time.sleep(0.05)
        
        self.logger.warning("等待移动超时")
        
        with self.lock:
            self.control_status = ControlStatus.ERROR
            self.current_state.error_code = 3
            self.current_state.error_message = "移动超时"
        
        return False
    
    def wait_for_trajectory(self, timeout: float = 10.0) -> bool:
        """
        等待轨迹完成
        
        Args:
            timeout: 超时时间 (秒)
            
        Returns:
            轨迹是否成功完成
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.trajectory_planner.is_trajectory_completed():
                with self.lock:
                    self.control_status = ControlStatus.HOLDING
                
                self.logger.info("轨迹完成")
                return True
            
            time.sleep(0.05)
        
        self.logger.warning("等待轨迹超时")
        
        with self.lock:
            self.control_status = ControlStatus.ERROR
            self.current_state.error_code = 4
            self.current_state.error_message = "轨迹执行超时"
        
        return False
    
    def wait_for_force(self, target_force: float, tolerance: float = 1.0,
                      timeout: float = 5.0) -> bool:
        """
        等待达到目标力
        
        Args:
            target_force: 目标力
            tolerance: 容忍误差
            timeout: 超时时间
            
        Returns:
            是否达到目标力
        """
        start_time = time.time()
        stable_count = 0
        
        while time.time() - start_time < timeout:
            # 更新状态（这里需要从传感器获取力数据）
            # 简化处理：假设力控制器正在工作
            time.sleep(0.1)
            
            # 检查是否稳定（简化）
            stable_count += 1
            if stable_count > 20:  # 2秒稳定
                self.logger.info(f"力控制稳定在 {target_force}N")
                return True
        
        self.logger.warning("力控制超时")
        return False
    
    def start_control(self):
        """开始控制线程"""
        if self.control_thread is None or not self.control_thread.is_alive():
            self.running = True
            self.paused = False
            self.control_thread = threading.Thread(
                target=self._control_loop,
                daemon=True
            )
            self.control_thread.start()
            self.logger.info("控制线程启动")
    
    def stop_control(self):
        """停止控制线程"""
        self.running = False
        self.paused = False
        
        # 通知等待的线程
        with self.condition:
            self.condition.notify_all()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
            self.logger.info("控制线程停止")
    
    def pause_control(self):
        """暂停控制线程"""
        self.paused = True
        self.logger.info("控制线程暂停")
    
    def resume_control(self):
        """恢复控制线程"""
        self.paused = False
        
        # 通知等待的线程
        with self.condition:
            self.condition.notify_all()
        
        self.logger.info("控制线程恢复")
    
    def _control_loop(self):
        """控制循环"""
        control_interval = 1.0 / self.control_rate
        
        while self.running:
            loop_start = time.time()
            
            # 检查暂停状态
            if self.paused:
                with self.condition:
                    self.condition.wait(timeout=control_interval)
                continue
            
            try:
                # 根据控制模式执行相应的控制
                if self.control_mode == ControlMode.FORCE:
                    self._force_control_iteration()
                elif self.control_mode == ControlMode.POSITION:
                    self._position_control_iteration()
                elif self.control_mode == ControlMode.VELOCITY:
                    self._velocity_control_iteration()
                elif self.control_mode == ControlMode.IMPEDANCE:
                    self._impedance_control_iteration()
                
                # 更新状态
                self.update_state()
                
            except Exception as e:
                self.logger.error(f"控制循环错误: {e}")
                
                with self.lock:
                    self.control_status = ControlStatus.ERROR
                    self.current_state.error_code = 5
                    self.current_state.error_message = f"控制错误: {str(e)}"
            
            # 控制频率
            loop_time = time.time() - loop_start
            if loop_time < control_interval:
                time.sleep(control_interval - loop_time)
    
    def _force_control_iteration(self):
        """力控制迭代"""
        # 需要从传感器获取当前力值
        # 这里简化处理，实际应该从传感器读取
        current_force = self.current_state.force  # 需要实际传感器数据
        
        # 计算位置调整
        position_adjustment = self.force_controller.update(
            current_force=current_force,
            target_force=self.target_force
        )
        
        # 计算新位置
        new_position = self.current_state.position + position_adjustment
        new_position = max(self.config.min_angle,
                          min(self.config.max_angle, new_position))
        
        # 移动夹爪
        self.hardware_interface.set_servo_position(new_position)
        
        # 更新目标位置（用于显示）
        with self.lock:
            self.target_position = new_position
    
    def _position_control_iteration(self):
        """位置控制迭代"""
        # 检查是否有轨迹规划
        if not self.trajectory_planner.is_trajectory_completed():
            # 获取下一个轨迹点
            point = self.trajectory_planner.get_trajectory_point()
            if point:
                target_pos, target_vel, target_acc = point
                
                # 发送位置命令
                self.hardware_interface.set_servo_position(target_pos)
                
                # 更新当前状态
                with self.lock:
                    self.current_state.position = target_pos
                    self.current_state.velocity = target_vel
                    self.current_state.acceleration = target_acc
            else:
                # 轨迹完成
                with self.lock:
                    self.control_status = ControlStatus.HOLDING
        else:
            # 使用轨迹规划器实时规划
            target_pos, target_vel, target_acc = self.trajectory_planner.get_next_point(
                self.target_position
            )
            
            # 发送位置命令
            self.hardware_interface.set_servo_position(target_pos)
            
            # 更新当前状态
            with self.lock:
                self.current_state.position = target_pos
                self.current_state.velocity = target_vel
                self.current_state.acceleration = target_acc
    
    def _velocity_control_iteration(self):
        """速度控制迭代"""
        # 获取当前位置
        current_pos = self.current_state.position
        current_vel = self.current_state.velocity
        
        # 计算位置调整
        position_adjustment = self.velocity_controller.update(
            current_velocity=current_vel,
            target_velocity=self.target_velocity
        )
        
        # 计算新位置
        new_position = current_pos + position_adjustment
        new_position = max(self.config.min_angle,
                          min(self.config.max_angle, new_position))
        
        # 移动夹爪
        self.hardware_interface.set_servo_position(new_position)
        
        # 更新目标位置（用于显示）
        with self.lock:
            self.target_position = new_position
    
    def _impedance_control_iteration(self):
        """阻抗控制迭代"""
        # 需要外部力输入
        # 这里简化处理
        external_force = 0.0  # 需要实际传感器数据
        
        # 计算阻抗控制输出
        target_pos, target_vel = self.impedance_controller.update(
            external_force=external_force,
            target_position=self.target_position
        )
        
        # 移动夹爪
        self.hardware_interface.set_servo_position(target_pos)
        
        # 更新当前状态
        with self.lock:
            self.current_state.position = target_pos
            self.current_state.velocity = target_vel
    
    def emergency_stop(self):
        """紧急停止"""
        self.logger.warning("夹爪紧急停止")
        
        # 停止控制线程
        self.stop_control()
        
        # 停止轨迹规划
        self.trajectory_planner.stop_trajectory()
        
        # 发送紧急停止命令
        self.hardware_interface.emergency_stop()
        
        # 重置控制器状态
        self.force_controller.reset()
        self.position_controller.reset()
        self.velocity_controller.reset()
        self.impedance_controller.reset()
        
        # 更新状态
        with self.lock:
            self.control_mode = ControlMode.POSITION
            self.control_status = ControlStatus.EMERGENCY_STOP
            self.target_position = self.current_state.position
            self.target_velocity = 0.0
            self.target_force = 0.0
        
        self.logger.info("紧急停止完成")
    
    def get_state(self) -> GripperState:
        """
        获取当前状态
        
        Returns:
            夹爪状态
        """
        with self.lock:
            return self.current_state
    
    def get_position(self) -> float:
        """
        获取当前位置
        
        Returns:
            当前位置
        """
        with self.lock:
            return self.current_state.position
    
    def get_force(self) -> float:
        """
        获取当前力
        
        Returns:
            当前力
        """
        with self.lock:
            return self.current_state.force
    
    def is_moving(self) -> bool:
        """
        检查是否在运动
        
        Returns:
            是否在运动
        """
        with self.lock:
            return self.current_state.moving
    
    def is_connected(self) -> bool:
        """
        检查是否连接
        
        Returns:
            是否连接
        """
        status = self.hardware_interface.get_status()
        return status["servo"]["connected"]
    
    def add_state_callback(self, callback: Callable[[GripperState], None]):
        """
        添加状态回调函数
        
        Args:
            callback: 回调函数，接收GripperState作为参数
        """
        self.state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[GripperState], None]):
        """
        移除状态回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
    
    def _notify_state_callbacks(self):
        """通知所有状态回调函数"""
        for callback in self.state_callbacks:
            try:
                callback(self.current_state)
            except Exception as e:
                self.logger.error(f"状态回调函数执行错误: {e}")
    
    def calibrate(self, calibration_type: str = "limits") -> bool:
        """
        校准夹爪
        
        Args:
            calibration_type: 校准类型（"limits"极限位置，"force"力传感器）
            
        Returns:
            校准是否成功
        """
        self.logger.info(f"校准夹爪，类型: {calibration_type}")
        
        with self.lock:
            self.control_status = ControlStatus.CALIBRATING
        
        try:
            if calibration_type == "limits":
                result = self._calibrate_limits()
            elif calibration_type == "force":
                result = self._calibrate_force()
            else:
                self.logger.error(f"未知的校准类型: {calibration_type}")
                result = False
            
            with self.lock:
                if result:
                    self.control_status = ControlStatus.CONNECTED
                else:
                    self.control_status = ControlStatus.ERROR
            
            return result
            
        except Exception as e:
            self.logger.error(f"校准失败: {e}")
            
            with self.lock:
                self.control_status = ControlStatus.ERROR
            
            return False
    
    def _calibrate_limits(self) -> bool:
        """极限位置校准"""
        try:
            # 打开夹爪到最大位置
            self.logger.info("校准最大位置...")
            self.open(block=True)
            time.sleep(1.0)
            
            max_pos = self.get_position()
            self.config.max_angle = max_pos
            self.logger.info(f"最大位置设置为: {max_pos}")
            
            # 关闭夹爪到最小位置
            self.logger.info("校准最小位置...")
            self.close(block=True)
            time.sleep(1.0)
            
            min_pos = self.get_position()
            self.config.min_angle = min_pos
            self.logger.info(f"最小位置设置为: {min_pos}")
            
            # 回到中间位置
            home_pos = (min_pos + max_pos) / 2
            self.config.home_position = home_pos
            self.move_to(home_pos, block=True)
            
            self.logger.info("极限位置校准完成")
            return True
            
        except Exception as e:
            self.logger.error(f"极限位置校准失败: {e}")
            return False
    
    def _calibrate_force(self) -> bool:
        """力传感器校准"""
        try:
            self.logger.info("开始力传感器校准...")
            
            # 这里应该实现力传感器的校准逻辑
            # 简化实现：记录当前力传感器读数作为零点
            
            # 假设力传感器已连接并可以读取
            # current_force = self.get_force()
            # self.force_controller.zero_offset = current_force
            
            self.logger.warning("力传感器校准尚未完全实现")
            return False
            
        except Exception as e:
            self.logger.error(f"力传感器校准失败: {e}")
            return False
    
    def set_config(self, config: ServoConfig):
        """
        设置新的配置
        
        Args:
            config: 新的舵机配置
        """
        self.config = config
        
        # 更新子模块
        self.trajectory_planner = TrajectoryPlanner(config)
        self.force_controller = ForceController(config)
        self.position_controller = PositionController(config)
        self.velocity_controller = VelocityController(config)
        self.impedance_controller = ImpedanceController(config)
        
        # 更新初始状态
        self.trajectory_planner.set_current_state(
            position=self.current_state.position,
            velocity=self.current_state.velocity,
            acceleration=self.current_state.acceleration
        )
    
    def save_calibration(self, filepath: str):
        """
        保存校准数据
        
        Args:
            filepath: 文件路径
        """
        calibration_data = {
            "min_angle": self.config.min_angle,
            "max_angle": self.config.max_angle,
            "home_position": self.config.home_position,
            "max_force": self.config.max_force,
            "min_pulse": self.config.min_pulse,
            "max_pulse": self.config.max_pulse,
            "speed": self.config.speed,
            "torque": self.config.torque,
            "timestamp": time.time()
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info(f"校准数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存校准数据失败: {e}")
            raise
    
    def load_calibration(self, filepath: str) -> bool:
        """
        加载校准数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载是否成功
        """
        try:
            import json
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            # 更新配置
            if "min_angle" in calibration_data:
                self.config.min_angle = calibration_data["min_angle"]
            if "max_angle" in calibration_data:
                self.config.max_angle = calibration_data["max_angle"]
            if "home_position" in calibration_data:
                self.config.home_position = calibration_data["home_position"]
            if "max_force" in calibration_data:
                self.config.max_force = calibration_data["max_force"]
            if "min_pulse" in calibration_data:
                self.config.min_pulse = calibration_data["min_pulse"]
            if "max_pulse" in calibration_data:
                self.config.max_pulse = calibration_data["max_pulse"]
            if "speed" in calibration_data:
                self.config.speed = calibration_data["speed"]
            if "torque" in calibration_data:
                self.config.torque = calibration_data["torque"]
            
            # 更新力控制器限制
            self.force_controller.set_limits(
                max_force=self.config.max_force,
                min_force=0.1
            )
            
            self.logger.info(f"校准数据已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载校准数据失败: {e}")
            return False
    
    def set_force_pid_params(self, kp: float, ki: float, kd: float):
        """
        设置力控制PID参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        """
        self.force_controller.set_pid_params(kp, ki, kd)
        self.logger.info(f"力控制PID参数设置为: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def set_position_pid_params(self, kp: float, ki: float, kd: float):
        """
        设置位置控制PID参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        """
        self.position_controller.set_pid_params(kp, ki, kd)
        self.logger.info(f"位置控制PID参数设置为: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def set_velocity_pi_params(self, kp: float, ki: float):
        """
        设置速度控制PI参数
        
        Args:
            kp: 比例系数
            ki: 积分系数
        """
        self.velocity_controller.set_pi_params(kp, ki)
        self.logger.info(f"速度控制PI参数设置为: Kp={kp}, Ki={ki}")
    
    def set_impedance_params(self, mass: float, damping: float, stiffness: float):
        """
        设置阻抗参数
        
        Args:
            mass: 质量
            damping: 阻尼
            stiffness: 刚度
        """
        self.impedance_controller.set_impedance_params(mass, damping, stiffness)
        self.logger.info(f"阻抗参数设置为: M={mass}, B={damping}, K={stiffness}")