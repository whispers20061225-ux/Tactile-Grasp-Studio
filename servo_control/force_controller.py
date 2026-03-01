"""
力控模块 - 基于触觉传感器的自适应力控制
支持PID控制、模型预测控制和强化学习控制策略
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import logging

from config.paxini_gen3_config import PaxiniConfig
from config.demo_config import ServoConfig
from utils.calibration import ForceCalibrator
from utils.transformations import low_pass_filter

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """力控制模式"""
    POSITION_CONTROL = "position"
    FORCE_CONTROL = "force"
    IMPEDANCE_CONTROL = "impedance"
    ADMITTANCE_CONTROL = "admittance"
    ADAPTIVE_CONTROL = "adaptive"
    LEARNING_CONTROL = "learning"


@dataclass
class ForceControlParams:
    """力控制参数"""
    # PID控制参数
    kp: float = 0.8          # 比例增益
    ki: float = 0.05         # 积分增益
    kd: float = 0.1          # 微分增益
    i_clamp: float = 1.0     # 积分项限幅
    dead_zone: float = 0.02  # 死区 (N)
    
    # 阻抗控制参数
    mass: float = 0.1        # 虚拟质量 (kg)
    damping: float = 5.0     # 虚拟阻尼 (Ns/m)
    stiffness: float = 100.0 # 虚拟刚度 (N/m)
    
    # 自适应控制参数
    learning_rate: float = 0.01
    forgetting_factor: float = 0.95
    
    # 安全限制
    max_force: float = 10.0   # 最大允许力 (N)
    min_force: float = 0.1    # 最小保持力 (N)
    force_rate_limit: float = 5.0  # 力变化率限制 (N/s)
    
    # 滤波参数
    low_pass_cutoff: float = 10.0  # 低通滤波截止频率 (Hz)
    median_filter_size: int = 5    # 中值滤波窗口大小
    
    # 控制参数
    control_freq: float = 100.0  # 控制频率 (Hz)
    timeout: float = 5.0         # 超时时间 (s)


class PIDForceController:
    """PID力控制器"""
    
    def __init__(self, params: ForceControlParams):
        self.params = params
        self.reset()
        
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        
    def compute(self, target_force: float, actual_force: float, dt: float = None) -> float:
        """
        计算控制输出
        
        Args:
            target_force: 目标力 (N)
            actual_force: 实际测量力 (N)
            dt: 时间步长 (s)，如果为None则自动计算
            
        Returns:
            控制输出（位置指令或速度指令）
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            dt = max(dt, 1e-6)  # 避免除零
            
        # 计算误差
        error = target_force - actual_force
        
        # 死区处理
        if abs(error) < self.params.dead_zone:
            error = 0.0
            
        # PID计算
        p_term = self.params.kp * error
        
        self.integral += error * dt
        # 积分限幅
        self.integral = np.clip(self.integral, 
                               -self.params.i_clamp / self.params.ki if self.params.ki > 0 else -np.inf,
                               self.params.i_clamp / self.params.ki if self.params.ki > 0 else np.inf)
        i_term = self.params.ki * self.integral
        
        d_term = self.params.kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        # 输出计算
        output = p_term + i_term + d_term
        
        # 输出限幅（基于力变化率限制）
        max_output_change = self.params.force_rate_limit * dt
        output = np.clip(output, -max_output_change, max_output_change)
        
        return output
    
    def update_params(self, kp: float = None, ki: float = None, kd: float = None):
        """更新PID参数"""
        if kp is not None:
            self.params.kp = kp
        if ki is not None:
            self.params.ki = ki
        if kd is not None:
            self.params.kd = kd


class ImpedanceController:
    """阻抗控制器"""
    
    def __init__(self, params: ForceControlParams):
        self.params = params
        self.reset()
        
    def reset(self):
        """重置控制器状态"""
        self.position = 0.0
        self.velocity = 0.0
        self.prev_force = 0.0
        self.prev_time = time.time()
        
    def compute(self, target_position: float, actual_force: float, 
                dt: float = None) -> float:
        """
        阻抗控制计算
        
        Args:
            target_position: 目标位置
            actual_force: 实际测量力 (N)
            dt: 时间步长 (s)
            
        Returns:
            修正后的位置指令
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            dt = max(dt, 1e-6)
            
        # 计算阻抗模型产生的位移
        force_error = actual_force - self.prev_force
        acceleration = (actual_force - self.params.damping * self.velocity - 
                       self.params.stiffness * (self.position - target_position)) / self.params.mass
        
        # 更新状态
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.prev_force = actual_force
        
        # 返回修正位置
        return self.position


class AdaptiveForceController:
    """自适应力控制器"""
    
    def __init__(self, params: ForceControlParams):
        self.params = params
        self.adaptive_gains = {
            'kp': params.kp,
            'ki': params.ki,
            'kd': params.kd
        }
        self.error_history = deque(maxlen=10)
        self.gain_history = deque(maxlen=20)
        self.reset()
        
    def reset(self):
        """重置控制器状态"""
        self.pid_controller = PIDForceController(
            ForceControlParams(
                kp=self.adaptive_gains['kp'],
                ki=self.adaptive_gains['ki'],
                kd=self.adaptive_gains['kd']
            )
        )
        
    def adapt_gains(self, error: float, error_derivative: float):
        """
        自适应调整增益
        
        Args:
            error: 当前误差
            error_derivative: 误差导数
        """
        self.error_history.append(error)
        
        # 基于误差统计调整增益
        if len(self.error_history) >= 5:
            error_mean = np.mean(self.error_history)
            error_std = np.std(self.error_history)
            
            # 如果误差波动大，减小增益
            if error_std > 0.5:
                self.adaptive_gains['kp'] *= 0.9
                self.adaptive_gains['ki'] *= 0.9
            # 如果误差持续较大，增加增益
            elif abs(error_mean) > 1.0:
                self.adaptive_gains['kp'] *= 1.1
                self.adaptive_gains['ki'] *= 1.1
                
            # 限制增益范围
            self.adaptive_gains['kp'] = np.clip(self.adaptive_gains['kp'], 0.1, 5.0)
            self.adaptive_gains['ki'] = np.clip(self.adaptive_gains['ki'], 0.01, 1.0)
            self.adaptive_gains['kd'] = np.clip(self.adaptive_gains['kd'], 0.01, 2.0)
            
            # 更新PID控制器参数
            self.pid_controller.update_params(
                kp=self.adaptive_gains['kp'],
                ki=self.adaptive_gains['ki'],
                kd=self.adaptive_gains['kd']
            )
            
            self.gain_history.append(self.adaptive_gains.copy())
            
    def compute(self, target_force: float, actual_force: float, 
                dt: float = None) -> float:
        """
        自适应力控制计算
        
        Args:
            target_force: 目标力
            actual_force: 实际力
            dt: 时间步长
            
        Returns:
            控制输出
        """
        # 计算误差
        error = target_force - actual_force
        
        # 自适应调整
        if dt is not None:
            error_derivative = error / dt
            self.adapt_gains(error, error_derivative)
        
        # 使用PID计算输出
        return self.pid_controller.compute(target_force, actual_force, dt)
    
    def get_gain_history(self) -> List[Dict]:
        """获取增益调整历史"""
        return list(self.gain_history)


class ForceController:
    """主力控制器 - 整合多种控制策略"""
    
    def __init__(self, 
                 sensor_config: PaxiniConfig,
                 servo_config: ServoConfig,
                 control_mode: ControlMode = ControlMode.FORCE_CONTROL):
        """
        初始化力控制器
        
        Args:
            sensor_config: 传感器配置
            servo_config: 舵机配置
            control_mode: 初始控制模式
        """
        self.sensor_config = sensor_config
        self.servo_config = servo_config
        self.control_mode = control_mode
        
        # 力控制参数
        self.params = ForceControlParams()
        
        # 初始化控制器
        self.pid_controller = PIDForceController(self.params)
        self.impedance_controller = ImpedanceController(self.params)
        self.adaptive_controller = AdaptiveForceController(self.params)
        
        # 力数据缓冲区
        self.force_buffer = deque(maxlen=100)
        self.filtered_forces = deque(maxlen=20)
        
        # 状态变量
        self.target_force = 2.0  # 默认目标力 2N
        self.current_force = 0.0
        self.filtered_force = 0.0
        self.control_output = 0.0
        
        # 安全监控
        self.safety_violations = 0
        self.last_violation_time = 0
        
        # 校准器
        self.calibrator = ForceCalibrator()
        self.is_calibrated = False
        
        # 性能监控
        self.control_start_time = time.time()
        self.control_cycles = 0
        self.avg_computation_time = 0.0
        
        # 学习控制器（可选）
        self.learning_controller = None
        self.use_learning_control = False
        
        logger.info(f"力控制器初始化完成，模式: {control_mode.value}")
        
    def set_control_mode(self, mode: ControlMode):
        """
        设置控制模式
        
        Args:
            mode: 控制模式
        """
        self.control_mode = mode
        self.reset_controllers()
        logger.info(f"控制模式切换为: {mode.value}")
        
    def reset_controllers(self):
        """重置所有控制器"""
        self.pid_controller.reset()
        self.impedance_controller.reset()
        self.adaptive_controller.reset()
        
    def calibrate_sensor(self, calibration_data: np.ndarray = None):
        """
        校准力传感器
        
        Args:
            calibration_data: 校准数据，如果为None则进行自动校准
        """
        if calibration_data is None:
            logger.info("开始自动校准力传感器...")
            # 自动校准逻辑
            self.is_calibrated = self.calibrator.auto_calibrate()
        else:
            self.is_calibrated = self.calibrator.calibrate(calibration_data)
            
        if self.is_calibrated:
            logger.info("力传感器校准完成")
        else:
            logger.warning("力传感器校准失败")
            
    def update_force_measurement(self, raw_force_data: Union[float, np.ndarray]):
        """
        更新力测量值
        
        Args:
            raw_force_data: 原始力数据，可以是单个值或数组
        """
        # 转换为标量力值
        if isinstance(raw_force_data, np.ndarray):
            # 如果是触觉阵列，计算平均力或最大力
            if raw_force_data.ndim > 1:
                # 多维阵列，计算有效区域的平均力
                center_region = raw_force_data[1:-1, 1:-1] if raw_force_data.shape[0] > 2 else raw_force_data
                self.current_force = np.mean(center_region)
            else:
                self.current_force = np.mean(raw_force_data)
        else:
            self.current_force = float(raw_force_data)
            
        # 应用校准
        if self.is_calibrated:
            self.current_force = self.calibrator.apply_calibration(self.current_force)
            
        # 低通滤波
        self.filtered_force = low_pass_filter(
            self.current_force, 
            self.filtered_force if len(self.filtered_forces) > 0 else self.current_force,
            self.params.low_pass_cutoff,
            1.0 / self.params.control_freq
        )
        
        # 更新缓冲区
        self.force_buffer.append(self.current_force)
        self.filtered_forces.append(self.filtered_force)
        
    def check_safety(self) -> Tuple[bool, str]:
        """
        安全检查
        
        Returns:
            (是否安全, 警告信息)
        """
        current_time = time.time()
        
        # 检查力是否超限
        if abs(self.filtered_force) > self.params.max_force:
            self.safety_violations += 1
            self.last_violation_time = current_time
            return False, f"力超过限制: {self.filtered_force:.2f} > {self.params.max_force}"
            
        # 检查力变化率
        if len(self.force_buffer) >= 2:
            force_rate = abs(self.force_buffer[-1] - self.force_buffer[-2]) * self.params.control_freq
            if force_rate > self.params.force_rate_limit:
                return False, f"力变化率过高: {force_rate:.2f} > {self.params.force_rate_limit}"
                
        return True, "安全状态正常"
    
    def set_target_force(self, target_force: float):
        """
        设置目标力
        
        Args:
            target_force: 目标力值 (N)
        """
        if target_force > self.params.max_force:
            logger.warning(f"目标力 {target_force} 超过最大限制 {self.params.max_force}")
            target_force = self.params.max_force
            
        self.target_force = target_force
        logger.debug(f"目标力设置为: {target_force:.2f} N")
        
    def compute_control(self, dt: float = None) -> float:
        """
        计算控制输出
        
        Args:
            dt: 时间步长，如果为None则自动计算
            
        Returns:
            控制输出（位置或速度指令）
        """
        start_time = time.time()
        
        # 安全检查
        is_safe, safety_msg = self.check_safety()
        if not is_safe:
            logger.warning(f"安全违规: {safety_msg}")
            # 进入安全模式，输出零或最小力
            self.control_output = 0.0
            return self.control_output
            
        # 根据控制模式选择控制器
        if self.control_mode == ControlMode.FORCE_CONTROL:
            # 标准PID力控制
            self.control_output = self.pid_controller.compute(
                self.target_force, self.filtered_force, dt
            )
            
        elif self.control_mode == ControlMode.IMPEDANCE_CONTROL:
            # 阻抗控制（假设目标位置为零）
            target_position = 0.0
            self.control_output = self.impedance_controller.compute(
                target_position, self.filtered_force, dt
            )
            
        elif self.control_mode == ControlMode.ADAPTIVE_CONTROL:
            # 自适应控制
            self.control_output = self.adaptive_controller.compute(
                self.target_force, self.filtered_force, dt
            )
            
        elif self.control_mode == ControlMode.LEARNING_CONTROL and self.use_learning_control:
            # 学习控制（需要外部学习控制器）
            if self.learning_controller is not None:
                self.control_output = self.learning_controller.compute(
                    self.target_force, self.filtered_force, dt
                )
            else:
                logger.warning("学习控制器未初始化，使用PID控制")
                self.control_output = self.pid_controller.compute(
                    self.target_force, self.filtered_force, dt
                )
        else:
            # 默认使用PID控制
            self.control_output = self.pid_controller.compute(
                self.target_force, self.filtered_force, dt
            )
            
        # 计算性能统计
        computation_time = time.time() - start_time
        self.control_cycles += 1
        self.avg_computation_time = (
            self.avg_computation_time * (self.control_cycles - 1) + computation_time
        ) / self.control_cycles
        
        return self.control_output
    
    def get_force_statistics(self) -> Dict:
        """获取力统计信息"""
        if len(self.force_buffer) == 0:
            return {}
            
        forces = list(self.force_buffer)
        filtered = list(self.filtered_forces)
        
        return {
            'current_force': self.current_force,
            'filtered_force': self.filtered_force,
            'target_force': self.target_force,
            'force_mean': np.mean(forces) if forces else 0.0,
            'force_std': np.std(forces) if len(forces) > 1 else 0.0,
            'force_min': np.min(forces) if forces else 0.0,
            'force_max': np.max(forces) if forces else 0.0,
            'force_error': self.target_force - self.filtered_force,
            'buffer_size': len(self.force_buffer),
            'safety_violations': self.safety_violations
        }
    
    def get_control_statistics(self) -> Dict:
        """获取控制统计信息"""
        total_time = time.time() - self.control_start_time
        
        return {
            'control_mode': self.control_mode.value,
            'control_cycles': self.control_cycles,
            'avg_computation_time_ms': self.avg_computation_time * 1000,
            'control_frequency_hz': self.control_cycles / total_time if total_time > 0 else 0.0,
            'total_time_s': total_time,
            'current_output': self.control_output
        }
    
    def save_calibration(self, filepath: str):
        """保存校准数据"""
        if self.is_calibrated:
            self.calibrator.save_calibration(filepath)
            logger.info(f"校准数据保存到: {filepath}")
            
    def load_calibration(self, filepath: str):
        """加载校准数据"""
        if self.calibrator.load_calibration(filepath):
            self.is_calibrated = True
            logger.info(f"从 {filepath} 加载校准数据成功")
        else:
            logger.warning(f"从 {filepath} 加载校准数据失败")
            
    def set_learning_controller(self, controller):
        """设置学习控制器"""
        self.learning_controller = controller
        self.use_learning_control = controller is not None
        if self.use_learning_control:
            logger.info("学习控制器已设置")


class LearningForceController(nn.Module):
    """基于深度学习的力控制器"""
    
    def __init__(self, 
                 input_dim: int = 3,  # [目标力, 当前力, 力误差]
                 hidden_dim: int = 64,
                 output_dim: int = 1):  # 控制输出
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
        
        # 历史数据用于序列学习
        self.history_buffer = deque(maxlen=10)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)
    
    def compute(self, target_force: float, actual_force: float, 
                dt: float = None) -> float:
        """
        计算控制输出
        
        Args:
            target_force: 目标力
            actual_force: 实际力
            dt: 时间步长
            
        Returns:
            控制输出
        """
        # 准备输入状态
        error = target_force - actual_force
        
        # 创建状态向量
        state = torch.tensor([target_force, actual_force, error], dtype=torch.float32)
        
        # 如果有历史数据，可以拼接
        if len(self.history_buffer) > 0:
            history = torch.cat(list(self.history_buffer))
            state = torch.cat([state, history])
            
        # 确保状态维度匹配网络输入
        if state.shape[0] < self.network[0].in_features:
            # 补零
            padding = torch.zeros(self.network[0].in_features - state.shape[0])
            state = torch.cat([state, padding])
        elif state.shape[0] > self.network[0].in_features:
            # 截断
            state = state[:self.network[0].in_features]
            
        # 网络推理
        with torch.no_grad():
            output = self.forward(state.unsqueeze(0)).squeeze().item()
            
        # 更新历史缓冲区
        self.history_buffer.append(torch.tensor([actual_force, error, output]))
        
        return output
    
    def train_step(self, states: torch.Tensor, targets: torch.Tensor, 
                   optimizer, criterion):
        """训练单步"""
        self.train()
        optimizer.zero_grad()
        predictions = self(states)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


# 工厂函数
def create_force_controller(mode: Union[str, ControlMode] = "force", **kwargs) -> ForceController:
    """
    创建力控制器工厂函数
    
    Args:
        mode: 控制模式字符串或枚举
        **kwargs: 其他参数
        
    Returns:
        力控制器实例
    """
    if isinstance(mode, str):
        mode = ControlMode(mode.lower())
        
    sensor_config = kwargs.get('sensor_config', PaxiniConfig())
    servo_config = kwargs.get('servo_config', ServoConfig())
    
    controller = ForceController(sensor_config, servo_config, mode)
    
    # 设置参数
    if 'params' in kwargs:
        controller.params = kwargs['params']
        
    # 校准
    if kwargs.get('auto_calibrate', False):
        controller.calibrate_sensor()
        
    return controller


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试控制器
    controller = create_force_controller("adaptive")
    
    # 模拟力数据
    time_steps = 100
    target_forces = [2.0] * time_steps
    actual_forces = []
    control_outputs = []
    
    # 模拟控制循环
    for i in range(time_steps):
        # 模拟实际力（带有噪声）
        actual_force = 2.0 + np.random.normal(0, 0.2)
        if i > 30 and i < 60:
            actual_force += 1.0  # 模拟干扰
            
        controller.update_force_measurement(actual_force)
        output = controller.compute_control(dt=0.01)
        
        actual_forces.append(actual_force)
        control_outputs.append(output)
        
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(target_forces, label='目标力', linestyle='--')
    ax1.plot(actual_forces, label='实际力')
    ax1.set_ylabel('力 (N)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(control_outputs, label='控制输出', color='red')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('控制输出')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    stats = controller.get_force_statistics()
    print("力统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")