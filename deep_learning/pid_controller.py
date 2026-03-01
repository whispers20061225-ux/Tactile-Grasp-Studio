import torch
import torch.nn as nn
import numpy as np

class NeuralPIDController(nn.Module):
    """神经网络PID控制器"""
    
    def __init__(self, input_dim=9, hidden_dim=32):
        super(NeuralPIDController, self).__init__()
        
        # 神经网络生成PID参数
        self.pid_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # Kp, Ki, Kd (位置和力各3个)
        )
        
        # PID控制变量
        self.position_error_integral = 0
        self.position_error_previous = 0
        self.force_error_integral = 0
        self.force_error_previous = 0
        
        # 积分限制
        self.integral_limit = 10.0
        
    def forward(self, tactile_input, setpoint, current_position, current_force):
        """
        前向传播计算控制输出
        
        Args:
            tactile_input: 触觉传感器数据 (batch, channels, height, width)
            setpoint: 目标位置和力
            current_position: 当前位置
            current_force: 当前力
        """
        # 提取触觉特征
        tactile_flat = tactile_input.view(tactile_input.size(0), -1)
        
        # 生成PID参数
        pid_params = self.pid_generator(tactile_flat)
        
        # 分离位置和力PID参数
        pos_kp = pid_params[:, 0].unsqueeze(1)
        pos_ki = pid_params[:, 1].unsqueeze(1)
        pos_kd = pid_params[:, 2].unsqueeze(1)
        force_kp = pid_params[:, 3].unsqueeze(1)
        force_ki = pid_params[:, 4].unsqueeze(1)
        force_kd = pid_params[:, 5].unsqueeze(1)
        
        # 计算误差
        pos_error = setpoint[:, 0:1] - current_position
        force_error = setpoint[:, 1:2] - current_force
        
        # 更新积分项（带限制）
        self.position_error_integral = torch.clamp(
            self.position_error_integral + pos_error,
            -self.integral_limit,
            self.integral_limit
        )
        
        self.force_error_integral = torch.clamp(
            self.force_error_integral + force_error,
            -self.integral_limit,
            self.integral_limit
        )
        
        # 计算微分项
        pos_error_derivative = pos_error - self.position_error_previous
        force_error_derivative = force_error - self.force_error_previous
        
        # 更新历史误差
        self.position_error_previous = pos_error
        self.force_error_previous = force_error
        
        # PID控制输出
        pos_output = (
            pos_kp * pos_error +
            pos_ki * self.position_error_integral +
            pos_kd * pos_error_derivative
        )
        
        force_output = (
            force_kp * force_error +
            force_ki * self.force_error_integral +
            force_kd * force_error_derivative
        )
        
        # 组合输出
        control_output = torch.cat([pos_output, force_output], dim=1)
        
        return {
            'control_output': control_output,
            'pid_parameters': pid_params,
            'errors': torch.cat([pos_error, force_error], dim=1)
        }
    
    def reset_integrators(self):
        """重置积分器"""
        self.position_error_integral = 0
        self.position_error_previous = 0
        self.force_error_integral = 0
        self.force_error_previous = 0


class AdaptivePID:
    """自适应PID控制器 - 结合传统PID和神经网络"""
    
    def __init__(self, base_kp=1.0, base_ki=0.1, base_kd=0.05):
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        
        # 误差历史
        self.error_history = []
        self.max_history = 100
        
        # 自适应参数
        self.learning_rate = 0.01
        
    def update_parameters(self, tactile_features, performance_metric):
        """
        根据触觉特征和性能指标更新PID参数
        
        Args:
            tactile_features: 触觉特征向量
            performance_metric: 性能指标（越小越好）
        """
        # 基于特征调整参数
        # 这里可以添加更复杂的自适应逻辑
        
        # 示例：根据压力分布调整
        pressure_mean = np.mean(tactile_features)
        pressure_std = np.std(tactile_features)
        
        # 调整KP
        if pressure_std > 0.2:  # 压力分布不均匀
            self.base_kp *= 1.1  # 增加响应速度
        else:
            self.base_kp *= 0.9  # 减少振荡
        
        # 调整KI
        if abs(pressure_mean) < 0.1:  # 接近目标
            self.base_ki *= 0.8  # 减少积分作用
        
        # 限制参数范围
        self.base_kp = np.clip(self.base_kp, 0.1, 5.0)
        self.base_ki = np.clip(self.base_ki, 0.01, 1.0)
        self.base_kd = np.clip(self.base_kd, 0.0, 2.0)
        
        return self.base_kp, self.base_ki, self.base_kd
    
    def compute_control(self, error, dt=0.01):
        """计算PID控制输出"""
        if len(self.error_history) == 0:
            self.error_history.append(error)
            integral = 0
            derivative = 0
        else:
            self.error_history.append(error)
            if len(self.error_history) > self.max_history:
                self.error_history.pop(0)
            
            # 计算积分
            integral = np.trapz(self.error_history, dx=dt)
            
            # 计算微分
            if len(self.error_history) >= 2:
                derivative = (self.error_history[-1] - self.error_history[-2]) / dt
            else:
                derivative = 0
        
        # PID公式
        output = (
            self.base_kp * error +
            self.base_ki * integral +
            self.base_kd * derivative
        )
        
        return output, integral, derivative