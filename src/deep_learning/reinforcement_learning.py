import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Dict, Optional, Any
import logging
import time
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# 定义经验元组
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class RLConfig:
    """强化学习配置"""
    # 网络参数
    state_dim: int = 100
    action_dim: int = 10
    hidden_dims: List[int] = None
    
    # 训练参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # 软更新参数
    
    # PPO参数
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    
    # 缓冲区参数
    buffer_size: int = 10000
    min_buffer_size: int = 1000
    
    # 探索参数
    exploration_noise: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class RLEnvWrapper:
    """
    强化学习环境包装器
    将机械臂控制环境转换为RL友好的接口
    """
    
    def __init__(self, system_controller, config: RLConfig):
        self.system_controller = system_controller
        self.config = config
        
        # 状态空间和动作空间
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        
        # 探索参数
        self.exploration_noise = config.exploration_noise
        self.exploration_decay = config.exploration_decay
        self.min_exploration = config.min_exploration
        
        # 当前状态
        self.current_state = None
        self.last_action = None
        self.episode_step = 0
        self.max_steps = 100
        self.episode_reward = 0
        
        # 状态统计
        self.state_mean = np.zeros(self.state_dim)
        self.state_std = np.ones(self.state_dim)
        self.state_count = 0
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置机械臂到初始位置
        self.system_controller.reset_arm()
        
        # 获取初始状态
        self.current_state = self._get_raw_state()
        
        # 更新状态统计
        self._update_state_stats(self.current_state)
        
        # 标准化状态
        state_normalized = self._normalize_state(self.current_state)
        
        # 重置统计
        self.last_action = None
        self.episode_step = 0
        self.episode_reward = 0
        
        logger.debug(f"环境重置，初始状态维度: {state_normalized.shape}")
        
        return state_normalized
    
    def step(self, action: np.ndarray, 
             add_noise: bool = True) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作并返回结果
        
        Args:
            action: 动作向量 [-1, 1]
            add_noise: 是否添加探索噪声
        
        Returns:
            next_state, reward, done, info
        """
        self.episode_step += 1
        
        # 添加探索噪声
        if add_noise and self.exploration_noise > self.min_exploration:
            noise = np.random.normal(0, self.exploration_noise, action.shape)
            action = np.clip(action + noise, -1, 1)
            self.exploration_noise *= self.exploration_decay
        
        self.last_action = action
        
        # 执行动作
        try:
            success = self._execute_action(action)
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            success = False
        
        # 获取新状态
        next_state_raw = self._get_raw_state()
        
        # 更新状态统计
        self._update_state_stats(next_state_raw)
        
        # 标准化状态
        next_state_normalized = self._normalize_state(next_state_raw)
        
        # 计算奖励
        reward = self._compute_reward(success, next_state_raw)
        self.episode_reward += reward
        
        # 检查是否结束
        done = self._check_done(success)
        
        # 信息字典
        info = {
            'success': bool(success),
            'step': self.episode_step,
            'total_reward': self.episode_reward,
            'exploration_noise': self.exploration_noise,
            'action_executed': action.tolist()
        }
        
        self.current_state = next_state_normalized
        
        return next_state_normalized, reward, done, info
    
    def _get_raw_state(self) -> np.ndarray:
        """获取原始状态（未标准化）"""
        state_components = []
        
        # 1. 机械臂状态
        arm_state = self.system_controller.get_arm_state()
        if arm_state:
            # 关节位置 (6) - 归一化到 [-1, 1]
            joint_pos = np.array(arm_state.get('joint_positions', np.zeros(6)))
            joint_pos_normalized = np.clip(joint_pos / np.pi, -1, 1)
            state_components.extend(joint_pos_normalized)
            
            # 关节速度 (6) - 归一化
            joint_vel = np.array(arm_state.get('joint_velocities', np.zeros(6)))
            joint_vel_normalized = np.tanh(joint_vel)
            state_components.extend(joint_vel_normalized)
            
            # 末端位姿 (6) - 位置归一化，姿态用四元数
            ee_pose = np.array(arm_state.get('end_effector_pose', np.zeros(7)))
            if len(ee_pose) == 7:  # 位置(3) + 四元数(4)
                # 位置归一化（假设工作空间在1米范围内）
                pos_normalized = ee_pose[:3] / 1.0
                quat = ee_pose[3:]
                state_components.extend(pos_normalized)
                state_components.extend(quat)
            elif len(ee_pose) == 6:  # 位置(3) + 欧拉角(3)
                pos_normalized = ee_pose[:3] / 1.0
                euler = ee_pose[3:]
                state_components.extend(pos_normalized)
                state_components.extend(np.sin(euler))  # 周期函数编码
                state_components.extend(np.cos(euler))
        
        # 2. 触觉特征
        tactile_data = self.system_controller.get_tactile_data()
        if tactile_data is not None:
            tactile_features = self._extract_tactile_features(tactile_data)
            state_components.extend(tactile_features)
        
        # 3. 视觉特征
        visual_features = self.system_controller.get_visual_features()
        if visual_features is not None:
            # 使用PCA降维后的特征
            state_components.extend(visual_features[:20])  # 取前20个主成分
        
        # 4. 目标信息
        target_info = self.system_controller.get_target_info()
        if target_info:
            target_pos = target_info.get('position', np.zeros(3))
            target_pos_normalized = target_pos / 1.0
            state_components.extend(target_pos_normalized)
            
            target_quat = target_info.get('orientation', np.array([1, 0, 0, 0]))
            state_components.extend(target_quat)
        
        # 5. 历史动作（如果可用）
        if self.last_action is not None:
            state_components.extend(self.last_action)
        
        # 转换为numpy数组并确保维度
        state_raw = np.array(state_components, dtype=np.float32)
        
        # 填充或截断到固定维度
        if len(state_raw) < self.state_dim:
            # 补零
            padding = np.zeros(self.state_dim - len(state_raw))
            state_raw = np.concatenate([state_raw, padding])
        elif len(state_raw) > self.state_dim:
            # 截断
            state_raw = state_raw[:self.state_dim]
        
        return state_raw
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """标准化状态"""
        if self.state_count < 2:
            return state
        
        # Z-score标准化
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        return np.clip(normalized, -5, 5)  # 防止异常值
    
    def _update_state_stats(self, state: np.ndarray):
        """更新状态统计（在线计算均值和标准差）"""
        self.state_count += 1
        
        # 在线更新均值
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        
        # 在线更新方差
        if self.state_count > 1:
            delta2 = state - self.state_mean
            self.state_std = np.sqrt(
                (self.state_std**2 * (self.state_count - 1) + delta * delta2) / self.state_count
            )
    
    def _execute_action(self, action: np.ndarray) -> bool:
        """执行动作"""
        try:
            # 解析动作 (假设动作在[-1, 1]范围内)
            # 前3个: 末端位置增量
            # 中间3个: 末端姿态增量 (欧拉角或轴角)
            # 最后4个: 夹爪控制 (开合度、力等)
            
            pos_increment = action[:3] * 0.05  # 最大5cm增量
            rot_increment = action[3:6] * 0.1  # 最大0.1弧度增量
            gripper_action = action[6:]
            
            # 获取当前位姿
            current_pose = self.system_controller.get_current_pose()
            
            # 计算新位姿
            new_position = current_pose[:3] + pos_increment
            new_orientation = self._update_orientation(
                current_pose[3:], rot_increment
            )
            
            # 移动机械臂
            success_move = self.system_controller.move_arm_to(
                new_position, new_orientation
            )
            
            # 控制夹爪
            if len(gripper_action) >= 2:
                gripper_width = (gripper_action[0] + 1) / 2  # [-1,1] -> [0,1]
                grasp_force = (gripper_action[1] + 1) / 2  # [-1,1] -> [0,1]
                success_gripper = self.system_controller.control_gripper(
                    gripper_width, grasp_force
                )
            else:
                success_gripper = True
            
            return success_move and success_gripper
            
        except Exception as e:
            logger.error(f"动作执行错误: {e}")
            return False
    
    def _update_orientation(self, current_quat: np.ndarray, 
                           rot_increment: np.ndarray) -> np.ndarray:
        """更新姿态四元数"""
        # 将旋转增量转换为四元数
        angle = np.linalg.norm(rot_increment)
        if angle < 1e-6:
            return current_quat
        
        axis = rot_increment / angle
        delta_quat = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])
        
        # 四元数乘法
        w1, x1, y1, z1 = current_quat
        w2, x2, y2, z2 = delta_quat
        
        new_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        return new_quat / np.linalg.norm(new_quat)
    
    def _extract_tactile_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """提取触觉特征"""
        if tactile_data.size == 0:
            return np.zeros(10)
        
        features = []
        
        # 基本统计特征
        features.append(np.mean(tactile_data))
        features.append(np.std(tactile_data))
        features.append(np.max(tactile_data))
        features.append(np.min(tactile_data))
        
        # 百分位数
        features.append(np.percentile(tactile_data, 25))
        features.append(np.percentile(tactile_data, 50))
        features.append(np.percentile(tactile_data, 75))
        
        # 偏度和峰度（需要scipy）
        try:
            from scipy.stats import skew, kurtosis
            features.append(skew(tactile_data.flatten()))
            features.append(kurtosis(tactile_data.flatten()))
        except ImportError:
            features.extend([0, 0])
        
        # 接触面积估计
        contact_threshold = 0.1
        contact_area = np.sum(tactile_data > contact_threshold) / tactile_data.size
        features.append(contact_area)
        
        # 压力中心
        if tactile_data.ndim == 2:
            h, w = tactile_data.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            total_pressure = np.sum(tactile_data)
            if total_pressure > 0:
                cog_x = np.sum(x_coords * tactile_data) / total_pressure
                cog_y = np.sum(y_coords * tactile_data) / total_pressure
                features.append(cog_x / w)
                features.append(cog_y / h)
            else:
                features.extend([0.5, 0.5])
        
        return np.array(features, dtype=np.float32)
    
    def _compute_reward(self, success: bool, next_state: np.ndarray) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 基础生存奖励（鼓励探索）
        reward += self.config.gamma ** self.episode_step * 0.01
        
        # 2. 成功抓取奖励
        if success:
            # 检查是否抓取到物体
            object_grasped = self.system_controller.check_object_grasped()
            if object_grasped:
                object_weight = self.system_controller.get_object_weight()
                reward += 10.0 * (1.0 + object_weight)
                logger.info(f"成功抓取物体，奖励: {reward}")
        
        # 3. 目标接近奖励
        target_info = self.system_controller.get_target_info()
        if target_info:
            current_pos = self.system_controller.get_current_position()
            target_pos = target_info['position']
            distance = np.linalg.norm(current_pos - target_pos)
            
            # 距离奖励（越近奖励越高）
            distance_reward = 1.0 / (1.0 + distance)
            reward += distance_reward * 0.1
        
        # 4. 碰撞惩罚
        if self.system_controller.check_collision():
            reward -= 5.0
            logger.warning("检测到碰撞，惩罚: -5.0")
        
        # 5. 关节限制惩罚
        joint_limits_violation = self.system_controller.check_joint_limits()
        if joint_limits_violation > 0:
            reward -= joint_limits_violation * 0.1
        
        # 6. 动作平滑奖励（惩罚大动作）
        if self.last_action is not None:
            action_magnitude = np.linalg.norm(self.last_action)
            reward -= action_magnitude * 0.01
        
        # 7. 触觉奖励（鼓励接触）
        tactile_data = self.system_controller.get_tactile_data()
        if tactile_data is not None:
            tactile_sum = np.sum(tactile_data)
            if 0.1 < tactile_sum < 0.9:  # 适中的接触力
                reward += tactile_sum * 0.1
            elif tactile_sum >= 0.9:  # 过大的接触力
                reward -= 0.5
        
        return float(reward)
    
    def _check_done(self, success: bool) -> bool:
        """检查是否结束"""
        # 超过最大步数
        if self.episode_step >= self.max_steps:
            logger.info(f"达到最大步数: {self.max_steps}")
            return True
        
        # 成功抓取
        if success:
            object_grasped = self.system_controller.check_object_grasped()
            if object_grasped and self.config.get('terminate_on_success', True):
                logger.info("成功抓取物体，结束回合")
                return True
        
        # 严重碰撞
        if self.system_controller.check_collision(threshold=0.5):
            logger.warning("严重碰撞，结束回合")
            return True
        
        # 关节超限
        if self.system_controller.check_joint_limits() > 0.5:
            logger.warning("关节超限，结束回合")
            return True
        
        return False
    
    def get_state_stats(self) -> Dict:
        """获取状态统计信息"""
        return {
            'mean': self.state_mean.tolist(),
            'std': self.state_std.tolist(),
            'count': self.state_count
        }


class ActorNetwork(nn.Module):
    """演员网络（策略网络）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 输出层
        self.feature_extractor = nn.Sequential(*layers)
        
        # 均值输出层
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        
        # 对数标准差输出层
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        mean = torch.tanh(self.mean_layer(features))
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, 
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        # 从正态分布采样
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # 使用重参数化技巧
        
        # 计算对数概率
        log_prob = normal.log_prob(action)
        
        # 应用tanh变换并修正对数概率
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action_tanh, log_prob
    
    def evaluate(self, state: torch.Tensor, 
                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作的对数概率"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        
        # 应用tanh变换的逆
        epsilon = 1e-6
        action_raw = torch.atanh(torch.clamp(action, -1 + epsilon, 1 - epsilon))
        
        log_prob = normal.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob, torch.tanh(mean)


class CriticNetwork(nn.Module):
    """评论家网络（值函数网络）"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.position = 0
    
    def push(self, experience: Experience):
        """存储经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.FloatTensor(np.array([e.action for e in batch]))
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor(np.array([e.done for e in batch])).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str):
        """保存缓冲区"""
        data = {
            'buffer': list(self.buffer),
            'position': self.position,
            'capacity': self.capacity
        }
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)
        logger.info(f"经验缓冲区已保存到 {path}")
    
    def load(self, path: str):
        """加载缓冲区"""
        try:
            with open(path, 'rb') as f:
                import pickle
                data = pickle.load(f)
            
            self.buffer = deque(data['buffer'], maxlen=data['capacity'])
            self.position = data['position']
            logger.info(f"经验缓冲区已从 {path} 加载，大小: {len(self.buffer)}")
        except Exception as e:
            logger.error(f"加载经验缓冲区失败: {e}")


class PPOAgent:
    """PPO算法代理"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, config.hidden_dims).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=config.learning_rate
        )
        
        # 经验缓冲区
        self.buffer = ReplayBuffer(config.buffer_size)
        
        # 训练统计
        self.training_steps = 0
        self.episode_rewards = []
        self.loss_history = []
        
        logger.info(f"PPO代理初始化完成，设备: {self.device}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = mean
            else:
                action, _ = self.actor.sample(state_tensor)
        
        return action.squeeze().cpu().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.push(experience)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) < self.config.min_buffer_size:
            logger.warning(f"缓冲区大小不足: {len(self.buffer)} < {self.config.min_buffer_size}")
            return {}
        
        # 采样批量数据
        batch = self.buffer.sample(self.config.batch_size)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones = batch
        
        # 移动到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算优势函数
        with torch.no_grad():
            current_values = self.critic(states)
            next_values = self.critic(next_states)
            
            # TD目标
            td_target = rewards + self.config.gamma * next_values * (1 - dones)
            
            # 优势函数
            advantages = td_target - current_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算旧策略的对数概率
        old_log_probs, _ = self.actor.evaluate(states, actions)
        
        # PPO更新
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.config.ppo_epochs):
            # 计算新策略的对数概率
            new_log_probs, entropy = self.actor.evaluate(states, actions)
            
            # 概率比
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 裁剪的PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                              1 + self.config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 熵奖励
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            
            # 总actor损失
            total_actor_loss = actor_loss + entropy_loss
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.config.max_grad_norm
            )
            self.actor_optimizer.step()
            
            # Critic损失
            current_values = self.critic(states)
            critic_loss = F.mse_loss(current_values, td_target)
            
            # 更新critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.config.max_grad_norm
            )
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        self.training_steps += 1
        
        loss_info = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'avg_reward': rewards.mean().item(),
            'buffer_size': len(self.buffer),
            'training_steps': self.training_steps
        }
        
        self.loss_history.append(loss_info)
        
        if self.training_steps % 100 == 0:
            logger.info(f"训练步骤 {self.training_steps}: "
                       f"actor_loss={loss_info['actor_loss']:.4f}, "
                       f"critic_loss={loss_info['critic_loss']:.4f}")
        
        return loss_info
    
    def train_episode(self, env: RLEnvWrapper, 
                     max_steps: int = 100) -> Dict[str, Any]:
        """训练一个回合"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < max_steps:
            # 选择动作
            action = self.select_action(state, deterministic=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转移
            self.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            if len(self.buffer) >= self.config.min_buffer_size:
                loss_info = self.update()
                info.update(loss_info)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # 记录回合奖励
        self.episode_rewards.append(episode_reward)
        
        episode_info = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'avg_reward_last_10': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'exploration_noise': env.exploration_noise
        }
        
        logger.info(f"回合结束: 奖励={episode_reward:.2f}, "
                   f"步数={episode_steps}, "
                   f"最近10轮平均奖励={episode_info['avg_reward_last_10']:.2f}")
        
        return episode_info
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, path)
        
        # 保存缓冲区
        buffer_path = path.replace('.pth', '_buffer.pkl')
        self.buffer.save(buffer_path)
        
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.loss_history = checkpoint.get('loss_history', [])
        
        # 加载缓冲区
        buffer_path = path.replace('.pth', '_buffer.pkl')
        try:
            self.buffer.load(buffer_path)
        except:
            logger.warning(f"无法加载缓冲区: {buffer_path}")
        
        logger.info(f"模型已从 {path} 加载，训练步数: {self.training_steps}")
    
    def evaluate(self, env: RLEnvWrapper, num_episodes: int = 10) -> Dict[str, Any]:
        """评估模型性能"""
        self.actor.eval()
        
        eval_rewards = []
        eval_steps = []
        successes = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 100:
                # 使用确定性策略
                action = self.select_action(state, deterministic=True)
                next_state, reward, done, info = env.step(action, add_noise=False)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done and info.get('success', False):
                    successes.append(1)
                elif done:
                    successes.append(0)
            
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)
            
            logger.info(f"评估回合 {episode + 1}/{num_episodes}: "
                       f"奖励={episode_reward:.2f}, 步数={episode_steps}")
        
        self.actor.train()
        
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'success_rate': np.mean(successes) if successes else 0,
            'eval_episodes': num_episodes
        }
        
        logger.info(f"评估完成: 平均奖励={eval_stats['mean_reward']:.2f}, "
                   f"成功率={eval_stats['success_rate']:.3f}")
        
        return eval_stats


class RLTrainer:
    """强化学习训练管理器"""
    
    def __init__(self, env: RLEnvWrapper, config: RLConfig):
        self.env = env
        self.config = config
        
        # 创建PPO代理
        self.agent = PPOAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            config=config
        )
        
        # 训练参数
        self.max_episodes = 1000
        self.eval_interval = 50
        self.save_interval = 100
        
        # 训练统计
        self.training_history = []
        
        logger.info("RL训练器初始化完成")
    
    def train(self, max_episodes: int = None):
        """训练循环"""
        if max_episodes is not None:
            self.max_episodes = max_episodes
        
        logger.info(f"开始训练，最大回合数: {self.max_episodes}")
        
        best_reward = -np.inf
        
        for episode in range(1, self.max_episodes + 1):
            # 训练一个回合
            episode_info = self.agent.train_episode(self.env)
            episode_info['episode'] = episode
            
            # 记录训练历史
            self.training_history.append(episode_info)
            
            # 定期评估
            if episode % self.eval_interval == 0:
                eval_stats = self.agent.evaluate(self.env, num_episodes=5)
                episode_info.update(eval_stats)
                
                # 保存最佳模型
                if eval_stats['mean_reward'] > best_reward:
                    best_reward = eval_stats['mean_reward']
                    self.agent.save(f"best_model_ep{episode}_reward{best_reward:.2f}.pth")
                    logger.info(f"新的最佳模型保存，奖励: {best_reward:.2f}")
            
            # 定期保存
            if episode % self.save_interval == 0:
                self.agent.save(f"checkpoint_ep{episode}.pth")
                self.save_training_history()
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean([h['episode_reward'] for h in self.training_history[-10:]])
                logger.info(f"进度: {episode}/{self.max_episodes}, "
                           f"最近10轮平均奖励: {avg_reward:.2f}")
        
        # 训练完成
        logger.info("训练完成")
        self.agent.save("final_model.pth")
        self.save_training_history()
        
        return self.training_history
    
    def save_training_history(self, path: str = "training_history.json"):
        """保存训练历史"""
        with open(path, 'w') as f:
            # 转换numpy类型为Python原生类型
            history_serializable = []
            for record in self.training_history:
                record_serializable = {}
                for key, value in record.items():
                    if isinstance(value, np.generic):
                        record_serializable[key] = value.item()
                    elif isinstance(value, (np.ndarray, list)):
                        record_serializable[key] = [v.item() if isinstance(v, np.generic) else v 
                                                   for v in value]
                    else:
                        record_serializable[key] = value
                history_serializable.append(record_serializable)
            
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"训练历史已保存到 {path}")
    
    def plot_training_curve(self, save_path: str = "training_curve.png"):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            episodes = [h['episode'] for h in self.training_history]
            rewards = [h['episode_reward'] for h in self.training_history]
            
            plt.figure(figsize=(12, 6))
            
            # 奖励曲线
            plt.subplot(1, 2, 1)
            plt.plot(episodes, rewards, 'b-', alpha=0.5, label='单回合奖励')
            
            # 滑动平均
            window_size = 20
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, 
                        label=f'{window_size}轮滑动平均')
            
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.title('训练奖励曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 成功率曲线（如果有评估数据）
            eval_episodes = [h['episode'] for h in self.training_history 
                            if 'success_rate' in h]
            success_rates = [h['success_rate'] for h in self.training_history 
                           if 'success_rate' in h]
            
            if eval_episodes:
                plt.subplot(1, 2, 2)
                plt.plot(eval_episodes, success_rates, 'g-', marker='o', linewidth=2)
                plt.xlabel('回合')
                plt.ylabel('成功率')
                plt.title('评估成功率')
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练曲线已保存到 {save_path}")
            
        except ImportError:
            logger.warning("无法导入matplotlib，跳过绘图")
        except Exception as e:
            logger.error(f"绘图失败: {e}")