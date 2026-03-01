"""
安全监控模块 - 监控系统安全状态并触发保护机制
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

class SafetyLevel(Enum):
    """安全等级枚举"""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"

@dataclass
class SafetyRule:
    """安全规则数据结构"""
    name: str
    condition: str  # 条件表达式
    level: SafetyLevel
    action: str  # 触发动作
    enabled: bool = True
    last_triggered: Optional[float] = None

@dataclass
class SafetyStatus:
    """安全状态数据结构"""
    level: SafetyLevel
    active_warnings: List[str]
    active_dangers: List[str]
    emergency_condition: bool
    last_update: float

class SafetyMonitor:
    """
    安全监控器
    监控系统安全状态，包括硬件状态、传感器数据、运动限制等
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化安全监控器
        
        Args:
            config: 安全监控配置
        """
        self.config = config
        self.safety_rules: List[SafetyRule] = []
        self.current_status = SafetyStatus(
            level=SafetyLevel.NORMAL,
            active_warnings=[],
            active_dangers=[],
            emergency_condition=False,
            last_update=time.time()
        )
        
        # 监控数据缓存
        self.monitor_data = {
            'joint_positions': None,
            'joint_velocities': None,
            'joint_torques': None,
            'gripper_force': None,
            'tactile_data': None,
            'temperature': None,
            'current': None,
            'voltage': None
        }
        
        # 阈值配置
        self.thresholds = {
            'joint_position_limits': config.get('joint_position_limits', {}),
            'joint_velocity_limits': config.get('joint_velocity_limits', {}),
            'joint_torque_limits': config.get('joint_torque_limits', {}),
            'gripper_force_limit': config.get('gripper_force_limit', 20.0),  # N
            'temperature_limit': config.get('temperature_limit', 60.0),  # °C
            'current_limit': config.get('current_limit', 2.0),  # A
            'voltage_limit': config.get('voltage_limit', {'min': 10.0, 'max': 14.0})  # V
        }
        
        # 状态历史
        self.status_history: List[SafetyStatus] = []
        self.max_history_size = 1000
        
        # 事件回调
        self.safety_callbacks = {
            'warning': [],
            'danger': [],
            'emergency': [],
            'status_changed': []
        }
        
        # 线程控制
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitoring_interval = 0.01  # 100Hz监控
        
        # 初始化安全规则
        self._initialize_safety_rules()
        
        logger.info("SafetyMonitor initialized")
    
    def _initialize_safety_rules(self):
        """初始化安全规则"""
        # 1. 关节位置限制
        self.safety_rules.append(SafetyRule(
            name="joint_position_limit",
            condition="any(abs(joint_positions - home_positions) > position_limits)",
            level=SafetyLevel.EMERGENCY,
            action="emergency_stop"
        ))
        
        # 2. 关节速度限制
        self.safety_rules.append(SafetyRule(
            name="joint_velocity_limit",
            condition="any(abs(joint_velocities) > velocity_limits)",
            level=SafetyLevel.DANGER,
            action="reduce_velocity"
        ))
        
        # 3. 关节扭矩限制
        self.safety_rules.append(SafetyRule(
            name="joint_torque_limit",
            condition="any(abs(joint_torques) > torque_limits)",
            level=SafetyLevel.WARNING,
            action="reduce_torque"
        ))
        
        # 4. 夹爪力限制
        self.safety_rules.append(SafetyRule(
            name="gripper_force_limit",
            condition="gripper_force > force_limit",
            level=SafetyLevel.WARNING,
            action="release_gripper"
        ))
        
        # 5. 温度限制
        self.safety_rules.append(SafetyRule(
            name="temperature_limit",
            condition="temperature > temperature_limit",
            level=SafetyLevel.DANGER,
            action="cool_down"
        ))
        
        # 6. 电流限制
        self.safety_rules.append(SafetyRule(
            name="current_limit",
            condition="current > current_limit",
            level=SafetyLevel.EMERGENCY,
            action="power_off"
        ))
        
        # 7. 电压限制
        self.safety_rules.append(SafetyRule(
            name="voltage_limit",
            condition="voltage < voltage_min or voltage > voltage_max",
            level=SafetyLevel.WARNING,
            action="check_power_supply"
        ))
        
        logger.info(f"Initialized {len(self.safety_rules)} safety rules")
    
    def start_monitoring(self):
        """开始安全监控"""
        if self.is_monitoring:
            logger.warning("Safety monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """停止安全监控"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 1. 更新监控数据（假设已从硬件接口获取）
                self._update_monitor_data()
                
                # 2. 检查所有安全规则
                self._check_safety_rules()
                
                # 3. 更新安全状态
                self._update_safety_status()
                
                # 4. 触发必要的安全动作
                self._trigger_safety_actions()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {str(e)}")
                time.sleep(0.1)
    
    def _update_monitor_data(self):
        """更新监控数据"""
        # 这里应该从硬件接口获取实际数据
        # 示例：self.monitor_data = self.hardware_interface.get_safety_data()
        pass
    
    def _check_safety_rules(self):
        """检查安全规则"""
        active_warnings = []
        active_dangers = []
        emergency_condition = False
        
        for rule in self.safety_rules:
            if not rule.enabled:
                continue
            
            try:
                # 评估规则条件
                is_triggered = self._evaluate_condition(rule.condition)
                
                if is_triggered:
                    rule.last_triggered = time.time()
                    
                    if rule.level == SafetyLevel.WARNING:
                        active_warnings.append(rule.name)
                        self._trigger_callback('warning', rule)
                        
                    elif rule.level == SafetyLevel.DANGER:
                        active_dangers.append(rule.name)
                        self._trigger_callback('danger', rule)
                        
                    elif rule.level == SafetyLevel.EMERGENCY:
                        emergency_condition = True
                        self._trigger_callback('emergency', rule)
                        
                    logger.warning(f"Safety rule triggered: {rule.name} - {rule.level.value}")
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {str(e)}")
        
        # 更新当前状态
        self.current_status.active_warnings = active_warnings
        self.current_status.active_dangers = active_dangers
        self.current_status.emergency_condition = emergency_condition
        
        # 确定整体安全等级
        if emergency_condition:
            self.current_status.level = SafetyLevel.EMERGENCY
        elif active_dangers:
            self.current_status.level = SafetyLevel.DANGER
        elif active_warnings:
            self.current_status.level = SafetyLevel.WARNING
        else:
            self.current_status.level = SafetyLevel.NORMAL
        
        self.current_status.last_update = time.time()
    
    def _evaluate_condition(self, condition: str) -> bool:
        """
        评估安全条件
        
        Args:
            condition: 条件表达式
            
        Returns:
            bool: 条件是否触发
        """
        # 这里实现条件表达式的解析和评估
        # 简化实现，实际应该使用更强大的表达式引擎
        try:
            # 获取当前监控数据
            data = self.monitor_data.copy()
            data.update(self.thresholds)
            
            # 安全评估（简化版）
            if "joint_position_limit" in condition:
                if data['joint_positions'] is not None:
                    limits = self.thresholds['joint_position_limits']
                    for i, pos in enumerate(data['joint_positions']):
                        if abs(pos) > limits.get(f'joint_{i}', 180.0):
                            return True
            elif "gripper_force_limit" in condition:
                if data['gripper_force'] is not None:
                    return data['gripper_force'] > self.thresholds['gripper_force_limit']
            
            return False
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {str(e)}")
            return False
    
    def _update_safety_status(self):
        """更新安全状态历史"""
        # 复制当前状态到历史
        self.status_history.append(
            SafetyStatus(
                level=self.current_status.level,
                active_warnings=self.current_status.active_warnings.copy(),
                active_dangers=self.current_status.active_dangers.copy(),
                emergency_condition=self.current_status.emergency_condition,
                last_update=self.current_status.last_update
            )
        )
        
        # 限制历史大小
        if len(self.status_history) > self.max_history_size:
            self.status_history = self.status_history[-self.max_history_size:]
        
        # 触发状态改变回调
        if len(self.status_history) > 1:
            old_status = self.status_history[-2]
            new_status = self.status_history[-1]
            
            if old_status.level != new_status.level:
                self._trigger_callback('status_changed', {
                    'old': old_status,
                    'new': new_status
                })
    
    def _trigger_safety_actions(self):
        """触发安全动作"""
        if self.current_status.emergency_condition:
            self._execute_emergency_actions()
        
        elif self.current_status.level == SafetyLevel.DANGER:
            self._execute_danger_actions()
        
        elif self.current_status.level == SafetyLevel.WARNING:
            self._execute_warning_actions()
    
    def _execute_emergency_actions(self):
        """执行紧急动作"""
        logger.critical("Executing emergency actions!")
        
        # 这里应该调用硬件接口的紧急停止
        # 例如：self.hardware_interface.emergency_stop()
        
        # 触发紧急回调
        for callback in self.safety_callbacks['emergency']:
            try:
                callback(self.current_status)
            except Exception as e:
                logger.error(f"Emergency callback error: {str(e)}")
    
    def _execute_danger_actions(self):
        """执行危险动作"""
        for danger in self.current_status.active_dangers:
            # 根据具体危险类型执行相应动作
            if "joint_velocity" in danger:
                logger.warning("Reducing joint velocities")
                # self.hardware_interface.reduce_velocity(0.5)
            
            elif "temperature" in danger:
                logger.warning("Activating cooling system")
                # self.hardware_interface.enable_cooling()
    
    def _execute_warning_actions(self):
        """执行警告动作"""
        for warning in self.current_status.active_warnings:
            if "gripper_force" in warning:
                logger.warning("Reducing gripper force")
                # self.hardware_interface.reduce_gripper_force()
            
            elif "voltage" in warning:
                logger.warning("Checking power supply")
                # self.hardware_interface.check_power_supply()
    
    def _trigger_callback(self, callback_type: str, data: Any):
        """触发安全回调"""
        if callback_type in self.safety_callbacks:
            for callback in self.safety_callbacks[callback_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Safety callback error: {str(e)}")
    
    def register_callback(self, callback_type: str, callback):
        """注册安全回调"""
        if callback_type in self.safety_callbacks:
            self.safety_callbacks[callback_type].append(callback)
        else:
            logger.warning(f"Unknown callback type: {callback_type}")
    
    def is_emergency(self) -> bool:
        """检查是否处于紧急状态"""
        return self.current_status.emergency_condition
    
    def get_status(self) -> SafetyStatus:
        """获取当前安全状态"""
        return self.current_status
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取安全状态摘要"""
        return {
            'safety_level': self.current_status.level.value,
            'active_warnings': len(self.current_status.active_warnings),
            'active_dangers': len(self.current_status.active_dangers),
            'emergency': self.current_status.emergency_condition,
            'last_update': self.current_status.last_update,
            'monitoring_active': self.is_monitoring
        }
    
    def add_safety_rule(self, rule: SafetyRule):
        """添加安全规则"""
        self.safety_rules.append(rule)
        logger.info(f"Added safety rule: {rule.name}")
    
    def enable_rule(self, rule_name: str, enabled: bool = True):
        """启用/禁用安全规则"""
        for rule in self.safety_rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                logger.info(f"{'Enabled' if enabled else 'Disabled'} rule: {rule_name}")
                return
        
        logger.warning(f"Rule not found: {rule_name}")
    
    def clear_history(self):
        """清除状态历史"""
        self.status_history.clear()
        logger.info("Safety history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        if not self.status_history:
            return {}
        
        # 计算各级别持续时间
        level_durations = {level.value: 0.0 for level in SafetyLevel}
        
        for i in range(1, len(self.status_history)):
            prev = self.status_history[i-1]
            curr = self.status_history[i]
            duration = curr.last_update - prev.last_update
            
            if prev.level.value in level_durations:
                level_durations[prev.level.value] += duration
        
        # 统计触发次数
        trigger_counts = {}
        for rule in self.safety_rules:
            if rule.last_triggered:
                trigger_counts[rule.name] = 1
        
        return {
            'level_durations': level_durations,
            'trigger_counts': trigger_counts,
            'total_records': len(self.status_history),
            'monitoring_duration': time.time() - self.status_history[0].last_update if self.status_history else 0.0
        }
