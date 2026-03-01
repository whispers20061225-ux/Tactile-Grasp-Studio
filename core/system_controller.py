"""
系统协调控制器 - 协调所有模块的交互和状态管理
"""

import time
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from config.demo_config import DemoConfig
from config.learm_arm_config import LearmArmConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)

class SystemState(Enum):
    """系统状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    DEMO_RUNNING = "demo_running"
    TEACHING = "teaching"
    LEARNING = "learning"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

@dataclass
class TaskStatus:
    """任务状态数据结构"""
    task_id: str
    status: str
    progress: float
    start_time: float
    current_step: str
    error_message: Optional[str] = None

class SystemController:
    """
    系统协调控制器
    负责协调硬件、感知、规划和控制模块的交互
    """
    
    def __init__(self, config: DemoConfig):
        """
        初始化系统控制器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.state = SystemState.INITIALIZING
        
        # 模块引用（将在运行时注入）
        self.hardware_interface = None
        self.demo_manager = None
        self.perception_module = None
        self.gripper_controller = None
        self.arm_controller = None
        self.safety_monitor = None
        self.motion_planner = None
        
        # 系统状态管理
        self.task_queue = queue.Queue()
        self.current_task = None
        self.task_history: List[TaskStatus] = []
        
        # 线程和同步
        self.main_thread = None
        self.is_running = False
        self.control_loop_interval = 0.01  # 100Hz控制循环
        
        # 数据缓存
        self.sensor_data_cache = {}
        self.control_command_cache = {}
        self.system_status_cache = {}
        
        # 事件回调
        self.event_callbacks = {
            'state_changed': [],
            'task_started': [],
            'task_completed': [],
            'error_occurred': []
        }
        
        logger.info("SystemController initialized")
    
    def register_modules(self, **modules):
        """
        注册系统模块
        
        Args:
            modules: 模块字典 {模块名: 模块实例}
        """
        for name, module in modules.items():
            if hasattr(self, name):
                setattr(self, name, module)
                logger.info(f"Registered module: {name}")
            else:
                logger.warning(f"Unknown module: {name}")
    
    def start(self):
        """启动系统控制器"""
        if self.state != SystemState.INITIALIZING:
            logger.error(f"Cannot start from state: {self.state}")
            return False
        
        self.is_running = True
        self.main_thread = threading.Thread(target=self._main_control_loop, daemon=True)
        self.main_thread.start()
        
        self._change_state(SystemState.READY)
        logger.info("SystemController started")
        return True
    
    def stop(self):
        """停止系统控制器"""
        self.is_running = False
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
        
        self._change_state(SystemState.INITIALIZING)
        logger.info("SystemController stopped")
    
    def execute_task(self, task_type: str, task_params: Dict[str, Any]) -> str:
        """
        执行任务
        
        Args:
            task_type: 任务类型 ('pick_and_place', 'force_control', 'learning', etc.)
            task_params: 任务参数
            
        Returns:
            task_id: 任务ID
        """
        task_id = f"task_{int(time.time())}_{task_type}"
        task = {
            'task_id': task_id,
            'type': task_type,
            'params': task_params,
            'status': 'pending',
            'created_at': time.time()
        }
        
        self.task_queue.put(task)
        
        # 记录任务状态
        task_status = TaskStatus(
            task_id=task_id,
            status='pending',
            progress=0.0,
            start_time=time.time(),
            current_step='queued'
        )
        self.task_history.append(task_status)
        
        self._trigger_event('task_started', task)
        logger.info(f"Task {task_id} ({task_type}) queued")
        
        return task_id
    
    def _main_control_loop(self):
        """主控制循环"""
        while self.is_running:
            try:
                # 1. 检查安全状态
                if self.safety_monitor and hasattr(self.safety_monitor, "is_emergency") and self.safety_monitor.is_emergency():
                    self._handle_emergency()
                    continue
                
                # 2. 处理任务队列
                if not self.task_queue.empty():
                    self._process_next_task()
                
                # 3. 更新系统状态
                self._update_system_status()
                
                # 4. 执行当前任务的控制循环
                if self.current_task:
                    self._execute_task_control_loop()
                
                # 5. 发布状态更新
                self._publish_status_update()
                
                time.sleep(self.control_loop_interval)
                
            except Exception as e:
                logger.error(f"Control loop error: {str(e)}")
                self._handle_error(e)
    
    def _process_next_task(self):
        """处理下一个任务"""
        try:
            task = self.task_queue.get_nowait()
            self.current_task = task
            
            task_id = task['task_id']
            task_type = task['type']
            
            # 更新任务状态
            self._update_task_status(task_id, 'running', 0.1, 'starting')
            
            logger.info(f"Starting task {task_id}: {task_type}")
            
            # 根据任务类型执行
            if task_type == 'pick_and_place':
                self._execute_pick_and_place(task)
            elif task_type == 'force_control_demo':
                self._execute_force_control_demo(task)
            elif task_type == 'learning_experiment':
                self._execute_learning_experiment(task)
            elif task_type == 'calibration':
                self._execute_calibration(task)
            else:
                logger.error(f"Unknown task type: {task_type}")
                self._update_task_status(task_id, 'failed', 0.0, 'unknown_task')
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Task processing error: {str(e)}")
    
    def _execute_pick_and_place(self, task):
        """执行抓取放置任务"""
        task_id = task['task_id']
        params = task['params']
        perception_result = None
        grasp_plan = None
        
        try:
            # 1. 感知物体
            self._update_task_status(task_id, 'running', 0.2, 'perceiving')
            if self.perception_module and hasattr(self.perception_module, "detect_objects"):
                perception_result = self.perception_module.detect_objects()
                
            # 2. 规划抓取
            self._update_task_status(task_id, 'running', 0.4, 'planning_grasp')
            if self.motion_planner and hasattr(self.motion_planner, "plan_grasp"):
                grasp_plan = self.motion_planner.plan_grasp(perception_result)
                
            # 3. 移动机械臂
            self._update_task_status(task_id, 'running', 0.6, 'moving_arm')
            if self.arm_controller and grasp_plan and hasattr(self.arm_controller, "move_to_pose"):
                self.arm_controller.move_to_pose(getattr(grasp_plan, "approach_pose", None))
                
            # 4. 执行抓取
            self._update_task_status(task_id, 'running', 0.8, 'grasping')
            if self.gripper_controller and grasp_plan and hasattr(self.gripper_controller, "execute_grasp"):
                self.gripper_controller.execute_grasp(getattr(grasp_plan, "grasp_parameters", None))
                
            # 5. 移动到目标位置
            if self.arm_controller and grasp_plan and hasattr(self.arm_controller, "move_to_pose"):
                self.arm_controller.move_to_pose(getattr(grasp_plan, "placement_pose", None))
                
            # 6. 释放物体
            if self.gripper_controller and hasattr(self.gripper_controller, "release"):
                self.gripper_controller.release()
            
            self._update_task_status(task_id, 'completed', 1.0, 'finished')
            self.current_task = None
            
        except Exception as e:
            logger.error(f"Pick and place error: {str(e)}")
            self._update_task_status(task_id, 'failed', 0.0, f'error: {str(e)}')
    
    def _execute_force_control_demo(self, task):
        """执行力控演示"""
        # 实现力控演示逻辑
        pass
    
    def _execute_learning_experiment(self, task):
        """执行学习实验"""
        # 实现学习实验逻辑
        pass
    
    def _execute_calibration(self, task):
        """执行校准任务"""
        # 实现校准逻辑
        pass
    
    def _execute_task_control_loop(self):
        """执行任务控制循环"""
        # 根据当前任务执行控制逻辑
        pass
    
    def _update_system_status(self):
        """更新系统状态"""
        # 收集各模块状态
        self.system_status_cache = {
            'timestamp': time.time(),
            'state': self.state.value,
            'hardware_connected': bool(
                self.hardware_interface
                and hasattr(self.hardware_interface, "is_connected")
                and self.hardware_interface.is_connected()
            ),
            'sensor_data_valid': bool(
                self.perception_module
                and hasattr(self.perception_module, "is_data_valid")
                and self.perception_module.is_data_valid()
            ),
            'safety_status': (
                self.safety_monitor.get_status_summary()
                if self.safety_monitor and hasattr(self.safety_monitor, "get_status_summary")
                else 'unknown'
            ),
            'current_task': self.current_task['task_id'] if self.current_task else None,
            'queue_size': self.task_queue.qsize()
        }
    
    def _publish_status_update(self):
        """发布状态更新"""
        # 可以通过ROS、MQTT或回调函数发布状态
        pass
    
    def _change_state(self, new_state: SystemState):
        """改变系统状态"""
        old_state = self.state
        self.state = new_state
        logger.info(f"System state changed: {old_state.value} -> {new_state.value}")
        self._trigger_event('state_changed', {'old': old_state, 'new': new_state})
    
    def _update_task_status(self, task_id: str, status: str, 
                           progress: float, current_step: str, 
                           error_message: Optional[str] = None):
        """更新任务状态"""
        for task_status in self.task_history:
            if task_status.task_id == task_id:
                task_status.status = status
                task_status.progress = progress
                task_status.current_step = current_step
                task_status.error_message = error_message
                
                if status in ['completed', 'failed', 'cancelled']:
                    self._trigger_event('task_completed', task_status)
                
                break
    
    def _handle_emergency(self):
        """处理紧急情况"""
        logger.warning("Emergency condition detected!")
        self._change_state(SystemState.EMERGENCY_STOP)
        
        # 停止所有运动
        if self.arm_controller and hasattr(self.arm_controller, "emergency_stop"):
            self.arm_controller.emergency_stop()
        if self.gripper_controller and hasattr(self.gripper_controller, "emergency_stop"):
            self.gripper_controller.emergency_stop()
        
        # 清空任务队列
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                self._update_task_status(
                    task['task_id'], 'cancelled', 0.0, 
                    'emergency_stop', 'Emergency stop triggered'
                )
            except queue.Empty:
                break
    
    def _handle_error(self, error: Exception):
        """处理错误"""
        logger.error(f"System error: {str(error)}")
        self._change_state(SystemState.ERROR)
        self._trigger_event('error_occurred', {'error': str(error), 'type': type(error).__name__})
    
    def _trigger_event(self, event_name: str, data: Any):
        """触发事件回调"""
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event callback error: {str(e)}")
    
    def register_event_callback(self, event_name: str, callback):
        """注册事件回调"""
        if event_name in self.event_callbacks:
            self.event_callbacks[event_name].append(callback)
        else:
            logger.warning(f"Unknown event: {event_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_state': self.state.value,
            'current_task': self.current_task,
            'task_queue_size': self.task_queue.qsize(),
            'task_history_count': len(self.task_history),
            'system_status': self.system_status_cache
        }
    
    def reset(self):
        """重置系统"""
        logger.info("Resetting system...")
        
        # 清空任务队列
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        self.current_task = None
        self._change_state(SystemState.READY)
        
        logger.info("System reset completed")
