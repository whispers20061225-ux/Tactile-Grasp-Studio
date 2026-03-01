"""
关节控制器：对 LearmInterface 的轻量封装。

功能目标：
- 统一关节角度控制入口
- 角度范围限制与状态缓存
- 对上层提供简洁、稳定的 API
"""

import logging
import threading
from typing import Dict, List, Optional

from .learm_interface import LearmInterface

logger = logging.getLogger(__name__)


class JointController:
    """关节控制器，面向角度控制场景。"""

    def __init__(self, arm_interface: LearmInterface, num_joints: int = 6):
        # 低层通信接口
        self.arm = arm_interface

        # 关节数量
        self.num_joints = int(num_joints)

        # 是否使用角度控制（默认 True）
        self.use_degrees = True

        # 关节软限制禁用列表（1-based）
        self._limit_disabled_joints = set(getattr(self.arm, "_limit_disabled_joints", set()))

        # 关节角度限制（可按实际机械臂修正）
        self.joint_limits = {
            "angle_min": [-125.0] * self.num_joints,
            "angle_max": [125.0] * self.num_joints,
        }
        # 优先使用配置中的关节限制，避免默认范围过窄导致运动被截断
        self._load_joint_limits_from_config()

        # 默认速度系数（0~1）
        self.default_speed = 0.5

        # 关节状态缓存
        self.current_positions = [0.0] * self.num_joints
        self.target_positions = [0.0] * self.num_joints

        # 线程安全锁
        self._state_lock = threading.RLock()

        logger.info("关节控制器初始化完成")

    def _load_joint_limits_from_config(self) -> None:
        """从配置中加载关节限制（若存在则覆盖默认值）。"""
        config = getattr(self.arm, "config", None)
        if not config:
            return

        limits = None
        # DemoConfig 风格：config.learm_arm.PHYSICAL["joint_limits"]
        if hasattr(config, "learm_arm") and config.learm_arm is not None:
            profile = config.learm_arm
            if isinstance(profile, dict):
                limits = profile.get("PHYSICAL", {}).get("joint_limits")
            else:
                limits = getattr(profile, "PHYSICAL", {}).get("joint_limits")
        # dict 风格兼容：config["PHYSICAL"]["joint_limits"]
        elif isinstance(config, dict):
            limits = config.get("PHYSICAL", {}).get("joint_limits")

        if not isinstance(limits, dict):
            return

        mins = list(self.joint_limits["angle_min"])
        maxs = list(self.joint_limits["angle_max"])

        for i in range(self.num_joints):
            key = f"joint{i + 1}"
            if key not in limits:
                continue
            value = limits.get(key)
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            mins[i] = float(value[0])
            maxs[i] = float(value[1])

        self.joint_limits["angle_min"] = mins
        self.joint_limits["angle_max"] = maxs

    def move_to_position(
        self,
        positions: List[float],
        speed: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        兼容旧接口：positions 默认视为角度（度）。

        Args:
            positions: 关节目标（默认角度）
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        if self.use_degrees:
            return self.move_to_angles(positions, speed=speed, wait=wait)

        # 如果禁用角度控制，则尝试走“原始位置值”路径
        return self._move_raw_positions(positions, speed=speed, wait=wait)

    def move_to_angles(
        self,
        angles: List[float],
        speed: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        移动到指定关节角度（度）。

        Args:
            angles: 关节角度列表
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        if len(angles) != self.num_joints:
            logger.error("角度列表长度错误: 期望 %d, 实际 %d", self.num_joints, len(angles))
            return False

        if speed is None:
            speed = self.default_speed
        self.arm.set_speed(speed)

        # 限制角度范围
        limited = self._limit_angles(angles)

        # 构造舵机指令
        position_dict = {i + 1: limited[i] for i in range(self.num_joints)}
        success = self.arm.move_joints(position_dict, wait=wait)

        if success:
            with self._state_lock:
                self.target_positions = list(limited)
                if wait:
                    self.current_positions = list(limited)
            logger.info("关节目标角度更新: %s", limited)

        return bool(success)

    def move_single_joint(
        self,
        joint_id: int,
        position: float,
        speed: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        移动单个关节。

        Args:
            joint_id: 关节索引（0~num_joints-1）
            position: 目标角度（度）
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        if joint_id < 0 or joint_id >= self.num_joints:
            logger.error("关节索引越界: %d", joint_id)
            return False

        if speed is None:
            speed = self.default_speed
        self.arm.set_speed(speed)

        # 限制单关节角度范围
        limited = self._limit_single_angle(joint_id, position)
        success = self.arm.move_joint(joint_id + 1, limited, wait=wait)

        if success:
            with self._state_lock:
                self.target_positions[joint_id] = limited
                if wait:
                    self.current_positions[joint_id] = limited
            logger.debug("关节 %d 目标角度: %.2f", joint_id, limited)

        return bool(success)

    def get_current_positions(self, update: bool = True) -> List[float]:
        """
        获取当前关节位置（默认角度）。

        Args:
            update: 是否从硬件刷新
        """
        if update:
            self.update_state()
        with self._state_lock:
            return list(self.current_positions)

    def get_current_angles(self, update: bool = True) -> List[float]:
        """
        获取当前关节角度。

        Args:
            update: 是否从硬件刷新
        """
        return self.get_current_positions(update=update)

    def update_state(self):
        """从硬件读取关节状态并更新缓存。"""
        if not self.arm.is_connected():
            return

        positions = self.arm.get_all_joint_positions(degrees=True)
        if not positions:
            return

        with self._state_lock:
            for i in range(self.num_joints):
                servo_id = i + 1
                if servo_id in positions:
                    self.current_positions[i] = float(positions[servo_id])

    def go_home(self, speed: Optional[float] = None, wait: bool = True) -> bool:
        """
        关节回零（默认全部 0 度）。

        Args:
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        home_angles = [0.0] * self.num_joints
        return self.move_to_angles(home_angles, speed=speed, wait=wait)

    def stop(self):
        """紧急停止：转交给底层接口处理。"""
        self.arm.emergency_stop()

    def set_control_mode(self, use_degrees: bool):
        """
        设置控制模式。

        Args:
            use_degrees: True 使用角度控制；False 使用原始位置值
        """
        self.use_degrees = bool(use_degrees)
        logger.info("控制模式设置为: %s", "角度" if self.use_degrees else "原始位置")

    def _limit_angles(self, angles: List[float]) -> List[float]:
        """限制所有关节角度范围。"""
        limited: List[float] = []
        for i, angle in enumerate(angles):
            limited.append(self._limit_single_angle(i, angle))
        return limited

    def _limit_single_angle(self, joint_id: int, angle: float) -> float:
        """限制单个关节角度范围。"""
        # 对指定关节禁用软限制（允许负角度/超范围）
        if (joint_id + 1) in self._limit_disabled_joints:
            return float(angle)
        min_angle = self.joint_limits["angle_min"][joint_id]
        max_angle = self.joint_limits["angle_max"][joint_id]
        return max(min_angle, min(max_angle, float(angle)))

    def _move_raw_positions(
        self,
        positions: List[float],
        speed: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        使用“原始位置值”移动关节（需要底层支持）。

        Args:
            positions: 原始位置列表
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        if not hasattr(self.arm, "move_joints_raw"):
            logger.error("底层接口未提供 move_joints_raw，无法使用原始位置控制")
            return False

        if len(positions) != self.num_joints:
            logger.error("位置列表长度错误: 期望 %d, 实际 %d", self.num_joints, len(positions))
            return False

        if speed is None:
            speed = self.default_speed
        self.arm.set_speed(speed)

        position_dict = {i + 1: positions[i] for i in range(self.num_joints)}
        return bool(self.arm.move_joints_raw(position_dict, wait=wait))
