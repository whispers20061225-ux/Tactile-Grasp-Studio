"""
示教模式：记录与回放机械臂的关节轨迹。
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional

from .cartesian_controller import CartesianController, Pose
from .joint_controller import JointController

logger = logging.getLogger(__name__)


class TeachModeState(Enum):
    """示教模式状态枚举。"""

    IDLE = "idle"
    RECORDING = "recording"
    PLAYBACK = "playback"
    PAUSED = "paused"


@dataclass
class TeachPoint:
    """示教点数据结构。"""

    timestamp: float
    joint_positions: List[float]
    cartesian_pose: Optional[Pose]
    label: str = ""
    speed: float = 0.3


class TeachMode:
    """示教模式管理器。"""

    def __init__(
        self,
        joint_controller: JointController,
        cartesian_controller: Optional[CartesianController] = None,
    ):
        # 控制器引用
        self.joint_ctrl = joint_controller
        self.cartesian_ctrl = cartesian_controller

        # 示教点缓存
        self.teach_points: List[TeachPoint] = []
        self.current_point_index = 0

        # 运行状态
        self.state = TeachModeState.IDLE
        self.record_interval = 0.5
        self.playback_speed = 1.0

        # 线程控制
        self._stop_flag = False
        self._record_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None

        logger.info("示教模式初始化完成")

    def start_recording(self, interval: Optional[float] = None) -> bool:
        """
        开始录制示教轨迹。

        Args:
            interval: 记录间隔（秒）
        """
        if self.state != TeachModeState.IDLE:
            logger.warning("示教模式忙，无法开始录制")
            return False

        if interval is not None:
            self.record_interval = float(interval)

        self.teach_points.clear()
        self.current_point_index = 0
        self.state = TeachModeState.RECORDING
        self._stop_flag = False

        # 立即记录首点
        self._record_point("起始点")

        # 启动后台录制线程
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        logger.info("开始录制示教点，间隔 %.2fs", self.record_interval)
        return True

    def stop_recording(self):
        """停止录制示教轨迹。"""
        if self.state != TeachModeState.RECORDING:
            return

        self._stop_flag = True
        self.state = TeachModeState.IDLE

        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)

        # 记录结束点
        self._record_point("结束点")
        logger.info("录制结束，共 %d 个示教点", len(self.teach_points))

    def _record_loop(self):
        """录制线程循环。"""
        while not self._stop_flag:
            time.sleep(self.record_interval)
            if self._stop_flag or self.state != TeachModeState.RECORDING:
                break
            self._record_point()

    def _record_point(self, label: str = ""):
        """记录单个示教点。"""
        joint_positions = self.joint_ctrl.get_current_positions(update=True)

        # 如果提供笛卡尔控制器，则记录位姿；否则置空
        cartesian_pose = None
        if self.cartesian_ctrl is not None:
            cartesian_pose = self.cartesian_ctrl.get_current_pose(update=True)

        point = TeachPoint(
            timestamp=time.time(),
            joint_positions=list(joint_positions),
            cartesian_pose=cartesian_pose,
            label=label,
        )

        self.teach_points.append(point)
        logger.debug("记录示教点 #%d %s", len(self.teach_points), label)

    def start_playback(self, speed: Optional[float] = None, loop: bool = False) -> bool:
        """
        开始回放示教轨迹（后台线程）。

        Args:
            speed: 回放速度系数
            loop: 是否循环回放
        """
        if not self.teach_points:
            logger.warning("没有示教点，无法回放")
            return False

        if self.state != TeachModeState.IDLE:
            logger.warning("示教模式忙，无法开始回放")
            return False

        if speed is not None:
            self.playback_speed = float(speed)

        self.state = TeachModeState.PLAYBACK
        self.current_point_index = 0
        self._stop_flag = False

        self._playback_thread = threading.Thread(
            target=self._playback_loop, args=(loop,), daemon=True
        )
        self._playback_thread.start()
        logger.info("开始回放示教轨迹，点数: %d", len(self.teach_points))
        return True

    def _playback_loop(self, loop: bool):
        """回放线程循环。"""
        while not self._stop_flag:
            if self.state == TeachModeState.PAUSED:
                time.sleep(0.05)
                continue

            has_next = self.play_next_point()
            if not has_next:
                if loop:
                    self.current_point_index = 0
                    continue
                break

        self.state = TeachModeState.IDLE

    def stop_playback(self):
        """停止回放示教轨迹。"""
        if self.state not in (TeachModeState.PLAYBACK, TeachModeState.PAUSED):
            return

        self._stop_flag = True
        self.state = TeachModeState.IDLE

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        logger.info("回放已停止")

    def pause_playback(self):
        """暂停回放。"""
        if self.state == TeachModeState.PLAYBACK:
            self.state = TeachModeState.PAUSED
            logger.info("回放暂停")

    def resume_playback(self):
        """继续回放。"""
        if self.state == TeachModeState.PAUSED:
            self.state = TeachModeState.PLAYBACK
            logger.info("回放继续")

    def play_next_point(self) -> bool:
        """
        执行下一个示教点。

        Returns:
            是否还有后续点
        """
        if self.current_point_index >= len(self.teach_points):
            return False

        point = self.teach_points[self.current_point_index]
        speed = max(0.0, point.speed * self.playback_speed)

        success = self.joint_ctrl.move_to_position(
            point.joint_positions,
            speed=speed,
            wait=True,
        )

        if success:
            self.current_point_index += 1

        return self.current_point_index < len(self.teach_points)

    def save_trajectory(self, filepath: str):
        """
        保存示教轨迹到文件。

        Args:
            filepath: 文件路径
        """
        try:
            data = {
                "points": [],
                "metadata": {
                    "record_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_points": len(self.teach_points),
                    "record_interval": self.record_interval,
                },
            }

            for point in self.teach_points:
                point_dict = asdict(point)
                if point.cartesian_pose is not None:
                    point_dict["cartesian_pose"] = asdict(point.cartesian_pose)
                data["points"].append(point_dict)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("示教轨迹已保存: %s", filepath)
        except Exception as exc:
            logger.error("保存示教轨迹失败: %s", exc)

    def load_trajectory(self, filepath: str) -> bool:
        """
        从文件加载示教轨迹。

        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.teach_points.clear()
            for point_dict in data.get("points", []):
                pose_dict = point_dict.get("cartesian_pose")
                pose = Pose(**pose_dict) if isinstance(pose_dict, dict) else None

                point = TeachPoint(
                    timestamp=point_dict.get("timestamp", time.time()),
                    joint_positions=point_dict.get("joint_positions", []),
                    cartesian_pose=pose,
                    label=point_dict.get("label", ""),
                    speed=point_dict.get("speed", 0.3),
                )
                self.teach_points.append(point)

            logger.info("示教轨迹加载完成，点数: %d", len(self.teach_points))
            return True

        except Exception as exc:
            logger.error("加载示教轨迹失败: %s", exc)
            return False

    def get_point_count(self) -> int:
        """获取示教点数量。"""
        return len(self.teach_points)

    def get_current_state(self) -> TeachModeState:
        """获取当前状态。"""
        return self.state

    def clear_trajectory(self):
        """清空示教轨迹。"""
        self.teach_points.clear()
        self.current_point_index = 0
        logger.info("示教轨迹已清空")

    def add_point_manually(self, label: str = "") -> bool:
        """手动添加当前点。"""
        try:
            self._record_point(label)
            return True
        except Exception as exc:
            logger.error("手动添加示教点失败: %s", exc)
            return False

    def goto_point(self, point_index: int, speed: float = 0.3) -> bool:
        """
        跳转到指定示教点。

        Args:
            point_index: 点索引
            speed: 速度系数
        """
        if point_index < 0 or point_index >= len(self.teach_points):
            logger.error("点索引越界: %d", point_index)
            return False

        point = self.teach_points[point_index]
        return self.joint_ctrl.move_to_position(point.joint_positions, speed=speed, wait=True)
