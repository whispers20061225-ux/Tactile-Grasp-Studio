"""
笛卡尔控制器：提供末端位姿控制与查询。

说明：
- 优先使用 URDF 模型（models/dofbot.urdf）进行正/逆运动学计算
- 若 URDF 解析失败，则回退到简化几何模型
"""

import logging
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .joint_controller import JointController

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """末端位姿（位置：mm，姿态：度）。"""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class UrdfKinematics:
    """基于 URDF 的简易运动学解析器。"""

    def __init__(
        self,
        urdf_path: str,
        joint_names: List[str],
        base_link: str,
        end_link: str,
    ):
        self.urdf_path = urdf_path
        self.joint_names = joint_names
        self.base_link = base_link
        self.end_link = end_link

        # 每个关节的固定变换与轴信息
        self._joints: List[Dict[str, object]] = []
        self._valid = False

        self._load_urdf()

    @property
    def valid(self) -> bool:
        """URDF 是否解析成功。"""
        return self._valid

    def get_joint_limits_deg(self) -> List[Tuple[float, float]]:
        """获取 URDF 中的关节角度限制（度）。"""
        limits: List[Tuple[float, float]] = []
        for joint in self._joints:
            lower = joint.get("lower")
            upper = joint.get("upper")
            if lower is None or upper is None:
                limits.append((-180.0, 180.0))
            else:
                limits.append((math.degrees(lower), math.degrees(upper)))
        return limits

    def forward(self, joint_angles_rad: List[float]) -> np.ndarray:
        """
        正运动学：计算从 base_link 到 end_link 的 4x4 变换矩阵。
        """
        T = np.eye(4, dtype=float)

        for idx, joint in enumerate(self._joints):
            angle = joint_angles_rad[idx] if idx < len(joint_angles_rad) else 0.0
            T = T @ joint["origin_T"]

            joint_type = joint.get("type", "revolute")
            if joint_type in ("revolute", "continuous"):
                R = self._rot_axis_angle(joint["axis"], angle)
                T = T @ R
            elif joint_type == "prismatic":
                T = T @ self._trans_along_axis(joint["axis"], angle)

        return T

    def _load_urdf(self):
        """解析 URDF 并建立关节链。"""
        if not os.path.exists(self.urdf_path):
            logger.error("URDF 文件不存在: %s", self.urdf_path)
            return

        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()

            joints_by_name: Dict[str, Dict[str, object]] = {}
            for joint in root.findall("joint"):
                name = joint.attrib.get("name", "")
                joint_type = joint.attrib.get("type", "revolute")

                origin = joint.find("origin")
                xyz = self._parse_xyz(origin.attrib.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)
                rpy = self._parse_xyz(origin.attrib.get("rpy", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)

                axis_el = joint.find("axis")
                axis = self._parse_xyz(axis_el.attrib.get("xyz", "0 0 1")) if axis_el is not None else (0.0, 0.0, 1.0)

                limit_el = joint.find("limit")
                lower = float(limit_el.attrib.get("lower")) if limit_el is not None and "lower" in limit_el.attrib else None
                upper = float(limit_el.attrib.get("upper")) if limit_el is not None and "upper" in limit_el.attrib else None

                joints_by_name[name] = {
                    "name": name,
                    "type": joint_type,
                    "origin_T": self._make_transform(xyz, rpy),
                    "axis": np.array(axis, dtype=float),
                    "lower": lower,
                    "upper": upper,
                }

            # 按 joint_names 构建链
            missing = [j for j in self.joint_names if j not in joints_by_name]
            if missing:
                logger.error("URDF 中缺少关节: %s", missing)
                return

            self._joints = [joints_by_name[name] for name in self.joint_names]
            self._valid = True

        except Exception as exc:
            logger.error("URDF 解析失败: %s", exc)
            self._valid = False

    @staticmethod
    def _parse_xyz(text: str) -> Tuple[float, float, float]:
        parts = [p for p in text.strip().split(" ") if p]
        values = [float(p) for p in parts]
        while len(values) < 3:
            values.append(0.0)
        return values[0], values[1], values[2]

    @staticmethod
    def _make_transform(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
        """构造 4x4 齐次变换矩阵（URDF 采用米，rpy 为弧度）。"""
        x, y, z = xyz
        roll, pitch, yaw = rpy

        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)],
            ],
            dtype=float,
        )
        Ry = np.array(
            [
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)],
            ],
            dtype=float,
        )
        Rz = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        # URDF 采用 R = Rz * Ry * Rx 顺序
        R = Rz @ Ry @ Rx

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, z], dtype=float)
        return T

    @staticmethod
    def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """轴角旋转 -> 齐次矩阵。"""
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        x, y, z = axis.tolist()
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1 - c

        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s, 0],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s, 0],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        return R

    @staticmethod
    def _trans_along_axis(axis: np.ndarray, distance: float) -> np.ndarray:
        """沿关节轴方向平移（用于 prismatic）。"""
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        T = np.eye(4, dtype=float)
        T[:3, 3] = axis * distance
        return T


class CartesianController:
    """笛卡尔空间控制器（URDF 优先，失败回退简化模型）。"""

    def __init__(self, joint_controller: JointController):
        # 关联的关节控制器
        self.joint_controller = joint_controller

        # 优先加载 URDF 模型
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        urdf_path, end_link, joint_names = self._select_urdf_model(project_root)
        self.urdf_joint_names = joint_names
        self._urdf_kin = UrdfKinematics(
            urdf_path=urdf_path,
            joint_names=self.urdf_joint_names,
            base_link="base_link",
            end_link=end_link,
        )
        self._model_joint_count = len(self.urdf_joint_names)

        # 若 URDF 可用，更新关节限制（角度）
        if self._urdf_kin.valid:
            limits = self._urdf_kin.get_joint_limits_deg()
            if limits:
                for idx, (lower, upper) in enumerate(limits):
                    if idx < len(self.joint_controller.joint_limits["angle_min"]):
                        self.joint_controller.joint_limits["angle_min"][idx] = lower
                        self.joint_controller.joint_limits["angle_max"][idx] = upper

        # 回退时的简化几何参数（单位：mm）
        self.base_height = 80.0
        self.link1 = 105.0
        self.link2 = 105.0

        # 工作空间限制（优先从配置读取）
        reach = self.link1 + self.link2
        self.workspace_limits = {
            "x_min": -reach,
            "x_max": reach,
            "y_min": -reach,
            "y_max": reach,
            "z_min": 0.0,
            "z_max": self.base_height + reach,
        }
        cfg_limits = self._load_workspace_limits_from_config()
        if cfg_limits:
            self.workspace_limits = cfg_limits

        # 逆运动学参数
        self.ik_max_iterations = 80
        self.ik_tolerance = 2.0  # mm
        self.ik_damping = 0.01

        # 当前/目标位姿缓存
        self.current_pose = Pose(reach, 0.0, self.base_height, 0.0, 0.0, 0.0)
        self.target_pose = Pose(reach, 0.0, self.base_height, 0.0, 0.0, 0.0)

        if self._urdf_kin.valid:
            logger.info("笛卡尔控制器初始化完成（URDF 模型）")
        else:
            logger.warning("URDF 模型不可用，使用简化几何模型")

    def _select_urdf_model(self, project_root: str) -> Tuple[str, str, List[str]]:
        """根据配置选择 URDF 模型与关节链。"""
        urdf_path = None
        end_link = "arm_link5"
        joint_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5"]

        profile = self._get_arm_profile()
        sim_cfg = {}
        if isinstance(profile, dict):
            sim_cfg = profile.get("SIMULATION", {}) or {}
        elif profile:
            sim_cfg = getattr(profile, "SIMULATION", {}) or {}

        candidate = sim_cfg.get("urdf_path") if isinstance(sim_cfg, dict) else None
        if candidate:
            urdf_path = candidate
            if not os.path.isabs(urdf_path):
                urdf_path = os.path.join(project_root, urdf_path)
            if not os.path.exists(urdf_path):
                urdf_path = None

        if not urdf_path:
            learm_path = os.path.join(project_root, "models", "learm_arm.urdf")
            if os.path.exists(learm_path):
                urdf_path = learm_path
            else:
                urdf_path = os.path.join(project_root, "models", "dofbot.urdf")

        if os.path.basename(urdf_path).lower().startswith("learm"):
            end_link = "link6"
            joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        return urdf_path, end_link, joint_names

    def _get_arm_profile(self) -> object:
        config = getattr(self.joint_controller.arm, "config", None)
        if not config:
            return {}
        if hasattr(config, "learm_arm") and config.learm_arm:
            return config.learm_arm
        if isinstance(config, dict):
            return config
        return {}

    def _load_workspace_limits_from_config(self) -> Optional[Dict[str, float]]:
        profile = self._get_arm_profile()
        safety = {}
        physical = {}
        if isinstance(profile, dict):
            safety = profile.get("SAFETY", {}) or {}
            physical = profile.get("PHYSICAL", {}) or {}
        elif profile:
            safety = getattr(profile, "SAFETY", {}) or {}
            physical = getattr(profile, "PHYSICAL", {}) or {}

        limits = safety.get("workspace_limits") if isinstance(safety, dict) else None
        if isinstance(limits, dict):
            if all(k in limits for k in ("x", "y", "z")):
                x_min, x_max = limits["x"]
                y_min, y_max = limits["y"]
                z_min, z_max = limits["z"]
                return {
                    "x_min": float(x_min),
                    "x_max": float(x_max),
                    "y_min": float(y_min),
                    "y_max": float(y_max),
                    "z_min": float(z_min),
                    "z_max": float(z_max),
                }
            if all(k in limits for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
                return {
                    "x_min": float(limits["x_min"]),
                    "x_max": float(limits["x_max"]),
                    "y_min": float(limits["y_min"]),
                    "y_max": float(limits["y_max"]),
                    "z_min": float(limits["z_min"]),
                    "z_max": float(limits["z_max"]),
                }

        reach = None
        if isinstance(physical, dict):
            reach = physical.get("reach")
        if reach is None:
            return None

        reach = float(reach)
        return {
            "x_min": -reach,
            "x_max": reach,
            "y_min": -reach,
            "y_max": reach,
            "z_min": 0.0,
            "z_max": self.base_height + reach,
        }

    def move_to_pose(self, pose: Pose, speed: float = 0.3, wait: bool = True) -> bool:
        """
        移动到指定笛卡尔位姿（使用简化 IK）。

        Args:
            pose: 目标位姿
            speed: 速度系数（0~1）
            wait: 是否等待动作完成
        """
        if not self._check_workspace(pose):
            logger.error("目标位姿超出工作空间: %s", pose)
            return False

        joint_angles = self.inverse_kinematics(pose)
        if joint_angles is None:
            logger.error("逆运动学失败，无法到达目标位姿")
            return False

        success = self.joint_controller.move_to_angles(joint_angles, speed=speed, wait=wait)
        if success:
            self.target_pose = pose
        return bool(success)

    def move_linear(
        self,
        target_pose: Pose,
        speed: float = 0.2,
        num_points: int = 10,
        wait: bool = True,
    ) -> bool:
        """
        直线插值运动到目标位姿。

        Args:
            target_pose: 目标位姿
            speed: 速度系数（0~1）
            num_points: 插值点数量
            wait: 是否等待每段完成
        """
        start_pose = self.get_current_pose(update=True)
        trajectory = self._generate_linear_trajectory(start_pose, target_pose, num_points)
        return self.follow_trajectory(trajectory, speed=speed, wait=wait)

    def move_to_point(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        speed: float = 0.3,
        wait: bool = True,
    ) -> bool:
        """
        便捷接口：通过参数创建位姿并移动。
        """
        return self.move_to_pose(Pose(x, y, z, roll, pitch, yaw), speed=speed, wait=wait)

    def get_current_pose(self, update: bool = True) -> Pose:
        """
        获取当前末端位姿（通过简化 FK 估算）。

        Args:
            update: 是否从关节角度刷新
        """
        if update:
            joint_angles = self.joint_controller.get_current_angles(update=True)
            self.current_pose = self.forward_kinematics(joint_angles)
        return self.current_pose

    def forward_kinematics(self, joint_angles: List[float]) -> Pose:
        """
        简化正运动学：
        - 关节1：底座旋转（yaw）
        - 关节2/3：平面二连杆
        - 关节4/5/6：姿态映射
        """
        if self._urdf_kin.valid:
            return self._forward_kinematics_urdf(joint_angles)

        if len(joint_angles) < 6:
            logger.error("关节角度长度不足，期望 6")
            return self.current_pose

        j1, j2, j3, j4, j5, j6 = [math.radians(a) for a in joint_angles[:6]]

        # 平面二连杆求 r, z
        r = self.link1 * math.cos(j2) + self.link2 * math.cos(j2 + j3)
        z = self.base_height + self.link1 * math.sin(j2) + self.link2 * math.sin(j2 + j3)

        # 底座旋转映射到 x, y
        x = r * math.cos(j1)
        y = r * math.sin(j1)

        # 姿态简化为关节 4/5/6 角度
        roll = math.degrees(j4)
        pitch = math.degrees(j5)
        yaw = math.degrees(j6)

        return Pose(x, y, z, roll, pitch, yaw)

    def inverse_kinematics(self, pose: Pose) -> Optional[List[float]]:
        """
        简化逆运动学（平面二连杆 + 底座旋转）。

        Returns:
            关节角度列表（度），失败返回 None
        """
        if self._urdf_kin.valid:
            return self._inverse_kinematics_urdf(pose)

        x, y, z = pose.x, pose.y, pose.z

        # 关节1：底座旋转
        j1 = math.atan2(y, x)

        # 平面二连杆
        r = math.sqrt(x * x + y * y)
        z_adj = z - self.base_height

        # 余弦定理
        denom = 2.0 * self.link1 * self.link2
        if denom == 0:
            return None
        cos_j3 = (r * r + z_adj * z_adj - self.link1 * self.link1 - self.link2 * self.link2) / denom
        if cos_j3 < -1.0 or cos_j3 > 1.0:
            return None
        j3 = math.acos(cos_j3)

        # 关节2
        j2 = math.atan2(z_adj, r) - math.atan2(self.link2 * math.sin(j3), self.link1 + self.link2 * math.cos(j3))

        # 姿态简单映射
        j4 = math.radians(pose.roll)
        j5 = math.radians(pose.pitch)
        j6 = math.radians(pose.yaw)

        angles_deg = [
            math.degrees(j1),
            math.degrees(j2),
            math.degrees(j3),
            math.degrees(j4),
            math.degrees(j5),
            math.degrees(j6),
        ]

        # 使用关节控制器做范围限制
        if not self._check_joint_limits(angles_deg):
            return None

        return angles_deg

    def follow_trajectory(self, trajectory: List[Pose], speed: float = 0.2, wait: bool = True) -> bool:
        """
        顺序执行位姿轨迹。
        """
        for pose in trajectory:
            if not self.move_to_pose(pose, speed=speed, wait=wait):
                return False
        return True

    def _check_workspace(self, pose: Pose) -> bool:
        """检查位姿是否在工作空间范围内。"""
        return (
            self.workspace_limits["x_min"] <= pose.x <= self.workspace_limits["x_max"]
            and self.workspace_limits["y_min"] <= pose.y <= self.workspace_limits["y_max"]
            and self.workspace_limits["z_min"] <= pose.z <= self.workspace_limits["z_max"]
        )

    def _check_joint_limits(self, angles: List[float]) -> bool:
        """使用关节控制器的限制做二次校验。"""
        limits = self.joint_controller.joint_limits
        for i, angle in enumerate(angles):
            if angle < limits["angle_min"][i] or angle > limits["angle_max"][i]:
                return False
        return True

    def _generate_linear_trajectory(self, start: Pose, end: Pose, num_points: int) -> List[Pose]:
        """生成简单线性插值轨迹。"""
        if num_points <= 1:
            return [end]

        trajectory: List[Pose] = []
        for i in range(num_points):
            t = i / (num_points - 1)
            trajectory.append(
                Pose(
                    x=start.x + t * (end.x - start.x),
                    y=start.y + t * (end.y - start.y),
                    z=start.z + t * (end.z - start.z),
                    roll=start.roll + t * (end.roll - start.roll),
                    pitch=start.pitch + t * (end.pitch - start.pitch),
                    yaw=start.yaw + t * (end.yaw - start.yaw),
                )
            )
        return trajectory

    def _forward_kinematics_urdf(self, joint_angles: List[float]) -> Pose:
        """基于 URDF 的正运动学（更精确）。"""
        angles_rad = [math.radians(a) for a in joint_angles[: self._model_joint_count]]
        T = self._urdf_kin.forward(angles_rad)

        # URDF 使用米，转为 mm
        x = T[0, 3] * 1000.0
        y = T[1, 3] * 1000.0
        z = T[2, 3] * 1000.0

        roll, pitch, yaw = self._rotation_to_rpy(T[:3, :3])
        return Pose(x, y, z, math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def _inverse_kinematics_urdf(self, pose: Pose) -> Optional[List[float]]:
        """基于 URDF 的数值逆运动学（位置优先）。"""
        target = np.array([pose.x, pose.y, pose.z], dtype=float) / 1000.0

        # 取当前角度作为初值
        current_angles = self.joint_controller.get_current_angles(update=True)
        q = np.array(
            [math.radians(a) for a in current_angles[: self._model_joint_count]],
            dtype=float,
        )

        for _ in range(self.ik_max_iterations):
            T = self._urdf_kin.forward(q.tolist())
            pos = T[:3, 3]
            error = target - pos
            err_norm = float(np.linalg.norm(error))
            if err_norm * 1000.0 <= self.ik_tolerance:
                break

            J = self._numerical_jacobian(q)
            if J is None:
                return None

            # 阻尼最小二乘
            JT = J.T
            damping = self.ik_damping
            dq = JT @ np.linalg.inv(J @ JT + (damping ** 2) * np.eye(3)) @ error
            q = q + dq

            # 限制关节角度范围
            q = self._clamp_joint_angles(q)

        # 输出角度（度）
        angles_deg = [math.degrees(v) for v in q.tolist()]

        # 将结果扩展到完整关节列表（夹爪角度保持原值）
        full_angles = self.joint_controller.get_current_angles(update=False)
        for i in range(min(len(full_angles), len(angles_deg))):
            full_angles[i] = angles_deg[i]
        return full_angles

    def _numerical_jacobian(self, q: np.ndarray, eps: float = 1e-4) -> Optional[np.ndarray]:
        """数值法计算雅可比矩阵（位置部分）。"""
        n = q.shape[0]
        J = np.zeros((3, n), dtype=float)

        T0 = self._urdf_kin.forward(q.tolist())
        p0 = T0[:3, 3]

        for i in range(n):
            dq = np.zeros_like(q)
            dq[i] = eps
            T1 = self._urdf_kin.forward((q + dq).tolist())
            p1 = T1[:3, 3]
            J[:, i] = (p1 - p0) / eps

        return J

    def _clamp_joint_angles(self, q: np.ndarray) -> np.ndarray:
        """将关节角度限制在 URDF 范围内（弧度）。"""
        limits = self._urdf_kin.get_joint_limits_deg()
        if not limits:
            return q

        q_clamped = q.copy()
        for i, (lower_deg, upper_deg) in enumerate(limits):
            lower = math.radians(lower_deg)
            upper = math.radians(upper_deg)
            if i < len(q_clamped):
                q_clamped[i] = max(lower, min(upper, q_clamped[i]))
        return q_clamped

    @staticmethod
    def _rotation_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
        """从旋转矩阵提取 RPY（弧度）。"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw
