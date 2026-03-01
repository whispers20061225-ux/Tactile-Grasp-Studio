"""
High-level PyBullet-backed simulator.

面向没有 PyBullet 经验的说明：
1) 这里封装了 PyBullet 的常用流程：连接 -> 加载场景 -> step 迭代 -> 读取状态。
2) GUI 里看到的“仿真显示”会通过这个类拿到真实的关节/物体/接触信息再绘制。
3) 如果没有安装 PyBullet，会在 start() 抛错提示。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import pybullet as p
    import pybullet_data
except Exception:  # pragma: no cover - optional dependency
    # 没有安装 PyBullet 时，p/pybullet_data 会是 None，start() 会直接报错提醒
    p = None
    pybullet_data = None

import numpy as np

# 简单的全局单例，便于不同模块复用同一个仿真器实例
_SIMULATOR = None


def get_simulator() -> Optional["Simulator"]:
    """获取全局仿真器单例（如果已创建）"""
    return _SIMULATOR


def create_simulator(config: Any = None) -> "Simulator":
    """创建或复用全局仿真器单例"""
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = Simulator(config)
    else:
        if config is not None:
            _SIMULATOR.update_config(config)
    return _SIMULATOR


def clear_simulator() -> None:
    """释放全局仿真器资源"""
    global _SIMULATOR
    if _SIMULATOR is not None:
        _SIMULATOR.stop()
    _SIMULATOR = None


class Simulator:
    def __init__(self, config: Any = None):
        # config 可以是 SimulationConfig 或 dict，内部统一用 _get_section 读取
        self.config = None         # 存储仿真配置对象（SimulationConfig），初始化为空
        self.client_id = None      # 存储PyBullet仿真引擎的客户端ID，用于标识当前仿真连接，初始化为空
        self.running = False       # 仿真运行状态标记：False-未运行，True-运行中
        self.robot_id = None       # 存储加载到仿真中的机器人模型ID，初始化为空
        self.object_ids: List[int] = []      # 存储仿真中所有物体（机器人、障碍物、地面等）的ID列表
        self.object_meta: List[Dict[str, Any]] = []      # 存储每个物体的元信息列表，每个元素是字典（包含名称、位置、类型等）
        self._joint_indices: List[int] = []        # （内部属性）存储机器人所有关节的索引列表
        self._joint_name_to_index: Dict[str, int] = {}      # （内部属性）关节名称到索引的映射字典，方便通过名称查索引
        self._joint_index_to_name: Dict[int, str] = {}      # （内部属性）关节索引到名称的映射字典，方便通过索引查名称
        self._control_joint_names: List[str] = []        # （内部属性）可控制的关节名称列表（比如机械臂的主动关节）
        self._control_joint_indices: List[int] = []         # （内部属性）可控制的关节索引列表，与_control_joint_names一一对应
        self.step_count = 0       # 仿真步数计数器，记录已执行的仿真步长数量
        self.sim_time = 0.0        # 仿真累计时间（单位：秒），随步长累加
        self._wall_start = None        # （内部属性）记录仿真开始的系统时间戳，用于计算真实耗时
        self._demo_motion = True         # 是否启用演示运动标记：True-自动运行演示轨迹，False-手动控制
        self._joint_targets: List[float] = []      # （内部属性）每个关节的目标角度列表，与关节索引一一对应
        self._gripper_opening = 0.1        # （内部属性）夹爪开合度（0-闭合，1-完全打开），初始为0.1（微开）
        self.update_config(config)        # 调用配置更新方法，用传入的config初始化/更新仿真配置

    def update_config(self, config: Any) -> None:
        """更新仿真配置；为空则尝试加载默认 SimulationConfig"""
        if config is None:
            try:
                from config.simulation_config import SimulationConfig
                config = SimulationConfig()
            except Exception:
                config = None
        self.config = config

    def start(self) -> bool:
        """启动仿真环境并加载场景"""
        if p is None:
            raise RuntimeError("PyBullet is not available. Install pybullet to run simulation.")
        if self.running:
            return True

        engine = self._get_section("ENGINE", {})
        mode = engine.get("mode", "gui")
        # GUI 模式会弹出 PyBullet 的原生窗口，DIRECT 模式则不显示窗口
        if mode == "gui":
            connect_mode = p.GUI
        elif mode == "direct":
            connect_mode = p.DIRECT
        else:
            connect_mode = p.DIRECT

        # 连接到物理引擎，返回 client_id（相当于一个“会话”句柄）
        self.client_id = p.connect(connect_mode)
        p.resetSimulation(physicsClientId=self.client_id)
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        # 设置重力和时间步长
        gravity = engine.get("gravity", [0.0, 0.0, -9.81])
        p.setGravity(gravity[0], gravity[1], gravity[2], physicsClientId=self.client_id)

        time_step = float(engine.get("time_step", 1.0 / 240.0))
        p.setTimeStep(time_step, physicsClientId=self.client_id)
        p.setRealTimeSimulation(1 if engine.get("real_time_simulation", True) else 0,
                                physicsClientId=self.client_id)
        # 恢复鼠标拖拽物体功能（GUI 里默认可用）
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.client_id)
        except Exception:
            pass

        # 高级物理参数 & 场景加载
        self._apply_physics_settings()
        self._load_scene()
        self._wall_start = time.time()
        self.running = True
        return True

    def step(self) -> None:
        """推进一步仿真"""
        if not self.running or self.client_id is None:
            return
        # 防止外部关闭窗口导致 PyBullet 连接失效
        if not self.is_connected():
            # 这里直接 stop，清理状态，避免继续触发异常
            self.stop()
            return
        try:
            if self._demo_motion:
                # 默认让机械臂做一个简单正弦运动，便于观察
                self._apply_demo_motion()
            else:
                # 使用外部设置的关节目标角进行驱动（UI 控制会走这里）
                self._apply_joint_targets()
            p.stepSimulation(physicsClientId=self.client_id)
        except Exception:
            # 连接断开时直接清理，避免重复异常
            self.stop()
            return
        self.step_count += 1
        engine = self._get_section("ENGINE", {})
        time_step = float(engine.get("time_step", 1.0 / 240.0))
        self.sim_time += time_step

    def stop(self) -> None:
        """停止仿真并释放引擎资源"""
        if self.client_id is not None:
            try:
                p.disconnect(self.client_id)
            except Exception:
                pass
        self.client_id = None
        self.running = False
        self.robot_id = None
        self.object_ids = []
        self.object_meta = []
        self._joint_indices = []

    def reset(self) -> None:
        """重置仿真，重新加载场景"""
        if not self.running:
            self.start()
            return
        p.resetSimulation(physicsClientId=self.client_id)
        self.step_count = 0
        self.sim_time = 0.0
        self._load_scene()

    def get_state(self) -> Dict[str, Any]:
        """获取用于可视化的状态：关节位置、夹爪状态、物体、接触信息"""
        robot_joints = self._get_robot_joint_positions()
        gripper_state = self._get_gripper_state(robot_joints)
        objects = self._get_object_states()
        collision_points, contact_forces = self._get_contact_states()
        joint_states = self._get_joint_states()
        return {
            "robot_joints": robot_joints,
            "joint_angles": joint_states["angles"],
            "joint_velocities": joint_states["velocities"],    #关节速度
            "joint_torques": joint_states["torques"],
            "joint_targets": list(self._joint_targets),    #关节力矩
            "control_joint_names": list(self._control_joint_names),
            "gripper_state": gripper_state,
            "objects": objects,
            "collision_points": collision_points,   #碰撞点
            "contact_forces": contact_forces,
            # 附带时间与步数，便于 UI 同步显示
            "meta": {
                "sim_time": self.sim_time,
                "step_count": self.step_count,
            },
        }

    def get_camera_image(self, width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
        """从仿真器里获取相机画面（RGB + 深度）"""
        if self.client_id is None:
            raise RuntimeError("Simulator is not running.")
        view_cfg = self._get_section("VISUALIZATION", {}).get("camera_view", {})
        target = view_cfg.get("camera_target", [0.0, 0.0, 0.5])
        distance = float(view_cfg.get("camera_distance", 1.5))
        yaw = float(view_cfg.get("camera_yaw", 45.0))
        pitch = float(view_cfg.get("camera_pitch", -30.0))
        # PyBullet 需要视图矩阵和投影矩阵
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=10.0,
        )
        _, _, rgba, depth, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        # PyBullet 返回的 depth 是 0~1 的非线性深度，需要转换成米
        color = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
        depth_buf = np.reshape(depth, (height, width))
        depth_m = self._depth_buffer_to_meters(depth_buf, near=0.01, far=10.0)
        return color, depth_m

    def set_joint_targets(self, targets: List[float]) -> None:
        """设置关节目标角度（会关闭默认正弦运动）"""
        if not targets:
            return
        # 仅保留可控关节数量，避免长度不一致导致控制异常
        target_list = list(targets)
        if self._control_joint_indices:
            expected = len(self._control_joint_indices)
            if len(target_list) < expected:
                # 目标不足时用 0 补齐，保证长度一致
                target_list += [0.0] * (expected - len(target_list))
            if len(target_list) > expected:
                target_list = target_list[:expected]
        self._joint_targets = target_list
        self._demo_motion = False

    def _apply_physics_settings(self) -> None:
        """应用高级物理引擎参数"""
        physics_cfg = self._get_section("PHYSICS_ADVANCED", {})
        if not physics_cfg or self.client_id is None:
            return
        params = {
            "numSolverIterations": int(physics_cfg.get("solver_iterations", 50)),
            "solverResidualThreshold": float(physics_cfg.get("solver_residual_threshold", 1e-7)),
            "contactSlop": float(physics_cfg.get("contact_slop", 0.001)),
            "numSubSteps": int(physics_cfg.get("max_sub_steps", 10)),
        }
        # 不同 PyBullet 版本参数名不一致，优先尝试 enableFileCaching
        if "enable_caching" in physics_cfg:
            params["enableFileCaching"] = bool(physics_cfg.get("enable_caching", True))

        try:
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **params)
        except TypeError:
            params.pop("enableFileCaching", None)
            p.setPhysicsEngineParameter(physicsClientId=self.client_id, **params)

    def _load_scene(self) -> None:
        """加载地面、机械臂和示例物体"""
        if self.client_id is None:
            return
        scene_cfg = self._get_section("SCENE", {})
        if scene_cfg.get("floor_enabled", True):
            try:
                p.loadURDF("plane.urdf", physicsClientId=self.client_id)
            except Exception:
                pass
        self._load_robot()
        self._spawn_objects()

    def _load_robot(self) -> None:
        """加载机械臂 URDF"""
        arm_cfg = self._get_section("ARM_SIMULATION", {})
        urdf_path = arm_cfg.get("urdf_path", "")
        resolved = self._resolve_path(urdf_path)
        if not resolved or not os.path.exists(resolved):
            # 如果项目内没有 URDF，就回退到 PyBullet 自带模型
            if pybullet_data is not None:
                resolved = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
            else:
                raise FileNotFoundError(f"URDF not found: {urdf_path}")

        base_pos = arm_cfg.get("base_position", [0.0, 0.0, 0.0])
        base_ori = arm_cfg.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        global_scaling = float(arm_cfg.get("global_scaling", 1.0))
        if global_scaling <= 0:
            global_scaling = 1.0
        # 配置里可能是欧拉角，也可能是四元数
        if len(base_ori) == 3:
            base_ori = p.getQuaternionFromEuler(base_ori)
        use_fixed = bool(arm_cfg.get("use_fixed_base", True))

        # 追加搜索路径，方便解析 URDF 引用的 mesh
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            models_dir = os.path.join(project_root, "models")
            urdf_dir = os.path.dirname(resolved)
            search_base = models_dir if os.path.isdir(models_dir) else urdf_dir
            if search_base:
                p.setAdditionalSearchPath(search_base, physicsClientId=self.client_id)
        except Exception:
            pass

        # loadURDF 返回 body id
        self.robot_id = p.loadURDF(
            resolved,
            basePosition=base_pos,
            baseOrientation=base_ori,
            useFixedBase=use_fixed,
            globalScaling=global_scaling,
            physicsClientId=self.client_id,
        )
        self._joint_indices = []
        self._joint_name_to_index = {}
        self._joint_index_to_name = {}
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            # 记录所有关节名称，后续根据名称筛选“可控关节”
            self._joint_name_to_index[joint_name] = joint_index
            self._joint_index_to_name[joint_index] = joint_name
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self._joint_indices.append(joint_index)

        # 解析可控关节（优先用名称，找不到再回退）
        self._resolve_control_joints()

        # 初始化关节位置（只对可控关节做初始化）
        self._joint_targets = self._get_initial_joint_targets(arm_cfg)
        if self._joint_targets:
            self.set_joint_targets(self._joint_targets)
        for joint_index, target in zip(self._control_joint_indices, self._joint_targets):
            p.resetJointState(self.robot_id, joint_index, target, physicsClientId=self.client_id)

    def _resolve_control_joints(self) -> None:
        """确定需要被外部控制的关节列表（按名称优先）"""
        arm_cfg = self._get_section("ARM_SIMULATION", {})
        # 优先使用配置里指定的关节名称列表
        control_names = arm_cfg.get("control_joint_names")
        if not control_names:
            # 默认匹配 dofbot 的 5 个关节 + 夹爪
            control_names = [
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "grip_joint",
            ]
        self._control_joint_names = list(control_names)

        # 按名称查找关节索引
        indices = []
        for name in self._control_joint_names:
            idx = self._joint_name_to_index.get(name)
            if idx is not None:
                indices.append(idx)

        # 如果名称找不到，就回退到前 N 个可动关节
        if not indices:
            fallback_count = min(6, len(self._joint_indices))
            indices = self._joint_indices[:fallback_count]
            self._control_joint_names = [
                self._joint_index_to_name.get(idx, f"joint_{idx}") for idx in indices
            ]

        self._control_joint_indices = indices

    def _get_initial_joint_targets(self, arm_cfg: Dict[str, Any]) -> List[float]:
        if not self._control_joint_indices:
            return []
        count = len(self._control_joint_indices)
        initial_deg = arm_cfg.get("initial_joint_positions_deg")
        if isinstance(initial_deg, (list, tuple)) and len(initial_deg) >= count:
            return [math.radians(float(v)) for v in initial_deg[:count]]
        initial_rad = arm_cfg.get("initial_joint_positions")
        if isinstance(initial_rad, (list, tuple)) and len(initial_rad) >= count:
            return [float(v) for v in initial_rad[:count]]
        urdf_path = str(arm_cfg.get("urdf_path", ""))
        if os.path.basename(urdf_path) == "dofbot.urdf" and count >= 6:
            default_deg = [-90.0, 0.0, 0.0, 0.0, -90.0, 0.0]
            return [math.radians(v) for v in default_deg[:count]]
        return [0.0 for _ in range(count)]

    def _spawn_objects(self) -> None:
        """生成几个简单物体，方便观察抓取/碰撞效果"""
        self.object_ids = []
        self.object_meta = []
        if self.client_id is None:
            return

        cube_size = 0.1
        cube_half = cube_size / 2.0
        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_half] * 3,
                                          physicsClientId=self.client_id)
        cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_half] * 3,
                                       rgbaColor=[0.2, 0.7, 0.3, 1.0],
                                       physicsClientId=self.client_id)
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=[0.3, 0.2, cube_half],
            physicsClientId=self.client_id,
        )
        self.object_ids.append(cube_id)
        self.object_meta.append({"type": "cube", "size": [cube_size, cube_size, cube_size], "color": "green"})

        sphere_radius = 0.05
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius,
                                            physicsClientId=self.client_id)
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius,
                                         rgbaColor=[0.9, 0.9, 0.2, 1.0],
                                         physicsClientId=self.client_id)
        sphere_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=sphere_col,
            baseVisualShapeIndex=sphere_vis,
            basePosition=[-0.2, 0.3, sphere_radius],
            physicsClientId=self.client_id,
        )
        self.object_ids.append(sphere_id)
        self.object_meta.append({"type": "sphere", "radius": sphere_radius, "color": "yellow"})

        cyl_radius = 0.04
        cyl_height = 0.1
        cyl_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cyl_radius, height=cyl_height,
                                         physicsClientId=self.client_id)
        cyl_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cyl_radius, length=cyl_height,
                                      rgbaColor=[0.9, 0.6, 0.2, 1.0],
                                      physicsClientId=self.client_id)
        cyl_id = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=cyl_col,
            baseVisualShapeIndex=cyl_vis,
            basePosition=[0.1, -0.3, cyl_height / 2.0],
            physicsClientId=self.client_id,
        )
        self.object_ids.append(cyl_id)
        self.object_meta.append({"type": "cylinder", "radius": cyl_radius, "height": cyl_height, "color": "orange"})

    def _get_robot_joint_positions(self) -> List[List[float]]:
        """拿到每个关节对应 link 的世界坐标，用于绘制连杆"""
        if self.client_id is None or self.robot_id is None:
            return []
        positions = []
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        positions.append([base_pos[0], base_pos[1], base_pos[2]])
        for joint_index in self._joint_indices:
            link_state = p.getLinkState(self.robot_id, joint_index, physicsClientId=self.client_id)
            pos = link_state[0]
            positions.append([pos[0], pos[1], pos[2]])
        return positions

    def _get_gripper_state(self, robot_joints: List[List[float]]) -> Optional[Dict[str, Any]]:
        """夹爪状态用于 UI 展示（目前用末端位置 + 最大开度模拟）"""
        if not robot_joints:
            return None
        opening = self._get_section("GRIPPER_SIMULATION", {}).get("max_opening", self._gripper_opening)
        return {
            "position": robot_joints[-1],
            "rotation": [0.0, 0.0, 0.0],
            "opening": opening,
            "fingers": 2,
        }

    def _get_object_states(self) -> List[Dict[str, Any]]:
        """读取物体的世界位置，给可视化用"""
        if self.client_id is None:
            return []
        objects = []
        for obj_id, meta in zip(self.object_ids, self.object_meta):
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
            obj = dict(meta)
            obj["position"] = [pos[0], pos[1], pos[2]]
            objects.append(obj)
        return objects

    def _get_joint_states(self) -> Dict[str, List[float]]:
        """读取可控关节的角度/速度/力矩（PyBullet 原生单位）"""
        if self.client_id is None or self.robot_id is None:
            return {
                "angles": [],
                "velocities": [],
                "torques": [],
            }
        angles = []
        velocities = []
        torques = []
        for joint_index in self._control_joint_indices:
            state = p.getJointState(self.robot_id, joint_index, physicsClientId=self.client_id)
            # state: (position, velocity, reaction_forces, applied_torque)
            angles.append(float(state[0]))
            velocities.append(float(state[1]))
            torques.append(float(state[3]))
        return {
            "angles": angles,
            "velocities": velocities,
            "torques": torques,
        }

    def _get_contact_states(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """读取接触点和力，用于 UI 显示碰撞/力箭头"""
        if self.client_id is None or self.robot_id is None:
            return [], []
        contact_points = []
        contact_forces = []
        for contact in p.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client_id):
            pos_on_a = contact[5]
            normal_on_b = contact[7]
            normal_force = contact[9]
            contact_points.append({
                "position": [pos_on_a[0], pos_on_a[1], pos_on_a[2]],
                "normal": [normal_on_b[0], normal_on_b[1], normal_on_b[2]],
                "force": normal_force,
            })
            contact_forces.append({
                "position": [pos_on_a[0], pos_on_a[1], pos_on_a[2]],
                "force": [
                    normal_on_b[0] * normal_force,
                    normal_on_b[1] * normal_force,
                    normal_on_b[2] * normal_force,
                ],
            })
        return contact_points, contact_forces

    def _apply_joint_targets(self) -> None:
        """把外部下发的关节目标角写入 PyBullet 电机"""
        if self.client_id is None or self.robot_id is None:
            return
        if not self._control_joint_indices or not self._joint_targets:
            return

        force_limit = float(self._get_section("ARM_SIMULATION", {}).get("drive_force", 50.0))
        position_gain = float(self._get_section("ARM_SIMULATION", {}).get("position_gain", 0.4))
        try:
            p.setJointMotorControlArray(
                self.robot_id,
                self._control_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=self._joint_targets,
                forces=[force_limit for _ in self._control_joint_indices],
                positionGains=[position_gain for _ in self._control_joint_indices],
                physicsClientId=self.client_id,
            )
        except TypeError:
            # 兼容旧版 PyBullet 参数签名
            p.setJointMotorControlArray(
                self.robot_id,
                self._control_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=self._joint_targets,
                forces=[force_limit for _ in self._control_joint_indices],
                physicsClientId=self.client_id,
            )
        except Exception:
            # 连接丢失或其他异常时，停止仿真避免反复报错
            self.stop()

    def _apply_demo_motion(self) -> None:
        """默认正弦轨迹，方便在没有控制指令时也能看到运动"""
        if self.client_id is None or self.robot_id is None:
            return
        joint_indices = self._control_joint_indices or self._joint_indices
        if not joint_indices:
            return
        t = self.sim_time
        targets = []
        for idx in range(len(joint_indices)):
            amp = 0.4 if idx % 2 == 0 else 0.25
            targets.append(amp * np.sin(t * 0.7 + idx * 0.6))
        # 力矩设大一些，避免被重力“压住”看不到运动
        force_limit = float(self._get_section("ARM_SIMULATION", {}).get("drive_force", 50.0))
        try:
            p.setJointMotorControlArray(
                self.robot_id,
                joint_indices,
                p.POSITION_CONTROL,
                targetPositions=targets,
                forces=[force_limit for _ in joint_indices],
                positionGains=[0.4 for _ in joint_indices],
                physicsClientId=self.client_id,
            )
        except TypeError:
            p.setJointMotorControlArray(
                self.robot_id,
                joint_indices,
                p.POSITION_CONTROL,
                targetPositions=targets,
                forces=[force_limit for _ in joint_indices],
                physicsClientId=self.client_id,
            )
        except Exception:
            # 连接丢失时直接停止，避免 GUI 进程崩溃
            self.stop()

    def is_connected(self) -> bool:
        """判断 PyBullet 连接是否仍然有效"""
        if self.client_id is None or p is None:
            return False
        try:
            return bool(p.isConnected(self.client_id))
        except Exception:
            return False

    def _get_section(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """从 config 里读取配置段（兼容 dict / 配置类）"""
        default = default or {}
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(name, default) or default
        return getattr(self.config, name, default) or default

    def _resolve_path(self, path: str) -> str:
        """把相对路径解析为项目内的绝对路径"""
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(project_root, path)

    @staticmethod
    def _depth_buffer_to_meters(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
        """将 PyBullet 归一化深度转换为米（OpenGL 深度缓冲转换公式）"""
        return (far * near) / (far - (far - near) * depth_buffer)
