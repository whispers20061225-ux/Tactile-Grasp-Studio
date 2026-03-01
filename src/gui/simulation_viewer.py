"""
仿真显示部件 - 用于显示PyBullet/CoppeliaSim仿真环境
"""

import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
import time
import logging

# 导入Matplotlib 3D组件
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QGroupBox, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QSplitter, QTabWidget, QScrollArea, QSlider,
    QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon

try:
    from config import DemoConfig, SimulationConfig
except ImportError:
    try:
        from ..config import DemoConfig, SimulationConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import DemoConfig, SimulationConfig

try:
    from simulation.simulator import create_simulator
except Exception:
    try:
        from ..simulation.simulator import create_simulator
    except Exception:
        # 没有仿真后端时，UI 仍可运行，但不连接 PyBullet
        create_simulator = None

try:
    from simulation.pybullet_process_client import PyBulletProcessClient
except Exception:
    try:
        from ..simulation.pybullet_process_client import PyBulletProcessClient
    except Exception:
        PyBulletProcessClient = None

logger = logging.getLogger(__name__)


class Arrow3D(FancyArrowPatch):
    """3D箭头类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class SimulationViewer(QWidget):
    """仿真显示部件类"""
    
    # 自定义信号
    start_simulation = pyqtSignal()
    pause_simulation = pyqtSignal()
    stop_simulation = pyqtSignal()
    reset_simulation = pyqtSignal()
    step_simulation = pyqtSignal()
    load_scene = pyqtSignal(str)
    save_state = pyqtSignal(str)
    
    def __init__(self, config: SimulationConfig, parent=None):
        """
        初始化仿真显示部件
        
        Args:
            config: 仿真配置
            parent: 父部件
        """
        super().__init__(parent)
        
        # 保存配置
        self.config = config
        
        # 仿真状态
        self.simulation_running = False
        self.simulation_paused = False
        self.simulation_time = 0.0
        self.simulation_speed = 1.0
        # PyBullet 后端实例（由 simulation/simulator.py 提供）
        self.simulator = None
        # PyBullet GUI 子进程客户端（用于独立窗口）
        self.simulator_client = None
        self._external_step_count = 0
        
        # 3D场景数据
        self.robot_joints = []  # 机器人关节位置
        self.robot_links = []   # 机器人连杆
        self.gripper_state = None  # 夹爪状态
        self.objects = []       # 场景中的物体
        self.collision_points = []  # 碰撞点
        self.contact_forces = []    # 接触力
        
        # 可视化参数
        self.show_grid = True
        self.show_axes = True
        self.show_wireframe = False
        self.show_collisions = True
        self.show_forces = True
        self.show_trajectories = True
        self.show_coordinate_frames = True
        
        # 相机视角
        self.view_elevation = 30
        self.view_azimuth = 45
        self.view_distance = 2.0
        
        # 轨迹历史
        self.trajectory_history = []
        self.max_trajectory_points = 1000
        
        # 初始化UI
        self.init_ui()
        self._apply_config_defaults()
        
        # 设置定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_simulation)
        self.update_timer.start(50)  # 20Hz更新频率

    def _apply_config_defaults(self):
        """根据配置文件设置 UI 默认值"""
        engine_cfg = self._get_config_section("ENGINE", {})
        time_step = engine_cfg.get("time_step", 0.01)
        try:
            self.timestep_spin.setValue(float(time_step))
        except Exception:
            pass
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：3D显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建Matplotlib图形
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建3D子图
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # 设置3D视图
        self.setup_3d_view()
        
        # 添加导航工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
        
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        
        # 视图控制工具栏
        view_controls = QWidget()
        view_controls_layout = QHBoxLayout(view_controls)
        view_controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # 视图预设按钮
        view_presets = ["前视图", "侧视图", "顶视图", "等轴测", "夹爪视图"]
        for preset in view_presets:
            btn = QPushButton(preset)
            btn.clicked.connect(lambda checked, p=preset: self.set_view_preset(p))
            btn.setFixedHeight(25)
            view_controls_layout.addWidget(btn)
        
        view_controls_layout.addStretch()
        
        left_layout.addWidget(view_controls)
        
        # 添加左侧部件到分割器
        splitter.addWidget(left_widget)
        
        # 右侧：控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建内容部件
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(10)
        
        # 仿真状态组
        self.create_simulation_status_group(content_layout)
        
        # 仿真控制组
        self.create_simulation_control_group(content_layout)
        
        # 场景管理组
        self.create_scene_management_group(content_layout)
        
        # 可视化选项组
        self.create_visualization_options_group(content_layout)
        
        # 物理参数组
        self.create_physics_parameters_group(content_layout)
        
        # 机器人状态组
        self.create_robot_status_group(content_layout)
        
        # 设置滚动区域的内容部件
        scroll_area.setWidget(content_widget)
        right_layout.addWidget(scroll_area)
        
        # 添加右侧部件到分割器
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
    
    def setup_3d_view(self):
        """设置3D视图"""
        # 清除之前的绘图
        self.ax.clear()
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 设置标题
        self.ax.set_title('仿真环境 - 机械臂与夹爪')
        
        # 设置坐标轴范围
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(0.0, 1.5)
        
        # 设置网格
        self.ax.grid(self.show_grid)
        
        # 设置视角
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        
        # 设置坐标轴比例
        self.ax.set_box_aspect([1, 1, 1])
    
    def create_simulation_status_group(self, layout):
        """创建仿真状态组"""
        group = QGroupBox("仿真状态")
        group_layout = QFormLayout(group)
        
        # 运行状态
        self.run_status_label = QLabel("停止")
        self.run_status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-weight: bold;
            }
        """)
        group_layout.addRow("运行状态:", self.run_status_label)
        
        # 仿真时间
        self.time_label = QLabel("0.00 s")
        group_layout.addRow("仿真时间:", self.time_label)
        
        # 仿真步数
        self.step_label = QLabel("0")
        group_layout.addRow("仿真步数:", self.step_label)
        
        # 实时因子
        self.realtime_factor_label = QLabel("0.0")
        group_layout.addRow("实时因子:", self.realtime_factor_label)
        
        # 碰撞次数
        self.collision_label = QLabel("0")
        group_layout.addRow("碰撞次数:", self.collision_label)
        
        # 接触点数
        self.contact_label = QLabel("0")
        group_layout.addRow("接触点数:", self.contact_label)
        
        layout.addWidget(group)
    
    def create_simulation_control_group(self, layout):
        """创建仿真控制组"""
        group = QGroupBox("仿真控制")
        group_layout = QVBoxLayout(group)
        
        # 控制按钮
        control_layout = QGridLayout()
        
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.on_start_simulation)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        control_layout.addWidget(self.start_btn, 0, 0)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.on_pause_simulation)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn, 0, 1)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.on_stop_simulation)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn, 0, 2)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.on_reset_simulation)
        control_layout.addWidget(self.reset_btn, 1, 0)
        
        self.step_btn = QPushButton("单步")
        self.step_btn.clicked.connect(self.on_step_simulation)
        control_layout.addWidget(self.step_btn, 1, 1)
        
        # 仿真速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.change_simulation_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        
        control_layout.addLayout(speed_layout, 2, 0, 1, 3)
        
        group_layout.addLayout(control_layout)
        
        # 仿真设置
        settings_layout = QFormLayout()
        
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.001, 0.1)
        self.timestep_spin.setSingleStep(0.001)
        self.timestep_spin.setValue(0.01)
        self.timestep_spin.setDecimals(3)
        settings_layout.addRow("时间步长 (s):", self.timestep_spin)
        
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(100, 1000000)
        self.max_steps_spin.setValue(10000)
        settings_layout.addRow("最大步数:", self.max_steps_spin)
        
        group_layout.addLayout(settings_layout)
        
        layout.addWidget(group)
    
    def create_scene_management_group(self, layout):
        """创建场景管理组"""
        group = QGroupBox("场景管理")
        group_layout = QVBoxLayout(group)
        
        # 场景选择
        scene_layout = QHBoxLayout()
        scene_layout.addWidget(QLabel("场景:"))
        
        self.scene_combo = QComboBox()
        self.scene_combo.addItems([
            "空场景", 
            "桌面场景", 
            "抓取测试", 
            "装配任务",
            "自定义场景"
        ])
        scene_layout.addWidget(self.scene_combo)
        
        self.load_scene_btn = QPushButton("加载")
        self.load_scene_btn.clicked.connect(self.on_load_scene)
        scene_layout.addWidget(self.load_scene_btn)
        
        group_layout.addLayout(scene_layout)
        
        # 物体管理
        obj_layout = QHBoxLayout()
        
        self.add_object_btn = QPushButton("添加物体")
        self.add_object_btn.clicked.connect(self.add_object)
        obj_layout.addWidget(self.add_object_btn)
        
        self.remove_object_btn = QPushButton("移除物体")
        self.remove_object_btn.clicked.connect(self.remove_object)
        obj_layout.addWidget(self.remove_object_btn)
        
        group_layout.addLayout(obj_layout)
        
        # 保存/加载状态
        state_layout = QHBoxLayout()
        
        self.save_state_btn = QPushButton("保存状态")
        self.save_state_btn.clicked.connect(self.on_save_state)
        state_layout.addWidget(self.save_state_btn)
        
        self.load_state_btn = QPushButton("加载状态")
        self.load_state_btn.clicked.connect(self.load_state)
        state_layout.addWidget(self.load_state_btn)
        
        group_layout.addLayout(state_layout)
        
        layout.addWidget(group)
    
    def create_visualization_options_group(self, layout):
        """创建可视化选项组"""
        group = QGroupBox("可视化选项")
        group_layout = QVBoxLayout(group)
        
        # 显示选项复选框
        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self.toggle_grid)
        group_layout.addWidget(self.show_grid_check)
        
        self.show_axes_check = QCheckBox("显示坐标轴")
        self.show_axes_check.setChecked(True)
        self.show_axes_check.stateChanged.connect(self.toggle_axes)
        group_layout.addWidget(self.show_axes_check)
        
        self.show_wireframe_check = QCheckBox("显示线框")
        self.show_wireframe_check.stateChanged.connect(self.toggle_wireframe)
        group_layout.addWidget(self.show_wireframe_check)
        
        self.show_collisions_check = QCheckBox("显示碰撞")
        self.show_collisions_check.setChecked(True)
        self.show_collisions_check.stateChanged.connect(self.toggle_collisions)
        group_layout.addWidget(self.show_collisions_check)
        
        self.show_forces_check = QCheckBox("显示力")
        self.show_forces_check.setChecked(True)
        self.show_forces_check.stateChanged.connect(self.toggle_forces)
        group_layout.addWidget(self.show_forces_check)
        
        self.show_trajectories_check = QCheckBox("显示轨迹")
        self.show_trajectories_check.setChecked(True)
        self.show_trajectories_check.stateChanged.connect(self.toggle_trajectories)
        group_layout.addWidget(self.show_trajectories_check)
        
        self.show_frames_check = QCheckBox("显示坐标系")
        self.show_frames_check.setChecked(True)
        self.show_frames_check.stateChanged.connect(self.toggle_coordinate_frames)
        group_layout.addWidget(self.show_frames_check)
        
        layout.addWidget(group)
    
    def create_physics_parameters_group(self, layout):
        """创建物理参数组"""
        group = QGroupBox("物理参数")
        group_layout = QFormLayout(group)
        
        # 重力
        self.gravity_x_spin = QDoubleSpinBox()
        self.gravity_x_spin.setRange(-20.0, 20.0)
        self.gravity_x_spin.setValue(0.0)
        group_layout.addRow("重力 X:", self.gravity_x_spin)
        
        self.gravity_y_spin = QDoubleSpinBox()
        self.gravity_y_spin.setRange(-20.0, 20.0)
        self.gravity_y_spin.setValue(0.0)
        group_layout.addRow("重力 Y:", self.gravity_y_spin)
        
        self.gravity_z_spin = QDoubleSpinBox()
        self.gravity_z_spin.setRange(-20.0, 20.0)
        self.gravity_z_spin.setValue(-9.81)
        group_layout.addRow("重力 Z:", self.gravity_z_spin)
        
        # 摩擦系数
        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 2.0)
        self.friction_spin.setValue(0.5)
        self.friction_spin.setSingleStep(0.1)
        group_layout.addRow("摩擦系数:", self.friction_spin)
        
        # 恢复系数
        self.restitution_spin = QDoubleSpinBox()
        self.restitution_spin.setRange(0.0, 1.0)
        self.restitution_spin.setValue(0.5)
        self.restitution_spin.setSingleStep(0.1)
        group_layout.addRow("恢复系数:", self.restitution_spin)
        
        # 应用按钮
        self.apply_physics_btn = QPushButton("应用物理参数")
        self.apply_physics_btn.clicked.connect(self.apply_physics_parameters)
        group_layout.addRow(self.apply_physics_btn)
        
        layout.addWidget(group)
    
    def create_robot_status_group(self, layout):
        """创建机器人状态组"""
        group = QGroupBox("机器人状态")
        group_layout = QVBoxLayout(group)
        
        # 关节角度显示
        self.joint_angles_label = QLabel("关节角度: N/A")
        self.joint_angles_label.setWordWrap(True)
        group_layout.addWidget(self.joint_angles_label)
        
        # 末端位姿显示
        self.end_effector_label = QLabel("末端位姿: N/A")
        self.end_effector_label.setWordWrap(True)
        group_layout.addWidget(self.end_effector_label)
        
        # 夹爪状态
        self.gripper_label = QLabel("夹爪开度: N/A")
        group_layout.addWidget(self.gripper_label)
        
        # 力/扭矩显示
        self.force_torque_label = QLabel("力/扭矩: N/A")
        self.force_torque_label.setWordWrap(True)
        group_layout.addWidget(self.force_torque_label)
        
        layout.addWidget(group)
    
    def set_view_preset(self, preset):
        """设置视图预设"""
        preset_views = {
            "前视图": (0, 0),
            "侧视图": (0, -90),
            "顶视图": (90, -90),
            "等轴测": (30, 45),
            "夹爪视图": (20, 60)
        }
        
        if preset in preset_views:
            self.view_elevation, self.view_azimuth = preset_views[preset]
            self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
            self.canvas.draw()
    
    def update_simulation(self):
        """更新仿真显示"""
        if not self.simulation_running or self.simulation_paused:
            return

        if self.simulator_client and self.simulator_client.running:
            # 子进程模式：只拉取状态，不在当前进程 step
            state = self.simulator_client.get_state()
            if state:
                self._sync_from_simulator_state(state)
                self.simulation_time = state.get("meta", {}).get("sim_time", self.simulation_time)
        elif self.simulator and self.simulator.running:
            # 推进 PyBullet 仿真，并同步状态到 UI
            self.simulator.step()
            self.simulation_time = self.simulator.sim_time
            self._sync_from_simulator()
        else:
            # 更新仿真时间
            self.simulation_time += self.timestep_spin.value() * self.simulation_speed
        
        # 更新显示
        self.update_3d_scene()
        
        # 更新状态显示
        self.update_status_display()
    
    def update_3d_scene(self):
        """更新3D场景"""
        self.ax.clear()
        
        # 设置视图
        self.setup_3d_view()
        
        # 绘制地面
        self.draw_ground()
        
        # 绘制机器人
        self.draw_robot()
        
        # 绘制夹爪
        self.draw_gripper()
        
        # 绘制物体
        self.draw_objects()
        
        # 绘制碰撞点
        if self.show_collisions and self.collision_points:
            self.draw_collisions()
        
        # 绘制力向量
        if self.show_forces and self.contact_forces:
            self.draw_forces()
        
        # 绘制轨迹
        if self.show_trajectories and self.trajectory_history:
            self.draw_trajectories()
        
        # 绘制坐标系
        if self.show_coordinate_frames:
            self.draw_coordinate_frames()
        
        # 更新画布
        self.canvas.draw()
    
    def draw_ground(self):
        """绘制地面"""
        # 绘制网格地面
        size = 1.0
        resolution = 10
        x = np.linspace(-size, size, resolution)
        y = np.linspace(-size, size, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # 地面颜色
        ground_color = (0.8, 0.8, 0.8, 0.5)
        
        # 绘制地面
        self.ax.plot_surface(X, Y, Z, color=ground_color, alpha=0.3)
        
        # 绘制网格线
        for i in range(resolution):
            self.ax.plot(x, np.ones_like(x) * y[i], 0, 'k-', alpha=0.2, linewidth=0.5)
            self.ax.plot(np.ones_like(y) * x[i], y, 0, 'k-', alpha=0.2, linewidth=0.5)
    
    def draw_robot(self):
        """绘制机器人"""
        # 如果没有机器人数据，绘制一个简单的机械臂
        if not self.robot_joints:
            # 生成示例关节位置
            self.generate_sample_robot()
        
        # 绘制关节（球体）
        for i, joint in enumerate(self.robot_joints):
            x, y, z = joint
            self.ax.scatter(x, y, z, color='red', s=50, alpha=0.8, label=f'关节{i}' if i == 0 else "")
        
        # 绘制连杆（线）
        if len(self.robot_joints) > 1:
            joints_array = np.array(self.robot_joints)
            self.ax.plot(joints_array[:, 0], joints_array[:, 1], joints_array[:, 2], 
                        'b-', linewidth=3, alpha=0.8, label='连杆')
    
    def draw_gripper(self):
        """绘制夹爪"""
        # 如果没有夹爪数据，绘制一个简单的夹爪模型
        if self.gripper_state is None:
            # 生成示例夹爪状态
            self.gripper_state = {
                'position': [0.5, 0.0, 0.5],  # 末端位置
                'rotation': [0, 0, 0],         # 旋转
                'opening': 0.1,                # 开度
                'fingers': 2                   # 手指数量
            }
        
        pos = self.gripper_state['position']
        opening = self.gripper_state['opening']
        fingers = self.gripper_state.get('fingers', 2)
        
        # 绘制夹爪基座
        base_size = 0.05
        x = [pos[0] - base_size/2, pos[0] + base_size/2]
        y = [pos[1] - base_size/2, pos[1] + base_size/2]
        z = [pos[2] - base_size/2, pos[2] + base_size/2]
        
        # 绘制立方体基座
        self.draw_cube(pos, base_size, color='gray', alpha=0.8)
        
        # 绘制手指
        finger_length = 0.1
        finger_width = 0.02
        
        if fingers >= 2:
            # 左手指
            finger1_pos = [
                pos[0],
                pos[1] - opening/2 - finger_width/2,
                pos[2] - finger_length/2
            ]
            self.draw_cube(finger1_pos, [finger_width, finger_width, finger_length], 
                          color='blue', alpha=0.8)
            
            # 右手指
            finger2_pos = [
                pos[0],
                pos[1] + opening/2 + finger_width/2,
                pos[2] - finger_length/2
            ]
            self.draw_cube(finger2_pos, [finger_width, finger_width, finger_length], 
                          color='blue', alpha=0.8)
        
        # 记录轨迹
        self.trajectory_history.append(pos)
        if len(self.trajectory_history) > self.max_trajectory_points:
            self.trajectory_history.pop(0)
    
    def draw_objects(self):
        """绘制物体"""
        # 如果没有物体数据，绘制一些示例物体
        if not self.objects:
            # 生成示例物体
            self.objects = [
                {'type': 'cube', 'position': [0.3, 0.2, 0.05], 'size': [0.1, 0.1, 0.1], 'color': 'green'},
                {'type': 'sphere', 'position': [-0.2, 0.3, 0.05], 'radius': 0.05, 'color': 'yellow'},
                {'type': 'cylinder', 'position': [0.1, -0.3, 0.05], 'radius': 0.04, 'height': 0.1, 'color': 'orange'}
            ]
        
        for obj in self.objects:
            obj_type = obj.get('type', 'cube')
            pos = obj.get('position', [0, 0, 0])
            color = obj.get('color', 'gray')
            
            if obj_type == 'cube':
                size = obj.get('size', [0.1, 0.1, 0.1])
                self.draw_cube(pos, size, color=color, alpha=0.7)
            
            elif obj_type == 'sphere':
                radius = obj.get('radius', 0.05)
                self.draw_sphere(pos, radius, color=color, alpha=0.7)
            
            elif obj_type == 'cylinder':
                radius = obj.get('radius', 0.04)
                height = obj.get('height', 0.1)
                self.draw_cylinder(pos, radius, height, color=color, alpha=0.7)
    
    def draw_collisions(self):
        """绘制碰撞点"""
        for collision in self.collision_points:
            pos = collision.get('position', [0, 0, 0])
            normal = collision.get('normal', [0, 0, 1])
            force = collision.get('force', 0.0)
            
            # 绘制碰撞点
            self.ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, alpha=0.7, marker='*')
            
            # 绘制法线
            normal_end = [
                pos[0] + normal[0] * 0.05,
                pos[1] + normal[1] * 0.05,
                pos[2] + normal[2] * 0.05
            ]
            self.ax.plot([pos[0], normal_end[0]], 
                        [pos[1], normal_end[1]], 
                        [pos[2], normal_end[2]], 
                        'r-', linewidth=2, alpha=0.7)
    
    def draw_forces(self):
        """绘制力向量"""
        for force_data in self.contact_forces:
            pos = force_data.get('position', [0, 0, 0])
            force = force_data.get('force', [0, 0, 0])
            magnitude = np.linalg.norm(force)
            
            if magnitude > 0.01:  # 只显示足够大的力
                # 标准化力向量并缩放
                force_dir = force / magnitude
                scaled_force = force_dir * 0.1  # 缩放以便可视化
                
                # 绘制力向量
                force_end = [
                    pos[0] + scaled_force[0],
                    pos[1] + scaled_force[1],
                    pos[2] + scaled_force[2]
                ]
                
                # 使用箭头
                arrow = Arrow3D([pos[0], force_end[0]], 
                               [pos[1], force_end[1]], 
                               [pos[2], force_end[2]], 
                               mutation_scale=15, arrowstyle='-|>', 
                               color='blue', alpha=0.8, linewidth=2)
                self.ax.add_artist(arrow)
    
    def draw_trajectories(self):
        """绘制轨迹"""
        if len(self.trajectory_history) > 1:
            traj_array = np.array(self.trajectory_history)
            self.ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                        'g-', linewidth=1, alpha=0.5, label='轨迹')
    
    def draw_coordinate_frames(self):
        """绘制坐标系"""
        # 绘制世界坐标系
        self.draw_frame([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], scale=0.2)
        
        # 绘制末端坐标系
        if self.gripper_state:
            # 简单的末端坐标系（假设与末端执行器对齐）
            self.draw_frame(self.gripper_state['position'], 
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                          scale=0.1)
    
    def draw_cube(self, center, size, color='gray', alpha=0.7):
        """绘制立方体"""
        if isinstance(size, (int, float)):
            size = [size, size, size]
        
        # 计算顶点
        x_min, x_max = center[0] - size[0]/2, center[0] + size[0]/2
        y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
        z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2
        
        # 定义立方体的8个顶点
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])
        
        # 定义立方体的6个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # 左面
        ]
        
        # 绘制面
        if self.show_wireframe:
            # 线框模式
            for face in faces:
                face_array = np.array(face)
                # 闭合多边形
                face_array = np.vstack([face_array, face_array[0]])
                self.ax.plot(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
                            color=color, alpha=alpha, linewidth=1)
        else:
            # 面模式
            for face in faces:
                poly = Poly3DCollection([face], alpha=alpha, linewidths=0)
                poly.set_facecolor(color)
                poly.set_edgecolor(color)
                self.ax.add_collection3d(poly)
    
    def draw_sphere(self, center, radius, color='yellow', alpha=0.7):
        """绘制球体"""
        # 生成球体的网格
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 绘制球体
        if self.show_wireframe:
            self.ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=1)
        else:
            self.ax.plot_surface(x, y, z, color=color, alpha=alpha)
    
    def draw_cylinder(self, center, radius, height, color='orange', alpha=0.7):
        """绘制圆柱体"""
        # 生成圆柱体的网格
        z = np.linspace(center[2] - height/2, center[2] + height/2, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x_grid = center[0] + radius * np.cos(theta_grid)
        y_grid = center[1] + radius * np.sin(theta_grid)
        
        # 绘制圆柱体侧面
        if self.show_wireframe:
            self.ax.plot_wireframe(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=1)
        else:
            self.ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)
        
        # 绘制顶部和底部
        for z_offset in [-height/2, height/2]:
            r = np.linspace(0, radius, 10)
            theta = np.linspace(0, 2 * np.pi, 20)
            r_grid, theta_grid = np.meshgrid(r, theta)
            
            x_grid = center[0] + r_grid * np.cos(theta_grid)
            y_grid = center[1] + r_grid * np.sin(theta_grid)
            z_grid = np.ones_like(x_grid) * (center[2] + z_offset)
            
            if self.show_wireframe:
                self.ax.plot_wireframe(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=1)
            else:
                self.ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)
    
    def draw_frame(self, origin, axes, scale=0.1):
        """绘制坐标系"""
        # 坐标轴颜色
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            axis = axes[i]
            end_point = [
                origin[0] + axis[0] * scale,
                origin[1] + axis[1] * scale,
                origin[2] + axis[2] * scale
            ]
            
            # 绘制坐标轴
            self.ax.plot([origin[0], end_point[0]], 
                        [origin[1], end_point[1]], 
                        [origin[2], end_point[2]], 
                        color=colors[i], linewidth=2, alpha=0.8)
            
            # 绘制坐标轴标签
            self.ax.text(end_point[0], end_point[1], end_point[2], 
                        labels[i], color=colors[i], fontsize=10)
    
    def generate_sample_robot(self):
        """生成示例机器人数据"""
        # 简单的6自由度机械臂
        base_height = 0.1
        link_lengths = [0.2, 0.3, 0.25, 0.15, 0.1, 0.05]
        
        # 起始位置（基座）
        joints = [[0, 0, base_height]]
        
        # 生成后续关节位置（简单的直线链）
        current_pos = [0, 0, base_height]
        for i, length in enumerate(link_lengths):
            # 简单的关节角度（随时间变化）
            angle = np.sin(self.simulation_time * (i+1) * 0.5) * 0.5
            
            # 计算新关节位置
            new_pos = [
                current_pos[0] + length * np.sin(angle),
                current_pos[1] + length * np.cos(angle),
                current_pos[2] + length * (i % 2) * 0.5
            ]
            joints.append(new_pos)
            current_pos = new_pos
        
        self.robot_joints = joints
    
    def update_status_display(self):
        """更新状态显示"""
        # 更新仿真时间
        self.time_label.setText(f"{self.simulation_time:.2f} s")
        
        # 更新关节角度
        if self.robot_joints:
            angles_text = "关节角度: "
            for i, joint in enumerate(self.robot_joints):
                if i < 3:  # 只显示前3个关节
                    angles_text += f"J{i}=({joint[0]:.2f}, {joint[1]:.2f}, {joint[2]:.2f}) "
            self.joint_angles_label.setText(angles_text)
        
        # 更新末端位姿
        if self.gripper_state:
            pos = self.gripper_state['position']
            self.end_effector_label.setText(
                f"末端位姿: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}"
            )
            self.gripper_label.setText(f"夹爪开度: {self.gripper_state.get('opening', 0):.3f} m")
        
        # 更新碰撞和接触计数
        self.collision_label.setText(str(len(self.collision_points)))
        self.contact_label.setText(str(len(self.contact_forces)))
        
        # 更新实时因子
        realtime_factor = self._get_realtime_factor()
        self.realtime_factor_label.setText(f"{realtime_factor:.1f}")
        if self.simulator_client:
            self.step_label.setText(str(self._external_step_count))
        elif self.simulator:
            self.step_label.setText(str(self.simulator.step_count))
        else:
            self.step_label.setText("0")
    
    # 槽函数
    def on_start_simulation(self):
        """开始仿真"""
        self.simulation_running = True
        self.simulation_paused = False
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        
        self.run_status_label.setText("运行中")
        self.run_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        # 启动 PyBullet 后端（如果可用）
        engine_cfg = self._get_config_section("ENGINE", {})
        if engine_cfg.get("mode", "gui") == "gui":
            self._start_external_gui_simulator()
        else:
            if create_simulator is None:
                logger.warning("Simulation backend not available")
                self._set_error_state()
            else:
                try:
                    self.simulator = create_simulator(self.config)
                    self.simulator.start()
                    self._sync_from_simulator()
                except Exception as exc:
                    logger.error("Failed to start simulator: %s", exc)
                    self._set_error_state()

        self.start_simulation.emit()
    
    def on_pause_simulation(self):
        """暂停仿真"""
        if self.simulation_paused:
            self.simulation_paused = False
            self.pause_btn.setText("暂停")
            self.run_status_label.setText("运行中")
            self.run_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.simulation_paused = True
            self.pause_btn.setText("继续")
            self.run_status_label.setText("暂停")
            self.run_status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        
        if self.simulator_client and self.simulator_client.running:
            if self.simulation_paused:
                self.simulator_client.pause()
            else:
                self.simulator_client.resume()

        self.pause_simulation.emit()
    
    def on_stop_simulation(self):
        """停止仿真"""
        self.simulation_running = False
        self.simulation_paused = False
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        
        self.run_status_label.setText("停止")
        self.run_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")

        if self.simulator_client:
            try:
                self.simulator_client.shutdown()
            except Exception:
                pass
            self.simulator_client = None
        if self.simulator:
            try:
                self.simulator.stop()
            except Exception:
                pass
        
        self.stop_simulation.emit()
    
    def on_reset_simulation(self):
        """重置仿真"""
        self.simulation_time = 0.0
        self.trajectory_history.clear()
        self.collision_points.clear()
        self.contact_forces.clear()

        if self.simulator_client:
            try:
                self.simulator_client.reset()
                state = self.simulator_client.get_state()
                if state:
                    self._sync_from_simulator_state(state)
            except Exception:
                pass
        if self.simulator:
            try:
                self.simulator.reset()
                self._sync_from_simulator()
            except Exception:
                pass
        
        self.update_3d_scene()
        self.update_status_display()
        
        self.reset_simulation.emit()
    
    def on_step_simulation(self):
        """单步仿真"""
        if self.simulator_client:
            self.simulator_client.step()
            state = self.simulator_client.get_state()
            if state:
                self._sync_from_simulator_state(state)
                self.simulation_time = state.get("meta", {}).get("sim_time", self.simulation_time)
            self.update_3d_scene()
            self.update_status_display()
        elif self.simulator:
            # 单步推进 PyBullet
            self.simulator.step()
            self.simulation_time = self.simulator.sim_time
            self._sync_from_simulator()
            self.update_3d_scene()
            self.update_status_display()
        elif not self.simulation_running:
            self.simulation_time += self.timestep_spin.value()
            self.update_3d_scene()
            self.update_status_display()
        
        self.step_simulation.emit()
    
    def on_load_scene(self):
        """加载场景"""
        scene_name = self.scene_combo.currentText()
        self.load_scene.emit(scene_name)
        
        # 更新场景中的物体
        self.update_scene_objects(scene_name)
    
    def on_save_state(self):
        """保存状态"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        state_name = f"state_{timestamp}"
        self.save_state.emit(state_name)
    
    def change_simulation_speed(self, value):
        """更改仿真速度"""
        # 将滑块值映射到速度倍数 (0.1x 到 10x)
        self.simulation_speed = value / 10.0
        self.speed_label.setText(f"{self.simulation_speed:.1f}x")

    def _sync_from_simulator(self):
        """从 PyBullet 读取状态并更新 UI 缓存"""
        if not self.simulator or not self.simulator.running:
            return
        state = self.simulator.get_state()
        self._sync_from_simulator_state(state)

    def _sync_from_simulator_state(self, state):
        """统一处理状态字典"""
        self.robot_joints = state.get("robot_joints", [])
        self.gripper_state = state.get("gripper_state", None)
        self.objects = state.get("objects", [])
        self.collision_points = state.get("collision_points", [])
        self.contact_forces = state.get("contact_forces", [])
        meta = state.get("meta", {})
        if isinstance(meta, dict):
            self._external_step_count = meta.get("step_count", self._external_step_count)
            self.simulation_time = meta.get("sim_time", self.simulation_time)

    def _get_realtime_factor(self) -> float:
        """根据仿真时间 / 墙钟时间估算实时因子"""
        if self.simulator and self.simulator.running:
            wall_start = getattr(self.simulator, "_wall_start", None)
            if wall_start:
                wall_elapsed = max(time.time() - wall_start, 1e-6)
                return self.simulator.sim_time / wall_elapsed
        return np.random.uniform(0.8, 1.2)

    def _start_external_gui_simulator(self):
        """启动 PyBullet GUI 子进程"""
        if PyBulletProcessClient is None:
            logger.warning("PyBullet process client not available")
            self._set_error_state()
            return

        try:
            self.simulator_client = PyBulletProcessClient(self._build_sim_config_dict())
            self.simulator_client.start()
            state = self.simulator_client.get_state()
            if state:
                self._sync_from_simulator_state(state)
        except Exception as exc:
            logger.error("Failed to start PyBullet GUI process: %s", exc)
            self.simulator_client = None
            self._set_error_state()

    def _build_sim_config_dict(self):
        """把 SimulationConfig 转成 dict，便于子进程使用"""
        cfg = {}
        for key in (
            "ENGINE",
            "SCENE",
            "ARM_SIMULATION",
            "GRIPPER_SIMULATION",
            "OBJECT_SIMULATION",
            "PHYSICS_ADVANCED",
            "VISUALIZATION",
        ):
            section = self._get_config_section(key, None)
            if isinstance(section, dict):
                cfg[key] = dict(section)

        engine = cfg.setdefault("ENGINE", {})
        engine["mode"] = "gui"
        return cfg

    def _set_error_state(self):
        """仿真启动失败时的 UI 状态回退"""
        self.simulation_running = False
        self.simulator = None
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.run_status_label.setText("错误")
        self.run_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")

    def _get_config_section(self, name, default):
        """读取配置段（仅支持类属性 dict）"""
        if self.config is None:
            return default
        section = getattr(self.config, name, default)
        if isinstance(section, dict):
            return section
        return default
    
    def toggle_grid(self, state):
        """切换网格显示"""
        self.show_grid = (state == Qt.Checked)
        self.ax.grid(self.show_grid)
        self.canvas.draw()
    
    def toggle_axes(self, state):
        """切换坐标轴显示"""
        self.show_axes = (state == Qt.Checked)
        self.ax.set_axis_on() if self.show_axes else self.ax.set_axis_off()
        self.canvas.draw()
    
    def toggle_wireframe(self, state):
        """切换线框显示"""
        self.show_wireframe = (state == Qt.Checked)
    
    def toggle_collisions(self, state):
        """切换碰撞显示"""
        self.show_collisions = (state == Qt.Checked)
    
    def toggle_forces(self, state):
        """切换力显示"""
        self.show_forces = (state == Qt.Checked)
    
    def toggle_trajectories(self, state):
        """切换轨迹显示"""
        self.show_trajectories = (state == Qt.Checked)
    
    def toggle_coordinate_frames(self, state):
        """切换坐标系显示"""
        self.show_coordinate_frames = (state == Qt.Checked)
    
    def add_object(self):
        """添加物体"""
        # 生成随机物体
        obj_types = ['cube', 'sphere', 'cylinder']
        obj_type = np.random.choice(obj_types)
        
        # 随机位置
        pos = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.05, 0.2)
        ]
        
        # 随机颜色
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        color = np.random.choice(colors)
        
        if obj_type == 'cube':
            size = [np.random.uniform(0.05, 0.15) for _ in range(3)]
            self.objects.append({
                'type': 'cube',
                'position': pos,
                'size': size,
                'color': color
            })
        elif obj_type == 'sphere':
            radius = np.random.uniform(0.03, 0.08)
            self.objects.append({
                'type': 'sphere',
                'position': pos,
                'radius': radius,
                'color': color
            })
        elif obj_type == 'cylinder':
            radius = np.random.uniform(0.02, 0.06)
            height = np.random.uniform(0.05, 0.15)
            self.objects.append({
                'type': 'cylinder',
                'position': pos,
                'radius': radius,
                'height': height,
                'color': color
            })
    
    def remove_object(self):
        """移除物体"""
        if self.objects:
            self.objects.pop()
    
    def update_scene_objects(self, scene_name):
        """根据场景名称更新物体"""
        self.objects.clear()
        
        if scene_name == "空场景":
            # 没有物体
            pass
        elif scene_name == "桌面场景":
            # 桌面和几个物体
            self.objects = [
                # 桌子
                {'type': 'cube', 'position': [0, 0, -0.05], 'size': [1.0, 1.0, 0.1], 'color': 'brown'},
                # 物体
                {'type': 'cube', 'position': [0.3, 0.2, 0.05], 'size': [0.1, 0.1, 0.1], 'color': 'green'},
                {'type': 'sphere', 'position': [-0.2, 0.3, 0.05], 'radius': 0.05, 'color': 'yellow'},
                {'type': 'cylinder', 'position': [0.1, -0.3, 0.05], 'radius': 0.04, 'height': 0.1, 'color': 'orange'}
            ]
        elif scene_name == "抓取测试":
            # 多个抓取测试物体
            self.objects = [
                # 桌子
                {'type': 'cube', 'position': [0, 0, -0.05], 'size': [1.0, 1.0, 0.1], 'color': 'brown'},
                # 测试物体
                {'type': 'cube', 'position': [0, 0, 0.05], 'size': [0.08, 0.08, 0.08], 'color': 'blue'},
                {'type': 'cube', 'position': [0.2, 0, 0.05], 'size': [0.06, 0.12, 0.04], 'color': 'red'},
                {'type': 'cylinder', 'position': [-0.2, 0, 0.05], 'radius': 0.05, 'height': 0.1, 'color': 'green'}
            ]
        elif scene_name == "装配任务":
            # 装配任务场景
            self.objects = [
                # 工作台
                {'type': 'cube', 'position': [0, 0, -0.05], 'size': [1.5, 1.0, 0.1], 'color': 'gray'},
                # 零件
                {'type': 'cube', 'position': [-0.4, 0, 0.05], 'size': [0.1, 0.1, 0.05], 'color': 'blue'},
                {'type': 'cube', 'position': [-0.2, 0, 0.05], 'size': [0.08, 0.08, 0.08], 'color': 'red'},
                {'type': 'cylinder', 'position': [0, 0, 0.05], 'radius': 0.04, 'height': 0.08, 'color': 'green'},
                # 装配位置
                {'type': 'cube', 'position': [0.4, 0, 0.05], 'size': [0.15, 0.15, 0.02], 'color': 'yellow', 'alpha': 0.3}
            ]
    
    def load_state(self):
        """加载状态（模拟）"""
        # 这里应该从文件加载状态
        print("加载仿真状态")
    
    def apply_physics_parameters(self):
        """应用物理参数"""
        # 这里应该将物理参数应用到仿真引擎
        gravity = [
            self.gravity_x_spin.value(),
            self.gravity_y_spin.value(),
            self.gravity_z_spin.value()
        ]
        print(f"应用物理参数: 重力={gravity}, 摩擦={self.friction_spin.value()}")
    
    def update_robot_state(self, joint_positions, end_effector_pose, gripper_state):
        """
        更新机器人状态
        
        Args:
            joint_positions: 关节位置列表
            end_effector_pose: 末端位姿 [x, y, z, rx, ry, rz]
            gripper_state: 夹爪状态字典
        """
        self.robot_joints = joint_positions
        self.gripper_state = gripper_state
        
        # 更新力/扭矩显示
        if 'force' in gripper_state and 'torque' in gripper_state:
            force = gripper_state['force']
            torque = gripper_state['torque']
            self.force_torque_label.setText(
                f"力/扭矩: F=({force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f}) N, "
                f"T=({torque[0]:.2f}, {torque[1]:.2f}, {torque[2]:.2f}) Nm"
            )
    
    def update_collision_data(self, collision_points, contact_forces):
        """
        更新碰撞数据
        
        Args:
            collision_points: 碰撞点列表
            contact_forces: 接触力列表
        """
        self.collision_points = collision_points
        self.contact_forces = contact_forces


if __name__ == "__main__":
    # 测试仿真显示部件
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # 创建简单的配置类用于测试
    class TestSimulationConfig:
        pass
    
    app = QApplication(sys.argv)
    
    config = TestSimulationConfig()
    viewer = SimulationViewer(config)
    viewer.setWindowTitle("仿真显示测试")
    viewer.resize(1200, 800)
    viewer.show()
    
    sys.exit(app.exec_())
