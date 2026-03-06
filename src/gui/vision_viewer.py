"""
视觉显示部件 - 用于显示相机图像、目标检测和姿态估计结果
"""

import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import time

# 导入Matplotlib组件
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - register 3D projection

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QGroupBox, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QSplitter, QTabWidget, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

try:
    from config import DemoConfig, CameraConfig
except ImportError:
    try:
        from ..config import DemoConfig, CameraConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import DemoConfig, CameraConfig
from utils.transformations import rotation_matrix_to_euler


class VisionViewer(QWidget):
    """视觉显示部件类"""
    
    # 自定义信号
    capture_request = pyqtSignal()  # 请求捕获图像
    detection_request = pyqtSignal()  # 请求目标检测
    calibration_request = pyqtSignal()  # 请求相机校准
    save_image_request = pyqtSignal(str)  # 请求保存图像
    auto_detect_toggled = pyqtSignal(bool)  # 切换实时检测
    connect_request = pyqtSignal()  # 请求连接相机
    disconnect_request = pyqtSignal()  # 请求断开相机
    self_check_request = pyqtSignal()  # 请求设备自检
    # 点云相关信号：触发显示/保存动作，由主窗口执行融合流程
    pointcloud_request = pyqtSignal()  # 请求显示点云
    pointcloud_save_request = pyqtSignal(str)  # 请求保存点云
    
    def __init__(self, config: CameraConfig, parent=None):
        """
        初始化视觉显示部件
        
        Args:
            config: 相机配置
            parent: 父部件
        """
        super().__init__(parent)
        
        # 保存配置
        self.config = config
        
        # 图像数据
        self.current_image = None  # 当前显示的图像
        self.original_image = None  # 原始图像
        self.detection_results = None  # 检测结果
        self.pose_estimations = None  # 姿态估计结果
        # 点云渲染缓存：保存融合点云的点与颜色，供 Matplotlib 绘制
        self.pointcloud_data = None  # 点云数据（points/colors）
        
        # 显示参数
        self.show_detections = True
        self.show_bounding_boxes = True
        self.show_keypoints = True
        self.show_pose_axes = True
        self.show_depth_map = True
        self.detection_confidence = 0.5
        self.auto_detect_enabled = False
        self.pose_info = None
        
        # 显示缩放
        self.image_scale = 1.0
        
        # 相机状态
        self.camera_connected = False
        self.camera_streaming = False
        self.camera_calibrated = False
        self.connect_pending = False
        
        # 初始化UI
        self.init_ui()
        
        # 设置定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 10Hz更新频率
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建标签页
        self.image_tabs = QTabWidget()
        
        # RGB图像标签页
        self.rgb_tab = QWidget()
        rgb_layout = QVBoxLayout(self.rgb_tab)
        
        # RGB图像显示标签
        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumSize(640, 480)
        self.rgb_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
        """)
        rgb_layout.addWidget(self.rgb_label)
        
        self.image_tabs.addTab(self.rgb_tab, "RGB图像")
        
        # 深度图像标签页
        self.depth_tab = QWidget()
        depth_layout = QVBoxLayout(self.depth_tab)
        
        self.depth_label = QLabel()
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(640, 480)
        self.depth_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
        """)
        depth_layout.addWidget(self.depth_label)
        
        self.image_tabs.addTab(self.depth_tab, "深度图")
        
        # 检测结果标签页
        self.detection_tab = QWidget()
        detection_layout = QVBoxLayout(self.detection_tab)
        
        self.detection_label = QLabel()
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_label.setMinimumSize(640, 480)
        self.detection_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
        """)
        detection_layout.addWidget(self.detection_label)
        
        self.image_tabs.addTab(self.detection_tab, "检测结果")
        
        # 姿态估计标签页
        self.pose_tab = QWidget()
        pose_layout = QVBoxLayout(self.pose_tab)
        
        self.pose_label = QLabel()
        self.pose_label.setAlignment(Qt.AlignCenter)
        self.pose_label.setMinimumSize(640, 480)
        self.pose_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
        """)
        pose_layout.addWidget(self.pose_label)
        
        self.image_tabs.addTab(self.pose_tab, "姿态估计")

        # 点云显示标签页（多视角融合结果）
        # 使用 Matplotlib 3D 坐标轴绘制融合点云，便于快速预览
        self.pointcloud_tab = QWidget()
        pointcloud_layout = QVBoxLayout(self.pointcloud_tab)
        # Figure/Canvas 组合：Figure 承载 3D Axes，Canvas 嵌入 Qt
        self.pointcloud_figure = Figure(figsize=(5, 4))
        self.pointcloud_canvas = FigureCanvas(self.pointcloud_figure)
        self.pointcloud_ax = self.pointcloud_figure.add_subplot(111, projection="3d")
        # 基础坐标轴与标题
        self.pointcloud_ax.set_title("融合点云")
        self.pointcloud_ax.set_xlabel("X")
        self.pointcloud_ax.set_ylabel("Y")
        self.pointcloud_ax.set_zlabel("Z")
        pointcloud_layout.addWidget(self.pointcloud_canvas)
        
        self.image_tabs.addTab(self.pointcloud_tab, "点云")
        
        left_layout.addWidget(self.image_tabs)
        
        # 图像控制工具栏
        image_controls = QWidget()
        image_controls_layout = QHBoxLayout(image_controls)
        image_controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # 缩放控制
        image_controls_layout.addWidget(QLabel("缩放:"))
        
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["25%", "50%", "100%", "150%", "200%"])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.currentTextChanged.connect(self.change_zoom)
        image_controls_layout.addWidget(self.zoom_combo)
        
        # 图像控制按钮
        self.capture_btn = QPushButton("捕获")
        self.capture_btn.clicked.connect(self.capture_image)
        image_controls_layout.addWidget(self.capture_btn)
        
        self.detect_btn = QPushButton("检测")
        self.detect_btn.clicked.connect(self.detect_objects)
        image_controls_layout.addWidget(self.detect_btn)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_image)
        image_controls_layout.addWidget(self.save_btn)
        
        image_controls_layout.addStretch()
        
        left_layout.addWidget(image_controls)
        
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
        
        # 相机状态组
        self.create_camera_status_group(content_layout)
        
        # 检测设置组
        self.create_detection_settings_group(content_layout)

        # 姿态估计信息组
        self.create_pose_info_group(content_layout)

        # 点云控制组
        self.create_pointcloud_group(content_layout)
        
        # 显示选项组
        self.create_display_options_group(content_layout)
        
        # 相机控制组
        self.create_camera_control_group(content_layout)
        
        # 设置滚动区域的内容部件
        scroll_area.setWidget(content_widget)
        right_layout.addWidget(scroll_area)
        
        # 添加右侧部件到分割器
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
    
    def create_camera_status_group(self, layout):
        """创建相机状态组"""
        group = QGroupBox("相机状态")
        group_layout = QFormLayout(group)
        
        # 连接状态
        self.connection_status_label = QLabel("未连接")
        self.connection_status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-weight: bold;
            }
        """)
        group_layout.addRow("连接状态:", self.connection_status_label)
        
        # 流状态
        self.stream_status_label = QLabel("未流式传输")
        self.stream_status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
            }
        """)
        group_layout.addRow("流状态:", self.stream_status_label)
        
        # 校准状态
        self.calibration_status_label = QLabel("未校准")
        self.calibration_status_label.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
            }
        """)
        group_layout.addRow("校准状态:", self.calibration_status_label)
        
        # 分辨率
        self.resolution_label = QLabel("N/A")
        group_layout.addRow("分辨率:", self.resolution_label)
        
        # 帧率
        self.fps_label = QLabel("0 FPS")
        group_layout.addRow("帧率:", self.fps_label)

        # 设备序列号（由自检按钮触发刷新）
        self.serial_label = QLabel("N/A")
        group_layout.addRow("序列号:", self.serial_label)

        # 深度状态（由自检按钮触发刷新）
        self.depth_status_label = QLabel("N/A")
        group_layout.addRow("深度状态:", self.depth_status_label)
        
        layout.addWidget(group)

        # 实时检测开关
        auto_group = QGroupBox("实时检测")
        auto_layout = QHBoxLayout(auto_group)
        self.auto_detect_checkbox = QCheckBox("启用实时检测")
        self.auto_detect_checkbox.setChecked(False)
        self.auto_detect_checkbox.stateChanged.connect(self.toggle_auto_detect)
        auto_layout.addWidget(self.auto_detect_checkbox)
        layout.addWidget(auto_group)
    
    def create_detection_settings_group(self, layout):
        """创建检测设置组"""
        group = QGroupBox("检测设置")
        group_layout = QVBoxLayout(group)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("检测模型:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv5", "YOLOv8", "SSD", "Faster R-CNN", "自定义模型"])
        default_model = getattr(self.config, "detection_model", "YOLOv5")
        self.model_combo.setCurrentText(default_model)
        model_layout.addWidget(self.model_combo)
        
        group_layout.addLayout(model_layout)
        
        # 置信度阈值
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度:"))
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(getattr(self.config, "confidence_threshold", 0.5))
        self.confidence_spin.valueChanged.connect(self.update_detection_confidence)
        confidence_layout.addWidget(self.confidence_spin)
        
        group_layout.addLayout(confidence_layout)
        
        # NMS阈值
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS阈值:"))
        
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(getattr(self.config, "nms_threshold", 0.5))
        nms_layout.addWidget(self.nms_spin)
        
        group_layout.addLayout(nms_layout)
        
        # 检测类别
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(QLabel("检测类别:"))
        
        self.classes_combo = QComboBox()
        self.classes_combo.addItems(["所有类别", "特定对象", "自定义"])
        classes_layout.addWidget(self.classes_combo)
        
        group_layout.addLayout(classes_layout)
        
        # 姿态估计选项
        pose_check = QCheckBox("启用6D姿态估计")
        pose_check.setChecked(True)
        group_layout.addWidget(pose_check)
        
        layout.addWidget(group)
    
    def create_display_options_group(self, layout):
        """创建显示选项组"""
        group = QGroupBox("显示选项")
        group_layout = QVBoxLayout(group)
        
        # 显示选项复选框
        self.show_bbox_check = QCheckBox("显示边界框")
        self.show_bbox_check.setChecked(True)
        self.show_bbox_check.stateChanged.connect(self.toggle_bounding_boxes)
        group_layout.addWidget(self.show_bbox_check)
        
        self.show_labels_check = QCheckBox("显示标签")
        self.show_labels_check.setChecked(True)
        group_layout.addWidget(self.show_labels_check)
        
        self.show_confidence_check = QCheckBox("显示置信度")
        self.show_confidence_check.setChecked(True)
        group_layout.addWidget(self.show_confidence_check)
        
        self.show_keypoints_check = QCheckBox("显示关键点")
        self.show_keypoints_check.setChecked(True)
        self.show_keypoints_check.stateChanged.connect(self.toggle_keypoints)
        group_layout.addWidget(self.show_keypoints_check)
        
        self.show_pose_axes_check = QCheckBox("显示姿态轴")
        self.show_pose_axes_check.setChecked(True)
        self.show_pose_axes_check.stateChanged.connect(self.toggle_pose_axes)
        group_layout.addWidget(self.show_pose_axes_check)

        self.show_depth_check = QCheckBox("显示深度图")
        self.show_depth_check.setChecked(self.show_depth_map)
        self.show_depth_check.stateChanged.connect(self.toggle_depth_map)
        group_layout.addWidget(self.show_depth_check)
        
        layout.addWidget(group)

    def create_pose_info_group(self, layout):
        """创建姿态估计信息组"""
        group = QGroupBox("姿态估计结果")
        group_layout = QFormLayout(group)

        self.pose_object_label = QLabel("N/A")
        self.pose_method_label = QLabel("N/A")
        self.pose_position_label = QLabel("N/A")
        self.pose_rotation_label = QLabel("N/A")
        self.pose_confidence_label = QLabel("N/A")

        group_layout.addRow("目标:", self.pose_object_label)
        group_layout.addRow("方法:", self.pose_method_label)
        group_layout.addRow("位置(x,y,z):", self.pose_position_label)
        group_layout.addRow("姿态(r,p,y):", self.pose_rotation_label)
        group_layout.addRow("置信度:", self.pose_confidence_label)

        layout.addWidget(group)

    def create_pointcloud_group(self, layout):
        """创建点云控制组"""
        group = QGroupBox("点云")
        group_layout = QVBoxLayout(group)

        # “显示点云”按钮：触发一次融合并刷新点云标签页
        self.show_pointcloud_btn = QPushButton("显示点云")
        self.show_pointcloud_btn.clicked.connect(self.request_pointcloud_display)
        group_layout.addWidget(self.show_pointcloud_btn)

        # “保存融合点云”按钮：由主窗口保存 PLY 文件
        self.save_pointcloud_btn = QPushButton("保存融合点云")
        self.save_pointcloud_btn.clicked.connect(self.request_pointcloud_save)
        group_layout.addWidget(self.save_pointcloud_btn)

        layout.addWidget(group)
    
    def create_camera_control_group(self, layout):
        """创建相机控制组"""
        group = QGroupBox("相机控制")
        group_layout = QVBoxLayout(group)
        
        # 连接/断开按钮
        btn_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("连接相机")
        self.connect_btn.clicked.connect(self.connect_camera)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        btn_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开相机")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        btn_layout.addWidget(self.disconnect_btn)
        
        group_layout.addLayout(btn_layout)
        
        # 流控制按钮
        stream_layout = QHBoxLayout()
        
        self.start_stream_btn = QPushButton("开始流式传输")
        self.start_stream_btn.clicked.connect(self.start_stream)
        self.start_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.start_stream_btn)
        
        self.stop_stream_btn = QPushButton("停止流式传输")
        self.stop_stream_btn.clicked.connect(self.stop_stream)
        self.stop_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.stop_stream_btn)
        
        group_layout.addLayout(stream_layout)
        
        # 校准按钮
        self.calibrate_btn = QPushButton("校准相机")
        self.calibrate_btn.clicked.connect(self.calibrate_camera)
        self.calibrate_btn.setEnabled(False)
        group_layout.addWidget(self.calibrate_btn)

        # 设备自检按钮：用于检查序列号/深度是否正常
        self.self_check_btn = QPushButton("设备自检")
        self.self_check_btn.clicked.connect(self.request_self_check)
        self.self_check_btn.setEnabled(True)
        group_layout.addWidget(self.self_check_btn)
        
        # 相机参数设置
        params_group = QGroupBox("相机参数")
        params_layout = QFormLayout(params_group)
        
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 100)
        self.brightness_spin.setValue(50)
        params_layout.addRow("亮度:", self.brightness_spin)
        
        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(0, 100)
        self.contrast_spin.setValue(50)
        params_layout.addRow("对比度:", self.contrast_spin)
        
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(-10, 10)
        self.exposure_spin.setValue(0)
        params_layout.addRow("曝光:", self.exposure_spin)
        
        group_layout.addWidget(params_group)
        
        layout.addWidget(group)
    
    @pyqtSlot()
    def update_display(self):
        """更新显示"""
        # 如果有当前图像，更新显示
        if self.current_image is not None:
            # 根据当前标签页选择显示内容
            current_tab = self.image_tabs.currentIndex()
            
            if current_tab == 0:  # RGB图像
                self.display_rgb_image()
            elif current_tab == 1:  # 深度图
                self.display_depth_image()
            elif current_tab == 2:  # 检测结果
                self.display_detection_image()
            elif current_tab == 3:  # 姿态估计
                self.display_pose_image()
            elif current_tab == 4:  # 点云
                # 点云渲染依赖 pointcloud_data（由主窗口更新）
                self.display_pointcloud()
    
    def display_rgb_image(self):
        """显示RGB图像"""
        if self.current_image is not None:
            # 调整图像大小
            display_image = self.resize_image(self.current_image)
            
            # 转换为QPixmap并显示
            pixmap = self.numpy_to_pixmap(display_image)
            self.rgb_label.setPixmap(pixmap)
    
    def display_depth_image(self):
        """显示深度图像"""
        if hasattr(self, 'depth_image') and self.depth_image is not None:
            # 归一化深度图像用于显示
            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            
            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            
            # 调整图像大小
            display_image = self.resize_image(depth_colored)
            
            # 转换为QPixmap并显示
            pixmap = self.numpy_to_pixmap(display_image)
            self.depth_label.setPixmap(pixmap)
        elif self.show_depth_map and self.current_image is not None:
            # 如果没有深度图像，显示提示
            height, width = self.current_image.shape[:2]
            text_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(text_image, "深度数据不可用", 
                       (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            display_image = self.resize_image(text_image)
            pixmap = self.numpy_to_pixmap(display_image)
            self.depth_label.setPixmap(pixmap)
    
    def display_detection_image(self):
        """显示检测结果图像"""
        if self.current_image is not None:
            # 创建副本用于绘制
            display_image = self.current_image.copy()
            
            # 如果检测结果可用，绘制检测框
            if self.detection_results is not None and self.show_bounding_boxes:
                display_image = self.draw_detections(display_image)
            
            # 调整图像大小
            display_image = self.resize_image(display_image)
            
            # 转换为QPixmap并显示
            pixmap = self.numpy_to_pixmap(display_image)
            self.detection_label.setPixmap(pixmap)
    
    def display_pose_image(self):
        """显示姿态估计图像"""
        if self.current_image is not None:
            # 创建副本用于绘制
            display_image = self.current_image.copy()
            
            # 如果姿态估计结果可用，绘制姿态
            if self.pose_estimations is not None:
                display_image = self.draw_poses(display_image)
            
            # 调整图像大小
            display_image = self.resize_image(display_image)
            
            # 转换为QPixmap并显示
            pixmap = self.numpy_to_pixmap(display_image)
            self.pose_label.setPixmap(pixmap)

    def display_pointcloud(self):
        """显示融合点云（Matplotlib 3D散点）"""
        if self.pointcloud_data is None:
            # 未收到点云数据时，显示提示信息
            self.pointcloud_ax.clear()
            self.pointcloud_ax.set_title("暂无点云数据")
            self.pointcloud_canvas.draw_idle()
            return

        points = self.pointcloud_data.get("points")
        colors = self.pointcloud_data.get("colors")
        if points is None or len(points) == 0:
            # 点云为空时直接提示
            self.pointcloud_ax.clear()
            self.pointcloud_ax.set_title("点云为空")
            self.pointcloud_canvas.draw_idle()
            return

        # 清空并绘制新的点云散点
        self.pointcloud_ax.clear()
        if colors is not None and len(colors) == len(points):
            # 使用每个点的颜色渲染
            self.pointcloud_ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                       c=colors, s=1, depthshade=False)
        else:
            # 没有颜色信息时使用统一颜色
            self.pointcloud_ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                       c="tab:blue", s=1, depthshade=False)

        # 设置坐标轴与视角
        self.pointcloud_ax.set_title("融合点云")
        self.pointcloud_ax.set_xlabel("X")
        self.pointcloud_ax.set_ylabel("Y")
        self.pointcloud_ax.set_zlabel("Z")
        self.pointcloud_ax.view_init(elev=20, azim=45)
        self.pointcloud_canvas.draw_idle()
    
    def draw_detections(self, image):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            
        Returns:
            绘制了检测结果的图像
        """
        # 这里应该根据实际的检测结果格式进行绘制
        # 假设detection_results是一个包含边界框、类别、置信度的列表
        
        if not self.detection_results:
            return image
        
        for detection in self.detection_results:
            # 提取检测信息
            bbox = detection.get('bbox', [])  # [x1, y1, x2, y2]
            label = detection.get('label', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            color = detection.get('color', (0, 255, 0))  # BGR格式
            
            # 检查置信度阈值
            if confidence < self.detection_confidence:
                continue
            
            # 绘制边界框
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if self.show_labels_check.isChecked():
                    label_text = f"{label}: {confidence:.2f}" if self.show_confidence_check.isChecked() else label
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 标签背景
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # 标签文本
                    cv2.putText(image, label_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def draw_poses(self, image):
        """
        在图像上绘制姿态估计结果
        
        Args:
            image: 输入图像
            
        Returns:
            绘制了姿态估计结果的图像
        """
        if not self.pose_estimations:
            return image
        
        for pose in self.pose_estimations:
            # 提取姿态信息
            keypoints = pose.get('keypoints', [])  # 关键点列表 [x1, y1, x2, y2, ...]
            bbox = pose.get('bbox', [])  # 边界框
            rotation = pose.get('rotation', None)  # 旋转矩阵
            translation = pose.get('translation', None)  # 平移向量
            
            # 绘制关键点
            if self.show_keypoints and keypoints:
                for i in range(0, len(keypoints), 2):
                    if i+1 < len(keypoints):
                        x, y = int(keypoints[i]), int(keypoints[i+1])
                        if x > 0 and y > 0:
                            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # 绘制姿态轴
            if self.show_pose_axes and rotation is not None and translation is not None:
                # 简化：在边界框中心绘制坐标轴
                if len(bbox) == 4:
                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)
                    
                    # 绘制坐标轴（简化为固定长度）
                    axis_length = 30
                    # X轴 (红色)
                    cv2.line(image, (x_center, y_center), 
                            (x_center + axis_length, y_center), (0, 0, 255), 2)
                    # Y轴 (绿色)
                    cv2.line(image, (x_center, y_center), 
                            (x_center, y_center - axis_length), (0, 255, 0), 2)
                    # Z轴 (蓝色)
                    cv2.line(image, (x_center, y_center), 
                            (x_center, y_center + axis_length), (255, 0, 0), 2)
        
        return image
    
    def resize_image(self, image):
        """
        调整图像大小
        
        Args:
            image: 输入图像
            
        Returns:
            调整大小后的图像
        """
        if image is None:
            return None
        
        # 获取缩放比例
        scale_percent = self.image_scale * 100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        
        # 调整大小
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized
    
    def numpy_to_pixmap(self, image):
        """
        将numpy数组转换为QPixmap
        
        Args:
            image: numpy数组图像
            
        Returns:
            QPixmap对象
        """
        if image is None:
            return QPixmap()
        
        # 转换颜色空间（如果需要）
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 强制为连续内存 + uint8，避免 QImage 缓冲区报错
            rgb_image = np.require(rgb_image, dtype=np.uint8, requirements=["C"])
            height, width, _ = rgb_image.shape
            bytes_per_line = rgb_image.strides[0]

            # 使用 bytes 构建 QImage，并 copy 以确保内部数据自持
            buffer = rgb_image.tobytes()
            q_image = QImage(buffer, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(q_image)
        elif len(image.shape) == 2:
            # 灰度图像
            gray_image = np.require(image, dtype=np.uint8, requirements=["C"])
            height, width = gray_image.shape
            bytes_per_line = gray_image.strides[0]
            buffer = gray_image.tobytes()
            q_image = QImage(buffer, width, height, bytes_per_line, QImage.Format_Grayscale8).copy()
            return QPixmap.fromImage(q_image)
        
        return QPixmap()
    
    def update_image(self, image, image_type="rgb"):
        """
        更新图像数据
        
        Args:
            image: 新图像
            image_type: 图像类型 ("rgb", "depth", "detection", "pose")
        """
        if image_type == "rgb":
            self.current_image = image
            self.original_image = image.copy()
        elif image_type == "depth":
            self.depth_image = image
        elif image_type == "detection":
            self.detection_results = image
        elif image_type == "pose":
            self.pose_estimations = image
            self.update_pose_info()
        elif image_type == "pointcloud":
            # 点云数据由主窗口融合后推送
            self.pointcloud_data = image

    def update_pose_info(self):
        """更新姿态信息显示"""
        if not self.pose_estimations:
            self.pose_object_label.setText("N/A")
            self.pose_method_label.setText("N/A")
            self.pose_position_label.setText("N/A")
            self.pose_rotation_label.setText("N/A")
            self.pose_confidence_label.setText("N/A")
            return

        pose = self.pose_estimations[0]
        obj = pose.get("object_class", "object")
        method = pose.get("method", "stub")
        conf = pose.get("confidence", 0.0)
        translation = pose.get("translation")
        rotation = pose.get("rotation")

        self.pose_object_label.setText(str(obj))
        self.pose_method_label.setText(str(method))
        if translation is not None and hasattr(translation, "__len__") and len(translation) >= 3:
            self.pose_position_label.setText(f"{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}")
        else:
            self.pose_position_label.setText("N/A")

        if rotation is not None:
            try:
                euler = rotation_matrix_to_euler(rotation)
                roll, pitch, yaw = np.degrees(euler)
                self.pose_rotation_label.setText(f"{roll:.1f}, {pitch:.1f}, {yaw:.1f}")
            except Exception:
                self.pose_rotation_label.setText("N/A")
        else:
            self.pose_rotation_label.setText("N/A")

        self.pose_confidence_label.setText(f"{conf:.2f}")
    
    def update_camera_status(self, connected=False, streaming=False, calibrated=False, 
                            resolution="N/A", fps=0):
        """
        更新相机状态
        
        Args:
            connected: 是否连接
            streaming: 是否在流式传输
            calibrated: 是否已校准
            resolution: 分辨率
            fps: 帧率
        """
        self.camera_connected = connected
        self.camera_streaming = streaming
        self.camera_calibrated = calibrated
        
        # 更新状态标签
        if connected:
            self.connection_status_label.setText("已连接")
            self.connection_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.connection_status_label.setText("未连接")
            self.connection_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            self.connect_pending = False  # 重置挂起状态
        
        if streaming:
            self.stream_status_label.setText("流式传输中")
            self.stream_status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.stream_status_label.setText("未流式传输")
            self.stream_status_label.setStyleSheet("color: #ff6b6b;")
        
        if calibrated:
            self.calibration_status_label.setText("已校准")
            self.calibration_status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.calibration_status_label.setText("未校准")
            self.calibration_status_label.setStyleSheet("color: #ff6b6b;")
        
        self.resolution_label.setText(resolution)
        self.fps_label.setText(f"{fps} FPS")

        # 断开时清空自检结果，避免显示过期信息
        if not connected:
            self.serial_label.setText("N/A")
            self.depth_status_label.setText("N/A")
            self.depth_status_label.setStyleSheet("color: #ff6b6b;")
        
        # 更新按钮状态
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.start_stream_btn.setEnabled(connected and not streaming)
        self.stop_stream_btn.setEnabled(connected and streaming)
        self.calibrate_btn.setEnabled(connected)
        self.capture_btn.setEnabled(streaming)
        self.detect_btn.setEnabled(streaming)
    
    # 槽函数
    def change_zoom(self, zoom_text):
        """更改缩放级别"""
        scale_map = {
            "25%": 0.25,
            "50%": 0.5,
            "100%": 1.0,
            "150%": 1.5,
            "200%": 2.0
        }
        self.image_scale = scale_map.get(zoom_text, 1.0)
    
    def capture_image(self):
        """捕获图像"""
        self.capture_request.emit()
    
    def detect_objects(self):
        """检测对象"""
        self.detection_request.emit()
    
    def toggle_auto_detect(self, state):
        """切换实时检测"""
        enabled = state == Qt.Checked
        self.auto_detect_enabled = enabled
        self.auto_detect_toggled.emit(enabled)
    
    def save_image(self):
        """保存图像"""
        # 这里可以添加文件对话框来选择保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        self.save_image_request.emit(filename)
    
    def request_pointcloud_display(self):
        """请求显示点云（触发一次融合并切换到点云页）"""
        # 切换到点云标签页，便于用户直接看到渲染结果
        if self.image_tabs:
            self.image_tabs.setCurrentWidget(self.pointcloud_tab)
        # 通知主窗口执行融合与渲染
        self.pointcloud_request.emit()

    def request_pointcloud_save(self):
        """请求保存点云（默认保存为 PLY）"""
        # 生成带时间戳的文件名，交由主窗口保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_{timestamp}.ply"
        self.pointcloud_save_request.emit(filename)

    def connect_camera(self):
        """连接相机"""
        # 将请求抛给上层，由上层尝试实际连接并回调 update_camera_status
        self.connect_pending = True
        self.connect_request.emit()
        # 如果上层未在超时时间内设置为连接状态，显示失败提示
        QTimer.singleShot(1500, self._check_connect_result)
    
    def disconnect_camera(self):
        """断开相机连接"""
        self.disconnect_request.emit()

    def request_self_check(self):
        """触发设备自检（序列号/深度状态）"""
        self.self_check_request.emit()
    
    def start_stream(self):
        """开始流式传输"""
        # 仅由上层驱动开始流媒体，避免误标状态
        pass
    
    def stop_stream(self):
        """停止流式传输"""
        pass

    def show_connection_error(self, message: str = "连接失败"):
        """显示连接失败状态/提示"""
        self.update_camera_status(connected=False, streaming=False, calibrated=False,
                                  resolution="N/A", fps=0)
        try:
            QMessageBox.warning(self, "相机连接", message)
        except Exception:
            # 无法弹窗时直接更新状态标签
            self.connection_status_label.setText(message)
            self.connection_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")

    def _check_connect_result(self):
        """在连接超时后提示失败"""
        if self.connect_pending and not self.camera_connected:
            self.connect_pending = False
            self.show_connection_error("相机连接失败，请检查USB或配置")
    
    def calibrate_camera(self):
        """校准相机"""
        self.calibration_request.emit()
    
    def toggle_bounding_boxes(self, state):
        """切换边界框显示"""
        self.show_bounding_boxes = (state == Qt.Checked)
    
    def toggle_keypoints(self, state):
        """切换关键点显示"""
        self.show_keypoints = (state == Qt.Checked)
    
    def toggle_pose_axes(self, state):
        """切换姿态轴显示"""
        self.show_pose_axes = (state == Qt.Checked)
    
    def toggle_depth_map(self, state):
        """切换深度图显示"""
        self.show_depth_map = (state == Qt.Checked)
        if self.show_depth_map and not hasattr(self, 'depth_image'):
            # 如果没有深度图像，生成模拟深度图
            if self.current_image is not None:
                height, width = self.current_image.shape[:2]
                # 创建简单的模拟深度图（中间近，边缘远）
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                center_x, center_y = width // 2, height // 2
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                self.depth_image = (1 - distances / max_dist) * 255
                self.depth_image = self.depth_image.astype(np.uint8)
    
    def update_detection_confidence(self, value):
        """更新检测置信度阈值"""
        self.detection_confidence = value

    def get_detection_settings(self) -> Dict[str, Any]:
        model_text = self.model_combo.currentText() if self.model_combo else "YOLOv5"
        model_map = {
            "YOLOv5": "yolov5",
            "YOLOv8": "yolov8",
            "SSD": "ssd",
            "Faster R-CNN": "faster_rcnn",
            "自定义模型": "custom",
        }
        return {
            "model_name": model_map.get(model_text, "yolov5"),
            "confidence_threshold": float(self.confidence_spin.value()) if self.confidence_spin else 0.5,
            "iou_threshold": float(self.nms_spin.value()) if self.nms_spin else 0.45,
        }

    def update_self_check_result(self, serial_text: str, depth_status: str, depth_ok: Optional[bool] = None):
        """
        更新自检结果显示。

        Args:
            serial_text: 设备序列号/名称组合文本
            depth_status: 深度状态文本
            depth_ok: 是否正常（True/False/None），用于着色
        """
        self.serial_label.setText(serial_text or "N/A")
        self.depth_status_label.setText(depth_status or "N/A")
        if depth_ok is True:
            self.depth_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif depth_ok is False:
            self.depth_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        else:
            self.depth_status_label.setStyleSheet("color: #999999;")


if __name__ == "__main__":
    # 测试视觉显示部件
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # 创建简单的配置类用于测试
    class TestCameraConfig:
        detection_model = "YOLOv5"
        confidence_threshold = 0.5
        nms_threshold = 0.4
        enable_pose_estimation = True
    
    app = QApplication(sys.argv)
    
    config = TestCameraConfig()
    viewer = VisionViewer(config)
    viewer.setWindowTitle("视觉显示测试")
    viewer.resize(1200, 800)
    viewer.show()
    
    # 加载测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    viewer.update_image(test_image)
    
    sys.exit(app.exec_())
