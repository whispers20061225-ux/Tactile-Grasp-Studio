"""
对话框模块 - 包含系统使用的各种对话框
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QProgressBar, QProgressDialog, QApplication,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QDialogButtonBox, QGridLayout, QSplitter, QScrollArea,
    QSlider, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QThread
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap

try:
    from config import DemoConfig
except ImportError:
    try:
        from ..config import DemoConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import DemoConfig


class ConfigDialog(QDialog):
    """配置对话框"""
    
    def __init__(self, config: DemoConfig, parent=None):
        """
        初始化配置对话框
        
        Args:
            config: 系统配置
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.config = config
        self.modified_config = DemoConfig()  # 修改后的配置副本
        
        # 初始化UI
        self.init_ui()
        
        # 加载配置
        self.load_config()
        
        # 设置窗口属性
        self.setWindowTitle("系统配置")
        self.resize(800, 600)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 硬件配置标签页
        self.hardware_tab = self.create_hardware_tab()
        self.tab_widget.addTab(self.hardware_tab, "硬件配置")
        
        # 算法配置标签页
        self.algorithm_tab = self.create_algorithm_tab()
        self.tab_widget.addTab(self.algorithm_tab, "算法配置")
        
        # UI配置标签页
        self.ui_tab = self.create_ui_tab()
        self.tab_widget.addTab(self.ui_tab, "界面配置")
        
        # 通信配置标签页
        self.communication_tab = self.create_communication_tab()
        self.tab_widget.addTab(self.communication_tab, "通信配置")
        
        # 数据配置标签页
        self.data_tab = self.create_data_tab()
        self.tab_widget.addTab(self.data_tab, "数据配置")
        
        layout.addWidget(self.tab_widget)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_btn)
        
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_config)
        button_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_config)
        button_layout.addWidget(self.reset_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def create_hardware_tab(self):
        """创建硬件配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 传感器配置组
        sensor_group = QGroupBox("触觉传感器配置")
        sensor_layout = QFormLayout(sensor_group)
        
        self.sensor_type_combo = QComboBox()
        self.sensor_type_combo.addItems(["Paxini Gen3 M2020", "模拟传感器", "自定义"])
        sensor_layout.addRow("传感器类型:", self.sensor_type_combo)
        
        self.sensor_port_edit = QLineEdit()
        sensor_layout.addRow("串口端口:", self.sensor_port_edit)
        
        self.sensor_baud_spin = QSpinBox()
        self.sensor_baud_spin.setRange(9600, 115200)
        self.sensor_baud_spin.setSingleStep(9600)
        sensor_layout.addRow("波特率:", self.sensor_baud_spin)
        
        self.sensor_rate_spin = QSpinBox()
        self.sensor_rate_spin.setRange(1, 1000)
        self.sensor_rate_spin.setSingleStep(10)
        sensor_layout.addRow("采样率 (Hz):", self.sensor_rate_spin)
        
        content_layout.addWidget(sensor_group)
        
        # 舵机配置组
        servo_group = QGroupBox("舵机配置")
        servo_layout = QFormLayout(servo_group)
        
        self.servo_type_combo = QComboBox()
        self.servo_type_combo.addItems(["ST3215-C018", "模拟舵机", "自定义"])
        servo_layout.addRow("舵机类型:", self.servo_type_combo)
        
        self.servo_port_edit = QLineEdit()
        servo_layout.addRow("串口端口:", self.servo_port_edit)
        
        self.servo_id_spin = QSpinBox()
        self.servo_id_spin.setRange(1, 254)
        servo_layout.addRow("舵机ID:", self.servo_id_spin)
        
        self.servo_min_spin = QSpinBox()
        self.servo_min_spin.setRange(0, 180)
        servo_layout.addRow("最小角度:", self.servo_min_spin)
        
        self.servo_max_spin = QSpinBox()
        self.servo_max_spin.setRange(0, 180)
        servo_layout.addRow("最大角度:", self.servo_max_spin)
        
        self.servo_speed_spin = QSpinBox()
        self.servo_speed_spin.setRange(1, 100)
        servo_layout.addRow("速度:", self.servo_speed_spin)
        
        content_layout.addWidget(servo_group)
        
        # 机械臂配置组
        arm_group = QGroupBox("机械臂配置")
        arm_layout = QFormLayout(arm_group)
        
        self.arm_type_combo = QComboBox()
        self.arm_type_combo.addItems(["LEArm", "UR5", "Franka", "自定义"])
        arm_layout.addRow("机械臂类型:", self.arm_type_combo)
        
        self.arm_ip_edit = QLineEdit()
        arm_layout.addRow("IP地址:", self.arm_ip_edit)
        
        self.arm_port_spin = QSpinBox()
        self.arm_port_spin.setRange(1, 65535)
        arm_layout.addRow("端口:", self.arm_port_spin)
        
        content_layout.addWidget(arm_group)
        
        # 相机配置组
        camera_group = QGroupBox("相机配置")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(["RealSense", "USB相机", "模拟相机", "自定义"])
        camera_layout.addRow("相机类型:", self.camera_type_combo)
        
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        camera_layout.addRow("相机ID:", self.camera_id_spin)
        
        self.camera_width_spin = QSpinBox()
        self.camera_width_spin.setRange(320, 3840)
        camera_layout.addRow("宽度:", self.camera_width_spin)
        
        self.camera_height_spin = QSpinBox()
        self.camera_height_spin.setRange(240, 2160)
        camera_layout.addRow("高度:", self.camera_height_spin)
        
        content_layout.addWidget(camera_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def create_algorithm_tab(self):
        """创建算法配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 深度学习配置组
        dl_group = QGroupBox("深度学习配置")
        dl_layout = QFormLayout(dl_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("模型文件路径")
        dl_layout.addRow("模型路径:", self.model_path_edit)
        
        self.model_browse_btn = QPushButton("浏览...")
        self.model_browse_btn.clicked.connect(self.browse_model_path)
        dl_layout.addRow("", self.model_browse_btn)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        dl_layout.addRow("置信度阈值:", self.confidence_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        dl_layout.addRow("批处理大小:", self.batch_size_spin)
        
        self.use_gpu_check = QCheckBox("使用GPU加速")
        dl_layout.addRow(self.use_gpu_check)
        
        content_layout.addWidget(dl_group)
        
        # PID控制配置组
        pid_group = QGroupBox("PID控制配置")
        pid_layout = QFormLayout(pid_group)
        
        self.kp_spin = QDoubleSpinBox()
        self.kp_spin.setRange(0.0, 10.0)
        self.kp_spin.setSingleStep(0.1)
        pid_layout.addRow("比例系数 Kp:", self.kp_spin)
        
        self.ki_spin = QDoubleSpinBox()
        self.ki_spin.setRange(0.0, 10.0)
        self.ki_spin.setSingleStep(0.01)
        pid_layout.addRow("积分系数 Ki:", self.ki_spin)
        
        self.kd_spin = QDoubleSpinBox()
        self.kd_spin.setRange(0.0, 10.0)
        self.kd_spin.setSingleStep(0.01)
        pid_layout.addRow("微分系数 Kd:", self.kd_spin)
        
        content_layout.addWidget(pid_group)
        
        # 滤波配置组
        filter_group = QGroupBox("滤波配置")
        filter_layout = QFormLayout(filter_group)
        
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["无", "低通滤波", "中值滤波", "卡尔曼滤波"])
        filter_layout.addRow("滤波类型:", self.filter_type_combo)
        
        self.filter_cutoff_spin = QDoubleSpinBox()
        self.filter_cutoff_spin.setRange(0.1, 100.0)
        filter_layout.addRow("截止频率 (Hz):", self.filter_cutoff_spin)
        
        content_layout.addWidget(filter_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def create_ui_tab(self):
        """创建UI配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 界面显示组
        display_group = QGroupBox("界面显示配置")
        display_layout = QFormLayout(display_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色主题", "深色主题", "科技主题", "自定义"])
        display_layout.addRow("主题:", self.theme_combo)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "英文", "日语", "韩语"])
        display_layout.addRow("语言:", self.language_combo)
        
        self.refresh_rate_spin = QSpinBox()
        self.refresh_rate_spin.setRange(1, 120)
        display_layout.addRow("刷新率 (Hz):", self.refresh_rate_spin)
        
        self.auto_refresh_check = QCheckBox("自动刷新")
        display_layout.addRow(self.auto_refresh_check)
        
        content_layout.addWidget(display_group)
        
        # 窗口配置组
        window_group = QGroupBox("窗口配置")
        window_layout = QFormLayout(window_group)
        
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 3840)
        window_layout.addRow("窗口宽度:", self.window_width_spin)
        
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 2160)
        window_layout.addRow("窗口高度:", self.window_height_spin)
        
        self.fullscreen_check = QCheckBox("全屏模式")
        window_layout.addRow(self.fullscreen_check)
        
        content_layout.addWidget(window_group)
        
        # 图表配置组
        chart_group = QGroupBox("图表配置")
        chart_layout = QFormLayout(chart_group)
        
        self.chart_history_spin = QSpinBox()
        self.chart_history_spin.setRange(100, 10000)
        chart_layout.addRow("历史数据点数:", self.chart_history_spin)
        
        self.chart_smoothing_check = QCheckBox("启用平滑")
        chart_layout.addRow(self.chart_smoothing_check)
        
        self.chart_animation_check = QCheckBox("启用动画")
        chart_layout.addRow(self.chart_animation_check)
        
        content_layout.addWidget(chart_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def create_communication_tab(self):
        """创建通信配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # ROS配置组
        ros_group = QGroupBox("ROS配置")
        ros_layout = QFormLayout(ros_group)
        
        self.ros_enable_check = QCheckBox("启用ROS")
        ros_layout.addRow(self.ros_enable_check)
        
        self.ros_master_edit = QLineEdit()
        self.ros_master_edit.setPlaceholderText("http://localhost:11311")
        ros_layout.addRow("ROS Master:", self.ros_master_edit)
        
        self.ros_namespace_edit = QLineEdit()
        self.ros_namespace_edit.setPlaceholderText("/tactile_gripper")
        ros_layout.addRow("命名空间:", self.ros_namespace_edit)
        
        content_layout.addWidget(ros_group)
        
        # MQTT配置组
        mqtt_group = QGroupBox("MQTT配置")
        mqtt_layout = QFormLayout(mqtt_group)
        
        self.mqtt_enable_check = QCheckBox("启用MQTT")
        mqtt_layout.addRow(self.mqtt_enable_check)
        
        self.mqtt_broker_edit = QLineEdit()
        self.mqtt_broker_edit.setPlaceholderText("localhost")
        mqtt_layout.addRow("MQTT Broker:", self.mqtt_broker_edit)
        
        self.mqtt_port_spin = QSpinBox()
        self.mqtt_port_spin.setRange(1, 65535)
        self.mqtt_port_spin.setValue(1883)
        mqtt_layout.addRow("端口:", self.mqtt_port_spin)
        
        self.mqtt_topic_edit = QLineEdit()
        self.mqtt_topic_edit.setPlaceholderText("tactile_gripper/data")
        mqtt_layout.addRow("主题:", self.mqtt_topic_edit)
        
        content_layout.addWidget(mqtt_group)
        
        # API配置组
        api_group = QGroupBox("API配置")
        api_layout = QFormLayout(api_group)
        
        self.api_enable_check = QCheckBox("启用REST API")
        api_layout.addRow(self.api_enable_check)
        
        self.api_host_edit = QLineEdit()
        self.api_host_edit.setPlaceholderText("0.0.0.0")
        api_layout.addRow("主机:", self.api_host_edit)
        
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1, 65535)
        self.api_port_spin.setValue(8080)
        api_layout.addRow("端口:", self.api_port_spin)
        
        content_layout.addWidget(api_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def create_data_tab(self):
        """创建数据配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 数据记录组
        recording_group = QGroupBox("数据记录配置")
        recording_layout = QFormLayout(recording_group)
        
        self.auto_record_check = QCheckBox("自动记录")
        recording_layout.addRow(self.auto_record_check)
        
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("数据保存目录")
        recording_layout.addRow("数据目录:", self.data_dir_edit)
        
        self.data_browse_btn = QPushButton("浏览...")
        self.data_browse_btn.clicked.connect(self.browse_data_dir)
        recording_layout.addRow("", self.data_browse_btn)
        
        self.data_format_combo = QComboBox()
        self.data_format_combo.addItems(["CSV", "JSON", "HDF5", "二进制"])
        recording_layout.addRow("数据格式:", self.data_format_combo)
        
        self.max_file_size_spin = QSpinBox()
        self.max_file_size_spin.setRange(1, 10000)
        self.max_file_size_spin.setSuffix(" MB")
        recording_layout.addRow("最大文件大小:", self.max_file_size_spin)
        
        content_layout.addWidget(recording_group)
        
        # 数据集配置组
        dataset_group = QGroupBox("数据集配置")
        dataset_layout = QFormLayout(dataset_group)
        
        self.dataset_split_spin = QDoubleSpinBox()
        self.dataset_split_spin.setRange(0.0, 1.0)
        self.dataset_split_spin.setSingleStep(0.05)
        self.dataset_split_spin.setValue(0.8)
        dataset_layout.addRow("训练集比例:", self.dataset_split_spin)
        
        self.augmentation_check = QCheckBox("数据增强")
        dataset_layout.addRow(self.augmentation_check)
        
        self.normalization_check = QCheckBox("数据归一化")
        dataset_layout.addRow(self.normalization_check)
        
        content_layout.addWidget(dataset_group)
        
        # 导出配置组
        export_group = QGroupBox("导出配置")
        export_layout = QFormLayout(export_group)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["CSV", "JSON", "Excel", "MATLAB"])
        export_layout.addRow("导出格式:", self.export_format_combo)
        
        self.include_timestamp_check = QCheckBox("包含时间戳")
        export_layout.addRow(self.include_timestamp_check)
        
        self.include_metadata_check = QCheckBox("包含元数据")
        export_layout.addRow(self.include_metadata_check)
        
        content_layout.addWidget(export_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def load_config(self):
        """加载配置到UI"""
        # 硬件配置
        if hasattr(self.config, 'hardware'):
            hardware = self.config.hardware
            
            # 传感器配置
            if hasattr(hardware, 'sensor'):
                sensor = hardware.sensor
                self.sensor_port_edit.setText(getattr(sensor, 'port', 'COM3'))
                self.sensor_baud_spin.setValue(getattr(sensor, 'baud_rate', 115200))
                self.sensor_rate_spin.setValue(getattr(sensor, 'sampling_rate', 100))
            
            # 舵机配置
            if hasattr(hardware, 'servo'):
                servo = hardware.servo
                self.servo_port_edit.setText(getattr(servo, 'port', 'COM4'))
                self.servo_id_spin.setValue(getattr(servo, 'id', 1))
                self.servo_min_spin.setValue(getattr(servo, 'min_angle', 0))
                self.servo_max_spin.setValue(getattr(servo, 'max_angle', 180))
                self.servo_speed_spin.setValue(getattr(servo, 'speed', 50))
        
        # 算法配置
        if hasattr(self.config, 'algorithm'):
            algorithm = self.config.algorithm
            self.confidence_spin.setValue(getattr(algorithm, 'confidence_threshold', 0.7))
            self.batch_size_spin.setValue(getattr(algorithm, 'batch_size', 32))
            self.use_gpu_check.setChecked(getattr(algorithm, 'use_gpu', False))
            
            # PID配置
            if hasattr(algorithm, 'pid'):
                pid = algorithm.pid
                self.kp_spin.setValue(getattr(pid, 'kp', 1.0))
                self.ki_spin.setValue(getattr(pid, 'ki', 0.1))
                self.kd_spin.setValue(getattr(pid, 'kd', 0.01))
        
        # UI配置
        if hasattr(self.config, 'ui'):
            ui = self.config.ui
            self.window_width_spin.setValue(getattr(ui, 'window_width', 1200))
            self.window_height_spin.setValue(getattr(ui, 'window_height', 800))
            self.refresh_rate_spin.setValue(getattr(ui, 'refresh_rate', 30))
            self.fullscreen_check.setChecked(getattr(ui, 'fullscreen', False))
            
            # 图表配置
            if hasattr(ui, 'chart'):
                chart = ui.chart
                self.chart_history_spin.setValue(getattr(chart, 'history_points', 1000))
                self.chart_smoothing_check.setChecked(getattr(chart, 'smoothing', True))
                self.chart_animation_check.setChecked(getattr(chart, 'animation', True))
    
    def save_config(self):
        """保存配置"""
        try:
            # 更新配置对象
            self.update_config_from_ui()
            
            # 保存到文件
            config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
            os.makedirs(config_dir, exist_ok=True)
            
            config_path = os.path.join(config_dir, "current_config.yaml")
            self.modified_config.save(config_path)
            
            QMessageBox.information(self, "保存成功", f"配置已保存到 {config_path}")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存配置时出错: {str(e)}")
    
    def apply_config(self):
        """应用配置"""
        try:
            self.update_config_from_ui()
            QMessageBox.information(self, "应用成功", "配置已应用")
        except Exception as e:
            QMessageBox.critical(self, "应用失败", f"应用配置时出错: {str(e)}")
    
    def reset_config(self):
        """重置配置"""
        reply = QMessageBox.question(
            self, "重置配置",
            "是否重置所有配置为默认值？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.modified_config = DemoConfig()
            self.load_config()
    
    def update_config_from_ui(self):
        """从UI更新配置对象"""
        # 硬件配置
        if not hasattr(self.modified_config, 'hardware'):
            class Hardware:
                class sensor:
                    pass
                class servo:
                    pass
                class camera:
                    pass
            
            self.modified_config.hardware = Hardware()
        
        # 传感器配置
        self.modified_config.hardware.sensor.port = self.sensor_port_edit.text()
        self.modified_config.hardware.sensor.baud_rate = self.sensor_baud_spin.value()
        self.modified_config.hardware.sensor.sampling_rate = self.sensor_rate_spin.value()
        
        # 舵机配置
        self.modified_config.hardware.servo.port = self.servo_port_edit.text()
        self.modified_config.hardware.servo.id = self.servo_id_spin.value()
        self.modified_config.hardware.servo.min_angle = self.servo_min_spin.value()
        self.modified_config.hardware.servo.max_angle = self.servo_max_spin.value()
        self.modified_config.hardware.servo.speed = self.servo_speed_spin.value()
        
        # 算法配置
        if not hasattr(self.modified_config, 'algorithm'):
            class Algorithm:
                class pid:
                    pass
            
            self.modified_config.algorithm = Algorithm()
        
        self.modified_config.algorithm.confidence_threshold = self.confidence_spin.value()
        self.modified_config.algorithm.batch_size = self.batch_size_spin.value()
        self.modified_config.algorithm.use_gpu = self.use_gpu_check.isChecked()
        
        # PID配置
        self.modified_config.algorithm.pid.kp = self.kp_spin.value()
        self.modified_config.algorithm.pid.ki = self.ki_spin.value()
        self.modified_config.algorithm.pid.kd = self.kd_spin.value()
        
        # UI配置
        if not hasattr(self.modified_config, 'ui'):
            class UI:
                class chart:
                    pass
            
            self.modified_config.ui = UI()
        
        self.modified_config.ui.window_width = self.window_width_spin.value()
        self.modified_config.ui.window_height = self.window_height_spin.value()
        self.modified_config.ui.refresh_rate = self.refresh_rate_spin.value()
        self.modified_config.ui.fullscreen = self.fullscreen_check.isChecked()
        
        # 图表配置
        self.modified_config.ui.chart.history_points = self.chart_history_spin.value()
        self.modified_config.ui.chart.smoothing = self.chart_smoothing_check.isChecked()
        self.modified_config.ui.chart.animation = self.chart_animation_check.isChecked()
    
    def browse_model_path(self):
        """浏览模型路径"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件",
            "", "模型文件 (*.pt *.pth *.onnx);;所有文件 (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_data_dir(self):
        """浏览数据目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据目录",
            "", QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            self.data_dir_edit.setText(dir_path)
    
    def get_config(self):
        """获取配置"""
        return self.modified_config


class CalibrationDialog(QDialog):
    """校准对话框"""
    
    calibration_complete = pyqtSignal(dict)
    
    def __init__(self, demo_manager, config: DemoConfig, parent=None):
        """
        初始化校准对话框
        
        Args:
            demo_manager: 演示管理器
            config: 系统配置
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.demo_manager = demo_manager
        self.config = config
        
        # 校准状态
        self.calibration_in_progress = False
        self.current_step = 0
        self.total_steps = 5
        self.calibration_results = {}
        
        # 初始化UI
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle("硬件校准")
        self.resize(600, 500)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 传感器校准标签页
        self.sensor_tab = self.create_sensor_calibration_tab()
        self.tab_widget.addTab(self.sensor_tab, "传感器校准")
        
        # 舵机校准标签页
        self.servo_tab = self.create_servo_calibration_tab()
        self.tab_widget.addTab(self.servo_tab, "舵机校准")
        
        # 相机校准标签页
        self.camera_tab = self.create_camera_calibration_tab()
        self.tab_widget.addTab(self.camera_tab, "相机校准")
        
        # 机械臂校准标签页
        self.arm_tab = self.create_arm_calibration_tab()
        self.tab_widget.addTab(self.arm_tab, "机械臂校准")
        
        layout.addWidget(self.tab_widget)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备开始校准")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始校准")
        self.start_btn.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止校准")
        self.stop_btn.clicked.connect(self.stop_calibration)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def create_sensor_calibration_tab(self):
        """创建传感器校准标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 校准说明
        instructions = QLabel(
            "触觉传感器校准步骤:\n\n"
            "1. 确保传感器处于自由状态（无接触）\n"
            "2. 点击'开始校准'按钮\n"
            "3. 等待校准完成\n"
            "4. 校准完成后可以保存结果\n\n"
            "校准将记录零偏和灵敏度参数。"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # 校准参数
        params_group = QGroupBox("校准参数")
        params_layout = QFormLayout(params_group)
        
        self.sensor_samples_spin = QSpinBox()
        self.sensor_samples_spin.setRange(10, 1000)
        self.sensor_samples_spin.setValue(100)
        params_layout.addRow("采样点数:", self.sensor_samples_spin)
        
        self.sensor_noise_threshold_spin = QDoubleSpinBox()
        self.sensor_noise_threshold_spin.setRange(0.0, 10.0)
        self.sensor_noise_threshold_spin.setValue(0.1)
        self.sensor_noise_threshold_spin.setSuffix(" N")
        params_layout.addRow("噪声阈值:", self.sensor_noise_threshold_spin)
        
        layout.addWidget(params_group)
        
        # 校准结果显示
        self.sensor_results_text = QTextEdit()
        self.sensor_results_text.setReadOnly(True)
        self.sensor_results_text.setMaximumHeight(150)
        layout.addWidget(QLabel("校准结果:"))
        layout.addWidget(self.sensor_results_text)
        
        layout.addStretch()
        
        return tab
    
    def create_servo_calibration_tab(self):
        """创建舵机校准标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 校准说明
        instructions = QLabel(
            "舵机校准步骤:\n\n"
            "1. 确保夹爪处于安全位置\n"
            "2. 点击'开始校准'按钮\n"
            "3. 舵机将自动移动到各个位置\n"
            "4. 记录位置反馈进行校准\n\n"
            "校准将确定位置映射和死区补偿。"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # 校准参数
        params_group = QGroupBox("校准参数")
        params_layout = QFormLayout(params_group)
        
        self.servo_test_positions_spin = QSpinBox()
        self.servo_test_positions_spin.setRange(3, 20)
        self.servo_test_positions_spin.setValue(10)
        params_layout.addRow("测试位置数:", self.servo_test_positions_spin)
        
        self.servo_position_tolerance_spin = QDoubleSpinBox()
        self.servo_position_tolerance_spin.setRange(0.1, 10.0)
        self.servo_position_tolerance_spin.setValue(1.0)
        self.servo_position_tolerance_spin.setSuffix(" °")
        params_layout.addRow("位置容差:", self.servo_position_tolerance_spin)
        
        layout.addWidget(params_group)
        
        # 舵机控制
        control_group = QGroupBox("手动控制")
        control_layout = QHBoxLayout(control_group)
        
        self.servo_position_slider = QSlider(Qt.Horizontal)
        self.servo_position_slider.setRange(0, 180)
        self.servo_position_slider.setValue(90)
        control_layout.addWidget(self.servo_position_slider)
        
        self.servo_position_label = QLabel("90")
        control_layout.addWidget(self.servo_position_label)
        
        self.move_servo_btn = QPushButton("移动")
        self.move_servo_btn.clicked.connect(self.move_servo)
        control_layout.addWidget(self.move_servo_btn)
        
        layout.addWidget(control_group)
        
        # 校准结果显示
        self.servo_results_text = QTextEdit()
        self.servo_results_text.setReadOnly(True)
        self.servo_results_text.setMaximumHeight(150)
        layout.addWidget(QLabel("校准结果:"))
        layout.addWidget(self.servo_results_text)
        
        layout.addStretch()
        
        # 连接信号
        self.servo_position_slider.valueChanged.connect(
            lambda v: self.servo_position_label.setText(str(v))
        )
        
        return tab
    
    def create_camera_calibration_tab(self):
        """创建相机校准标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 校准说明
        instructions = QLabel(
            "相机校准步骤:\n\n"
            "1. 准备棋盘格标定板\n"
            "2. 从不同角度拍摄标定板图像\n"
            "3. 点击'开始校准'按钮\n"
            "4. 系统将自动检测角点并计算参数\n\n"
            "校准将确定相机内参和畸变系数。"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # 标定板参数
        board_group = QGroupBox("标定板参数")
        board_layout = QFormLayout(board_group)
        
        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(2, 20)
        self.board_width_spin.setValue(9)
        board_layout.addRow("棋盘格宽度:", self.board_width_spin)
        
        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(2, 20)
        self.board_height_spin.setValue(6)
        board_layout.addRow("棋盘格高度:", self.board_height_spin)
        
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 100.0)
        self.square_size_spin.setValue(25.0)
        self.square_size_spin.setSuffix(" mm")
        board_layout.addRow("方格大小:", self.square_size_spin)
        
        layout.addWidget(board_group)
        
        # 图像采集
        capture_group = QGroupBox("图像采集")
        capture_layout = QVBoxLayout(capture_group)
        
        self.capture_images_btn = QPushButton("采集图像")
        self.capture_images_btn.clicked.connect(self.capture_calibration_images)
        capture_layout.addWidget(self.capture_images_btn)
        
        self.image_count_label = QLabel("已采集图像: 0")
        capture_layout.addWidget(self.image_count_label)
        
        self.clear_images_btn = QPushButton("清除图像")
        self.clear_images_btn.clicked.connect(self.clear_calibration_images)
        capture_layout.addWidget(self.clear_images_btn)
        
        layout.addWidget(capture_group)
        
        # 校准结果显示
        self.camera_results_text = QTextEdit()
        self.camera_results_text.setReadOnly(True)
        self.camera_results_text.setMaximumHeight(150)
        layout.addWidget(QLabel("校准结果:"))
        layout.addWidget(self.camera_results_text)
        
        layout.addStretch()
        
        return tab
    
    def create_arm_calibration_tab(self):
        """创建机械臂校准标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 校准说明
        instructions = QLabel(
            "机械臂校准步骤:\n\n"
            "1. 确保机械臂在安全工作空间内\n"
            "2. 点击'开始校准'按钮\n"
            "3. 机械臂将执行标准运动序列\n"
            "4. 记录关节位置和末端位姿\n\n"
            "校准将确定DH参数和运动学模型。"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # 校准类型
        type_group = QGroupBox("校准类型")
        type_layout = QVBoxLayout(type_group)
        
        self.arm_calibration_type_combo = QComboBox()
        self.arm_calibration_type_combo.addItems([
            "关节零位校准",
            "工具中心点校准",
            "工作空间标定",
            "完全校准"
        ])
        type_layout.addWidget(self.arm_calibration_type_combo)
        
        layout.addWidget(type_group)
        
        # 校准点管理
        points_group = QGroupBox("校准点")
        points_layout = QVBoxLayout(points_group)
        
        self.record_point_btn = QPushButton("记录当前点")
        self.record_point_btn.clicked.connect(self.record_calibration_point)
        points_layout.addWidget(self.record_point_btn)
        
        self.clear_points_btn = QPushButton("清除所有点")
        self.clear_points_btn.clicked.connect(self.clear_calibration_points)
        points_layout.addWidget(self.clear_points_btn)
        
        self.points_count_label = QLabel("已记录点: 0")
        points_layout.addWidget(self.points_count_label)
        
        layout.addWidget(points_group)
        
        # 校准结果显示
        self.arm_results_text = QTextEdit()
        self.arm_results_text.setReadOnly(True)
        self.arm_results_text.setMaximumHeight(150)
        layout.addWidget(QLabel("校准结果:"))
        layout.addWidget(self.arm_results_text)
        
        layout.addStretch()
        
        return tab
    
    def start_calibration(self):
        """开始校准"""
        if self.calibration_in_progress:
            return
        
        self.calibration_in_progress = True
        self.current_step = 0
        self.calibration_results.clear()
        
        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        
        # 获取当前标签页
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # 传感器校准
            self.start_sensor_calibration()
        elif current_tab == 1:  # 舵机校准
            self.start_servo_calibration()
        elif current_tab == 2:  # 相机校准
            self.start_camera_calibration()
        elif current_tab == 3:  # 机械臂校准
            self.start_arm_calibration()
    
    def stop_calibration(self):
        """停止校准"""
        self.calibration_in_progress = False
        self.update_status("校准已停止")
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 重置进度条
        self.progress_bar.setValue(0)
    
    def save_calibration(self):
        """保存校准结果"""
        if not self.calibration_results:
            QMessageBox.warning(self, "保存失败", "没有可保存的校准结果")
            return
        
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存校准结果",
                f"calibration_{timestamp}.json",
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if file_path:
                # 保存到文件
                with open(file_path, 'w') as f:
                    json.dump(self.calibration_results, f, indent=2)
                
                QMessageBox.information(self, "保存成功", f"校准结果已保存到 {file_path}")
                
                # 发射信号
                self.calibration_complete.emit(self.calibration_results)
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存校准结果时出错: {str(e)}")
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
        QApplication.processEvents()  # 确保UI更新
    
    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()
    
    def start_sensor_calibration(self):
        """开始传感器校准"""
        # 模拟校准过程
        self.update_status("开始传感器校准...")
        
        # 模拟校准步骤
        steps = [
            "检查传感器连接...",
            "采集零偏数据...",
            "计算灵敏度...",
            "验证校准结果...",
            "校准完成"
        ]
        
        # 启动校准线程
        self.calibration_thread = CalibrationThread(steps)
        self.calibration_thread.progress_updated.connect(self.update_progress)
        self.calibration_thread.status_updated.connect(self.update_status)
        self.calibration_thread.finished.connect(self.on_sensor_calibration_finished)
        self.calibration_thread.start()
    
    def start_servo_calibration(self):
        """开始舵机校准"""
        self.update_status("开始舵机校准...")
        
        # 模拟校准过程
        steps = [
            "检查舵机连接...",
            "移动到最小位置...",
            "记录位置反馈...",
            "移动到最大位置...",
            "计算位置映射...",
            "校准完成"
        ]
        
        # 模拟结果
        self.calibration_results['servo'] = {
            'min_position': 0,
            'max_position': 180,
            'position_error': 0.5,
            'calibrated_at': datetime.now().isoformat()
        }
        
        # 显示结果
        self.servo_results_text.setText(
            f"舵机校准结果:\n"
            f"最小位置: {self.calibration_results['servo']['min_position']}°\n"
            f"最大位置: {self.calibration_results['servo']['max_position']}°\n"
            f"位置误差: {self.calibration_results['servo']['position_error']}°\n"
            f"校准时间: {self.calibration_results['servo']['calibrated_at']}"
        )
        
        # 完成校准
        self.on_calibration_finished()
    
    def start_camera_calibration(self):
        """开始相机校准"""
        self.update_status("开始相机校准...")
        
        # 模拟校准过程
        steps = [
            "检查相机连接...",
            "检测标定板角点...",
            "计算相机内参...",
            "计算畸变系数...",
            "验证校准结果...",
            "校准完成"
        ]
        
        # 模拟结果
        self.calibration_results['camera'] = {
            'fx': 500.0,
            'fy': 500.0,
            'cx': 320.0,
            'cy': 240.0,
            'distortion_coeffs': [0.1, -0.2, 0.0, 0.0, 0.0],
            'reprojection_error': 0.15,
            'calibrated_at': datetime.now().isoformat()
        }
        
        # 显示结果
        self.camera_results_text.setText(
            f"相机校准结果:\n"
            f"焦距: fx={self.calibration_results['camera']['fx']:.2f}, "
            f"fy={self.calibration_results['camera']['fy']:.2f}\n"
            f"主点: cx={self.calibration_results['camera']['cx']:.2f}, "
            f"cy={self.calibration_results['camera']['cy']:.2f}\n"
            f"重投影误差: {self.calibration_results['camera']['reprojection_error']:.3f}\n"
            f"校准时间: {self.calibration_results['camera']['calibrated_at']}"
        )
        
        # 完成校准
        self.on_calibration_finished()
    
    def start_arm_calibration(self):
        """开始机械臂校准"""
        self.update_status("开始机械臂校准...")
        
        # 模拟校准过程
        steps = [
            "检查机械臂连接...",
            "记录关节零位...",
            "标定工具中心点...",
            "测量工作空间...",
            "计算运动学参数...",
            "校准完成"
        ]
        
        # 模拟结果
        self.calibration_results['arm'] = {
            'joint_offsets': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'tool_center_point': [0.0, 0.0, 0.1],
            'workspace_volume': 0.5,
            'calibrated_at': datetime.now().isoformat()
        }
        
        # 显示结果
        self.arm_results_text.setText(
            f"机械臂校准结果:\n"
            f"关节零位: {self.calibration_results['arm']['joint_offsets']}\n"
            f"工具中心点: {self.calibration_results['arm']['tool_center_point']}\n"
            f"工作空间体积: {self.calibration_results['arm']['workspace_volume']} m³\n"
            f"校准时间: {self.calibration_results['arm']['calibrated_at']}"
        )
        
        # 完成校准
        self.on_calibration_finished()
    
    def on_sensor_calibration_finished(self):
        """传感器校准完成"""
        # 模拟结果
        self.calibration_results['sensor'] = {
            'zero_offset': [0.1, 0.2, -0.1, 0.0, 0.1, -0.1, 0.0, 0.0, 0.0],
            'sensitivity': 1.05,
            'noise_level': 0.05,
            'calibrated_at': datetime.now().isoformat()
        }
        
        # 显示结果
        self.sensor_results_text.setText(
            f"传感器校准结果:\n"
            f"零偏: {self.calibration_results['sensor']['zero_offset'][:3]}...\n"
            f"灵敏度: {self.calibration_results['sensor']['sensitivity']}\n"
            f"噪声水平: {self.calibration_results['sensor']['noise_level']} N\n"
            f"校准时间: {self.calibration_results['sensor']['calibrated_at']}"
        )
        
        # 完成校准
        self.on_calibration_finished()
    
    def on_calibration_finished(self):
        """校准完成"""
        self.calibration_in_progress = False
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # 更新进度条
        self.progress_bar.setValue(100)
        self.update_status("校准完成")
    
    def move_servo(self):
        """移动舵机"""
        position = self.servo_position_slider.value()
        self.update_status(f"移动舵机到位置: {position}°")
        
        # 这里应该调用实际的舵机控制函数
        # 例如: self.demo_manager.move_servo(position)
    
    def capture_calibration_images(self):
        """采集校准图像"""
        self.update_status("采集校准图像...")
        # 这里应该调用相机采集函数
    
    def clear_calibration_images(self):
        """清除校准图像"""
        self.image_count_label.setText("已采集图像: 0")
        self.update_status("已清除所有图像")
    
    def record_calibration_point(self):
        """记录校准点"""
        # 这里应该记录当前机械臂位姿
        current_points = int(self.points_count_label.text().split(": ")[1])
        self.points_count_label.setText(f"已记录点: {current_points + 1}")
        self.update_status(f"已记录点 {current_points + 1}")
    
    def clear_calibration_points(self):
        """清除校准点"""
        self.points_count_label.setText("已记录点: 0")
        self.update_status("已清除所有点")


class CalibrationThread(QThread):
    """校准线程"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, steps):
        """
        初始化校准线程
        
        Args:
            steps: 校准步骤列表
        """
        super().__init__()
        self.steps = steps
    
    def run(self):
        """运行线程"""
        total_steps = len(self.steps)
        
        for i, step in enumerate(self.steps):
            # 更新状态
            self.status_updated.emit(step)
            
            # 更新进度
            progress = int((i + 1) / total_steps * 100)
            self.progress_updated.emit(progress)
            
            # 模拟处理时间
            self.msleep(500)
        
        # 完成
        self.finished.emit()


class DemoSelectionDialog(QDialog):
    """演示选择对话框"""
    
    def __init__(self, config: DemoConfig, parent=None):
        """
        初始化演示选择对话框
        
        Args:
            config: 系统配置
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.config = config
        self.selected_demo = None
        self.demo_params = {}
        
        # 初始化UI
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle("选择演示")
        self.resize(500, 400)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 演示列表
        demos_group = QGroupBox("可用演示")
        demos_layout = QVBoxLayout(demos_group)
        
        self.demos_list = QListWidget()
        
        # 添加演示项目
        demo_items = [
            ("校准演示", "calibration", "系统硬件校准"),
            ("抓取演示", "grasping", "基本抓取操作"),
            ("滑动检测", "slip_detection", "检测物体滑动"),
            ("物体分类", "object_classification", "基于触觉的物体识别"),
            ("力控制", "force_control", "精确力控制演示"),
            ("学习控制", "learning", "基于学习的控制策略"),
            ("矢量可视化", "vector_visualization", "三维力矢量显示"),
            ("触觉映射", "tactile_mapping", "触觉数据空间映射"),
            ("自主抓取", "autonomous_grasping", "完整的自主抓取流程")
        ]
        
        for name, demo_id, description in demo_items:
            item_text = f"{name}\n  ({description})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, demo_id)
            self.demos_list.addItem(item)
        
        self.demos_list.itemSelectionChanged.connect(self.on_demo_selected)
        demos_layout.addWidget(self.demos_list)
        
        layout.addWidget(demos_group)
        
        # 参数设置
        params_group = QGroupBox("演示参数")
        params_layout = QFormLayout(params_group)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" 秒")
        params_layout.addRow("持续时间:", self.duration_spin)
        
        self.force_limit_spin = QDoubleSpinBox()
        self.force_limit_spin.setRange(0.0, 100.0)
        self.force_limit_spin.setValue(30.0)
        self.force_limit_spin.setSuffix(" N")
        params_layout.addRow("力限制:", self.force_limit_spin)
        
        self.save_data_check = QCheckBox("保存数据")
        params_layout.addRow(self.save_data_check)
        
        self.visualize_check = QCheckBox("实时可视化")
        self.visualize_check.setChecked(True)
        params_layout.addRow(self.visualize_check)
        
        layout.addWidget(params_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton("开始演示")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def on_demo_selected(self):
        """演示选择变化"""
        selected_items = self.demos_list.selectedItems()
        if selected_items:
            self.selected_demo = selected_items[0].data(Qt.UserRole)
            self.ok_btn.setEnabled(True)
        else:
            self.selected_demo = None
            self.ok_btn.setEnabled(False)
    
    def get_selection(self):
        """获取选择"""
        # 收集参数
        self.demo_params = {
            'duration': self.duration_spin.value(),
            'force_limit': self.force_limit_spin.value(),
            'save_data': self.save_data_check.isChecked(),
            'visualize': self.visualize_check.isChecked()
        }
        
        return self.selected_demo, self.demo_params


class LogViewerDialog(QDialog):
    """日志查看器对话框"""
    
    def __init__(self, parent=None):
        """
        初始化日志查看器对话框
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 日志数据
        self.log_entries = []
        
        # 初始化UI
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle("系统日志")
        self.resize(800, 600)
        
        # 加载日志
        self.load_logs()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.load_logs)
        toolbar_layout.addWidget(self.refresh_btn)
        
        self.clear_btn = QPushButton("清除")
        self.clear_btn.clicked.connect(self.clear_logs)
        toolbar_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_logs)
        toolbar_layout.addWidget(self.save_btn)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["所有级别", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.filter_combo.currentTextChanged.connect(self.filter_logs)
        toolbar_layout.addWidget(QLabel("过滤:"))
        toolbar_layout.addWidget(self.filter_combo)
        
        toolbar_layout.addStretch()
        
        layout.addLayout(toolbar_layout)
        
        # 日志表格
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels([
            "时间", "级别", "模块", "消息", "详细信息"
        ])
        
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.log_table)
        
        # 详细信息
        details_group = QGroupBox("详细信息")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(100)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # 连接信号
        self.log_table.itemSelectionChanged.connect(self.show_log_details)
    
    def load_logs(self):
        """加载日志"""
        # 清空表格
        self.log_table.setRowCount(0)
        self.log_entries.clear()
        
        # 模拟日志数据
        import logging
        import random
        from datetime import datetime, timedelta
        
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_modules = ['main', 'hardware', 'sensor', 'servo', 'algorithm', 'gui']
        log_messages = [
            "系统启动",
            "传感器连接成功",
            "舵机初始化完成",
            "开始数据采集",
            "检测到物体接触",
            "力超过阈值",
            "滑动检测触发",
            "保存数据到文件",
            "系统错误",
            "硬件连接断开"
        ]
        
        # 生成模拟日志
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
            level = random.choice(log_levels)
            module = random.choice(log_modules)
            message = random.choice(log_messages)
            details = f"详细信息 {i+1}: 这是{level}级别的日志消息"
            
            self.log_entries.append({
                'timestamp': timestamp,
                'level': level,
                'module': module,
                'message': message,
                'details': details
            })
        
        # 显示所有日志
        self.display_logs()
    
    def display_logs(self, filter_level="所有级别"):
        """显示日志"""
        self.log_table.setRowCount(0)
        
        row = 0
        for entry in self.log_entries:
            # 应用过滤
            if filter_level != "所有级别" and entry['level'] != filter_level:
                continue
            
            self.log_table.insertRow(row)
            
            # 时间
            time_item = QTableWidgetItem(entry['timestamp'].strftime("%H:%M:%S"))
            self.log_table.setItem(row, 0, time_item)
            
            # 级别
            level_item = QTableWidgetItem(entry['level'])
            
            # 根据级别设置颜色
            if entry['level'] == 'ERROR' or entry['level'] == 'CRITICAL':
                level_item.setBackground(QColor(255, 200, 200))
            elif entry['level'] == 'WARNING':
                level_item.setBackground(QColor(255, 255, 200))
            elif entry['level'] == 'INFO':
                level_item.setBackground(QColor(200, 255, 200))
            elif entry['level'] == 'DEBUG':
                level_item.setBackground(QColor(200, 200, 255))
            
            self.log_table.setItem(row, 1, level_item)
            
            # 模块
            module_item = QTableWidgetItem(entry['module'])
            self.log_table.setItem(row, 2, module_item)
            
            # 消息
            message_item = QTableWidgetItem(entry['message'])
            self.log_table.setItem(row, 3, message_item)
            
            # 详细信息
            details_item = QTableWidgetItem(entry['details'][:50] + "...")
            self.log_table.setItem(row, 4, details_item)
            
            # 保存完整信息
            details_item.setData(Qt.UserRole, entry['details'])
            
            row += 1
    
    def filter_logs(self, filter_text):
        """过滤日志"""
        self.display_logs(filter_text)
    
    def show_log_details(self):
        """显示日志详细信息"""
        selected_items = self.log_table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        details_item = self.log_table.item(row, 4)
        
        if details_item:
            full_details = details_item.data(Qt.UserRole)
            self.details_text.setText(full_details)
    
    def clear_logs(self):
        """清除日志"""
        reply = QMessageBox.question(
            self, "清除日志",
            "是否清除所有日志？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_table.setRowCount(0)
            self.log_entries.clear()
            self.details_text.clear()
    
    def save_logs(self):
        """保存日志"""
        if not self.log_entries:
            QMessageBox.warning(self, "保存失败", "没有可保存的日志")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存日志",
                f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "文本文件 (*.txt);;所有文件 (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for entry in self.log_entries:
                        line = f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} " \
                               f"[{entry['level']}] {entry['module']}: {entry['message']}\n"
                        f.write(line)
                
                QMessageBox.information(self, "保存成功", f"日志已保存到 {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存日志时出错: {str(e)}")


class AboutDialog(QDialog):
    """关于对话框"""
    
    def __init__(self, parent=None):
        """
        初始化关于对话框
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 初始化UI
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle("关于")
        self.resize(400, 300)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel("触觉夹爪控制系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # 版本
        version_label = QLabel("版本 2.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 描述
        description = QLabel(
            "基于三维力传感器的智能夹爪控制系统\n\n"
            "功能特性:\n"
            "• 三维力触觉数据采集与可视化\n"
            "• 机械臂运动控制与状态监控\n"
            "• 基于深度学习的抓取策略\n"
            "• 实时视觉与仿真显示\n"
            "• 模块化系统架构\n\n"
            "支持硬件:\n"
            "• 触觉传感器: Paxini Gen3 M2020\n"
            "• 舵机: ST3215-C018\n"
            "• 控制器: STM32嵌入式系统"
        )
        description.setAlignment(Qt.AlignLeft)
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)
        
        # 版权信息
        copyright_label = QLabel("© 2023 触觉夹爪研发团队")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        # 联系方式
        contact_label = QLabel("联系方式: tactile.gripper@example.com")
        contact_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(contact_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 设置样式
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
        """)


if __name__ == "__main__":
    # 测试对话框
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # 测试配置对话框
    config = DemoConfig()
    dialog = ConfigDialog(config)
    dialog.show()
    
    sys.exit(app.exec_())
