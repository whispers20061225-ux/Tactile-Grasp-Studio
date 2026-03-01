"""
触觉夹爪演示系统 - 图形用户界面模块
提供完整的用户界面，包括控制面板、数据可视化和系统监控。
"""

from .main_window import MainWindow
from .control_panel import ControlPanel
from .plot_widget import (
    PlotWidget,
    TactilePlotWidget,
    ForcePlotWidget,
    TactileSurfaceWidget,
    VectorFieldWidget,
)
from .dialogs import (
    ConfigDialog,
    CalibrationDialog,
    DemoSelectionDialog,
    LogViewerDialog,
    AboutDialog
)
from .vision_viewer import VisionViewer
from .simulation_viewer import SimulationViewer
from .arm_status_panel import ArmStatusPanel

# 版本信息
__version__ = "3.0.0"
__author__ = "Tactile Gripper Team"
__description__ = "触觉夹爪演示系统图形用户界面模块（含视觉、仿真与机械臂状态）"

# 导出列表
__all__ = [
    # 主窗口
    "MainWindow",
    
    # 控制面板
    "ControlPanel",
    
    # 绘图部件
    "PlotWidget",
    "TactilePlotWidget",
    "ForcePlotWidget",
    "TactileSurfaceWidget",
    "VectorFieldWidget",
    "VisionViewer",
    "SimulationViewer",
    "ArmStatusPanel",
    
    # 对话框
    "ConfigDialog",
    "CalibrationDialog",
    "DemoSelectionDialog",
    "LogViewerDialog",
    "AboutDialog",
    
    # 元信息
    "__version__",
    "__author__",
    "__description__",
]


def get_ui_info() -> dict:
    """
    获取UI模块信息
    
    Returns:
        包含UI模块信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "widgets": [
            "MainWindow",
            "ControlPanel",
        "PlotWidget",
        "TactilePlotWidget",
        "ForcePlotWidget",
        "TactileSurfaceWidget",
        "VectorFieldWidget",
        "VisionViewer",
        "SimulationViewer",
        "ArmStatusPanel",
        ],
        "dialogs": [
            "ConfigDialog",
            "CalibrationDialog",
            "DemoSelectionDialog",
            "LogViewerDialog",
            "AboutDialog",
        ]
    }


# 样式表管理器
class StyleManager:
    """样式管理器，管理应用的主题和样式"""
    
    @staticmethod
    def get_light_theme() -> str:
        """获取浅色主题样式表"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QWidget {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 10pt;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #45a049;
        }
        
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        
        QPushButton.emergency {
            background-color: #f44336;
        }
        
        QPushButton.emergency:hover {
            background-color: #da190b;
        }
        
        QLabel {
            color: #333333;
        }
        
        QLabel.title {
            font-size: 14pt;
            font-weight: bold;
            color: #2196F3;
        }
        
        QLabel.status {
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 3px;
        }
        
        QLabel.status-ok {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        QLabel.status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        QLabel.status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 3px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 2px;
        }
        
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #e9e9e9;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #f0f0f0;
        }
        
        QTextEdit, QPlainTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
            background-color: white;
        }
        
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 4px;
            background-color: white;
        }
        
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
            border: 1px solid #2196F3;
        }
        
        QComboBox::drop-down {
            border: none;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #666666;
            width: 0;
            height: 0;
            margin-right: 5px;
        }
        
        QScrollBar:vertical {
            border: none;
            background-color: #f0f0f0;
            width: 10px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #cccccc;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #aaaaaa;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        """
    
    @staticmethod
    def get_dark_theme() -> str:
        """获取深色主题样式表"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
        }
        
        QWidget {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 10pt;
            color: #ffffff;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            color: #ffffff;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #64b5f6;
        }
        
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #1976D2;
        }
        
        QPushButton:pressed {
            background-color: #0d47a1;
        }
        
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        
        QPushButton.emergency {
            background-color: #f44336;
        }
        
        QPushButton.emergency:hover {
            background-color: #d32f2f;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QLabel.title {
            font-size: 14pt;
            font-weight: bold;
            color: #64b5f6;
        }
        
        QLabel.status {
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 3px;
        }
        
        QLabel.status-ok {
            background-color: #388e3c;
            color: #c8e6c9;
            border: 1px solid #2e7d32;
        }
        
        QLabel.status-warning {
            background-color: #f57c00;
            color: #fff3e0;
            border: 1px solid #ef6c00;
        }
        
        QLabel.status-error {
            background-color: #d32f2f;
            color: #ffcdd2;
            border: 1px solid #c62828;
        }
        
        QProgressBar {
            border: 1px solid #555555;
            border-radius: 3px;
            text-align: center;
            color: white;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 2px;
        }
        
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #3c3c3c;
        }
        
        QTabBar::tab {
            background-color: #444444;
            color: #cccccc;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #3c3c3c;
            color: white;
            border-bottom: 2px solid #2196F3;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #505050;
        }
        
        QTextEdit, QPlainTextEdit {
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
            color: white;
        }
        
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 4px;
            background-color: #3c3c3c;
            color: white;
        }
        
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
            border: 1px solid #2196F3;
        }
        
        QComboBox::drop-down {
            border: none;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #cccccc;
            width: 0;
            height: 0;
            margin-right: 5px;
        }
        
        QScrollBar:vertical {
            border: none;
            background-color: #3c3c3c;
            width: 10px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #666666;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #888888;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        """
    
    @staticmethod
    def get_theme_names() -> list:
        """获取可用的主题名称"""
        return ["light", "dark"]
    
    @staticmethod
    def apply_theme(app, theme_name: str = "light"):
        """
        应用主题到Qt应用
        
        Args:
            app: QApplication实例
            theme_name: 主题名称（"light"或"dark"）
        """
        if theme_name == "light":
            app.setStyleSheet(StyleManager.get_light_theme())
        elif theme_name == "dark":
            app.setStyleSheet(StyleManager.get_dark_theme())
        else:
            raise ValueError(f"未知的主题: {theme_name}")


# 图标资源管理器
class IconManager:
    """图标管理器，提供各种图标资源"""
    
    # 图标名称常量
    ICON_PLAY = "play"
    ICON_PAUSE = "pause"
    ICON_STOP = "stop"
    ICON_RECORD = "record"
    ICON_SAVE = "save"
    ICON_LOAD = "load"
    ICON_SETTINGS = "settings"
    ICON_CALIBRATE = "calibrate"
    ICON_HOME = "home"
    ICON_EMERGENCY = "emergency"
    ICON_GRAPH = "graph"
    ICON_SENSOR = "sensor"
    ICON_SERVO = "servo"
    ICON_INFO = "info"
    ICON_WARNING = "warning"
    ICON_ERROR = "error"
    
    @staticmethod
    def get_icon(icon_name: str, size: int = 24):
        """
        获取图标
        
        Args:
            icon_name: 图标名称
            size: 图标大小
            
        Returns:
            QIcon实例
        """
        from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
        from PyQt5.QtCore import Qt, QRect, QPointF
        
        # 创建一个空图标
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置颜色
        color = QColor("#2196F3")  # 默认蓝色
        
        if icon_name == IconManager.ICON_EMERGENCY:
            color = QColor("#f44336")  # 红色
        elif icon_name in [IconManager.ICON_WARNING, IconManager.ICON_ERROR]:
            color = QColor("#ff9800")  # 橙色
        elif icon_name in [IconManager.ICON_SAVE, IconManager.ICON_LOAD]:
            color = QColor("#4CAF50")  # 绿色
        
        painter.setPen(color)
        painter.setBrush(color)
        
        # 根据图标名称绘制
        if icon_name == IconManager.ICON_PLAY:
            # 绘制三角形（播放）
            points = [
                (size * 0.3, size * 0.2),
                (size * 0.3, size * 0.8),
                (size * 0.8, size * 0.5)
            ]
            painter.drawPolygon(*[QPointF(x, y) for x, y in points])
            
        elif icon_name == IconManager.ICON_PAUSE:
            # 绘制两个矩形（暂停）
            rect1 = QRect(size * 0.25, size * 0.2, size * 0.2, size * 0.6)
            rect2 = QRect(size * 0.55, size * 0.2, size * 0.2, size * 0.6)
            painter.drawRect(rect1)
            painter.drawRect(rect2)
            
        elif icon_name == IconManager.ICON_STOP:
            # 绘制正方形（停止）
            rect = QRect(size * 0.2, size * 0.2, size * 0.6, size * 0.6)
            painter.drawRect(rect)
            
        elif icon_name == IconManager.ICON_RECORD:
            # 绘制圆形（录制）
            painter.drawEllipse(size * 0.2, size * 0.2, size * 0.6, size * 0.6)
            
        elif icon_name == IconManager.ICON_SETTINGS:
            # 绘制齿轮（设置）
            painter.drawEllipse(size * 0.25, size * 0.25, size * 0.5, size * 0.5)
            # 添加齿轮齿（简化）
            for i in range(8):
                angle = i * 45
                # 绘制齿轮齿
                pass
            
        elif icon_name == IconManager.ICON_EMERGENCY:
            # 绘制感叹号（紧急）
            painter.drawEllipse(size * 0.2, size * 0.2, size * 0.6, size * 0.6)
            painter.setPen(Qt.white)
            painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "!")
            
        elif icon_name == IconManager.ICON_INFO:
            # 绘制字母i（信息）
            painter.drawEllipse(size * 0.2, size * 0.2, size * 0.6, size * 0.6)
            painter.setPen(Qt.white)
            painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "i")
            
        painter.end()
        
        return QIcon(pixmap)


# 初始化代码
if __name__ == "__main__":
    print(f"触觉夹爪GUI模块 v{__version__}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")
    
    # 显示UI模块信息
    ui_info = get_ui_info()
    print(f"\n可用部件: {', '.join(ui_info['widgets'])}")
    print(f"可用对话框: {', '.join(ui_info['dialogs'])}")
    
    # 显示可用的主题
    theme_names = StyleManager.get_theme_names()
    print(f"\n可用主题: {', '.join(theme_names)}")
    
    # 显示可用的图标
    print("\n可用图标:")
    for attr in dir(IconManager):
        if attr.startswith('ICON_'):
            print(f"  - {getattr(IconManager, attr)}")
