# tactile_perception/__init__.py
"""
触觉感知模块
提供完整的触觉数据读取、处理、分析和可视化功能
"""

from .sensor_reader import SensorReader, AsyncSensorReader, TactileData, SensorStatus
from .data_processor import DataProcessor, ProcessedTactileData
from .visualization import TactileVisualizer
from .tactile_analyzer import TactileAnalyzer
from .tactile_mapper import TactileMapper

__version__ = "1.0.0"
__all__ = [
    "SensorReader",
    "AsyncSensorReader",
    "TactileData",
    "SensorStatus",
    "DataProcessor",
    "ProcessedTactileData",
    "TactileVisualizer",
    "TactileAnalyzer",
    "TactileMapper"
]

def get_version():
    """获取模块版本"""
    return __version__

def create_default_pipeline(port: str = "COM3", baudrate: int = 115200,
                           num_taxels: int = 9, rows: int = 3, cols: int = 3):
    """
    创建默认的触觉处理流水线
    
    Args:
        port: 串口端口
        baudrate: 波特率
        num_taxels: 触觉单元数量 (默认9个，3x3)
        rows: 行数
        cols: 列数
    """
    # 创建读取器
    reader = SensorReader(port=port, baudrate=baudrate, num_taxels=num_taxels)
    
    # 创建处理器
    processor = DataProcessor(num_taxels=num_taxels)
    
    # 创建分析器
    analyzer = TactileAnalyzer(num_taxels=num_taxels, rows=rows, cols=cols)
    
    # 创建可视化器
    visualizer = TactileVisualizer(rows=rows, cols=cols)
    
    # 创建映射器
    mapper = TactileMapper(rows=rows, cols=cols)
    
    return {
        'reader': reader,
        'processor': processor,
        'analyzer': analyzer,
        'visualizer': visualizer,
        'mapper': mapper
    }