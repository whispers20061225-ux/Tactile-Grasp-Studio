"""
触觉夹爪演示系统 - 数据采集模块
负责从传感器采集数据并进行预处理。
"""

import threading
import time
import queue
import numpy as np
import logging
import sys
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用绝对导入
try:
    from config import DemoConfig
except ImportError as e:
    print(f"警告: 无法导入DemoConfig: {e}")
    # 定义备用配置类
    @dataclass
    class DemoConfig:
        hardware: Any = None
        algorithm: Any = None
        ui: Any = None

# 导入硬件接口
try:
    from core.hardware_interface import HardwareInterface, SensorReading
except ImportError as e:
    print(f"警告: 无法导入HardwareInterface: {e}")
    # 定义备用类
    @dataclass
    class SensorReading:
        timestamp: float = 0.0
        force_data: List[float] = None
        temperature: Optional[float] = None
        status: int = 0
    
    class HardwareInterface:
        pass


@dataclass
class SensorData:
    """传感器数据结构"""
    timestamp: float
    force_data: List[float]
    force_vectors: Optional[List[List[float]]] = None  # 三维力向量 [Fx, Fy, Fz]
    temperature: Optional[float] = None
    status: int = 0
    sequence_id: int = 0
    
    # 计算属性
    total_force: float = field(init=False)
    average_force: float = field(init=False)
    max_force: float = field(init=False)
    min_force: float = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.force_data is None:
            self.force_data = []
        # 如果未提供三维力向量，使用 force_data 生成 [0, 0, f]
        if self.force_vectors is None:
            self.force_vectors = [[0.0, 0.0, f] for f in self.force_data]
        
        # 计算统计属性
        if self.force_data:
            self.total_force = sum(self.force_data)
            self.average_force = self.total_force / len(self.force_data)
            self.max_force = max(self.force_data)
            self.min_force = min(self.force_data)
        else:
            self.total_force = 0.0
            self.average_force = 0.0
            self.max_force = 0.0
            self.min_force = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'force_data': self.force_data,
            'force_vectors': self.force_vectors,
            'temperature': self.temperature,
            'status': self.status,
            'sequence_id': self.sequence_id,
            'total_force': self.total_force,
            'average_force': self.average_force,
            'max_force': self.max_force,
            'min_force': self.min_force
        }
    
    @classmethod
    def from_sensor_reading(cls, reading: SensorReading, sequence_id: int = 0) -> 'SensorData':
        """从SensorReading创建"""
        return cls(
            timestamp=reading.timestamp,
            force_data=reading.force_data.copy() if reading.force_data else [],
            force_vectors=reading.force_vectors.copy() if hasattr(reading, 'force_vectors') and reading.force_vectors else None,
            temperature=reading.temperature,
            status=reading.status,
            sequence_id=sequence_id
        )


class DataBuffer:
    """数据缓冲区"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化数据缓冲区
        
        Args:
            max_size: 最大缓冲区大小
        """
        self.max_size = max_size
        self.buffer = []
        self.sequence_id = 0
        self.lock = threading.Lock()
    
    def add_data(self, data: SensorData) -> bool:
        """
        添加数据到缓冲区
        
        Args:
            data: 传感器数据
            
        Returns:
            是否成功添加
        """
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            
            # 设置序列ID
            data.sequence_id = self.sequence_id
            self.sequence_id += 1
            
            self.buffer.append(data)
            return True
    
    def get_latest(self) -> Optional[SensorData]:
        """
        获取最新数据
        
        Returns:
            最新传感器数据
        """
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
            return None
    
    def get_range(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[SensorData]:
        """
        获取数据范围
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引（None表示到最后）
            
        Returns:
            数据列表
        """
        with self.lock:
            if not self.buffer:
                return []
            
            if end_idx is None or end_idx > len(self.buffer):
                end_idx = len(self.buffer)
            
            return self.buffer[start_idx:end_idx]
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self.lock:
            if not self.buffer:
                return {
                    'size': 0,
                    'data_rate': 0.0,
                    'force_stats': {}
                }
            
            # 计算数据率
            if len(self.buffer) >= 2:
                time_span = self.buffer[-1].timestamp - self.buffer[0].timestamp
                if time_span > 0:
                    data_rate = len(self.buffer) / time_span
                else:
                    data_rate = 0.0
            else:
                data_rate = 0.0
            
            # 计算力统计
            if self.buffer:
                latest_data = self.buffer[-1]
                force_stats = {
                    'total': latest_data.total_force,
                    'average': latest_data.average_force,
                    'max': latest_data.max_force,
                    'min': latest_data.min_force
                }
            else:
                force_stats = {}
            
            return {
                'size': len(self.buffer),
                'data_rate': data_rate,
                'force_stats': force_stats
            }


class DataAcquisitionThread(QThread):
    """数据采集线程"""
    
    # 信号定义
    new_data = pyqtSignal(object)  # 发送新数据
    status_changed = pyqtSignal(str, dict)  # 状态变化
    error_occurred = pyqtSignal(str, dict)  # 错误发生
    
    def __init__(self, hardware_interface: HardwareInterface, config: DemoConfig):
        """
        初始化数据采集线程
        
        Args:
            hardware_interface: 硬件接口
            config: 系统配置
        """
        super().__init__()
        
        self.hardware_interface = hardware_interface
        self.config = config
        
        # 线程控制
        self.running = False
        self.paused = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 数据缓冲区
        self.data_buffer = DataBuffer(max_size=1000)
        
        # 统计信息
        self.read_count = 0
        self.error_count = 0
        self.start_time = 0.0
        self.idle_fail_count = 0
        
        # 数据处理
        self.filter_enabled = config.hardware.sensor.filter_enabled if hasattr(config.hardware, 'sensor') else False
        self.filter_type = config.hardware.sensor.filter_type if hasattr(config.hardware, 'sensor') else "none"
        self.filter_window = 5
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据过滤器
        self._init_filters()
    
    def _init_filters(self):
        """初始化数据过滤器"""
        self.filter_buffer = []
        self.filter_size = self.filter_window
        
        # 卡尔曼滤波器状态
        if self.filter_type == "kalman":
            self.kalman_state = None
            self.kalman_covariance = None

    def _is_sensor_connected(self) -> bool:
        """检查硬件接口传感器状态"""
        try:
            status = self.hardware_interface.get_status()
            return status.get("sensor", {}).get("connected", False)
        except Exception:
            return False
    
    def run(self):
        """线程主循环"""
        self.logger.info("数据采集线程启动")
        self.running = True
        self.start_time = time.time()
        
        # 发送状态信号
        self.status_changed.emit("starting", {"message": "数据采集启动"})
        
        while self.running:
            try:
                # 检查暂停状态
                self.mutex.lock()
                if self.paused:
                    self.condition.wait(self.mutex)
                self.mutex.unlock()
                
                # 计算采集间隔
                sampling_rate = getattr(self.config.hardware.sensor, 'sampling_rate', 100) if hasattr(self.config.hardware, 'sensor') else 100
                target_interval = 1.0 / sampling_rate
                loop_start_time = time.time()
                
                # 采集数据
                self._acquire_data()
                
                # 控制采集频率
                loop_time = time.time() - loop_start_time
                if loop_time < target_interval:
                    time.sleep(target_interval - loop_time)
                    
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"数据采集错误: {e}")
                self.error_occurred.emit("acquisition_error", {"error": str(e)})
                
                # 短暂暂停后重试
                time.sleep(0.1)
        
        # 线程结束
        self.logger.info("数据采集线程停止")
        self.status_changed.emit("stopped", {"message": "数据采集停止"})
    
    def _acquire_data(self):
        """采集数据"""
        try:
            # 未连接时降低轮询频率，避免空转卡顿
            if not self._is_sensor_connected():
                self.idle_fail_count += 1
                if self.idle_fail_count % 20 == 0:
                    self.status_changed.emit("idle", {
                        "message": "传感器未连接或模拟中，降低采样频率",
                        "error_count": self.error_count
                    })
                time.sleep(0.2)
                return

            # 读取传感器数据
            reading = self.hardware_interface.read_sensor() if hasattr(self.hardware_interface, 'read_sensor') else None
            
            if reading:
                # 处理数据
                processed_data = self._process_data(reading)
                
                # 添加到缓冲区
                self.data_buffer.add_data(processed_data)
                
                # 更新统计
                self.read_count += 1
                
                # 发送信号
                self.new_data.emit(processed_data)
                
                # 定期发送状态更新
                if self.read_count % 10 == 0:
                    stats = self.data_buffer.get_statistics()
                    self.status_changed.emit("running", {
                        "read_count": self.read_count,
                        "error_count": self.error_count,
                        "data_rate": stats['data_rate'],
                        "buffer_size": stats['size']
                    })
            else:
                # 读取失败
                self.error_count += 1
                if self.error_count % 10 == 0:
                    self.logger.warning(f"连续读取失败次数: {self.error_count}")
                time.sleep(0.05)
                
        except Exception as e:
            raise Exception(f"数据采集失败: {e}")
    
    def _process_data(self, reading: SensorReading) -> SensorData:
        """
        处理传感器数据
        
        Args:
            reading: 原始传感器读数
            
        Returns:
            处理后的数据
        """
        # 转换为SensorData
        sensor_data = SensorData.from_sensor_reading(reading, self.read_count)

        # 确保三维力数据长度与配置一致
        target_taxels = getattr(self.config.hardware.sensor, 'num_taxels', None) if hasattr(self.config.hardware, 'sensor') else None
        if target_taxels:
            # force_data 补齐/裁剪
            if sensor_data.force_data is None:
                sensor_data.force_data = []
            if len(sensor_data.force_data) < target_taxels:
                sensor_data.force_data += [0.0] * (target_taxels - len(sensor_data.force_data))
            elif len(sensor_data.force_data) > target_taxels:
                sensor_data.force_data = sensor_data.force_data[:target_taxels]
            # force_vectors 补齐/裁剪
            if sensor_data.force_vectors is None:
                sensor_data.force_vectors = []
            if len(sensor_data.force_vectors) < target_taxels:
                sensor_data.force_vectors += [[0.0, 0.0, 0.0]] * (target_taxels - len(sensor_data.force_vectors))
            elif len(sensor_data.force_vectors) > target_taxels:
                sensor_data.force_vectors = sensor_data.force_vectors[:target_taxels]
        
        # 应用滤波
        if self.filter_enabled and len(sensor_data.force_data) > 0:
            sensor_data = self._apply_filter(sensor_data)
        
        # 归一化（如果需要）
        normalize_data = getattr(self.config.hardware.sensor, 'normalize_data', False) if hasattr(self.config.hardware, 'sensor') else False
        if normalize_data:
            sensor_data = self._normalize_data(sensor_data)
        
        return sensor_data
    
    def _apply_filter(self, data: SensorData) -> SensorData:
        """
        应用滤波器
        
        Args:
            data: 原始数据
            
        Returns:
            滤波后的数据
        """
        if self.filter_type == "lowpass":
            return self._apply_lowpass_filter(data)
        elif self.filter_type == "median":
            return self._apply_median_filter(data)
        elif self.filter_type == "kalman":
            return self._apply_kalman_filter(data)
        else:
            return data
    
    def _apply_lowpass_filter(self, data: SensorData) -> SensorData:
        """应用低通滤波器"""
        # 简单的移动平均滤波器
        self.filter_buffer.append(data.force_data.copy())
        
        if len(self.filter_buffer) > self.filter_size:
            self.filter_buffer.pop(0)
        
        if len(self.filter_buffer) >= 3:
            # 计算移动平均
            filtered_forces = []
            for i in range(len(data.force_data)):
                values = [buf[i] for buf in self.filter_buffer if i < len(buf)]
                if values:
                    filtered_forces.append(sum(values) / len(values))
                else:
                    filtered_forces.append(data.force_data[i])
            
            data.force_data = filtered_forces
        
        return data
    
    def _apply_median_filter(self, data: SensorData) -> SensorData:
        """应用中值滤波器"""
        self.filter_buffer.append(data.force_data.copy())
        
        if len(self.filter_buffer) > self.filter_size:
            self.filter_buffer.pop(0)
        
        if len(self.filter_buffer) >= 3:
            # 计算中值
            filtered_forces = []
            for i in range(len(data.force_data)):
                values = [buf[i] for buf in self.filter_buffer if i < len(buf)]
                if values:
                    sorted_values = sorted(values)
                    median_idx = len(sorted_values) // 2
                    filtered_forces.append(sorted_values[median_idx])
                else:
                    filtered_forces.append(data.force_data[i])
            
            data.force_data = filtered_forces
        
        return data
    
    def _apply_kalman_filter(self, data: SensorData) -> SensorData:
        """应用卡尔曼滤波器"""
        # 简化的卡尔曼滤波器实现
        # 这里使用一阶低通滤波器近似
        if self.kalman_state is None:
            self.kalman_state = [0.0] * len(data.force_data)
            self.kalman_covariance = [1.0] * len(data.force_data)
        
        # 卡尔曼滤波器参数
        Q = 0.01  # 过程噪声
        R = 0.1   # 测量噪声
        
        filtered_forces = []
        for i, measurement in enumerate(data.force_data):
            if i >= len(self.kalman_state):
                filtered_forces.append(measurement)
                continue
            
            # 预测
            predicted_state = self.kalman_state[i]
            predicted_covariance = self.kalman_covariance[i] + Q
            
            # 更新
            kalman_gain = predicted_covariance / (predicted_covariance + R)
            new_state = predicted_state + kalman_gain * (measurement - predicted_state)
            new_covariance = (1 - kalman_gain) * predicted_covariance
            
            self.kalman_state[i] = new_state
            self.kalman_covariance[i] = new_covariance
            
            filtered_forces.append(new_state)
        
        data.force_data = filtered_forces
        return data
    
    def _normalize_data(self, data: SensorData) -> SensorData:
        """
        归一化数据
        
        Args:
            data: 原始数据
            
        Returns:
            归一化后的数据
        """
        if not data.force_data:
            return data
        
        # 简单的最大最小值归一化
        force_range = getattr(self.config.hardware.sensor, 'force_range', (0, 100)) if hasattr(self.config.hardware, 'sensor') else (0, 100)
        max_force = force_range[1]
        
        if max_force > 0:
            normalized_forces = [f / max_force for f in data.force_data]
            data.force_data = normalized_forces
        
        return data
    
    def start_acquisition(self):
        """开始数据采集"""
        if not self.running:
            self.start()
        elif self.paused:
            self.resume_acquisition()
    
    def stop_acquisition(self):
        """停止数据采集"""
        self.running = False
        self.resume_acquisition()  # 确保线程不会在暂停状态下卡住
        self.wait()
    
    def pause_acquisition(self):
        """暂停数据采集"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
        self.status_changed.emit("paused", {"message": "数据采集暂停"})
        self.logger.info("数据采集暂停")
    
    def resume_acquisition(self):
        """恢复数据采集"""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
        self.status_changed.emit("running", {"message": "数据采集恢复"})
        self.logger.info("数据采集恢复")
    
    def get_latest_data(self) -> Optional[SensorData]:
        """
        获取最新数据
        
        Returns:
            最新传感器数据
        """
        return self.data_buffer.get_latest()
    
    def get_data_range(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[SensorData]:
        """
        获取数据范围
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            
        Returns:
            数据列表
        """
        return self.data_buffer.get_range(start_idx, end_idx)
    
    def clear_buffer(self):
        """清空数据缓冲区"""
        self.data_buffer.clear()
        self.logger.info("数据缓冲区已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        buffer_stats = self.data_buffer.get_statistics()
        
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            'runtime': runtime,
            'read_count': self.read_count,
            'error_count': self.error_count,
            'read_rate': self.read_count / runtime if runtime > 0 else 0,
            'buffer_stats': buffer_stats
        }
    
    def save_data(self, filepath: str):
        """
        保存数据到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            import json
            
            # 获取所有数据
            all_data = self.data_buffer.get_range()
            
            # 转换为可序列化的格式
            serializable_data = []
            for data in all_data:
                data_dict = data.to_dict()
                data_dict['timestamp_str'] = datetime.fromtimestamp(data.timestamp).isoformat()
                serializable_data.append(data_dict)
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise
    
    def load_data(self, filepath: str) -> bool:
        """
        从文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载是否成功
        """
        try:
            import json
            
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # 清空当前缓冲区
            self.data_buffer.clear()
            
            # 加载数据到缓冲区
            for item in loaded_data:
                # 从字典恢复数据
                sensor_data = SensorData(
                    timestamp=item['timestamp'],
                    force_data=item['force_data'],
                    temperature=item.get('temperature'),
                    status=item.get('status', 0),
                    sequence_id=item.get('sequence_id', 0)
                )
                self.data_buffer.add_data(sensor_data)
            
            self.logger.info(f"数据已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return False
    
    def is_running(self) -> bool:
        """检查是否在运行"""
        return self.running and not self.paused
    
    def is_paused(self) -> bool:
        """检查是否暂停"""
        return self.paused


# 测试函数
def test_data_acquisition():
    """测试数据采集模块"""
    import time
    
    print("测试数据采集模块...")
    
    # 创建模拟硬件接口
    class MockHardwareInterface:
        def __init__(self):
            self.counter = 0
            
        def read_sensor(self):
            self.counter += 1
            return SensorReading(
                timestamp=time.time(),
                force_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                temperature=25.0,
                status=0
            )
    
    # 创建模拟配置
    class MockConfig:
        class Hardware:
            class Sensor:
                def __init__(self):
                    self.filter_enabled = True
                    self.filter_type = "lowpass"
                    self.sampling_rate = 10
                    self.normalize_data = True
                    self.force_range = (0, 100)
        
        def __init__(self):
            self.hardware = MockConfig.Hardware()
            self.hardware.sensor = MockConfig.Hardware.Sensor()
    
    # 创建数据采集线程
    hardware = MockHardwareInterface()
    config = MockConfig()
    
    thread = DataAcquisitionThread(hardware, config)
    
    # 设置信号处理
    def handle_new_data(data):
        print(f"收到数据: ID={data.sequence_id}, 时间={data.timestamp:.3f}, 总力={data.total_force:.2f}")
    
    def handle_status(status, info):
        print(f"状态变化: {status}, 信息={info}")
    
    thread.new_data.connect(handle_new_data)
    thread.status_changed.connect(handle_status)
    
    # 启动数据采集
    thread.start_acquisition()
    
    # 运行几秒钟
    print("数据采集运行中...")
    time.sleep(3)
    
    # 获取统计信息
    stats = thread.get_statistics()
    print(f"统计信息: {stats}")
    
    # 停止数据采集
    thread.stop_acquisition()
    
    print("数据采集测试完成")


if __name__ == "__main__":
    test_data_acquisition()
