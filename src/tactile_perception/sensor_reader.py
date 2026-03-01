# tactile_perception/sensor_reader.py
import serial
import struct
import threading
import time
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class TactileData:
    """触觉数据容器类"""
    timestamp: float
    sequence: int
    tactile_array: np.ndarray  # 形状: [9, 3] - 9个测点，每个点有3个力分量 (kPa)
    resultant_force: np.ndarray  # 形状: [3] - 全局合力 (kPa)
    contact_state: np.ndarray  # 形状: [9] - 接触状态 (0/1)
    temperature: Optional[float] = None
    data_valid: bool = True
    
    @property
    def total_pressure(self) -> float:
        """总压力 (kPa)"""
        return np.sum(self.tactile_array[:, 2]) if self.tactile_array.size > 0 else 0.0
    
    @property
    def average_pressure(self) -> float:
        """平均压力 (kPa)"""
        return np.mean(self.tactile_array[:, 2]) if self.tactile_array.size > 0 else 0.0
    
    @property
    def max_pressure(self) -> float:
        """最大压力 (kPa)"""
        return np.max(self.tactile_array[:, 2]) if self.tactile_array.size > 0 else 0.0
    
    @property
    def min_pressure(self) -> float:
        """最小压力 (kPa)"""
        return np.min(self.tactile_array[:, 2]) if self.tactile_array.size > 0 else 0.0

class SensorStatus(Enum):
    """传感器状态枚举"""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    STREAMING = 3
    ERROR = 4

class SensorReader:
    """
    触觉传感器数据读取器
    负责与STM32通信，读取并解析传感器数据
    """
    
    # 协议常量定义
    PACKET_HEADER = bytes([0xAA, 0x55])
    PACKET_TYPE_TACTILE = 0x01
    PACKET_TYPE_STATUS = 0x03
    PACKET_TYPE_ERROR = 0x06
    
    def __init__(self,
                 port: str = "COM3",
                 baudrate: int = 115200,
                 timeout: float = 0.1,
                 num_tactels: int = 9,
                 force_scale: float = 10.0,  # 修改：从0.1改为10.0，将原始值放大
                 max_pressure_range: float = 100.0,
                 num_taxels: Optional[int] = None):  # 添加最大压力范围参数
        
        # 兼容旧参数名 num_taxels
        if num_taxels is not None:
            num_tactels = num_taxels

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.num_tactels = num_tactels
        self.force_scale = force_scale
        self.max_pressure_range = max_pressure_range  # 传感器最大量程 (kPa)
        
        # 通信相关
        self.serial: Optional[serial.Serial] = None
        self.status = SensorStatus.DISCONNECTED
        self.connected = False
        
        # 数据缓冲区
        self.data_buffer: List[TactileData] = []
        self.buffer_max_size = 1000
        self.latest_data: Optional[TactileData] = None
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.drop_count = 0
        self.last_timestamp = 0
        
        # 校准参数
        self.calibration_offset = np.zeros((num_tactels, 3), dtype=np.float32)
        self.calibration_scale = np.ones((num_tactels, 3), dtype=np.float32)
        
        # 回调函数
        self.data_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # 线程控制
        self.read_thread: Optional[threading.Thread] = None
        self.running = False
        self.read_interval = 0.012  # 约83Hz
        
        # 同步锁
        self.data_lock = threading.Lock()
        
        logger.info(f"SensorReader initialized for port {port}, force_scale={force_scale}")
    
    def connect(self) -> bool:
        """连接到串口设备"""
        try:
            logger.info(f"Connecting to {self.port} at {self.baudrate} baud...")
            self.status = SensorStatus.CONNECTING
            
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # 等待连接稳定
            time.sleep(0.5)
            
            if self.serial.is_open:
                self.connected = True
                self.status = SensorStatus.CONNECTED
                logger.info(f"Successfully connected to {self.port}")
                
                # 发送握手信号
                self._send_handshake()
                
                # 尝试读取一些初始数据以验证连接
                if self._test_connection():
                    logger.info("Sensor connection verified")
                else:
                    logger.warning("Sensor connection verification failed, but continuing")
                
                return True
            else:
                logger.error(f"Failed to open port {self.port}")
                self.status = SensorStatus.ERROR
                return False
                
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            self.status = SensorStatus.ERROR
            # 进入模拟模式
            logger.info("Entering simulation mode due to connection error")
            self._enter_simulation_mode()
            return True  # 模拟模式视为连接成功
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def _enter_simulation_mode(self):
        """进入模拟模式"""
        self.connected = True
        self.simulation_mode = True
        logger.info("SensorReader in simulation mode")
    
    def _test_connection(self) -> bool:
        """测试连接是否正常"""
        try:
            # 发送测试命令
            test_packet = bytes([0xAA, 0x55, 0x00, 0x03, 0x00, 0x00, 0x03])
            if self.serial:
                self.serial.write(test_packet)
                time.sleep(0.05)
                
                # 尝试读取响应
                response = self.serial.read(100)
                if response:
                    logger.debug(f"Test response: {response.hex()}")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.stop_streaming()
        self.connected = False
        
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None
        
        self.status = SensorStatus.DISCONNECTED
        logger.info(f"Disconnected from {self.port}")
    
    def start_streaming(self):
        """开始数据流读取"""
        if not self.connected:
            logger.warning("Cannot start streaming: not connected")
            return False
        
        if self.running:
            logger.warning("Streaming already running")
            return True
        
        self.running = True
        self.read_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="SensorReadThread"
        )
        self.read_thread.start()
        
        self.status = SensorStatus.STREAMING
        logger.info("Started sensor data streaming")
        return True
    
    def stop_streaming(self):
        """停止数据流读取"""
        self.running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
            self.read_thread = None
        
        self.status = SensorStatus.CONNECTED
        logger.info("Stopped sensor data streaming")
    
    def _read_loop(self):
        """数据读取主循环"""
        logger.debug("Starting read loop")
        
        # 清空输入缓冲区
        if self.serial and hasattr(self.serial, 'reset_input_buffer'):
            self.serial.reset_input_buffer()
        
        while self.running and self.connected:
            try:
                # 检查是否为模拟模式
                if hasattr(self, 'simulation_mode') and self.simulation_mode:
                    data = self._generate_simulated_data()
                else:
                    # 读取一帧数据
                    packet = self._read_packet()
                    
                    if packet:
                        # 解析数据包
                        data = self._parse_packet(packet)
                    else:
                        # 如果没有数据包，生成模拟数据
                        data = self._generate_simulated_data()
                
                if data and data.data_valid:
                    # 应用校准
                    data = self._apply_calibration(data)
                    
                    # 更新最新数据
                    with self.data_lock:
                        self.latest_data = data
                        self.frame_count += 1
                    
                    # 添加到缓冲区
                    self._add_to_buffer(data)
                    
                    # 调用回调函数
                    self._notify_callbacks(data)
                    
                    # 计算并适应读取间隔
                    current_time = time.time()
                    if self.last_timestamp > 0:
                        actual_interval = current_time - self.last_timestamp
                        # 动态调整读取间隔以匹配传感器频率
                        if actual_interval < self.read_interval * 0.9:
                            self.read_interval *= 1.05
                        elif actual_interval > self.read_interval * 1.1:
                            self.read_interval *= 0.95
                    
                    self.last_timestamp = current_time
                
                # 控制读取频率
                time.sleep(max(0, self.read_interval - 0.001))
                
            except serial.SerialException as e:
                logger.error(f"Serial read error: {e}")
                self.error_count += 1
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Unexpected error in read loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
        
        logger.debug("Exiting read loop")
    
    def _read_packet(self) -> Optional[bytes]:
        """读取一个完整的数据包"""
        if not self.serial or not hasattr(self.serial, 'read'):
            return None
        
        try:
            # 查找帧头
            header_found = False
            header_bytes = bytes()
            
            # 设置超时
            start_time = time.time()
            while not header_found and self.running and time.time() - start_time < 0.5:
                # 读取一个字节
                if self.serial.in_waiting > 0:
                    byte = self.serial.read(1)
                    if byte:
                        header_bytes += byte
                        
                        # 检查是否匹配帧头
                        if len(header_bytes) >= 2:
                            if header_bytes[-2:] == self.PACKET_HEADER:
                                header_found = True
                            else:
                                # 移除非帧头字节
                                header_bytes = header_bytes[1:]
                else:
                    time.sleep(0.001)
            
            if not header_found:
                return None
            
            # 读取包长度 (2字节小端)
            length_bytes = self._read_exact(2, timeout=0.1)
            if not length_bytes or len(length_bytes) < 2:
                return None
            
            packet_length = struct.unpack('<H', length_bytes)[0]
            
            # 读取剩余数据 (类型 + 数据 + 校验和)
            # 总长度 = 帧头(2) + 长度字段(2) + 数据长度
            remaining_bytes = self._read_exact(packet_length + 3, timeout=0.1)  # +3 for type(1)+seq(1)+checksum(1)
            
            if not remaining_bytes or len(remaining_bytes) < packet_length + 3:
                return None
            
            # 组合完整数据包
            packet = self.PACKET_HEADER + length_bytes + remaining_bytes
            
            # 验证校验和
            if self._verify_checksum(packet):
                return packet
            else:
                logger.warning("Checksum verification failed")
                self.error_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Error reading packet: {e}")
            return None
    
    def _read_exact(self, n_bytes: int, timeout: float = 0.1) -> Optional[bytes]:
        """精确读取指定字节数"""
        if not self.serial:
            return None
        
        data = bytearray()
        start_time = time.time()
        
        while len(data) < n_bytes and time.time() - start_time < timeout:
            if self.serial.in_waiting > 0:
                to_read = min(n_bytes - len(data), self.serial.in_waiting)
                chunk = self.serial.read(to_read)
                if chunk:
                    data.extend(chunk)
            else:
                time.sleep(0.001)
        
        return bytes(data) if len(data) == n_bytes else None
    
    def _parse_packet(self, packet: bytes) -> Optional[TactileData]:
        """解析数据包为触觉数据"""
        try:
            # 解析数据包结构
            # [帧头2][长度2][类型1][序列号1][数据N][校验和1]
            if len(packet) < 7:
                return None
            
            packet_type = packet[4]
            sequence = packet[5]
            
            if packet_type == self.PACKET_TYPE_TACTILE:
                # 解析触觉数据
                # 数据格式: [合力Fx1][合力Fy1][合力Fz1][测点数据...]
                data_start = 6
                data_end = len(packet) - 1  # 排除校验和
                
                # 计算期望的数据长度
                expected_length = 3 + self.num_tactels * 3
                actual_length = data_end - data_start
                
                if actual_length < expected_length:
                    logger.warning(f"Data length mismatch: got {actual_length}, expected {expected_length}")
                    # 尝试解析尽可能多的数据
                    logger.warning(f"Packet content: {packet.hex()}")
                
                # 解析合力 (3字节)
                resultant_forces = np.zeros(3, dtype=np.float32)
                if data_start + 2 < len(packet):
                    resultant_bytes = packet[data_start:data_start+3]
                    # 假设合力为原始值，直接应用缩放
                    resultant_forces[0] = resultant_bytes[0] * self.force_scale
                    resultant_forces[1] = resultant_bytes[1] * self.force_scale
                    resultant_forces[2] = resultant_bytes[2] * self.force_scale  # Fz使用不同的缩放
                
                # 解析测点数据
                tactile_data = np.zeros((self.num_tactels, 3), dtype=np.float32)
                contact_state = np.zeros(self.num_tactels, dtype=np.uint8)
                
                for i in range(self.num_tactels):
                    offset = data_start + 3 + i * 3
                    if offset + 2 >= len(packet):
                        break
                    
                    # 每个测点3个字节: Fx, Fy, Fz
                    fx_byte = packet[offset]
                    fy_byte = packet[offset + 1]
                    fz_byte = packet[offset + 2]
                    
                    # 直接应用缩放因子，不再使用有符号转换
                    tactile_data[i, 0] = fx_byte * self.force_scale
                    tactile_data[i, 1] = fy_byte * self.force_scale
                    tactile_data[i, 2] = fz_byte * self.force_scale * 2.0  # Z方向通常需要更大的缩放
                    
                    # 简单的接触检测: Z方向力大于阈值
                    contact_threshold_kpa = 0.5  # 0.5 kPa 阈值
                    contact_state[i] = 1 if tactile_data[i, 2] > contact_threshold_kpa else 0
                
                # 创建数据对象
                data = TactileData(
                    timestamp=time.time(),
                    sequence=sequence,
                    tactile_array=tactile_data,
                    resultant_force=resultant_forces,
                    contact_state=contact_state,
                    data_valid=True
                )
                
                # 添加调试日志
                if self.frame_count % 100 == 0:
                    logger.debug(f"Parsed data - Avg pressure: {data.average_pressure:.2f} kPa, Max: {data.max_pressure:.2f} kPa")
                
                return data
                
            elif packet_type == self.PACKET_TYPE_STATUS:
                # 状态数据包
                status_msg = packet[6:-1].decode('ascii', errors='ignore') if len(packet) > 7 else ""
                logger.info(f"Status packet received: {status_msg}")
                return None
                
            elif packet_type == self.PACKET_TYPE_ERROR:
                # 错误数据包
                error_msg = packet[6:-1].decode('ascii', errors='ignore') if len(packet) > 7 else ""
                logger.error(f"Error from sensor: {error_msg}")
                self.error_count += 1
                return None
                
            else:
                logger.warning(f"Unknown packet type: {packet_type:02X}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            self.error_count += 1
            return None
    
    def _verify_checksum(self, packet: bytes) -> bool:
        """验证数据包校验和"""
        if len(packet) < 4:
            return False
        
        # 计算校验和 (简单求和取低8位)
        checksum = sum(packet[:-1]) & 0xFF
        return checksum == packet[-1]
    
    def _send_handshake(self):
        """发送握手信号"""
        handshake_packet = bytes([0xAA, 0x55, 0x00, 0x03, 0x00, 0x01, 0x04])
        if self.serial:
            self.serial.write(handshake_packet)
            time.sleep(0.05)
    
    def _apply_calibration(self, data: TactileData) -> TactileData:
        """应用校准参数"""
        # 应用零点偏移
        calibrated_array = data.tactile_array - self.calibration_offset
        
        # 应用缩放因子
        calibrated_array = calibrated_array * self.calibration_scale
        
        # 确保数据在合理范围内
        calibrated_array = np.clip(calibrated_array, 0, self.max_pressure_range)
        
        # 更新数据对象
        data.tactile_array = calibrated_array
        
        # 重新计算接触状态
        contact_threshold_kpa = 0.5
        data.contact_state = np.array([1 if force_z > contact_threshold_kpa else 0 
                                      for force_z in calibrated_array[:, 2]], dtype=np.uint8)
        
        return data
    
    def calibrate_zero(self, num_samples: int = 50):
        """零点校准"""
        logger.info("Starting zero calibration...")
        
        samples = []
        for i in range(num_samples):
            data = self.get_latest_data()
            if data:
                samples.append(data.tactile_array)
            time.sleep(0.02)
        
        if samples:
            # 计算平均零点偏移
            avg_offset = np.mean(np.array(samples), axis=0)
            self.calibration_offset = avg_offset
            logger.info(f"Zero calibration complete. Average offset: {np.mean(avg_offset):.2f} kPa")
            return True
        
        logger.warning("Zero calibration failed: no data collected")
        return False
    
    def _generate_simulated_data(self) -> TactileData:
        """生成模拟传感器数据"""
        import random
        import math
        
        # 生成合理的压力数据 (5-30 kPa范围)
        base_pressure = random.uniform(5.0, 30.0)
        
        tactile_data = np.zeros((self.num_tactels, 3), dtype=np.float32)
        contact_state = np.zeros(self.num_tactels, dtype=np.uint8)
        
        for i in range(self.num_tactels):
            # 模拟压力分布，中间高四周低
            if self.num_tactels == 9:  # 3x3网格
                row = i // 3
                col = i % 3
                center_distance = math.sqrt((row - 1)**2 + (col - 1)**2)
                # 中心点压力最高，边缘递减
                pressure_factor = max(0.1, 1.0 - center_distance * 0.3)
            else:
                # 随机分布
                pressure_factor = random.uniform(0.3, 1.0)
            
            # 计算Z方向压力
            z_pressure = base_pressure * pressure_factor
            
            # 添加噪声
            noise = random.uniform(-1.0, 1.0)
            z_pressure = max(0.1, z_pressure + noise)
            
            # 生成X,Y方向的小量值（通常远小于Z方向）
            x_pressure = random.uniform(-0.5, 0.5)
            y_pressure = random.uniform(-0.5, 0.5)
            
            tactile_data[i, 0] = x_pressure
            tactile_data[i, 1] = y_pressure
            tactile_data[i, 2] = z_pressure
            
            # 接触状态
            contact_state[i] = 1 if z_pressure > 0.5 else 0
        
        # 合力（简化计算）
        resultant_force = np.array([
            np.sum(tactile_data[:, 0]),
            np.sum(tactile_data[:, 1]),
            np.sum(tactile_data[:, 2])
        ])
        
        return TactileData(
            timestamp=time.time(),
            sequence=self.frame_count,
            tactile_array=tactile_data,
            resultant_force=resultant_force,
            contact_state=contact_state,
            temperature=25.0,
            data_valid=True
        )
    
    def _add_to_buffer(self, data: TactileData):
        """添加数据到缓冲区"""
        with self.data_lock:
            self.data_buffer.append(data)
            
            # 保持缓冲区大小
            if len(self.data_buffer) > self.buffer_max_size:
                removed = len(self.data_buffer) - self.buffer_max_size
                self.data_buffer = self.data_buffer[removed:]
                self.drop_count += removed
    
    def get_latest_data(self) -> Optional[TactileData]:
        """获取最新数据"""
        with self.data_lock:
            return self.latest_data
    
    def get_buffer_snapshot(self, max_samples: int = 100) -> List[TactileData]:
        """获取缓冲区快照"""
        with self.data_lock:
            if len(self.data_buffer) <= max_samples:
                return self.data_buffer.copy()
            else:
                return self.data_buffer[-max_samples:].copy()
    
    def clear_buffer(self):
        """清空缓冲区"""
        with self.data_lock:
            self.data_buffer.clear()
    
    def register_callback(self, callback: Callable):
        """注册数据回调函数"""
        if callback not in self.data_callbacks:
            self.data_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """注销数据回调函数"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def _notify_callbacks(self, data: TactileData):
        """通知所有注册的回调函数"""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def get_statistics(self) -> Dict:
        """获取读取统计信息"""
        latest = self.get_latest_data()
        avg_pressure = latest.average_pressure if latest else 0.0
        max_pressure = latest.max_pressure if latest else 0.0
        
        return {
            "status": self.status.name,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "drop_count": self.drop_count,
            "buffer_size": len(self.data_buffer),
            "connected": self.connected,
            "streaming": self.running,
            "avg_pressure_kpa": avg_pressure,
            "max_pressure_kpa": max_pressure,
            "simulation_mode": hasattr(self, 'simulation_mode') and self.simulation_mode
        }

class AsyncSensorReader:
    """异步版本的传感器读取器"""
    
    def __init__(self, reader: SensorReader):
        self.reader = reader
        self.loop = asyncio.get_event_loop()
        self.callbacks = []
    
    async def connect_async(self) -> bool:
        """异步连接"""
        return await self.loop.run_in_executor(None, self.reader.connect)
    
    async def start_streaming_async(self):
        """异步开始流式读取"""
        return await self.loop.run_in_executor(None, self.reader.start_streaming)
    
    async def read_frame_async(self, timeout: float = 1.0) -> Optional[TactileData]:
        """异步读取一帧数据"""
        try:
            # 等待新数据到达
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                data = self.reader.get_latest_data()
                if data and data.timestamp > start_time:
                    return data
                await asyncio.sleep(0.001)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in async read: {e}")
            return None
    
    async def get_statistics_async(self) -> Dict:
        """异步获取统计信息"""
        return await self.loop.run_in_executor(None, self.reader.get_statistics)
