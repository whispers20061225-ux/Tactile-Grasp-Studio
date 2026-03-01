"""



触觉夹爪演示系统 - 演示管理器模块



负责管理各种演示模式和系统状态。



支持三维力数据和矢量图可视化



"""



import threading



import time



import json



import logging



import numpy as np



import math



from typing import Dict, Any, Optional, List, Callable, Tuple



from dataclasses import dataclass, field



from enum import Enum



from datetime import datetime



from PyQt5.QtCore import QObject, pyqtSignal



# 修改导入方式 - 使用绝对导入



try:



    # 方案1：先尝试绝对导入



    from config import DemoConfig



except ImportError:



    try:



        # 方案2：尝试相对导入



        from ..config import DemoConfig



    except ImportError:



        try:



            # 方案3：直接导入



            import sys



            import os



            # 添加父目录到路径



            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



            from config import DemoConfig



        except ImportError as e:



            raise ImportError(f"无法导入 DemoConfig: {e}")



# 导入其他模块



from .hardware_interface import HardwareInterface, SensorReading



from .data_acquisition import DataAcquisitionThread, SensorData



from .control_thread import ControlThread



# 尝试导入 tactile_perception



try:



    from ..tactile_perception import create_default_pipeline



    TACTILE_PERCEPTION_AVAILABLE = True



except ImportError:



    try:



        from tactile_perception import create_default_pipeline



        TACTILE_PERCEPTION_AVAILABLE = True



    except ImportError:



        TACTILE_PERCEPTION_AVAILABLE = False



        print("警告: tactile_perception 不可用，传感器管道功能将被禁用")



# 导入深度学习模型 - 使用绝对导入



try:



    from deep_learning.models import GripNet



    GRIPNET_AVAILABLE = True



except ImportError:



    try:



        from ..deep_learning.models import GripNet



        GRIPNET_AVAILABLE = True



    except ImportError:



        GRIPNET_AVAILABLE = False



        print("警告: GripNet 不可用，深度学习功能将被禁用")



class DemoMode(Enum):



    """演示模式枚举"""



    IDLE = "idle"



    CALIBRATION = "calibration"



    GRASPING = "grasping"



    SLIP_DETECTION = "slip_detection"



    OBJECT_CLASSIFICATION = "object_classification"



    FORCE_CONTROL = "force_control"



    LEARNING = "learning"



    VECTOR_VISUALIZATION = "vector_visualization"  # 新增：矢量可视化模式



    TACTILE_MAPPING = "tactile_mapping"  # 触觉映射/标定模式



class DemoStatus(Enum):



    """演示状态枚举"""



    IDLE = "idle"



    INITIALIZING = "initializing"



    RUNNING = "running"



    PAUSED = "paused"



    COMPLETED = "completed"



    ERROR = "error"



    STOPPED = "stopped"



@dataclass



class DemoResult:



    """演示结果"""



    demo_name: str



    start_time: float



    end_time: float



    success: bool



    data_points: int = 0



    metrics: Dict[str, Any] = field(default_factory=dict)



    error_message: str = ""



    



    @property



    def duration(self) -> float:



        """演示持续时间"""



        return self.end_time - self.start_time



    



    def to_dict(self) -> Dict[str, Any]:



        """转换为字典"""



        return {



            'demo_name': self.demo_name,



            'start_time': self.start_time,



            'end_time': self.end_time,



            'duration': self.duration,



            'success': self.success,



            'data_points': self.data_points,



            'metrics': self.metrics,



            'error_message': self.error_message



        }



class DemoManager(QObject):



    """演示管理器 - 支持三维力数据和矢量可视化"""



    



    # 信号定义



    status_changed = pyqtSignal(str, dict)  # 状态变化 (status, info)



    demo_started = pyqtSignal(str, dict)  # 演示开始 (demo_name, params)



    demo_stopped = pyqtSignal(str, dict)  # 演示停止 (demo_name, result)



    demo_progress = pyqtSignal(float, dict)  # 演示进度 (progress, info)



    error_occurred = pyqtSignal(str, dict)  # 错误发生 (error_type, info)



    



    # 新增：三维力数据信号



    vector_data_updated = pyqtSignal(object)  # 三维力向量数据更新



    force_statistics_updated = pyqtSignal(dict)  # 力统计信息更新



    contact_map_updated = pyqtSignal(object)  # 接触地图更新



    tactile_mapping_ready = pyqtSignal(list)  # 触觉映射结果，用于主线程显示



    



    def __init__(self, config: DemoConfig,



                 hardware_interface: HardwareInterface,



                 data_acquisition: DataAcquisitionThread,



                 control_thread: ControlThread):



        """



        初始化演示管理器



        



        Args:



            config: 系统配置



            hardware_interface: 硬件接口



            data_acquisition: 数据采集线程



            control_thread: 控制线程



        """



        super().__init__()



        



        self.config = config



        self.hardware_interface = hardware_interface



        self.data_acquisition = data_acquisition



        self.control_thread = control_thread



        



        # 演示状态



        self.current_demo = None



        self.demo_status = DemoStatus.IDLE



        self.demo_params = {}



        self.demo_start_time = 0.0



        self.demo_progress_value = 0.0



        



        # 三维力数据状态



        self.force_history = {



            'fx': [], 'fy': [], 'fz': [],  # 合力历史



            'fx_taxels': [], 'fy_taxels': [], 'fz_taxels': [],  # 各触点力历史



            'timestamps': []



        }



        self.max_history_points = 1000  # 历史数据点最大数量



        self.force_statistics = {



            'current': {'fx': 0, 'fy': 0, 'fz': 0, 'magnitude': 0},



            'avg': {'fx': 0, 'fy': 0, 'fz': 0, 'magnitude': 0},



            'max': {'fx': 0, 'fy': 0, 'fz': 0, 'magnitude': 0},



            'min': {'fx': 0, 'fy': 0, 'fz': 0, 'magnitude': 0}



        }



        



        # 接触状态



        self.contact_map = np.zeros(9, dtype=bool)  # 3x3网格的接触状态



        self.contact_forces = np.zeros((9, 3), dtype=np.float32)  # 各触点三维力



        



        # 演示结果



        self.demo_results = []



        self.max_results_history = 10



        



        # 触觉感知管道



        self.sensor_pipeline = None



        



        # 深度学习模型



        self.grip_net = None



        self.model_loaded = False



        



        # 线程同步



        self.lock = threading.Lock()



        



        # 演示处理器映射 - 添加矢量可视化处理器



        self.demo_handlers = {



            DemoMode.CALIBRATION.value: self._run_calibration_demo,



            DemoMode.GRASPING.value: self._run_grasping_demo,



            DemoMode.SLIP_DETECTION.value: self._run_slip_detection_demo,



            DemoMode.OBJECT_CLASSIFICATION.value: self._run_object_classification_demo,



            DemoMode.FORCE_CONTROL.value: self._run_force_control_demo,



            DemoMode.LEARNING.value: self._run_learning_demo,



            DemoMode.VECTOR_VISUALIZATION.value: self._run_vector_visualization_demo,  # 新增



            DemoMode.TACTILE_MAPPING.value: self._run_tactile_mapping_demo  # 新增触觉映射演示



        }



        



        # 日志



        self.logger = logging.getLogger(__name__)



        



        # 初始化



        self._init_sensor_pipeline()



        self._init_model()



        



        # 启动三维力数据更新线程



        self.vector_update_thread = threading.Thread(target=self._update_vector_data, daemon=True)



        self.vector_update_running = True



        self.vector_update_thread.start()



        



        self.logger.info("三维力演示管理器初始化完成")



    



    def _init_sensor_pipeline(self):



        """初始化传感器管道"""



        try:



            if not TACTILE_PERCEPTION_AVAILABLE:



                self.logger.warning("tactile_perception 不可用，跳过传感器管道初始化")



                self.sensor_pipeline = None



                return



            



            # 获取传感器端口和波特率



            sensor_port = getattr(self.config.hardware.sensor, 'port', 'COM3')



            sensor_baudrate = getattr(self.config.hardware.sensor, 'baudrate', 115200)



            # 兼容旧版 create_default_pipeline，只传常见参数



            self.sensor_pipeline = create_default_pipeline(



                port=sensor_port,



                baudrate=sensor_baudrate



            )



            self.logger.info("传感器管道初始化成功")



        except Exception as e:



            self.logger.error(f"传感器管道初始化失败: {e}")



            self.sensor_pipeline = None



    



    def _init_model(self):



        """初始化深度学习模型"""



        try:



            if not GRIPNET_AVAILABLE:



                self.logger.warning("GripNet不可用，跳过模型初始化")



                self.model_loaded = False



                return



            



            # 尝试从配置获取模型路径



            model_path = None



            if hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'model_path'):



                model_path = self.config.algorithm.model_path



            elif hasattr(self.config, 'deep_learning') and hasattr(self.config.deep_learning, 'model_path'):



                model_path = self.config.deep_learning.model_path



            



            # 获取输入输出大小 - 适应三维力数据



            input_size = 30  # 默认时间步长 * 3维



            output_size = 3  # 控制输出（位置、速度、力）



            num_taxels = 9   # Paxini Gen3 M2020 有9个taxel



            



            # 从配置获取参数



            if hasattr(self.config, 'algorithm'):



                if hasattr(self.config.algorithm, 'input_size'):



                    input_size = self.config.algorithm.input_size



                if hasattr(self.config.algorithm, 'output_size'):



                    output_size = self.config.algorithm.output_size



            



            if hasattr(self.config, 'sensor') and hasattr(self.config.sensor, 'num_taxels'):



                num_taxels = self.config.sensor.num_taxels



            



            # 检查是否配置了三维力



            force_dimensions = getattr(self.config.algorithm, 'force_dimensions', 3)



            



            if model_path:



                # GripNet 初始化参数按模型定义适配，移除 force_dimensions 不支持的参数



                self.grip_net = GripNet(



                    input_size=input_size,



                    output_size=output_size,



                    num_taxels=num_taxels



                )



                # 这里应该加载预训练权重



                # self.grip_net.load_model(model_path)



                self.model_loaded = True



                self.logger.info(f"深度学习模型初始化成功")



            else:



                self.logger.warning("没有指定模型路径，使用空模型")



                # 即使没有模型路径，也创建一个模型实例



                self.grip_net = GripNet(



                    input_size=input_size,



                    output_size=output_size,



                    num_taxels=num_taxels



                )



                self.model_loaded = True



                



        except Exception as e:



            self.logger.warning(f"深度学习模型初始化失败: {e}")



            self.model_loaded = False



    



    def _update_vector_data(self):



        """更新三维力数据（独立线程）"""



        update_interval = 0.05  # 20Hz更新频率



        



        while self.vector_update_running:



            try:



                # 获取最新的传感器数据



                latest_data = self.data_acquisition.get_latest_data()



                



                if latest_data and hasattr(latest_data, 'force_vectors'):



                    force_vectors = np.asarray(latest_data.force_vectors)



                    if force_vectors.size == 0:



                        time.sleep(update_interval)



                        continue



                    



                    # 计算合力



                    resultant_force = np.sum(force_vectors, axis=0)



                    fx, fy, fz = resultant_force



                    magnitude = math.sqrt(fx**2 + fy**2 + fz**2)



                    



                    current_time = time.time()



                    



                    # 更新历史数据



                    with self.lock:



                        self.force_history['fx'].append(fx)



                        self.force_history['fy'].append(fy)



                        self.force_history['fz'].append(fz)



                        self.force_history['timestamps'].append(current_time)



                        



                        # 保存各触点力



                        if len(self.force_history['fx_taxels']) < 100:  # 只保存最近100帧



                            self.force_history['fx_taxels'].append(force_vectors[:, 0])



                            self.force_history['fy_taxels'].append(force_vectors[:, 1])



                            self.force_history['fz_taxels'].append(force_vectors[:, 2])



                        



                        # 限制历史数据大小



                        for key in ['fx', 'fy', 'fz', 'timestamps']:



                            if len(self.force_history[key]) > self.max_history_points:



                                self.force_history[key] = self.force_history[key][-self.max_history_points:]



                        



                        # 更新统计信息



                        if self.force_history['fx']:



                            self.force_statistics['current'] = {



                                'fx': fx, 'fy': fy, 'fz': fz, 'magnitude': magnitude



                            }



                            



                            # 计算平均值



                            self.force_statistics['avg'] = {



                                'fx': np.mean(self.force_history['fx'][-100:]),



                                'fy': np.mean(self.force_history['fy'][-100:]),



                                'fz': np.mean(self.force_history['fz'][-100:]),



                                'magnitude': np.mean([math.sqrt(fx**2 + fy**2 + fz**2) 



                                                    for fx, fy, fz in zip(self.force_history['fx'][-100:], 



                                                                          self.force_history['fy'][-100:], 



                                                                          self.force_history['fz'][-100:])])



                            }



                            



                            # 计算最大值和最小值



                            self.force_statistics['max'] = {



                                'fx': np.max(self.force_history['fx'][-100:]),



                                'fy': np.max(self.force_history['fy'][-100:]),



                                'fz': np.max(self.force_history['fz'][-100:]),



                                'magnitude': np.max([math.sqrt(fx**2 + fy**2 + fz**2) 



                                                   for fx, fy, fz in zip(self.force_history['fx'][-100:], 



                                                                         self.force_history['fy'][-100:], 



                                                                         self.force_history['fz'][-100:])])



                            }



                            



                            self.force_statistics['min'] = {



                                'fx': np.min(self.force_history['fx'][-100:]),



                                'fy': np.min(self.force_history['fy'][-100:]),



                                'fz': np.min(self.force_history['fz'][-100:]),



                                'magnitude': np.min([math.sqrt(fx**2 + fy**2 + fz**2) 



                                                   for fx, fy, fz in zip(self.force_history['fx'][-100:], 



                                                                         self.force_history['fy'][-100:], 



                                                                         self.force_history['fz'][-100:])])



                            }



                        



                        # 更新接触状态，逐触点计算



                        contact_threshold = 0.05  # 接触阈值 (N) 降低阈值便于显示小力



                        contact_flags = force_vectors[:, 2] > contact_threshold



                        self.contact_map = contact_flags



                        self.contact_forces = force_vectors.copy()



                    



                    # 发送信号



                    vector_data = {



                        'force_vectors': force_vectors.tolist(),



                        'resultant_force': [fx, fy, fz],



                        'force_magnitude': magnitude,



                        'timestamp': current_time,



                        'contact_flags': contact_flags.tolist()



                    }



                    



                    self.vector_data_updated.emit(vector_data)



                    self.force_statistics_updated.emit(self._format_force_statistics(force_vectors, resultant_force, magnitude))



                    self.contact_map_updated.emit(contact_flags)



                



            except Exception as e:



                self.logger.debug(f"更新矢量数据时出错: {e}")



            



            time.sleep(update_interval)



    def _format_force_statistics(self, force_vectors: np.ndarray, resultant_force: np.ndarray, magnitude: float) -> dict:



        """将力统计转换为 UI 组件需要的键格式"""



        stats = {}



        if force_vectors.size == 0:



            return stats



        stats['mean_fx'] = float(np.mean(force_vectors[:, 0]))



        stats['mean_fy'] = float(np.mean(force_vectors[:, 1]))



        stats['mean_fz'] = float(np.mean(force_vectors[:, 2]))



        stats['std_fx'] = float(np.std(force_vectors[:, 0]))



        stats['std_fy'] = float(np.std(force_vectors[:, 1]))



        stats['std_fz'] = float(np.std(force_vectors[:, 2]))



        stats['max_fx'] = float(np.max(force_vectors[:, 0]))



        stats['max_fy'] = float(np.max(force_vectors[:, 1]))



        stats['max_fz'] = float(np.max(force_vectors[:, 2]))



        stats['min_fx'] = float(np.min(force_vectors[:, 0]))



        stats['min_fy'] = float(np.min(force_vectors[:, 1]))



        stats['min_fz'] = float(np.min(force_vectors[:, 2]))



        force_magnitudes = np.linalg.norm(force_vectors, axis=1)



        stats['mean_magnitude'] = float(np.mean(force_magnitudes))



        stats['std_magnitude'] = float(np.std(force_magnitudes))



        stats['max_magnitude'] = float(np.max(force_magnitudes))



        stats['min_magnitude'] = float(np.min(force_magnitudes))



        stats['total_force'] = [float(x) for x in resultant_force]



        stats['resultant_magnitude'] = float(magnitude)



        contact_thresholds = [0, 5, 15, 30]



        contact_counts = [0, 0, 0, 0]



        for mag in force_magnitudes:



            if mag < contact_thresholds[1]:



                contact_counts[0] += 1



            elif mag < contact_thresholds[2]:



                contact_counts[1] += 1



            elif mag < contact_thresholds[3]:



                contact_counts[2] += 1



            else:



                contact_counts[3] += 1



        stats['contact_distribution'] = contact_counts



        hist, bins = np.histogram(force_magnitudes, bins=np.linspace(0, 100, 11))



        stats['force_magnitude_hist'] = hist.tolist()



        stats['force_magnitude_bins'] = bins.tolist()



        return stats



    



    def start_demo(self, demo_name: str, params: Dict[str, Any] = None) -> bool:



        """



        开始演示



        



        Args:



            demo_name: 演示名称



            params: 演示参数



            



        Returns:



            是否成功启动



        """



        with self.lock:



            if self.demo_status != DemoStatus.IDLE:



                self.logger.error(f"无法开始演示，当前状态: {self.demo_status}")



                self.error_occurred.emit("demo_busy", {



                    "current_status": self.demo_status.value



                })



                return False



            



            if demo_name not in self.demo_handlers:



                self.logger.error(f"未知的演示模式: {demo_name}")



                self.error_occurred.emit("demo_not_found", {



                    "demo_name": demo_name



                })



                return False



            



            # 更新状态



            self.current_demo = demo_name



            self.demo_params = params or {}



            self.demo_status = DemoStatus.INITIALIZING



            self.demo_start_time = time.time()



            self.demo_progress_value = 0.0



        



        self.logger.info(f"开始演示: {demo_name}, 参数: {params}")



        



        # 发送信号



        self.status_changed.emit("demo_starting", {



            "demo_name": demo_name,



            "params": self.demo_params



        })



        self.demo_started.emit(demo_name, self.demo_params)



        



        # 启动演示线程



        demo_thread = threading.Thread(



            target=self._run_demo_thread,



            args=(demo_name, self.demo_params),



            daemon=True



        )



        demo_thread.start()



        



        return True



    



    def _run_demo_thread(self, demo_name: str, params: Dict[str, Any]):



        """运行演示线程"""



        try:



            with self.lock:



                self.demo_status = DemoStatus.RUNNING



            



            self.status_changed.emit("demo_running", {



                "demo_name": demo_name,



                "start_time": self.demo_start_time



            })



            



            # 执行演示



            handler = self.demo_handlers.get(demo_name)



            if handler:



                result = handler(params)



            else:



                raise ValueError(f"没有找到演示处理器: {demo_name}")



            



            # 演示完成



            with self.lock:



                self.demo_status = DemoStatus.COMPLETED if result.success else DemoStatus.ERROR



            



            # 保存结果



            self._add_demo_result(result)



            



            # 发送完成信号



            self.demo_stopped.emit(demo_name, result.to_dict())



            



            if result.success:



                self.status_changed.emit("demo_completed", {



                    "demo_name": demo_name,



                    "duration": result.duration,



                    "metrics": result.metrics



                })



                self.logger.info(f"演示完成: {demo_name}, 耗时: {result.duration:.2f}秒")



            else:



                self.status_changed.emit("demo_failed", {



                    "demo_name": demo_name,



                    "error": result.error_message



                })



                self.logger.error(f"演示失败: {demo_name}, 错误: {result.error_message}")



            



        except Exception as e:



            with self.lock:



                self.demo_status = DemoStatus.ERROR



            



            error_msg = str(e)



            result = DemoResult(



                demo_name=demo_name,



                start_time=self.demo_start_time,



                end_time=time.time(),



                success=False,



                error_message=error_msg



            )



            



            # 保存结果



            self._add_demo_result(result)



            



            # 发送错误信号



            self.demo_stopped.emit(demo_name, result.to_dict())



            self.error_occurred.emit("demo_error", {



                "demo_name": demo_name,



                "error": error_msg



            })



            self.logger.error(f"演示执行错误: {demo_name}, 错误: {error_msg}")



        



        finally:



            # 重置状态



            with self.lock:



                # 无论成功/失败，都回到空闲，便于再次启动



                if self.demo_status not in [DemoStatus.COMPLETED, DemoStatus.ERROR]:



                    self.demo_status = DemoStatus.STOPPED



                # 允许下次演示



                self.demo_status = DemoStatus.IDLE



            



                self.current_demo = None



                self.demo_params = {}



                self.demo_progress_value = 0.0



    



    def stop_demo(self) -> bool:



        """



        停止当前演示



        



        Returns:



            是否成功停止



        """



        with self.lock:



            if self.demo_status not in [DemoStatus.RUNNING, DemoStatus.PAUSED]:



                self.logger.warning(f"没有运行的演示可停止，当前状态: {self.demo_status}")



                return False



            



            previous_status = self.demo_status



            self.demo_status = DemoStatus.STOPPED



        



        self.logger.info(f"停止演示: {self.current_demo}")



        



        # 发送停止信号



        result = DemoResult(



            demo_name=self.current_demo or "unknown",



            start_time=self.demo_start_time,



            end_time=time.time(),



            success=False,



            error_message="手动停止"



        )



        



        self.demo_stopped.emit(self.current_demo or "unknown", result.to_dict())



        self.status_changed.emit("demo_stopped", {



            "demo_name": self.current_demo,



            "duration": result.duration



        })



        



        return True



    



    def pause_demo(self) -> bool:



        """



        暂停当前演示



        



        Returns:



            是否成功暂停



        """



        with self.lock:



            if self.demo_status != DemoStatus.RUNNING:



                self.logger.warning(f"无法暂停演示，当前状态: {self.demo_status}")



                return False



            



            self.demo_status = DemoStatus.PAUSED



        



        self.logger.info(f"暂停演示: {self.current_demo}")



        self.status_changed.emit("demo_paused", {"demo_name": self.current_demo})



        



        return True



    



    def resume_demo(self) -> bool:



        """



        恢复当前演示



        



        Returns:



            是否成功恢复



        """



        with self.lock:



            if self.demo_status != DemoStatus.PAUSED:



                self.logger.warning(f"无法恢复演示，当前状态: {self.demo_status}")



                return False



            



            self.demo_status = DemoStatus.RUNNING



        



        self.logger.info(f"恢复演示: {self.current_demo}")



        self.status_changed.emit("demo_resumed", {"demo_name": self.current_demo})



        



        return True



    



    def update_progress(self, progress: float, info: Dict[str, Any] = None):



        """



        更新演示进度



        



        Args:



            progress: 进度值 (0.0-1.0)



            info: 附加信息



        """



        progress = max(0.0, min(1.0, progress))



        



        with self.lock:



            self.demo_progress_value = progress



        



        self.demo_progress.emit(progress, info or {})



    



    def _add_demo_result(self, result: DemoResult):



        """添加演示结果"""



        self.demo_results.append(result)



        



        # 限制历史记录大小



        if len(self.demo_results) > self.max_results_history:



            self.demo_results.pop(0)



    



    # 演示处理器实现



    def _run_calibration_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行校准演示"""



        self.logger.info("开始校准演示")



        



        start_time = time.time()



        data_points = 0



        



        try:



            # 发送校准命令



            calibration_type = params.get("type", "all")



            



            self.update_progress(0.1, {"message": "开始校准"})



            



            # 执行校准



            success = True



            



            if calibration_type in ["sensor", "all"]:



                self.update_progress(0.3, {"message": "校准传感器"})



                



                # 校准传感器



                if not self.hardware_interface.calibrate_sensor("zero"):



                    success = False



                    raise Exception("传感器零点校准失败")



                



                time.sleep(1)



                



                if not self.hardware_interface.calibrate_sensor("scale"):



                    success = False



                    raise Exception("传感器量程校准失败")



                



                # 新增：三维力校准



                if hasattr(self.hardware_interface, '_calibrate_sensor_vector'):



                    self.update_progress(0.5, {"message": "校准三维力"})



                    if not self.hardware_interface._calibrate_sensor_vector():



                        self.logger.warning("三维力校准失败，继续使用默认校准")



                



                self.update_progress(0.6, {"message": "传感器校准完成"})



            



            if calibration_type in ["servo", "all"] and success:



                self.update_progress(0.7, {"message": "校准舵机"})



                



                # 校准舵机



                if not self.hardware_interface.calibrate_servo("limits"):



                    success = False



                    raise Exception("舵机极限位置校准失败")



                



                self.update_progress(0.9, {"message": "舵机校准完成"})



            



            # 保存校准数据



            if success:



                self.hardware_interface.save_calibration("calibration/latest.json")



                self.update_progress(1.0, {"message": "校准完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            return DemoResult(



                demo_name="calibration",



                start_time=start_time,



                end_time=time.time(),



                success=success,



                data_points=data_points,



                metrics={



                    "calibration_type": calibration_type,



                    "sensor_calibrated": calibration_type in ["sensor", "all"],



                    "servo_calibrated": calibration_type in ["servo", "all"],



                    "force_dimensions": 3  # 三维力校准状态



                }



            )



            



        except Exception as e:



            return DemoResult(



                demo_name="calibration",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_grasping_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行抓取演示 - 使用三维力数据"""



        self.logger.info("开始三维力抓取演示")



        



        start_time = time.time()



        data_points = 0



        



        try:



            # 获取参数



            target_force_z = params.get("force_z", 15.0)  # 法向力 (N)



            target_force_shear = params.get("force_shear", 5.0)  # 剪切力阈值 (N)



            grasp_duration = params.get("duration", 5.0)



            object_type = params.get("object", "unknown")



            



            # 步骤1: 打开夹爪



            self.update_progress(0.1, {"message": "打开夹爪"})



            self.control_thread.send_command("move_gripper", {"position": 100})



            time.sleep(1)



            



            # 步骤2: 接近物体



            self.update_progress(0.3, {"message": "接近物体"})



            self.control_thread.send_command("move_gripper", {"position": 60})



            time.sleep(1)



            



            # 步骤3: 开始抓取（使用三维力控制）



            self.update_progress(0.5, {"message": "开始抓取（三维力）"})



            



            grasp_start = time.time()



            slip_count = 0



            force_history = []



            



            while time.time() - grasp_start < grasp_duration:



                # 获取三维力数据



                latest_data = self.data_acquisition.get_latest_data()



                if latest_data and hasattr(latest_data, 'force_vectors'):



                    force_vectors = latest_data.force_vectors



                    



                    # 计算合力



                    resultant = np.sum(force_vectors, axis=0)



                    fx, fy, fz = resultant



                    shear_force = math.sqrt(fx**2 + fy**2)  # 剪切力大小



                    



                    # 记录力数据



                    force_history.append({



                        'time': time.time() - grasp_start,



                        'fx': fx, 'fy': fy, 'fz': fz,



                        'shear': shear_force



                    })



                    



                    # 检测滑动（基于剪切力变化）



                    if len(force_history) > 10:



                        recent_shear = [fh['shear'] for fh in force_history[-10:]]



                        shear_change = abs(recent_shear[-1] - np.mean(recent_shear[:-1]))



                        if shear_change > target_force_shear * 0.3:  # 剪切力变化超过30%



                            slip_count += 1



                            self.logger.warning(f"检测到滑动，剪切力变化: {shear_change:.2f}N, 计数: {slip_count}")



                



                time.sleep(0.1)



                



                # 更新进度



                elapsed = time.time() - grasp_start



                progress = 0.5 + 0.3 * (elapsed / grasp_duration)



                self.update_progress(progress, {



                    "message": f"抓取中... {elapsed:.1f}/{grasp_duration:.1f}秒",



                    "slip_count": slip_count,



                    "current_force": f"{fz:.1f}N" if 'fz' in locals() else "未知"



                })



            



            # 步骤4: 释放物体



            self.update_progress(0.9, {"message": "释放物体"})



            self.control_thread.send_command("move_gripper", {"position": 100})



            time.sleep(1)



            



            self.update_progress(1.0, {"message": "三维力抓取演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            # 计算性能指标



            if force_history:



                z_forces = [fh['fz'] for fh in force_history]



                avg_z_force = np.mean(z_forces) if z_forces else 0



                max_z_force = np.max(z_forces) if z_forces else 0



            else:



                avg_z_force = 0



                max_z_force = 0



            



            return DemoResult(



                demo_name="grasping",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "target_force_z": target_force_z,



                    "target_force_shear": target_force_shear,



                    "grasp_duration": grasp_duration,



                    "object_type": object_type,



                    "slip_count": slip_count,



                    "avg_z_force": avg_z_force,



                    "max_z_force": max_z_force,



                    "force_history_count": len(force_history)



                }



            )



            



        except Exception as e:



            # 确保释放物体



            try:



                self.control_thread.send_command("move_gripper", {"position": 100})



            except:



                pass



            



            return DemoResult(



                demo_name="grasping",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_slip_detection_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行滑动检测演示 - 使用三维力数据"""



        self.logger.info("开始三维力滑动检测演示")



        



        start_time = time.time()



        data_points = 0



        slip_events = []



        force_variations = []



        



        try:



            # 获取参数



            test_duration = params.get("duration", 10.0)



            slip_threshold = params.get("threshold", 3.0)  # 剪切力变化阈值 (N)



            



            # 步骤1: 抓取物体



            self.update_progress(0.2, {"message": "抓取物体"})



            self.control_thread.send_command("move_gripper", {"position": 40})



            time.sleep(1)



            



            # 步骤2: 开始滑动检测



            self.update_progress(0.4, {"message": "开始三维力滑动检测"})



            detection_start = time.time()



            force_history = []



            



            while time.time() - detection_start < test_duration:



                # 获取三维力数据



                latest_data = self.data_acquisition.get_latest_data()



                if latest_data and hasattr(latest_data, 'force_vectors'):



                    force_vectors = latest_data.force_vectors



                    resultant = np.sum(force_vectors, axis=0)



                    



                    force_history.append({



                        'timestamp': time.time(),



                        'fx': resultant[0],



                        'fy': resultant[1],



                        'fz': resultant[2],



                        'shear': math.sqrt(resultant[0]**2 + resultant[1]**2)



                    })



                    



                    # 检测滑动（基于剪切力和法向力变化）



                    if len(force_history) >= 5:



                        current = force_history[-1]



                        previous = force_history[-5]



                        



                        # 计算力变化



                        shear_change = abs(current['shear'] - previous['shear'])



                        normal_change = abs(current['fz'] - previous['fz'])



                        total_change = math.sqrt(shear_change**2 + normal_change**2)



                        



                        force_variations.append(total_change)



                        



                        if total_change > slip_threshold:



                            slip_time = time.time() - start_time



                            slip_events.append({



                                'time': slip_time,



                                'shear_change': shear_change,



                                'normal_change': normal_change,



                                'total_change': total_change



                            })



                            self.logger.info(f"检测到滑动事件，总变化: {total_change:.2f}N")



                



                # 更新进度



                elapsed = time.time() - detection_start



                progress = 0.4 + 0.5 * (elapsed / test_duration)



                self.update_progress(progress, {



                    "message": f"检测中... {elapsed:.1f}/{test_duration:.1f}秒",



                    "slip_events": len(slip_events),



                    "current_variation": force_variations[-1] if force_variations else 0



                })



                



                time.sleep(0.1)



            



            # 步骤3: 释放物体



            self.update_progress(0.95, {"message": "释放物体"})



            self.control_thread.send_command("move_gripper", {"position": 100})



            time.sleep(1)



            



            self.update_progress(1.0, {"message": "三维力滑动检测演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            # 计算统计信息



            avg_variation = np.mean(force_variations) if force_variations else 0



            max_variation = np.max(force_variations) if force_variations else 0



            



            return DemoResult(



                demo_name="slip_detection",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "test_duration": test_duration,



                    "slip_threshold": slip_threshold,



                    "slip_events": len(slip_events),



                    "slip_details": slip_events,



                    "avg_force_variation": avg_variation,



                    "max_force_variation": max_variation,



                    "force_variations_count": len(force_variations)



                }



            )



            



        except Exception as e:



            # 确保释放物体



            try:



                self.control_thread.send_command("move_gripper", {"position": 100})



            except:



                pass



            



            return DemoResult(



                demo_name="slip_detection",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_object_classification_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行物体分类演示"""



        self.logger.info("开始物体分类演示")



        



        start_time = time.time()



        data_points = 0



        classifications = []



        



        try:



            # 检查模型是否加载



            if not self.model_loaded:



                raise Exception("深度学习模型未加载")



            



            # 获取参数



            num_trials = params.get("trials", 5)



            confidence_threshold = params.get("confidence_threshold", 0.7)



            



            for trial in range(num_trials):



                # 步骤1: 抓取物体



                self.update_progress(trial / num_trials, {"message": f"抓取物体 ({trial+1}/{num_trials})"})



                self.control_thread.send_command("move_gripper", {"position": 40})



                time.sleep(1)



                



                # 步骤2: 收集触觉数据



                self.update_progress((trial + 0.3) / num_trials, {"message": "收集三维力数据"})



                time.sleep(2)



                



                # 步骤3: 获取数据并进行分类



                self.update_progress((trial + 0.6) / num_trials, {"message": "进行分类"})



                



                # 获取最近的数据



                recent_data = self.data_acquisition.get_data_range(start_idx=-50)



                if recent_data:



                    # 准备输入数据 - 使用三维力数据



                    input_data = self._prepare_classification_input(recent_data)



                    



                    # 使用模型进行分类



                    if self.grip_net:



                        # 这里应该调用模型的推理方法



                        # classification_result = self.grip_net.predict(input_data)



                        # 模拟结果



                        import random



                        classes = ["球体", "立方体", "圆柱体", "不规则物体", "未知"]



                        predicted_class = random.choice(classes)



                        confidence = random.uniform(0.5, 0.95)



                        



                        if confidence >= confidence_threshold:



                            classification = {



                                "trial": trial + 1,



                                "predicted_class": predicted_class,



                                "confidence": confidence,



                                "timestamp": time.time(),



                                "force_dimensions": 3  # 三维力分类



                            }



                            classifications.append(classification)



                            self.logger.info(f"三维力分类结果: {predicted_class}, 置信度: {confidence:.2f}")



                



                # 步骤4: 释放物体



                self.update_progress((trial + 0.9) / num_trials, {"message": "释放物体"})



                self.control_thread.send_command("move_gripper", {"position": 100})



                time.sleep(1)



            



            self.update_progress(1.0, {"message": "物体分类演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            return DemoResult(



                demo_name="object_classification",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "num_trials": num_trials,



                    "confidence_threshold": confidence_threshold,



                    "classifications": classifications,



                    "successful_classifications": len([c for c in classifications if c["confidence"] >= confidence_threshold]),



                    "force_dimensions": 3



                }



            )



            



        except Exception as e:



            # 确保释放物体



            try:



                self.control_thread.send_command("move_gripper", {"position": 100})



            except:



                pass



            



            return DemoResult(



                demo_name="object_classification",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_force_control_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行三维力控制演示"""



        self.logger.info("开始三维力控制演示")



        



        start_time = time.time()



        data_points = 0



        force_history = []



        



        try:



            # 获取参数



            target_forces = params.get("forces", [



                {'fx': 0, 'fy': 0, 'fz': 10},



                {'fx': 2, 'fy': 0, 'fz': 15},



                {'fx': 0, 'fy': 2, 'fz': 20},



                {'fx': -2, 'fy': -2, 'fz': 15},



                {'fx': 0, 'fy': 0, 'fz': 10}



            ])



            hold_time = params.get("hold_time", 3.0)



            



            for i, target_force in enumerate(target_forces):



                # 步骤1: 设置目标力



                self.update_progress(i / len(target_forces), {



                    "message": f"设置目标力: Fx={target_force['fx']}N, Fy={target_force['fy']}N, Fz={target_force['fz']}N ({i+1}/{len(target_forces)})"



                })



                



                # 发送三维力控制命令



                control_params = {



                    'target_force_x': target_force['fx'],



                    'target_force_y': target_force['fy'],



                    'target_force_z': target_force['fz']



                }



                self.control_thread.send_command("start_force_control_3d", control_params)



                



                # 步骤2: 保持力



                self.update_progress((i + 0.5) / len(target_forces), {"message": "保持目标三维力"})



                hold_start = time.time()



                



                while time.time() - hold_start < hold_time:



                    # 记录三维力数据



                    latest_data = self.data_acquisition.get_latest_data()



                    if latest_data and hasattr(latest_data, 'force_vectors'):



                        force_vectors = latest_data.force_vectors



                        resultant = np.sum(force_vectors, axis=0)



                        



                        force_history.append({



                            "timestamp": time.time(),



                            "target_fx": target_force['fx'],



                            "target_fy": target_force['fy'],



                            "target_fz": target_force['fz'],



                            "actual_fx": resultant[0],



                            "actual_fy": resultant[1],



                            "actual_fz": resultant[2]



                        })



                    



                    time.sleep(0.1)



            



            # 步骤3: 停止力控制



            self.update_progress(0.95, {"message": "停止三维力控制"})



            self.control_thread.send_command("stop_force_control", {})



            



            # 步骤4: 打开夹爪



            self.control_thread.send_command("move_gripper", {"position": 100})



            time.sleep(1)



            



            self.update_progress(1.0, {"message": "三维力控制演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            # 计算性能指标



            if force_history:



                errors_x = [abs(fh["actual_fx"] - fh["target_fx"]) for fh in force_history]



                errors_y = [abs(fh["actual_fy"] - fh["target_fy"]) for fh in force_history]



                errors_z = [abs(fh["actual_fz"] - fh["target_fz"]) for fh in force_history]



                



                avg_error_x = sum(errors_x) / len(errors_x) if errors_x else 0



                avg_error_y = sum(errors_y) / len(errors_y) if errors_y else 0



                avg_error_z = sum(errors_z) / len(errors_z) if errors_z else 0



                



                max_error_x = max(errors_x) if errors_x else 0



                max_error_y = max(errors_y) if errors_y else 0



                max_error_z = max(errors_z) if errors_z else 0



                



                # 计算总误差



                total_errors = [math.sqrt(ex**2 + ey**2 + ez**2) for ex, ey, ez in zip(errors_x, errors_y, errors_z)]



                avg_total_error = sum(total_errors) / len(total_errors) if total_errors else 0



                max_total_error = max(total_errors) if total_errors else 0



            else:



                avg_error_x = avg_error_y = avg_error_z = 0



                max_error_x = max_error_y = max_error_z = 0



                avg_total_error = max_total_error = 0



            



            return DemoResult(



                demo_name="force_control",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "target_forces": target_forces,



                    "hold_time": hold_time,



                    "force_history": force_history[:10],  # 只返回前10个点



                    "avg_error_x": avg_error_x,



                    "avg_error_y": avg_error_y,



                    "avg_error_z": avg_error_z,



                    "avg_total_error": avg_total_error,



                    "max_error_x": max_error_x,



                    "max_error_y": max_error_y,



                    "max_error_z": max_error_z,



                    "max_total_error": max_total_error,



                    "force_dimensions": 3



                }



            )



            



        except Exception as e:



            # 确保停止力控制



            try:



                self.control_thread.send_command("stop_force_control", {})



                self.control_thread.send_command("move_gripper", {"position": 100})



            except:



                pass



            



            return DemoResult(



                demo_name="force_control",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_learning_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行学习演示"""



        self.logger.info("开始学习演示")



        



        start_time = time.time()



        data_points = 0



        



        try:



            # 获取参数



            learning_time = params.get("duration", 30.0)



            save_model = params.get("save_model", True)



            



            # 步骤1: 开始数据收集



            self.update_progress(0.1, {"message": "开始三维力数据收集"})



            self.data_acquisition.clear_buffer()



            



            # 步骤2: 收集训练数据



            self.update_progress(0.3, {"message": "收集三维力训练数据"})



            collection_start = time.time()



            



            while time.time() - collection_start < learning_time:



                elapsed = time.time() - collection_start



                progress = 0.3 + 0.6 * (elapsed / learning_time)



                self.update_progress(progress, {



                    "message": f"收集三维力数据中... {elapsed:.1f}/{learning_time:.1f}秒"



                })



                time.sleep(0.1)



            



            # 步骤3: 训练模型



            self.update_progress(0.9, {"message": "训练三维力模型"})



            



            # 获取收集的数据



            training_data = self.data_acquisition.get_data_range()



            if len(training_data) >= 100:  # 有足够的数据



                # 检查是否为三维力数据



                force_3d_available = False



                for data in training_data:



                    if hasattr(data, 'force_vectors') and data.force_vectors:



                        force_3d_available = True



                        break



                



                self.logger.info(f"收集了 {len(training_data)} 个数据点用于训练，三维力数据: {'可用' if force_3d_available else '不可用'}")



                



                if save_model and self.grip_net and force_3d_available:



                    # 保存模型（简化）



                    # self.grip_net.save_model("models/latest_3d_model.pth")



                    pass



            else:



                self.logger.warning(f"数据不足，只有 {len(training_data)} 个数据点")



            



            self.update_progress(1.0, {"message": "三维力学习演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            return DemoResult(



                demo_name="learning",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "learning_time": learning_time,



                    "data_collected": len(training_data),



                    "save_model": save_model,



                    "force_3d_available": force_3d_available



                }



            )



            



        except Exception as e:



            return DemoResult(



                demo_name="learning",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _run_vector_visualization_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行矢量可视化演示"""



        self.logger.info("开始矢量可视化演示")



        



        start_time = time.time()



        data_points = 0



        visualization_data = []



        



        try:



            # 获取参数



            demo_duration = params.get("duration", 15.0)



            visualization_type = params.get("type", "vector_field")  # vector_field, force_history, 3d_surface



            



            # 步骤1: 抓取物体



            self.update_progress(0.2, {"message": "抓取物体进行矢量可视化"})



            self.control_thread.send_command("move_gripper", {"position": 50})



            time.sleep(1)



            



            # 步骤2: 开始矢量数据收集和可视化



            self.update_progress(0.4, {"message": "开始矢量数据收集"})



            demo_start = time.time()



            



            while time.time() - demo_start < demo_duration:



                # 获取三维力数据



                latest_data = self.data_acquisition.get_latest_data()



                if latest_data and hasattr(latest_data, 'force_vectors'):



                    force_vectors = np.asarray(latest_data.force_vectors)



                    resultant = np.sum(force_vectors, axis=0)



                    



                    # 记录可视化数据



                    visualization_data.append({



                        'timestamp': time.time() - demo_start,



                        'force_vectors': force_vectors.tolist(),



                        'resultant_force': resultant.tolist(),



                        'resultant_magnitude': math.sqrt(resultant[0]**2 + resultant[1]**2 + resultant[2]**2),



                        'visualization_type': visualization_type



                    })



                    



                    # 发送矢量数据信号（UI将使用这些数据绘制矢量图）



                    vector_plot_data = {



                        'force_vectors': force_vectors.tolist(),



                        'resultant_force': resultant.tolist(),



                        'taxel_positions': self._get_taxel_positions(),  # 触点位置



                        'contact_map': self.contact_map,



                        'timestamp': time.time() - demo_start



                    }



                    self.vector_data_updated.emit(vector_plot_data)



                



                # 更新进度



                elapsed = time.time() - demo_start



                progress = 0.4 + 0.5 * (elapsed / demo_duration)



                self.update_progress(progress, {



                    "message": f"可视化演示中... {elapsed:.1f}/{demo_duration:.1f}秒",



                    "data_points": len(visualization_data),



                    "visualization_type": visualization_type



                })



                



                time.sleep(0.1)



            



            # 步骤3: 释放物体



            self.update_progress(0.95, {"message": "释放物体"})



            self.control_thread.send_command("move_gripper", {"position": 100})



            time.sleep(1)



            



            self.update_progress(1.0, {"message": "矢量可视化演示完成"})



            



            # 收集数据点



            data_points = self.data_acquisition.read_count



            



            return DemoResult(



                demo_name="vector_visualization",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "demo_duration": demo_duration,



                    "visualization_type": visualization_type,



                    "data_collected": len(visualization_data),



                    "vector_data_samples": min(5, len(visualization_data)),  # 只返回少量样本



                    "force_dimensions": 3



                }



            )



            



        except Exception as e:



            # 确保释放物体



            try:



                self.control_thread.send_command("move_gripper", {"position": 100})



            except:



                pass



            



            return DemoResult(



                demo_name="vector_visualization",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    def _run_tactile_mapping_demo(self, params: Dict[str, Any]) -> DemoResult:



        """运行触觉映射演示（简版，收集/显示触觉网格）"""



        self.logger.info("开始触觉映射演示")



        start_time = time.time()



        data_points = 0



        duration = params.get("duration", 10.0)



        save_mapping = params.get("save", False)



        show_plot = params.get("show_plot", True)



        mapping = []



        try:



            self.update_progress(0.1, {"message": "收集触觉数据中"})



            demo_start = time.time()



            while time.time() - demo_start < duration:



                latest = self.data_acquisition.get_latest_data()



                if latest:



                    # 转换 force_vectors 为可序列化格式



                    data_dict = latest.to_dict()



                    if 'force_vectors' in data_dict and isinstance(data_dict['force_vectors'], np.ndarray):



                        data_dict['force_vectors'] = data_dict['force_vectors'].tolist()



                    mapping.append(data_dict)



                    data_points += 1



                elapsed = time.time() - demo_start



                progress = min(0.9, 0.1 + 0.8 * (elapsed / duration))



                self.update_progress(progress, {"message": f"触觉映射 {elapsed:.1f}/{duration:.1f}s"})



                time.sleep(0.05)



            if save_mapping and mapping:



                import json



                filepath = params.get("filepath", "logs/tactile_mapping.json")



                with open(filepath, "w", encoding="utf-8") as f:



                    json.dump(mapping, f, indent=2, ensure_ascii=False)



                self.logger.info(f"触觉映射数据已保存到: {filepath}")



            self.update_progress(1.0, {"message": "触觉映射完成"})



            if show_plot:



                try:



                    # 交给主线程显示，避免跨线程创建窗口



                    self.tactile_mapping_ready.emit(mapping)



                except Exception as e:



                    self.logger.warning(f"显示触觉映射图时出错: {e}")



            return DemoResult(



                demo_name="tactile_mapping",



                start_time=start_time,



                end_time=time.time(),



                success=True,



                data_points=data_points,



                metrics={



                    "duration": duration,



                    "saved": save_mapping,



                    "points": data_points



                }



            )



        except Exception as e:



            return DemoResult(



                demo_name="tactile_mapping",



                start_time=start_time,



                end_time=time.time(),



                success=False,



                data_points=data_points,



                error_message=str(e)



            )



    



    def _prepare_classification_input(self, sensor_data_list: List[SensorData]) -> np.ndarray:



        """准备分类输入数据 - 支持三维力数据"""



        if not sensor_data_list:



            return np.zeros((1, 30))



        



        # 将传感器数据转换为模型输入格式



        # 使用三维力数据：每个数据点有3个力分量，9个触点 => 27个特征



        # 加上时间序列 => 30个特征



        features = []



        



        for data in sensor_data_list[-10:]:  # 使用最近10个时间步



            if hasattr(data, 'force_vectors') and data.force_vectors:



                # 三维力数据



                force_vectors = np.array(data.force_vectors)



                flat_features = force_vectors.flatten()  # 27个特征 (9x3)



                



                # 如果特征不足，填充0



                if len(flat_features) < 27:



                    flat_features = np.pad(flat_features, (0, 27 - len(flat_features)))



                elif len(flat_features) > 27:



                    flat_features = flat_features[:27]



            else:



                # 只有Z方向力



                force_data = data.force_data if hasattr(data, 'force_data') else []



                flat_features = np.zeros(27)



                if len(force_data) >= 9:



                    flat_features[:9] = force_data[:9]  # 只使用前9个触点



        



            features.extend(flat_features)



        



        # 如果特征不足，填充0



        if len(features) < 30:



            features.extend([0] * (30 - len(features)))



        elif len(features) > 30:



            features = features[:30]



        



        return np.array(features).reshape(1, -1)



    



    def _get_taxel_positions(self) -> np.ndarray:



        """获取触点位置（归一化坐标）"""



        # 3x3网格的触点位置



        positions = []



        rows, cols = 3, 3



        



        for i in range(rows):



            for j in range(cols):



                x = j / (cols - 1) if cols > 1 else 0.5



                y = i / (rows - 1) if rows > 1 else 0.5



                positions.append([x, y])



        



        return np.array(positions)



    def _visualize_mapping(self, mapping_data: List[Dict[str, Any]]):



        """显示触觉映射结果：2D热力图 + 3D曲面"""



        if not mapping_data:



            return



        import matplotlib.pyplot as plt



        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



        import numpy as np



        latest = mapping_data[-1]



        fv = np.array(latest.get('force_vectors', []))



        if fv.size == 0:



            return



        magnitudes = np.linalg.norm(fv, axis=1)



        # 触点坐标映射到 [-10,10]mm 范围



        taxel_pos = self._get_taxel_positions()



        coords = (taxel_pos - 0.5) * 10.0



        # 网格插值



        grid_lin = np.linspace(-10, 10, 60)



        grid_x, grid_y = np.meshgrid(grid_lin, grid_lin)



        grid_z = self._interpolate_grid(coords, magnitudes, grid_x, grid_y)



        fig = plt.figure(figsize=(12, 5))



        # 2D 热力图



        ax1 = fig.add_subplot(1, 2, 1)



        im = ax1.imshow(grid_z, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis')



        ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=50, label='触点')



        ax1.set_title('触觉映射 - 2D视图')



        ax1.set_xlabel('X 位置 (mm)')



        ax1.set_ylabel('Y 位置 (mm)')



        ax1.legend(loc='upper right')



        fig.colorbar(im, ax=ax1, label='力 (N)')



        # 3D 曲面



        ax2 = fig.add_subplot(1, 2, 2, projection='3d')



        ax2.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)



        ax2.scatter(coords[:, 0], coords[:, 1], magnitudes, c='red', s=40)



        ax2.set_title('触觉映射 - 3D视图')



        ax2.set_xlabel('X 位置 (mm)')



        ax2.set_ylabel('Y 位置 (mm)')



        ax2.set_zlabel('力 (N)')



        fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2, fraction=0.046, pad=0.04, label='力 (N)')



        plt.tight_layout()



        plt.show(block=False)



    def _interpolate_grid(self, coords: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:



        """简单反距离加权插值，避免依赖scipy"""



        eps = 1e-6



        grid_z = np.zeros_like(grid_x, dtype=float)



        for i in range(grid_x.shape[0]):



            for j in range(grid_x.shape[1]):



                dx = coords[:, 0] - grid_x[i, j]



                dy = coords[:, 1] - grid_y[i, j]



                dist = np.sqrt(dx * dx + dy * dy) + eps



                w = 1.0 / dist



                grid_z[i, j] = np.sum(w * values) / np.sum(w)



        return grid_z



    



    def _detect_slip(self, force_data: List[float], threshold: float = 5.0) -> bool:



        """检测滑动 - 向后兼容"""



        if len(force_data) < 2:



            return False



        



        # 简单的滑动检测：检查力的变化



        variations = []



        for i in range(1, len(force_data)):



            variation = abs(force_data[i] - force_data[i-1])



            variations.append(variation)



        



        # 如果有明显的力变化，认为发生了滑动



        if variations and max(variations) > threshold:



            return True



        



        return False



    



    # 公共方法



    def get_status(self) -> Dict[str, Any]:



        """



        获取演示管理器状态



        



        Returns:



            状态字典



        """



        with self.lock:



            return {



                "current_demo": self.current_demo,



                "demo_status": self.demo_status.value,



                "demo_params": self.demo_params.copy(),



                "progress": self.demo_progress_value,



                "start_time": self.demo_start_time,



                "duration": time.time() - self.demo_start_time if self.demo_start_time > 0 else 0,



                "force_statistics": self.force_statistics,



                "contact_points": np.sum(self.contact_map)



            }



    



    def get_demo_results(self, limit: int = 5) -> List[Dict[str, Any]]:



        """



        获取演示结果



        



        Args:



            limit: 返回结果数量限制



            



        Returns:



            演示结果列表



        """



        recent_results = self.demo_results[-limit:] if self.demo_results else []



        return [result.to_dict() for result in recent_results]



    



    def get_force_history(self, limit: int = 100) -> Dict[str, Any]:



        """



        获取力历史数据



        



        Args:



            limit: 返回数据点数量限制



            



        Returns:



            力历史数据



        """



        with self.lock:



            history = {}



            for key in ['fx', 'fy', 'fz', 'timestamps']:



                history[key] = self.force_history[key][-limit:] if self.force_history[key] else []



            



            return history



    



    def get_force_vectors(self) -> Optional[np.ndarray]:



        """获取当前力向量数据"""



        with self.lock:



            if hasattr(self, 'contact_forces'):



                return self.contact_forces.copy()



            return None



    



    def get_contact_map(self) -> np.ndarray:



        """获取接触地图"""



        with self.lock:



            return self.contact_map.copy()



    



    def is_demo_running(self) -> bool:



        """检查是否有演示在运行"""



        with self.lock:



            return self.demo_status in [DemoStatus.RUNNING, DemoStatus.PAUSED]



    



    def is_sensor_connected(self) -> bool:



        """检查传感器是否连接"""



        status = self.hardware_interface.get_status()



        return status["sensor"]["connected"]



    



    def is_servo_connected(self) -> bool:



        """检查舵机是否连接"""



        status = self.hardware_interface.get_status()



        return status["servo"]["connected"]



    



    def is_stm32_connected(self) -> bool:



        """检查 STM32 是否已连接"""



        status = self.hardware_interface.get_status()



        return status["servo"]["connected"]



    def is_tactile_connected(self) -> bool:



        """检查触觉传感器是否已连接"""



        status = self.hardware_interface.get_status()



        return status["sensor"]["connected"]



    def get_hardware_status(self) -> Dict[str, Any]:



        """获取硬件状态"""



        return self.hardware_interface.get_status()



    def export_data(self, filepath: str):



        """导出数据



        Args:



            filepath: 导出文件路径



        """



        try:



            # 组织导出数据



            all_data = self.data_acquisition.get_data_range()



            export_data = {



                "metadata": {



                    "export_time": datetime.now().isoformat(),



                    "system_config": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),



                    "demo_results": self.get_demo_results(),



                    "force_statistics": self.force_statistics,



                    "contact_map": self.contact_map.tolist()



                },



                "sensor_data": [data.to_dict() for data in all_data],



                "force_history": self.get_force_history(500)



            }



            import json



            with open(filepath, 'w', encoding='utf-8') as f:



                json.dump(export_data, f, indent=2, ensure_ascii=False)



            self.logger.info(f"数据导出成功: {filepath}")



        except Exception as e:



            self.logger.error(f"数据导出失败: {e}")



            raise



    def handle_control_command(self, command: str, params: Dict[str, Any] = None):



        """



        处理控制命令



        



        Args:



            command: 命令类型



            params: 命令参数



        """



        self.logger.info(f"处理控制命令: {command}, 参数: {params}")



        



        if command == "connect_hardware":



            self.control_thread.send_command("connect_hardware", params)



        elif command == "disconnect_hardware":

            self.control_thread.send_command("disconnect_hardware", params)



        elif command == "connect_stm32":



            self.control_thread.send_command("connect_stm32", params)



        elif command == "disconnect_stm32":



            self.control_thread.send_command("disconnect_stm32", params)



        elif command == "connect_tactile":



            self.control_thread.send_command("connect_tactile", params)



        elif command == "disconnect_tactile":



            self.control_thread.send_command("disconnect_tactile", params)



            self.control_thread.send_command("disconnect_hardware", params)



        elif command == "calibrate_hardware":



            self.control_thread.send_command("calibrate_hardware", params)



        elif command == "start_demo":



            demo_name = params.get("demo_name")



            demo_params = params.get("params", {})



            self.start_demo(demo_name, demo_params)



        elif command == "stop_demo":



            self.stop_demo()



        elif command == "pause_demo":



            self.pause_demo()



        elif command == "resume_demo":



            self.resume_demo()



        elif command == "emergency_stop":



            # 先停止当前演示与数据采集，再下发紧急停止



            try:



                self.stop_demo()



            except Exception:



                pass



            try:



                if hasattr(self.data_acquisition, "stop_acquisition"):



                    self.data_acquisition.stop_acquisition()



            except Exception:



                pass



            with self.lock:



                self.demo_status = DemoStatus.IDLE



                self.current_demo = None



            self.control_thread.send_command("emergency_stop", params)



        elif command == "apply_parameters":



            self._apply_parameters(params or {})



        elif command == "update_vector_visualization":



            # 手动更新矢量可视化



            self.vector_data_updated.emit(params)



        elif command == "connect_arm":



            self.control_thread.send_command("connect_arm", params)



        elif command == "disconnect_arm":



            self.control_thread.send_command("disconnect_arm", params)



        elif command == "arm_enable":



            self.control_thread.send_command("arm_enable", params)



        elif command == "arm_disable":



            self.control_thread.send_command("arm_disable", params)



        elif command == "auto_grasp":

            self.control_thread.send_command("auto_grasp", params)

        elif command == "arm_home":



            self.control_thread.send_command("arm_home", params)



        elif command == "move_arm_joint":



            self.control_thread.send_command("move_arm_joint", params)



        elif command == "move_arm_joints":



            self.control_thread.send_command("move_arm_joints", params)



        else:



            self.logger.warning(f"未知的控制命令: {command}")



    def _apply_parameters(self, params: Dict[str, Any]):



        """应用来自UI的参数，包括仿真参数"""



        try:



            if not params:



                return



            sensor_params = params.get("sensor", {})



            servo_params = params.get("servo", {})



            algo_params = params.get("algorithm", {})



            sim_params = params.get("simulation", {})



            # 更新配置对象



            if hasattr(self.config, "hardware") and hasattr(self.config.hardware, "sensor"):



                for k, v in sensor_params.items():



                    setattr(self.config.hardware.sensor, k, v)



                for k, v in sim_params.items():



                    setattr(self.config.hardware.sensor, k, v)



            if hasattr(self.config, "hardware") and hasattr(self.config.hardware, "servo"):



                for k, v in servo_params.items():



                    setattr(self.config.hardware.servo, k, v)



            if hasattr(self.config, "algorithm"):



                for k, v in algo_params.items():



                    setattr(self.config.algorithm, k, v)



            # 将仿真参数应用到硬件接口（模拟模式）



            if sim_params and hasattr(self.hardware_interface, "set_simulation_params"):



                self.hardware_interface.set_simulation_params(sim_params)



            self.status_changed.emit("parameters_applied", {"message": "参数已更新"})



            self.logger.info(f"参数已应用: {params}")



        except Exception as e:



            self.logger.error(f"应用参数失败: {e}")



    def close(self):



        """关闭演示管理器"""



        self.vector_update_running = False



        if hasattr(self, 'vector_update_thread') and self.vector_update_thread.is_alive():



            self.vector_update_thread.join(timeout=1.0)



        



        self.logger.info("演示管理器已关闭")



