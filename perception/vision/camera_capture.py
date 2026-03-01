"""
相机捕获模块 - 支持 OpenCV（USB）、RealSense 与仿真相机
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

# RealSense Python SDK是可选依赖，未安装时保留为None以给出明确提示
try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - 运行环境可能未安装
    rs = None

from config.camera_config import CameraConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)

class CameraType(Enum):
    """相机类型枚举"""
    OPENCV = "opencv"
    REALSENSE = "realsense"
    SIMULATION = "simulation"

@dataclass
class CameraFrame:
    """相机帧数据结构"""
    timestamp: float
    color_image: np.ndarray
    depth_image: Optional[np.ndarray] = None
    infrared_image: Optional[np.ndarray] = None
    camera_pose: Optional[np.ndarray] = None  # 相机姿态 (4x4矩阵)
    intrinsics: Optional[Dict[str, Any]] = None  # 相机内参
    
class CameraCapture:
    """
    相机捕获模块
    支持多种相机接口的统一样式
    """
    
    def __init__(self, config: CameraConfig):
        """
        初始化相机捕获
        
        Args:
            config: 相机配置
        """
        self.config = config
        self.camera_type = self._normalize_camera_type(config.camera_type)
        
        # 相机对象
        self.camera = None
        self.pipeline = None  # 保留属性名以兼容外部引用
        self.align = None
        self.connected = False  # 仅在成功获取首帧后置为True
        # RealSense 专用状态（仅在 realsense 模式下使用）
        self.rs_profile = None
        self.rs_config = None
        self.depth_scale = None
        self.rs_color_format = None
        self.rs_color_is_bgr = False
        
        # 线程控制
        self.capture_thread = None
        self.is_capturing = False
        self.latest_frame: Optional[CameraFrame] = None
        self.frame_lock = threading.Lock()
        
        # 统计信息
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.failure_count = 0
        self.max_failures = getattr(config, "max_failures", 5)
        
        # 相机内参
        self.intrinsics = {
            'fx': getattr(config, "fx", None),
            'fy': getattr(config, "fy", None),
            'cx': getattr(config, "cx", None),
            'cy': getattr(config, "cy", None),
            'width': getattr(config, "width", 640),
            'height': getattr(config, "height", 480),
            'distortion_coeffs': getattr(config, "distortion_coeffs", []),
            'fov': getattr(config, "fov", None),
        }
        # 如果未提供内参且有视场角信息，估算 fx/fy
        if (self.intrinsics["fx"] is None or self.intrinsics["fy"] is None) and self.intrinsics["fov"]:
            fov_deg = self.intrinsics["fov"]
            fx_fy = 0.5 * self.intrinsics["width"] / np.tan(np.deg2rad(fov_deg) / 2.0)
            self.intrinsics["fx"] = fx_fy
            self.intrinsics["fy"] = fx_fy
            self.intrinsics.setdefault("note", "fx/fy estimated from FOV")
        
        # 初始化相机
        self._initialize_camera()
        
        logger.info(f"CameraCapture initialized with type: {self.camera_type}")
    
    def _initialize_camera(self):
        """初始化相机"""
        try:
            if self.camera_type == CameraType.OPENCV:
                self._initialize_opencv_camera()
            elif self.camera_type == CameraType.REALSENSE:
                self._initialize_realsense_camera()
            elif self.camera_type == CameraType.SIMULATION:
                self._initialize_simulation_camera()
            else:
                raise ValueError(f"Unsupported camera type: {self.camera_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            raise
    
    def _initialize_opencv_camera(self):
        """初始化OpenCV相机"""
        self.camera = cv2.VideoCapture(self.config.camera_index)
        
        if not self.camera.isOpened():
            self.connected = False
            raise RuntimeError(f"Failed to open camera at index {self.config.camera_index}")
        
        # 设置相机参数
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # 读取实际生效的设置，方便低成本 USB 相机调试
        actual_w = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        if (actual_w, actual_h) != (self.config.width, self.config.height):
            logger.warning(f"Camera resolution fallback to {int(actual_w)}x{int(actual_h)}")
        if actual_fps != self.config.fps:
            logger.warning(f"Camera FPS fallback to {actual_fps:.1f}")
        
        logger.info(f"OpenCV camera initialized: {self.config.width}x{self.config.height}@{self.config.fps}fps")

    def _initialize_realsense_camera(self):
        """初始化RealSense相机"""
        if rs is None:
            raise RuntimeError("未检测到 pyrealsense2，请先安装 RealSense Python SDK")

        # 检查设备是否存在，便于给出更清晰的错误信息
        ctx = rs.context()
        devices = list(ctx.devices)
        if not devices:
            raise RuntimeError("未检测到RealSense设备，请检查USB连接与供电")

        # 若指定序列号，确保目标设备存在
        serial = self._get_realsense_serial()
        if serial:
            if not any(dev.get_info(rs.camera_info.serial_number) == serial for dev in devices):
                raise RuntimeError(f"未找到序列号为 {serial} 的RealSense设备")

        # 创建 pipeline/config 并启用彩色/深度流
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        if serial:
            self.rs_config.enable_device(serial)

        color_format = self._map_realsense_color_format(getattr(self.config, "color_format", "rgb8"))
        self.rs_color_format = color_format
        # 用于后续是否需要BGR->RGB转换
        self.rs_color_is_bgr = (color_format == rs.format.bgr8)
        self.rs_config.enable_stream(
            rs.stream.color,
            int(self.config.width),
            int(self.config.height),
            color_format,
            int(self.config.fps)
        )

        # 深度流按需启用；深度分辨率可独立于彩色分辨率
        if getattr(self.config, "enable_depth", False):
            depth_w, depth_h = self._get_realsense_depth_resolution()
            depth_format = self._map_realsense_depth_format(getattr(self.config, "depth_format", "z16"))
            self.rs_config.enable_stream(
                rs.stream.depth,
                int(depth_w),
                int(depth_h),
                depth_format,
                int(self.config.fps)
            )

        # 启动 pipeline
        self.rs_profile = self.pipeline.start(self.rs_config)

        # 尝试设置曝光/白平衡等参数（若设备支持）
        self._configure_realsense_sensors(self.rs_profile)

        # 获取相机内参与深度尺度（用于后续深度单位转换）
        self._update_intrinsics_from_profile(self.rs_profile)
        if getattr(self.config, "enable_depth", False):
            try:
                depth_sensor = self.rs_profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
            except Exception:
                # 兜底为默认毫米->米比例
                self.depth_scale = 0.001

            # 若配置要求，启用深度对齐到彩色
            if getattr(self.config, "align_depth_to_color", True):
                self.align = rs.align(rs.stream.color)

        logger.info(f"RealSense camera initialized: {self.config.width}x{self.config.height}@{self.config.fps}fps")
    
    def _initialize_simulation_camera(self):
        """初始化仿真相机"""
        # 仿真相机在simulator模块中实现
        logger.info("Simulation camera initialized - will connect to simulator")
    
    def start_capture(self):
        """开始捕获图像"""
        if self.is_capturing:
            logger.warning("Camera capture already running")
            return
        
        self.is_capturing = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Camera capture started")
    
    def stop_capture(self):
        """停止捕获图像"""
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # 释放相机资源
        if self.camera_type == CameraType.OPENCV and self.camera:
            self.camera.release()
            self.connected = False
        elif self.camera_type == CameraType.REALSENSE and self.pipeline:
            # RealSense 停止 pipeline，避免设备被占用
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.connected = False
        
        logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """捕获循环"""
        while self.is_capturing:
            try:
                frame = self._capture_single_frame()
                
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_count += 1
                    self.failure_count = 0  # 成功读取后重置失败计数
                
                # 更新FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # RealSense 的 wait_for_frames 自带阻塞，避免再额外 sleep
                if self.camera_type in (CameraType.OPENCV, CameraType.SIMULATION):
                    time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                self.failure_count += 1
                logger.warning(f"Capture error ({self.failure_count}): {str(e)}")
                if self.failure_count >= self.max_failures:
                    recovered = self._handle_capture_error(e)
                    self.failure_count = 0
                    if not recovered:
                        logger.error("Stopping capture loop due to repeated failures")
                        self.is_capturing = False
                        break
                time.sleep(0.1)
    
    def _capture_single_frame(self) -> CameraFrame:
        """捕获单帧图像"""
        timestamp = time.time()
        
        if self.camera_type == CameraType.OPENCV:
            return self._capture_opencv_frame(timestamp)
        elif self.camera_type == CameraType.REALSENSE:
            return self._capture_realsense_frame(timestamp)
        elif self.camera_type == CameraType.SIMULATION:
            return self._capture_simulation_frame(timestamp)
        else:
            raise ValueError(f"Unsupported camera type for capture: {self.camera_type}")

    def is_connected(self) -> bool:
        """是否已连接相机（首帧成功获取后为True）"""
        return bool(self.connected)

    @staticmethod
    def _normalize_camera_type(camera_type) -> CameraType:
        """将配置中的相机类型字符串/枚举映射为内部枚举"""
        if isinstance(camera_type, CameraType):
            return camera_type
        name = str(camera_type).lower()
        if name in ("opencv", "usb", "usb_cam", "usb_camera"):
            return CameraType.OPENCV
        if name in ("realsense", "rs", "d455", "d435", "d415"):
            return CameraType.REALSENSE
        if name in ("simulation", "sim"):
            return CameraType.SIMULATION
        logger.warning(f"Unknown camera_type '{camera_type}', fallback to OpenCV")
        return CameraType.OPENCV
    
    def _capture_opencv_frame(self, timestamp: float) -> CameraFrame:
        """捕获OpenCV帧"""
        ret, color_image = self.camera.read()
        
        if not ret:
            raise RuntimeError("Failed to capture frame from OpenCV camera")
        
        # 转换为RGB
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # 首帧成功则认为已连接
        if not self.connected:
            self.connected = True
        
        return CameraFrame(
            timestamp=timestamp,
            color_image=color_image_rgb,
            intrinsics=self.intrinsics
        )

    def _capture_realsense_frame(self, timestamp: float) -> CameraFrame:
        """捕获RealSense帧（彩色+深度）"""
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline 未初始化")

        # wait_for_frames 本身会阻塞，超时后会抛异常
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("未获取到RealSense彩色帧")

        # RealSense返回的内存由SDK管理，copy避免后续被覆盖
        color_image = np.asanyarray(color_frame.get_data()).copy()
        if self.rs_color_is_bgr:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        depth_image = None
        if getattr(self.config, "enable_depth", False):
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_scale = self.depth_scale if self.depth_scale else 0.001
                depth_image = depth_raw * depth_scale  # 统一输出为“米”

                # 应用深度范围裁剪，超出范围的深度置0，避免影响显示与后续处理
                depth_min = getattr(self.config, "depth_min", None)
                depth_max = getattr(self.config, "depth_max", None)
                if depth_min is not None or depth_max is not None:
                    invalid = np.zeros(depth_image.shape, dtype=bool)
                    if depth_min is not None:
                        invalid |= depth_image < float(depth_min)
                    if depth_max is not None:
                        invalid |= depth_image > float(depth_max)
                    depth_image[invalid] = 0.0

        # 首帧成功则认为已连接
        if not self.connected:
            self.connected = True

        return CameraFrame(
            timestamp=timestamp,
            color_image=color_image,
            depth_image=depth_image,
            intrinsics=self.intrinsics
        )

    def _handle_capture_error(self, error: Exception):
        """处理连续捕获失败，尝试重启摄像头"""
        if self.camera_type == CameraType.OPENCV:
            logger.warning("Restarting OpenCV camera after consecutive failures")
            try:
                if self.camera:
                    self.camera.release()
                self._initialize_opencv_camera()
                return True
            except Exception as e:
                logger.error(f"Failed to restart camera: {e}")
                self.connected = False
                return False
        if self.camera_type == CameraType.REALSENSE:
            logger.warning("Restarting RealSense pipeline after consecutive failures")
            try:
                if self.pipeline:
                    self.pipeline.stop()
                self._initialize_realsense_camera()
                return True
            except Exception as e:
                logger.error(f"Failed to restart RealSense pipeline: {e}")
                self.connected = False
                return False
        return False

    def _get_realsense_serial(self) -> Optional[str]:
        """从配置中获取RealSense序列号（优先serial，其次camera_index字符串）"""
        serial = getattr(self.config, "serial", None)
        if serial is not None:
            serial_str = str(serial).strip()
            if serial_str:
                return serial_str
        camera_id = getattr(self.config, "camera_index", None)
        if isinstance(camera_id, str):
            camera_id = camera_id.strip()
            if camera_id:
                return camera_id
        return None

    def _map_realsense_color_format(self, fmt_name: str):
        """将配置的颜色格式字符串映射为 RealSense 格式常量"""
        if rs is None:
            raise RuntimeError("pyrealsense2 不可用")
        fmt = str(fmt_name).lower()
        mapping = {
            "rgb8": rs.format.rgb8,
            "bgr8": rs.format.bgr8,
            "yuyv": rs.format.yuyv,
        }
        if fmt not in mapping:
            logger.warning(f"未知的颜色格式 {fmt_name}，回退为 rgb8")
        return mapping.get(fmt, rs.format.rgb8)

    def _map_realsense_depth_format(self, fmt_name: str):
        """将配置的深度格式字符串映射为 RealSense 格式常量"""
        if rs is None:
            raise RuntimeError("pyrealsense2 不可用")
        fmt = str(fmt_name).lower()
        mapping = {
            "z16": rs.format.z16,
            "disparity16": rs.format.disparity16,
        }
        if fmt not in mapping:
            logger.warning(f"未知的深度格式 {fmt_name}，回退为 z16")
        return mapping.get(fmt, rs.format.z16)

    def _get_realsense_depth_resolution(self) -> Tuple[int, int]:
        """获取深度分辨率，未配置时退回彩色分辨率"""
        depth_w = getattr(self.config, "depth_width", None)
        depth_h = getattr(self.config, "depth_height", None)
        if depth_w and depth_h:
            return int(depth_w), int(depth_h)
        return int(self.config.width), int(self.config.height)

    def _configure_realsense_sensors(self, profile):
        """
        设置RealSense相机参数（自动曝光/曝光/白平衡）。
        某些型号或驱动可能不支持部分选项，需要做容错。
        """
        if rs is None or profile is None:
            return
        try:
            device = profile.get_device()
        except Exception:
            return

        # 优先找到RGB传感器，避免对深度传感器设置无关选项
        color_sensor = None
        for sensor in device.sensors:
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                name = ""
            if "RGB" in name or "Color" in name or sensor.supports(rs.option.white_balance):
                color_sensor = sensor
                break

        if color_sensor is None:
            return

        try:
            auto_exposure = getattr(self.config, "auto_exposure", True)
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)

            if not auto_exposure and color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(getattr(self.config, "exposure_value", 100)))

            if color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, float(getattr(self.config, "white_balance", 4600)))
        except Exception as e:
            logger.debug(f"RealSense参数设置失败，已忽略: {e}")

    def _update_intrinsics_from_profile(self, profile):
        """从RealSense profile中读取内参并写入缓存/配置"""
        if rs is None or profile is None:
            return
        try:
            stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream_profile.get_intrinsics()
        except Exception as e:
            logger.debug(f"读取RealSense内参失败: {e}")
            return

        # 更新本地内参缓存，供绘图与姿态估计使用
        self.intrinsics.update({
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "width": intr.width,
            "height": intr.height,
            "distortion_coeffs": list(intr.coeffs),
        })

        # 同步实际分辨率，避免“请求分辨率与实际分辨率不一致”
        try:
            self.config.width = int(intr.width)
            self.config.height = int(intr.height)
        except Exception:
            pass

        # 若配置中未提供内参，则同步到config，便于后续模块读取
        for key, value in (("fx", intr.fx), ("fy", intr.fy), ("cx", intr.ppx), ("cy", intr.ppy)):
            if getattr(self.config, key, None) in (None, 0):
                setattr(self.config, key, value)
        if not getattr(self.config, "distortion_coeffs", None):
            setattr(self.config, "distortion_coeffs", list(intr.coeffs))
    
    def _capture_simulation_frame(self, timestamp: float) -> CameraFrame:
        """捕获仿真帧"""
        # 从仿真器获取图像
        # 假设有一个全局的仿真器实例
        from simulation.simulator import get_simulator
        
        simulator = get_simulator()
        if not simulator:
            raise RuntimeError("Simulator not available")
        
        color_image, depth_image = simulator.get_camera_image()
        
        return CameraFrame(
            timestamp=timestamp,
            color_image=color_image,
            depth_image=depth_image,
            intrinsics=self.intrinsics
        )
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """获取最新帧"""
        with self.frame_lock:
            return self.latest_frame
    
    def get_frame_with_timeout(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """带超时的获取帧"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.get_latest_frame()
            if frame:
                return frame
            time.sleep(0.01)
        
        logger.warning(f"Timeout waiting for camera frame")
        return None
    
    def capture_single(self) -> Optional[CameraFrame]:
        """捕获单帧（同步）"""
        if self.camera_type in (CameraType.OPENCV, CameraType.REALSENSE):
            return self._capture_single_frame(time.time())
        else:
            # 对于其他相机，等待新帧
            old_count = self.frame_count
            start_time = time.time()
            
            while time.time() - start_time < 2.0:
                if self.frame_count > old_count:
                    return self.get_latest_frame()
                time.sleep(0.01)
            
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """获取相机信息"""
        return {
            'type': self.camera_type.value,
            'is_capturing': self.is_capturing,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'intrinsics': self.intrinsics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }

    def get_device_info(self) -> Dict[str, Any]:
        """
        获取设备信息（序列号/名称/固件），用于UI自检展示。

        注意：RealSense 需要 pipeline 已启动并获取过 profile；
        其他相机类型仅返回基础名称。
        """
        info = {
            "camera_type": self.camera_type.value,
            "name": None,
            "serial": None,
            "firmware_version": None,
        }

        if self.camera_type == CameraType.REALSENSE:
            if rs is None or self.rs_profile is None:
                return info
            try:
                dev = self.rs_profile.get_device()
                info["name"] = dev.get_info(rs.camera_info.name)
                info["serial"] = dev.get_info(rs.camera_info.serial_number)
                info["firmware_version"] = dev.get_info(rs.camera_info.firmware_version)
            except Exception as exc:
                logger.debug(f"Failed to read RealSense device info: {exc}")
            return info

        if self.camera_type == CameraType.OPENCV:
            info["name"] = "OpenCV USB Camera"
        elif self.camera_type == CameraType.SIMULATION:
            info["name"] = "Simulation Camera"
        return info
    
    def save_calibration_data(self, filepath: str):
        """保存相机标定数据"""
        import json
        
        calibration_data = {
            'camera_type': self.camera_type.value,
            'intrinsics': self.intrinsics,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Camera calibration data saved to {filepath}")
    
    def load_calibration_data(self, filepath: str):
        """加载相机标定数据"""
        import json
        
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)
        
        self.intrinsics.update(calibration_data['intrinsics'])
        logger.info(f"Camera calibration data loaded from {filepath}")
