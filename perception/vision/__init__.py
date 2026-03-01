"""
Vision submodule stubs.
"""

from .camera_capture import CameraCapture
from .image_processor import ImageProcessor
from .object_detector import ObjectDetector
from .pose_estimator import PoseEstimator
from .depth_processor import DepthProcessor
from .pointcloud_fusion import MultiViewPointCloudFusion

__all__ = [
    "CameraCapture",
    "ImageProcessor",
    "ObjectDetector",
    "PoseEstimator",
    "DepthProcessor",
    "MultiViewPointCloudFusion",
]
