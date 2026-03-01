"""
多视角点云融合模块 - 基于深度图的实时点云拼接
"""

from typing import Dict, Any, Optional, Tuple
import time

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)


class MultiViewPointCloudFusion:
    """
    多视角点云融合器

    设计思路：
    - 每帧将深度图转为点云，作为当前帧点云。
    - 使用 ICP 将当前帧点云对齐到全局点云坐标系。
    - 融合后下采样与滤波，控制点数规模并降低噪声。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        # 融合参数：体素大小越小越精细，但计算量也越大
        self.voxel_size = float(self.config.get("voxel_size", 0.01))
        # ICP 最近邻距离阈值（米）
        self.icp_max_corr = float(self.config.get("fusion_max_correspondence", 0.05))
        # ICP 最大迭代次数
        self.icp_iterations = int(self.config.get("fusion_icp_iterations", 30))
        # 更新间隔（秒），用于限频
        self.min_update_interval = float(self.config.get("fusion_min_interval", 0.2))
        # 全局点云最大点数限制
        self.max_points = int(self.config.get("fusion_max_points", 200000))
        # 是否保留颜色信息
        self.use_color = bool(self.config.get("use_color", True))

        # 深度过滤范围（单位：米）
        self.min_depth = float(self.config.get("min_depth", 0.1))
        self.max_depth = float(self.config.get("max_depth", 3.0))

        # 离群点滤波参数
        outlier_cfg = self.config.get("statistical_outlier_removal", {}) or {}
        self.outlier_nb = int(outlier_cfg.get("nb_neighbors", 20))
        self.outlier_std = float(outlier_cfg.get("std_ratio", 2.0))

        # 运行状态：全局点云与最后更新时间戳
        self._global_pcd = None
        self._last_update_time = 0.0

        # 延迟导入 Open3D，避免缺依赖时直接崩溃
        try:
            import open3d as o3d  # type: ignore
            self.o3d = o3d
        except Exception as exc:  # pragma: no cover - 运行环境可能缺少 open3d
            self.o3d = None
            logger.warning(f"Open3D not available, point cloud fusion disabled: {exc}")

    def reset(self) -> None:
        """重置融合结果与内部状态"""
        self._global_pcd = None
        self._last_update_time = 0.0

    def get_global_cloud(self):
        """获取当前融合结果（Open3D PointCloud）"""
        return self._global_pcd

    def update(
        self,
        depth_image: np.ndarray,
        color_image: Optional[np.ndarray],
        intrinsics: Dict[str, Any],
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        输入一帧深度图，更新融合点云。

        Args:
            depth_image: 深度图（单位：米）
            color_image: 彩色图（RGB），用于为点云着色
            intrinsics: 相机内参字典（fx, fy, cx, cy, width, height）
            roi_bbox: 可选 ROI（x1, y1, x2, y2），仅融合目标区域
        """
        if self.o3d is None:
            # 缺少 Open3D 依赖时直接跳过
            return None

        if depth_image is None:
            # 没有深度数据时，返回当前融合结果
            return self._global_pcd

        now = time.time()
        if now - self._last_update_time < self.min_update_interval:
            # 限频：避免每帧都执行 ICP 导致卡顿
            return self._global_pcd

        self._last_update_time = now

        # 1) 生成当前帧点云（支持 ROI 裁剪）
        curr_pcd = self._create_pointcloud(depth_image, color_image, intrinsics, roi_bbox)
        if curr_pcd is None or len(curr_pcd.points) == 0:
            return self._global_pcd

        # 2) 首帧直接初始化全局点云
        if self._global_pcd is None:
            self._global_pcd = curr_pcd
            return self._global_pcd

        # 3) ICP 配准：将当前点云对齐到全局点云坐标系
        #    为提升速度，先体素下采样并估计法线
        source = curr_pcd.voxel_down_sample(self.voxel_size)
        target = self._global_pcd.voxel_down_sample(self.voxel_size)

        if len(source.points) < 50 or len(target.points) < 50:
            # 点数过少时不做 ICP，直接合并
            self._global_pcd += curr_pcd
            self._global_pcd = self._global_pcd.voxel_down_sample(self.voxel_size)
            return self._global_pcd

        source.estimate_normals()
        target.estimate_normals()

        # 4) 执行 ICP（点到面）并获取变换矩阵
        icp_result = self.o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.icp_max_corr,
            np.eye(4),
            self.o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            self.o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_iterations
            ),
        )

        # 5) 应用变换并融合到全局点云
        curr_pcd.transform(icp_result.transformation)
        self._global_pcd += curr_pcd

        # 6) 体素下采样 + 离群点剔除，控制规模与噪声
        self._global_pcd = self._global_pcd.voxel_down_sample(self.voxel_size)
        if len(self._global_pcd.points) > self.max_points:
            # 超过上限时按比例随机采样
            self._global_pcd = self._global_pcd.random_down_sample(
                self.max_points / float(len(self._global_pcd.points))
            )
        if self.outlier_nb > 0:
            # 统计离群点剔除
            self._global_pcd, _ = self._global_pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb,
                std_ratio=self.outlier_std,
            )

        return self._global_pcd

    def _create_pointcloud(
        self,
        depth_image: np.ndarray,
        color_image: Optional[np.ndarray],
        intrinsics: Dict[str, Any],
        roi_bbox: Optional[Tuple[int, int, int, int]],
    ):
        """将深度图转换为 Open3D 点云（支持 ROI 裁剪）"""
        if self.o3d is None:
            return None

        # Open3D 需要 float32 深度（单位：米）
        depth = depth_image.astype(np.float32)
        # 深度范围外的像素置零，避免引入远距离噪声
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0

        # 读取内参（缺失时回退到深度图尺寸）
        fx = float(intrinsics.get("fx", 0.0))
        fy = float(intrinsics.get("fy", 0.0))
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        width = int(intrinsics.get("width", depth.shape[1]))
        height = int(intrinsics.get("height", depth.shape[0]))

        if roi_bbox is not None:
            # ROI 裁剪：同步裁剪深度和彩色图，并平移主点坐标
            x1, y1, x2, y2 = roi_bbox
            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(x1 + 1, min(width, int(x2)))
            y2 = max(y1 + 1, min(height, int(y2)))
            depth = depth[y1:y2, x1:x2]
            if color_image is not None:
                color_image = color_image[y1:y2, x1:x2]
            # ROI 后的主点要平移到新坐标系
            cx -= x1
            cy -= y1
            width = depth.shape[1]
            height = depth.shape[0]

        # 确保缓冲区连续，避免 Open3D 报错
        depth = np.ascontiguousarray(depth)

        if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
            # 内参非法无法生成点云
            logger.warning("Invalid camera intrinsics, cannot create point cloud")
            return None

        # 构建 Open3D 相机内参对象
        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        # Open3D 图像封装
        depth_o3d = self.o3d.geometry.Image(depth)
        if color_image is not None and self.use_color:
            # 使用 RGBD 生成彩色点云
            color_u8 = color_image.astype(np.uint8)
            color_u8 = np.ascontiguousarray(color_u8)
            color_o3d = self.o3d.geometry.Image(color_u8)
            # 深度已转换为米，depth_scale 设为 1.0
            rgbd = self.o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=self.max_depth,
                convert_rgb_to_intensity=False,
            )
            pcd = self.o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        else:
            # 只使用深度生成点云
            pcd = self.o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic,
                depth_scale=1.0,
                depth_trunc=self.max_depth,
            )

        return pcd
