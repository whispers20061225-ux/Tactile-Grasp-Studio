# tactile_perception/tactile_mapper.py
import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Dict, Any
import logging
import time 

logger = logging.getLogger(__name__)

class TactileMapper:
    """
    触觉数据映射器
    将离散测点数据映射到连续空间，创建触觉图像
    """
    
    def __init__(self, 
                 grid_shape: Tuple[int, int] = (3, 3),
                 sensor_size: Tuple[float, float] = (20.0, 20.0),  # mm
                 interpolation_method: str = 'cubic'):
        
        self.grid_shape = grid_shape
        self.sensor_size = sensor_size
        self.interpolation_method = interpolation_method
        
        # 计算测点物理位置 (mm)
        self.tactile_positions = self._calculate_positions()
        
        # 插值网格
        self.interp_grid_x, self.interp_grid_y = self._create_interp_grid(resolution=0.5)
        
        logger.info(f"TactileMapper initialized with {grid_shape} grid")
    
    def _calculate_positions(self) -> np.ndarray:
        """计算测点的物理位置"""
        positions = []
        
        # 假设测点均匀分布在传感器表面
        x_spacing = self.sensor_size[0] / (self.grid_shape[1] + 1)
        y_spacing = self.sensor_size[1] / (self.grid_shape[0] + 1)
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                x = (j + 1) * x_spacing - self.sensor_size[0] / 2
                y = (i + 1) * y_spacing - self.sensor_size[1] / 2
                positions.append([x, y])
        
        return np.array(positions)
    
    def _create_interp_grid(self, resolution: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """创建插值网格"""
        # 网格边界
        x_min, x_max = -self.sensor_size[0]/2, self.sensor_size[0]/2
        y_min, y_max = -self.sensor_size[1]/2, self.sensor_size[1]/2
        
        # 创建网格
        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, int(self.sensor_size[0]/resolution)),
            y_min:y_max:complex(0, int(self.sensor_size[1]/resolution))
        ]
        
        return grid_x, grid_y
    
    def create_tactile_image(self, forces: np.ndarray, 
                            force_component: int = 2) -> np.ndarray:
        """
        创建触觉图像
        
        Args:
            forces: 力数据数组 [num_tactels, 3]
            force_component: 0=Fx, 1=Fy, 2=Fz
        
        Returns:
            插值后的触觉图像
        """
        try:
            # 提取指定力分量
            if len(forces.shape) == 2 and forces.shape[1] == 3:  # [Fx, Fy, Fz]
                force_values = forces[:, force_component]
            else:  # 假设已经是单分量
                force_values = forces.flatten()
            
            # 检查数据有效性
            if len(force_values) != len(self.tactile_positions):
                logger.error(f"Force data length {len(force_values)} "
                           f"does not match positions {len(self.tactile_positions)}")
                # 尝试填充或截断
                if len(force_values) > len(self.tactile_positions):
                    force_values = force_values[:len(self.tactile_positions)]
                else:
                    # 填充零值
                    force_values = np.pad(force_values, 
                                         (0, len(self.tactile_positions) - len(force_values)),
                                         'constant')
            
            # 插值到连续网格
            if self.interpolation_method == 'rbf':
                # 使用径向基函数插值
                interp_values = self._rbf_interpolation(force_values)
            else:
                # 使用griddata插值
                points = self.tactile_positions
                values = force_values
                
                # 创建三角剖分以确保凸包外的点有值
                try:
                    tri = Delaunay(points)
                except:
                    # 如果无法创建三角剖分，使用最近邻插值
                    self.interpolation_method = 'nearest'
                
                # 插值
                interp_values = griddata(
                    points, values,
                    (self.interp_grid_x, self.interp_grid_y),
                    method=self.interpolation_method,
                    fill_value=0.0
                )
            
            return interp_values
            
        except Exception as e:
            logger.error(f"Error creating tactile image: {e}")
            return np.zeros(self.interp_grid_x.shape)
    
    def _rbf_interpolation(self, force_values: np.ndarray) -> np.ndarray:
        """径向基函数插值"""
        try:
            # 创建RBF插值器
            rbf = Rbf(self.tactile_positions[:, 0],
                     self.tactile_positions[:, 1],
                     force_values,
                     function='multiquadric',  # 多二次函数
                     smooth=0.1)  # 平滑参数
            
            # 在网格上插值
            interp_values = rbf(self.interp_grid_x, self.interp_grid_y)
            
            # 确保非负（对于压力）
            interp_values = np.maximum(interp_values, 0)
            
            return interp_values
            
        except Exception as e:
            logger.error(f"RBF interpolation error: {e}")
            # 回退到线性插值
            return griddata(
                self.tactile_positions, force_values,
                (self.interp_grid_x, self.interp_grid_y),
                method='linear', fill_value=0.0
            )
    
    def create_3d_surface(self, forces: np.ndarray,
                         force_component: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建3D表面"""
        tactile_image = self.create_tactile_image(forces, force_component)
        
        return self.interp_grid_x, self.interp_grid_y, tactile_image
    
    def calculate_contact_area(self, tactile_image: np.ndarray,
                              threshold: float = 0.1) -> float:
        """计算接触面积"""
        # 二值化
        binary_image = tactile_image > threshold
        
        # 计算面积 (像素计数)
        contact_pixels = np.sum(binary_image)
        total_pixels = tactile_image.size
        
        # 转换为物理面积 (mm²)
        pixel_width = (self.sensor_size[0] / tactile_image.shape[0])
        pixel_height = (self.sensor_size[1] / tactile_image.shape[1])
        pixel_area = pixel_width * pixel_height
        contact_area = contact_pixels * pixel_area
        
        return contact_area
    
    def calculate_force_centroid(self, tactile_image: np.ndarray) -> Tuple[float, float]:
        """计算力分布的质心"""
        # 计算加权质心
        total_force = np.sum(tactile_image)
        
        if total_force > 0:
            # X方向质心
            centroid_x = np.sum(self.interp_grid_x * tactile_image) / total_force
            # Y方向质心
            centroid_y = np.sum(self.interp_grid_y * tactile_image) / total_force
        else:
            centroid_x, centroid_y = 0.0, 0.0
        
        return centroid_x, centroid_y
    
    def calculate_force_moments(self, tactile_image: np.ndarray) -> Dict[str, float]:
        """计算力分布矩"""
        centroid_x, centroid_y = self.calculate_force_centroid(tactile_image)
        total_force = np.sum(tactile_image)
        
        if total_force == 0:
            return {
                'Mxx': 0.0, 'Myy': 0.0, 'Mxy': 0.0,
                'eccentricity': 0.0, 'orientation': 0.0
            }
        
        # 计算二阶矩
        dx = self.interp_grid_x - centroid_x
        dy = self.interp_grid_y - centroid_y
        
        Mxx = np.sum(dx**2 * tactile_image) / total_force
        Myy = np.sum(dy**2 * tactile_image) / total_force
        Mxy = np.sum(dx * dy * tactile_image) / total_force
        
        # 计算偏心率和方向
        eccentricity = np.sqrt((Mxx - Myy)**2 + 4 * Mxy**2) / (Mxx + Myy + 1e-6)
        orientation = 0.5 * np.arctan2(2 * Mxy, Mxx - Myy)
        
        return {
            'Mxx': float(Mxx),
            'Myy': float(Myy),
            'Mxy': float(Mxy),
            'eccentricity': float(eccentricity),
            'orientation': float(np.degrees(orientation))
        }
    
    def segment_contact_regions(self, tactile_image: np.ndarray,
                               threshold: float = 0.1,
                               min_area: float = 1.0) -> List[Dict[str, Any]]:
        """分割接触区域"""
        try:
            # 尝试导入skimage，如果不可用则使用简单方法
            try:
                from skimage import measure, morphology
                use_skimage = True
            except ImportError:
                use_skimage = False
                logger.warning("scikit-image not available, using simple region segmentation")
            
            # 二值化
            binary_image = tactile_image > threshold
            
            if use_skimage:
                # 形态学操作去除噪声
                binary_image = morphology.binary_opening(binary_image)
                binary_image = morphology.binary_closing(binary_image)
                
                # 标记连通区域
                labeled_image = measure.label(binary_image)
                regions = measure.regionprops(labeled_image, intensity_image=tactile_image)
                
                # 转换区域属性
                segmented_regions = []
                pixel_width = (self.sensor_size[0] / tactile_image.shape[0])
                pixel_height = (self.sensor_size[1] / tactile_image.shape[1])
                pixel_area = pixel_width * pixel_height
                
                for region in regions:
                    # 计算区域面积
                    area_pixels = region.area
                    area_mm2 = area_pixels * pixel_area
                    
                    if area_mm2 < min_area:
                        continue  # 忽略太小的区域
                    
                    # 转换物理坐标
                    centroid_y, centroid_x = region.centroid
                    phys_centroid_x = (centroid_x / tactile_image.shape[1]) * self.sensor_size[0] - self.sensor_size[0]/2
                    phys_centroid_y = (centroid_y / tactile_image.shape[0]) * self.sensor_size[1] - self.sensor_size[1]/2
                    
                    # 区域属性
                    region_data = {
                        'area': area_mm2,
                        'centroid': (float(phys_centroid_x), float(phys_centroid_y)),
                        'mean_intensity': float(region.mean_intensity),
                        'max_intensity': float(region.max_intensity),
                        'min_intensity': float(region.min_intensity),
                        'bbox': region.bbox,
                        'eccentricity': float(region.eccentricity),
                        'orientation': float(np.degrees(region.orientation) if region.orientation else 0.0)
                    }
                    
                    segmented_regions.append(region_data)
                
                return segmented_regions
            else:
                # 简单区域分割方法
                return self._simple_region_segmentation(binary_image, tactile_image, min_area)
            
        except Exception as e:
            logger.error(f"Error in contact region segmentation: {e}")
            return []
    
    def _simple_region_segmentation(self, binary_image: np.ndarray, 
                                   tactile_image: np.ndarray,
                                   min_area: float = 1.0) -> List[Dict[str, Any]]:
        """简单区域分割方法（不依赖scikit-image）"""
        from collections import deque
        
        height, width = binary_image.shape
        visited = np.zeros_like(binary_image, dtype=bool)
        regions = []
        
        pixel_width = (self.sensor_size[0] / width)
        pixel_height = (self.sensor_size[1] / height)
        pixel_area = pixel_width * pixel_height
        
        # 定义4连通方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] and not visited[i, j]:
                    # 找到新的区域
                    queue = deque([(i, j)])
                    region_pixels = []
                    region_intensities = []
                    
                    while queue:
                        y, x = queue.popleft()
                        if 0 <= y < height and 0 <= x < width and binary_image[y, x] and not visited[y, x]:
                            visited[y, x] = True
                            region_pixels.append((y, x))
                            region_intensities.append(tactile_image[y, x])
                            
                            # 检查邻居
                            for dy, dx in directions:
                                queue.append((y + dy, x + dx))
                    
                    # 计算区域属性
                    area_pixels = len(region_pixels)
                    area_mm2 = area_pixels * pixel_area
                    
                    if area_mm2 < min_area:
                        continue
                    
                    # 计算质心
                    sum_y = sum(p[0] for p in region_pixels)
                    sum_x = sum(p[1] for p in region_pixels)
                    centroid_y = sum_y / area_pixels
                    centroid_x = sum_x / area_pixels
                    
                    # 转换为物理坐标
                    phys_centroid_x = (centroid_x / width) * self.sensor_size[0] - self.sensor_size[0]/2
                    phys_centroid_y = (centroid_y / height) * self.sensor_size[1] - self.sensor_size[1]/2
                    
                    # 计算边界框
                    min_y = min(p[0] for p in region_pixels)
                    max_y = max(p[0] for p in region_pixels)
                    min_x = min(p[1] for p in region_pixels)
                    max_x = max(p[1] for p in region_pixels)
                    
                    region_data = {
                        'area': area_mm2,
                        'centroid': (float(phys_centroid_x), float(phys_centroid_y)),
                        'mean_intensity': float(np.mean(region_intensities)),
                        'max_intensity': float(np.max(region_intensities)),
                        'min_intensity': float(np.min(region_intensities)),
                        'bbox': (min_y, min_x, max_y, max_x),
                        'eccentricity': 0.0,  # 简化版本
                        'orientation': 0.0    # 简化版本
                    }
                    
                    regions.append(region_data)
        
        return regions
    
    def visualize_tactile_image(self, tactile_image: np.ndarray,
                               title: str = "触觉图像",
                               show_positions: bool = True,
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """可视化触觉图像"""
        fig = plt.figure(figsize=(12, 5))
        
        # 1. 2D热图
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(tactile_image.T, 
                       extent=[-self.sensor_size[0]/2, self.sensor_size[0]/2,
                               -self.sensor_size[1]/2, self.sensor_size[1]/2],
                       origin='lower',
                       cmap='viridis',
                       aspect='auto')
        
        ax1.set_title(f"{title} - 2D视图")
        ax1.set_xlabel("X 位置 (mm)")
        ax1.set_ylabel("Y 位置 (mm)")
        plt.colorbar(im, ax=ax1, label='力 (N)')
        
        # 显示测点位置
        if show_positions:
            ax1.scatter(self.tactile_positions[:, 0],
                       self.tactile_positions[:, 1],
                       c='red', s=50, marker='o',
                       label='测点位置')
            ax1.legend()
        
        # 2. 3D表面图
        ax2 = fig.add_subplot(122, projection='3d')
        
        # 创建网格
        X, Y = np.meshgrid(np.linspace(-self.sensor_size[0]/2, self.sensor_size[0]/2, tactile_image.shape[1]),
                          np.linspace(-self.sensor_size[1]/2, self.sensor_size[1]/2, tactile_image.shape[0]))
        
        surf = ax2.plot_surface(X, Y, tactile_image,
                               cmap='viridis', alpha=0.8, linewidth=0)
        
        ax2.set_title(f"{title} - 3D视图")
        ax2.set_xlabel("X 位置 (mm)")
        ax2.set_ylabel("Y 位置 (mm)")
        ax2.set_zlabel("力 (N)")
        
        # 显示测点位置
        if show_positions:
            z_values = np.zeros(len(self.tactile_positions))
            ax2.scatter(self.tactile_positions[:, 0],
                       self.tactile_positions[:, 1],
                       z_values,
                       c='red', s=50, marker='o')
        
        plt.colorbar(surf, ax=ax2, label='力 (N)', shrink=0.5)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tactile image saved to {save_path}")
            plt.close(fig)
            return None
        else:
            plt.show()
            return fig
    
    def create_force_vector_field(self, forces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """创建力向量场"""
        # 分别插值Fx和Fy分量
        fx_image = self.create_tactile_image(forces, force_component=0)
        fy_image = self.create_tactile_image(forces, force_component=1)
        
        return self.interp_grid_x, self.interp_grid_y, fx_image, fy_image
    
    def calculate_force_divergence(self, fx_image: np.ndarray, fy_image: np.ndarray) -> np.ndarray:
        """计算力场的散度"""
        try:
            # 计算梯度
            fx_dx = np.gradient(fx_image, axis=0)
            fy_dy = np.gradient(fy_image, axis=1)
            
            # 散度 = ∂Fx/∂x + ∂Fy/∂y
            divergence = fx_dx + fy_dy
            
            return divergence
            
        except Exception as e:
            logger.error(f"Error calculating force divergence: {e}")
            return np.zeros_like(fx_image)
    
    def calculate_force_curl(self, fx_image: np.ndarray, fy_image: np.ndarray) -> np.ndarray:
        """计算力场的旋度"""
        try:
            # 计算梯度
            fx_dy = np.gradient(fx_image, axis=1)
            fy_dx = np.gradient(fy_image, axis=0)
            
            # 旋度 = ∂Fy/∂x - ∂Fx/∂y
            curl = fy_dx - fx_dy
            
            return curl
            
        except Exception as e:
            logger.error(f"Error calculating force curl: {e}")
            return np.zeros_like(fx_image)
    
    def get_contact_statistics(self, tactile_image: np.ndarray, 
                              threshold: float = 0.1) -> Dict[str, Any]:
        """获取接触统计信息"""
        try:
            # 计算接触面积
            contact_area = self.calculate_contact_area(tactile_image, threshold)
            
            # 计算力质心
            centroid_x, centroid_y = self.calculate_force_centroid(tactile_image)
            
            # 计算力矩
            moments = self.calculate_force_moments(tactile_image)
            
            # 计算总力
            total_force = np.sum(tactile_image)
            
            # 计算最大力和平均力
            max_force = np.max(tactile_image)
            mean_force = np.mean(tactile_image) if tactile_image.size > 0 else 0
            
            # 计算力标准差
            force_std = np.std(tactile_image) if tactile_image.size > 0 else 0
            
            return {
                'contact_area_mm2': contact_area,
                'centroid_x_mm': centroid_x,
                'centroid_y_mm': centroid_y,
                'total_force_n': total_force,
                'max_force_n': max_force,
                'mean_force_n': mean_force,
                'force_std_n': force_std,
                'moments': moments,
                'sensor_area_mm2': self.sensor_size[0] * self.sensor_size[1],
                'contact_ratio': contact_area / (self.sensor_size[0] * self.sensor_size[1]) if (self.sensor_size[0] * self.sensor_size[1]) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating contact statistics: {e}")
            return {
                'contact_area_mm2': 0.0,
                'centroid_x_mm': 0.0,
                'centroid_y_mm': 0.0,
                'total_force_n': 0.0,
                'max_force_n': 0.0,
                'mean_force_n': 0.0,
                'force_std_n': 0.0,
                'moments': {'Mxx': 0.0, 'Myy': 0.0, 'Mxy': 0.0, 'eccentricity': 0.0, 'orientation': 0.0},
                'sensor_area_mm2': self.sensor_size[0] * self.sensor_size[1],
                'contact_ratio': 0.0
            }

    def create_force_distribution_histogram(self, tactile_image: np.ndarray, 
                                          bins: int = 20) -> Dict[str, Any]:
        """创建力分布直方图"""
        try:
            # 展平图像
            flat_forces = tactile_image.flatten()
            
            # 移除零值（无接触区域）
            non_zero_forces = flat_forces[flat_forces > 0]
            
            if len(non_zero_forces) == 0:
                return {
                    'hist_values': [0] * bins,
                    'bin_edges': np.linspace(0, np.max(tactile_image), bins + 1).tolist(),
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0
                }
            
            # 计算直方图
            hist, bin_edges = np.histogram(non_zero_forces, bins=bins)
            
            # 计算统计量
            mean_force = np.mean(non_zero_forces)
            median_force = np.median(non_zero_forces)
            std_force = np.std(non_zero_forces)
            
            # 计算偏度和峰度
            from scipy import stats
            skewness = stats.skew(non_zero_forces) if len(non_zero_forces) > 2 else 0.0
            kurtosis = stats.kurtosis(non_zero_forces) if len(non_zero_forces) > 3 else 0.0
            
            return {
                'hist_values': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(mean_force),
                'median': float(median_force),
                'std': float(std_force),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'num_samples': len(non_zero_forces)
            }
            
        except Exception as e:
            logger.error(f"Error creating force distribution histogram: {e}")
            return {
                'hist_values': [],
                'bin_edges': [],
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'num_samples': 0
            }

    def create_contact_heatmap_overlay(self, tactile_image: np.ndarray,
                                      threshold: float = 0.1,
                                      alpha: float = 0.7) -> np.ndarray:
        """创建接触热图叠加层"""
        # 创建二值接触掩码
        contact_mask = tactile_image > threshold
        
        # 创建RGBA图像
        heatmap = np.zeros((*tactile_image.shape, 4))
        
        # 归一化力值用于颜色映射
        if np.max(tactile_image) > 0:
            normalized_force = tactile_image / np.max(tactile_image)
        else:
            normalized_force = np.zeros_like(tactile_image)
        
        # 应用颜色映射（使用matplotlib的viridis）
        cmap = plt.cm.viridis
        
        for i in range(tactile_image.shape[0]):
            for j in range(tactile_image.shape[1]):
                if contact_mask[i, j]:
                    # 有接触的区域使用viridis颜色
                    color = cmap(normalized_force[i, j])
                    heatmap[i, j] = [color[0], color[1], color[2], alpha]
                else:
                    # 无接触区域透明
                    heatmap[i, j] = [0, 0, 0, 0]
        
        return heatmap

    def export_tactile_data(self, tactile_image: np.ndarray,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """导出触觉数据"""
        try:
            # 基础数据
            data = {
                'tactile_image': tactile_image.tolist(),
                'image_shape': tactile_image.shape,
                'sensor_size_mm': self.sensor_size,
                'grid_shape': self.grid_shape,
                'tactile_positions': self.tactile_positions.tolist(),
                'interp_grid_x': self.interp_grid_x.tolist(),
                'interp_grid_y': self.interp_grid_y.tolist(),
                'timestamp': time.time()
            }
            
            # 统计信息
            stats = self.get_contact_statistics(tactile_image)
            data.update({'statistics': stats})
            
            # 力分布直方图
            hist_data = self.create_force_distribution_histogram(tactile_image)
            data.update({'force_distribution': hist_data})
            
            # 添加元数据
            if metadata:
                data['metadata'] = metadata
            
            return data
            
        except Exception as e:
            logger.error(f"Error exporting tactile data: {e}")
            return {'error': str(e)}

    def load_tactile_data(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """从导出的数据加载触觉图像"""
        try:
            if 'tactile_image' not in data:
                logger.error("No tactile image found in data")
                return None
            
            tactile_array = np.array(data['tactile_image'])
            
            # 验证形状
            if 'image_shape' in data:
                expected_shape = tuple(data['image_shape'])
                if tactile_array.shape != expected_shape:
                    logger.warning(f"Image shape mismatch: expected {expected_shape}, got {tactile_array.shape}")
                    # 尝试重塑
                    try:
                        tactile_array = tactile_array.reshape(expected_shape)
                    except:
                        logger.error("Cannot reshape tactile array")
                        return None
            
            return tactile_array
            
        except Exception as e:
            logger.error(f"Error loading tactile data: {e}")
            return None