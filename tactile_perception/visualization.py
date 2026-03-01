# tactile_perception/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict
import threading
import time
import logging

logger = logging.getLogger(__name__)

class TactileVisualizer:
    """
    触觉数据可视化器
    提供实时数据可视化界面
    """
    
    def __init__(self, 
                 update_interval: float = 0.1,  # 更新间隔 (秒)
                 figsize: tuple = (15, 10),
                 colormap: str = 'viridis'):
        
        self.update_interval = update_interval
        self.figsize = figsize
        self.colormap = colormap
        
        # 图形对象
        self.fig = None
        self.ax = None
        self.anim = None
        
        # 数据缓冲区
        self.data_buffer: List = []
        self.buffer_size = 100
        self.latest_data = None
        
        # 可视化元素
        self.scatter = None
        self.quiver = None
        self.heatmap = None
        self.line_plots = []
        self.text_boxes = []
        
        # 控制变量
        self.running = False
        self.update_thread = None
        
        # 颜色映射
        self.cmap = cm.get_cmap(colormap)
        
        logger.info("TactileVisualizer initialized")
    
    def start(self):
        """启动可视化"""
        if self.running:
            logger.warning("Visualizer already running")
            return
        
        # 创建图形
        self._create_figure()
        
        # 启动更新线程
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="VisualizerUpdateThread"
        )
        self.update_thread.start()
        
        # 显示图形
        plt.show(block=False)
        logger.info("Visualizer started")
    
    def stop(self):
        """停止可视化"""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        
        logger.info("Visualizer stopped")
    
    def _create_figure(self):
        """创建图形布局"""
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle('PaXini M2020 触觉传感器实时可视化', fontsize=16, fontweight='bold')
        
        # 使用GridSpec创建复杂布局
        gs = gridspec.GridSpec(3, 4, figure=self.fig, 
                               width_ratios=[1, 1, 1, 0.5],
                               height_ratios=[1, 1, 0.5])
        
        # 1. 3D力向量图
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_3d.set_title('3D力向量分布')
        self.ax_3d.set_xlabel('X (N)')
        self.ax_3d.set_ylabel('Y (N)')
        self.ax_3d.set_zlabel('Z (N)')
        self.ax_3d.set_xlim([-10, 10])
        self.ax_3d.set_ylim([-10, 10])
        self.ax_3d.set_zlim([0, 25])
        
        # 2. 压力分布热图
        self.ax_heatmap = self.fig.add_subplot(gs[0, 1])
        self.ax_heatmap.set_title('压力分布热图')
        self.ax_heatmap.set_xlabel('X 位置')
        self.ax_heatmap.set_ylabel('Y 位置')
        self.ax_heatmap.set_aspect('equal')
        
        # 3. 力时间序列图
        self.ax_force_time = self.fig.add_subplot(gs[1, 0:2])
        self.ax_force_time.set_title('力时间序列')
        self.ax_force_time.set_xlabel('时间 (s)')
        self.ax_force_time.set_ylabel('力 (N)')
        self.ax_force_time.grid(True, alpha=0.3)
        
        # 4. 测点力分布图
        self.ax_tactile = self.fig.add_subplot(gs[0, 2])
        self.ax_tactile.set_title('测点力分布')
        self.ax_tactile.set_xlabel('测点索引')
        self.ax_tactile.set_ylabel('力 (N)')
        self.ax_tactile.grid(True, alpha=0.3)
        
        # 5. 特征雷达图
        self.ax_radar = self.fig.add_subplot(gs[1, 2], polar=True)
        self.ax_radar.set_title('触觉特征雷达图')
        
        # 6. 统计信息文本
        self.ax_text = self.fig.add_subplot(gs[0:2, 3])
        self.ax_text.axis('off')
        
        # 7. 接触状态图
        self.ax_contact = self.fig.add_subplot(gs[2, 0])
        self.ax_contact.set_title('接触状态')
        self.ax_contact.set_aspect('equal')
        
        # 8. 滑移检测图
        self.ax_slip = self.fig.add_subplot(gs[2, 1])
        self.ax_slip.set_title('滑移检测')
        
        # 9. 合力向量图
        self.ax_resultant = self.fig.add_subplot(gs[2, 2])
        self.ax_resultant.set_title('合力向量')
        
        # 调整布局
        plt.tight_layout()
        
        # 初始化图形元素
        self._init_visual_elements()
    
    def _init_visual_elements(self):
        """初始化可视化元素"""
        # 3D散点图
        self.scatter_3d = self.ax_3d.scatter([], [], [], c=[], cmap=self.colormap, s=100)
        
        # 热图
        self.heatmap = self.ax_heatmap.imshow(
            np.zeros((3, 3)), 
            cmap=self.colormap,
            vmin=0, vmax=25,
            interpolation='bilinear'
        )
        plt.colorbar(self.heatmap, ax=self.ax_heatmap, label='压力 (N)')
        
        # 力时间序列线
        self.line_fx, = self.ax_force_time.plot([], [], 'r-', label='Fx', linewidth=2)
        self.line_fy, = self.ax_force_time.plot([], [], 'g-', label='Fy', linewidth=2)
        self.line_fz, = self.ax_force_time.plot([], [], 'b-', label='Fz', linewidth=2)
        self.ax_force_time.legend(loc='upper right')
        
        # 测点力分布柱状图
        x_positions = np.arange(9)
        self.bars_tactile = self.ax_tactile.bar(x_positions, np.zeros(9))
        
        # 雷达图
        self.radar_line, = self.ax_radar.plot([], [], 'b-', linewidth=2)
        self.radar_fill = self.ax_radar.fill([], [], 'b', alpha=0.25)
        
        # 接触状态点
        self.contact_scatter = self.ax_contact.scatter([], [], s=100, c='green')
        
        # 滑移检测指示器
        self.slip_indicator = patches.Circle((0.5, 0.5), 0.3, color='red')
        self.ax_slip.add_patch(self.slip_indicator)
        
        # 合力向量箭头
        self.resultant_arrow = self.ax_resultant.arrow(0, 0, 0, 0, 
                                                      head_width=0.3, head_length=0.5, 
                                                      fc='red', ec='red', linewidth=3)
        
        # 统计文本
        self.text_stats = self.ax_text.text(0.05, 0.95, '', 
                                           transform=self.ax_text.transAxes,
                                           fontsize=10,
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update(self, data):
        """更新可视化数据"""
        if data is None:
            return
        
        # 添加到缓冲区
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
        
        self.latest_data = data
        
        # 检查是否需要重绘
        if self.fig and plt.fignum_exists(self.fig.number):
            self._redraw()
    
    def _redraw(self):
        """重绘所有图形元素"""
        if not self.latest_data:
            return
        
        try:
            data = self.latest_data
            
            # 1. 更新3D力向量图
            self._update_3d_plot(data)
            
            # 2. 更新热图
            self._update_heatmap(data)
            
            # 3. 更新时间序列图
            self._update_time_series(data)
            
            # 4. 更新测点力分布
            self._update_tactile_plot(data)
            
            # 5. 更新雷达图
            self._update_radar_plot(data)
            
            # 6. 更新统计文本
            self._update_stats_text(data)
            
            # 7. 更新接触状态
            self._update_contact_plot(data)
            
            # 8. 更新滑移检测
            self._update_slip_plot(data)
            
            # 9. 更新合力向量
            self._update_resultant_plot(data)
            
            # 重绘图形
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error in redraw: {e}")
    
    def _update_3d_plot(self, data):
        """更新3D力向量图"""
        tactile_array = data.filtered_array
        forces_x = tactile_array[:, 0]
        forces_y = tactile_array[:, 1]
        forces_z = tactile_array[:, 2]
        
        # 更新散点图数据
        self.scatter_3d._offsets3d = (forces_x, forces_y, forces_z)
        
        # 根据Z方向力设置颜色
        colors = self.cmap(forces_z / 25.0)  # 归一化到0-25N范围
        self.scatter_3d.set_color(colors)
    
    def _update_heatmap(self, data):
        """更新热图"""
        pressure_map = data.pressure_map
        
        # 更新热图数据
        self.heatmap.set_data(pressure_map)
        
        # 更新颜色范围
        if pressure_map.max() > 0:
            self.heatmap.set_clim(vmin=0, vmax=pressure_map.max())
        
        # 添加文本标注
        self.ax_heatmap.clear()
        self.ax_heatmap.imshow(pressure_map, cmap=self.colormap, 
                              interpolation='bilinear', vmin=0, vmax=25)
        self.ax_heatmap.set_title('压力分布热图')
        self.ax_heatmap.set_xlabel('X 位置')
        self.ax_heatmap.set_ylabel('Y 位置')
        
        # 在热图上显示数值
        for i in range(pressure_map.shape[0]):
            for j in range(pressure_map.shape[1]):
                value = pressure_map[i, j]
                if value > 0.5:  # 只显示较大的值
                    self.ax_heatmap.text(j, i, f'{value:.1f}', 
                                        ha='center', va='center',
                                        color='white' if value > pressure_map.max()/2 else 'black')
    
    def _update_time_series(self, data):
        """更新时间序列图"""
        # 提取历史数据中的合力
        if len(self.data_buffer) > 1:
            timestamps = [d.timestamp for d in self.data_buffer]
            forces_x = [d.features.get('resultant_force_x', 0) for d in self.data_buffer]
            forces_y = [d.features.get('resultant_force_y', 0) for d in self.data_buffer]
            forces_z = [d.features.get('resultant_force_z', 0) for d in self.data_buffer]
            
            # 转换为相对时间
            if timestamps:
                rel_time = [t - timestamps[0] for t in timestamps]
                
                # 更新线图
                self.line_fx.set_data(rel_time, forces_x)
                self.line_fy.set_data(rel_time, forces_y)
                self.line_fz.set_data(rel_time, forces_z)
                
                # 调整坐标轴范围
                if rel_time:
                    self.ax_force_time.set_xlim(min(rel_time), max(rel_time))
                    
                    all_forces = forces_x + forces_y + forces_z
                    if all_forces:
                        force_min = min(all_forces)
                        force_max = max(all_forces)
                        padding = (force_max - force_min) * 0.1
                        self.ax_force_time.set_ylim(force_min - padding, force_max + padding)
    
    def _update_tactile_plot(self, data):
        """更新测点力分布图"""
        z_forces = data.filtered_array[:, 2]
        
        # 更新柱状图高度
        for i, bar in enumerate(self.bars_tactile):
            if i < len(z_forces):
                bar.set_height(z_forces[i])
                
                # 根据力值设置颜色
                color = self.cmap(z_forces[i] / 25.0)
                bar.set_color(color)
        
        self.ax_tactile.set_ylim(0, max(z_forces.max() * 1.2, 1.0))
    
    def _update_radar_plot(self, data):
        """更新雷达图"""
        # 选择要显示的特征
        features = data.features
        selected_features = [
            'mean_force_z',
            'contact_ratio',
            'force_std_z',
            'resultant_force_magnitude',
            'force_eccentricity',
            'force_change_rate'
        ]
        
        # 提取特征值
        values = []
        angles = []
        
        n_features = len(selected_features)
        for i, feature_name in enumerate(selected_features):
            value = features.get(feature_name, 0)
            
            # 归一化到0-1范围
            if feature_name == 'mean_force_z':
                norm_value = min(value / 25.0, 1.0)
            elif feature_name == 'contact_ratio':
                norm_value = value
            elif feature_name == 'force_std_z':
                norm_value = min(value / 5.0, 1.0)
            elif feature_name == 'resultant_force_magnitude':
                norm_value = min(value / 30.0, 1.0)
            elif feature_name == 'force_eccentricity':
                norm_value = value
            elif feature_name == 'force_change_rate':
                norm_value = min(value / 5.0, 1.0)
            else:
                norm_value = 0
            
            values.append(norm_value)
            
            # 计算角度
            angle = 2 * np.pi * i / n_features
            angles.append(angle)
        
        # 闭合多边形
        values.append(values[0])
        angles.append(angles[0])
        
        # 更新雷达图
        self.radar_line.set_data(angles, values)
        
        # 更新填充区域
        self.ax_radar.clear()
        self.ax_radar.plot(angles, values, 'b-', linewidth=2)
        self.ax_radar.fill(angles, values, 'b', alpha=0.25)
        
        # 设置雷达图刻度
        self.ax_radar.set_xticks(angles[:-1])
        self.ax_radar.set_xticklabels(selected_features, fontsize=8)
        self.ax_radar.set_ylim(0, 1)
        self.ax_radar.set_title('触觉特征雷达图')
        self.ax_radar.grid(True, alpha=0.3)
    
    def _update_stats_text(self, data):
        """更新统计文本"""
        features = data.features
        
        # 构建统计文本
        stats_text = (
            f"帧序列: {data.raw_data.sequence}\n"
            f"时间戳: {data.timestamp:.3f}s\n"
            f"总Z方向力: {features.get('total_force_z', 0):.2f} N\n"
            f"平均Z方向力: {features.get('mean_force_z', 0):.2f} N\n"
            f"最大Z方向力: {features.get('max_force_z', 0):.2f} N\n"
            f"接触测点数: {features.get('contact_count', 0)}/9\n"
            f"接触比例: {features.get('contact_ratio', 0):.2%}\n"
            f"合力大小: {features.get('resultant_force_magnitude', 0):.2f} N\n"
            f"力质心: ({features.get('force_centroid_x', 0):.2f}, "
            f"{features.get('force_centroid_y', 0):.2f})\n"
            f"滑移检测: {'是' if data.slip_detected else '否'}\n"
            f"振动检测: {'是' if data.vibration_detected else '否'}\n"
            f"缓冲区: {len(self.data_buffer)}/{self.buffer_size}"
        )
        
        self.text_stats.set_text(stats_text)
    
    def _update_contact_plot(self, data):
        """更新接触状态图"""
        contact_mask = data.contact_mask
        
        # 3x3网格坐标
        x_coords = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        y_coords = [1, 1, 1, 0, 0, 0, -1, -1, -1]
        
        # 分离接触和非接触点
        contact_x = [x_coords[i] for i in range(9) if contact_mask[i]]
        contact_y = [y_coords[i] for i in range(9) if contact_mask[i]]
        
        non_contact_x = [x_coords[i] for i in range(9) if not contact_mask[i]]
        non_contact_y = [y_coords[i] for i in range(9) if not contact_mask[i]]
        
        # 更新散点图
        self.ax_contact.clear()
        if contact_x:
            self.ax_contact.scatter(contact_x, contact_y, s=200, c='green', 
                                   label=f'接触 ({len(contact_x)})', alpha=0.8)
        if non_contact_x:
            self.ax_contact.scatter(non_contact_x, non_contact_y, s=100, c='red', 
                                   label=f'非接触 ({len(non_contact_x)})', alpha=0.3)
        
        self.ax_contact.set_xlim(-1.5, 1.5)
        self.ax_contact.set_ylim(-1.5, 1.5)
        self.ax_contact.set_title('接触状态')
        self.ax_contact.set_aspect('equal')
        self.ax_contact.legend(loc='upper right')
        self.ax_contact.grid(True, alpha=0.3)
    
    def _update_slip_plot(self, data):
        """更新滑移检测图"""
        # 更新滑移指示器颜色
        if data.slip_detected:
            self.slip_indicator.set_color('red')
            slip_text = '滑移检测: 是'
        else:
            self.slip_indicator.set_color('green')
            slip_text = '滑移检测: 否'
        
        # 添加文本
        self.ax_slip.clear()
        self.ax_slip.add_patch(self.slip_indicator)
        self.ax_slip.text(0.5, 0.2, slip_text, 
                         ha='center', va='center',
                         transform=self.ax_slip.transAxes,
                         fontsize=12, fontweight='bold')
        self.ax_slip.set_xlim(0, 1)
        self.ax_slim.set_ylim(0, 1)
        self.ax_slip.set_title('滑移检测')
        self.ax_slip.axis('off')
    
    def _update_resultant_plot(self, data):
        """更新合力向量图"""
        resultant = data.raw_data.resultant_force
        
        # 计算向量长度和角度
        magnitude = np.linalg.norm(resultant)
        if magnitude > 0:
            angle = np.arctan2(resultant[1], resultant[0])
        else:
            angle = 0
        
        # 更新箭头
        self.ax_resultant.clear()
        self.ax_resultant.arrow(0, 0, 
                               resultant[0]/5, resultant[1]/5,  # 缩放显示
                               head_width=0.3, head_length=0.5,
                               fc='red', ec='red', linewidth=3)
        
        # 添加文本
        self.ax_resultant.text(0.5, 0.9, f'合力大小: {magnitude:.2f} N',
                              ha='center', va='center',
                              transform=self.ax_resultant.transAxes)
        self.ax_resultant.text(0.5, 0.8, f'角度: {np.degrees(angle):.1f}°',
                              ha='center', va='center',
                              transform=self.ax_resultant.transAxes)
        
        self.ax_resultant.set_xlim(-2, 2)
        self.ax_resultant.set_ylim(-2, 2)
        self.ax_resultant.set_title('合力向量')
        self.ax_resultant.set_aspect('equal')
        self.ax_resultant.grid(True, alpha=0.3)
    
    def _update_loop(self):
        """更新循环"""
        while self.running:
            if self.latest_data and self.fig and plt.fignum_exists(self.fig.number):
                try:
                    self._redraw()
                except Exception as e:
                    logger.error(f"Error in update loop: {e}")
            
            time.sleep(self.update_interval)
    
    def save_snapshot(self, filename: str = None):
        """保存当前可视化快照"""
        if not self.fig:
            logger.warning("No figure to save")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tactile_visualization_{timestamp}.png"
        
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Snapshot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")