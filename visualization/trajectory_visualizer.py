"""
轨迹可视化器
提供机器人轨迹的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    from ..trajectory.trajectory_generator import Waypoint
except ImportError:
    from trajectory.trajectory_generator import Waypoint


class TrajectoryVisualizer:
    """
    轨迹可视化器
    提供多种轨迹可视化方式
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("visualization_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_trajectory_2d(self, waypoints: List[Waypoint], 
                               room_bounds: Dict[str, float],
                               furniture_list: Optional[List] = None,
                               title: str = "机器人轨迹 (俯视图)",
                               save_path: Optional[Path] = None) -> Path:
        """
        2D轨迹可视化（俯视图）
        
        Args:
            waypoints: 路径点列表
            room_bounds: 房间边界
            furniture_list: 家具列表
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制房间边界
        self._draw_room_bounds(ax, room_bounds)
        
        # 绘制家具
        if furniture_list:
            self._draw_furniture_2d(ax, furniture_list)
        
        # 绘制轨迹
        self._draw_trajectory_2d(ax, waypoints)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 添加图例
        ax.legend()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "trajectory_2d.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_trajectory_3d(self, waypoints: List[Waypoint],
                               room_bounds: Dict[str, float],
                               furniture_list: Optional[List] = None,
                               title: str = "机器人轨迹 (3D视图)",
                               save_path: Optional[Path] = None) -> Path:
        """
        3D轨迹可视化
        
        Args:
            waypoints: 路径点列表
            room_bounds: 房间边界
            furniture_list: 家具列表
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制房间边界
        self._draw_room_bounds_3d(ax, room_bounds)
        
        # 绘制家具
        if furniture_list:
            self._draw_furniture_3d(ax, furniture_list)
        
        # 绘制轨迹
        self._draw_trajectory_3d(ax, waypoints)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "trajectory_3d.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_trajectory_comparison(self, trajectories: Dict[str, List[Waypoint]],
                                      room_bounds: Dict[str, float],
                                      title: str = "轨迹策略对比",
                                      save_path: Optional[Path] = None) -> Path:
        """
        轨迹对比可视化
        
        Args:
            trajectories: 轨迹字典 {策略名: 路径点列表}
            room_bounds: 房间边界
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制房间边界
        self._draw_room_bounds(ax, room_bounds)
        
        # 绘制不同策略的轨迹
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
        
        for i, (strategy_name, waypoints) in enumerate(trajectories.items()):
            self._draw_trajectory_2d(ax, waypoints, color=colors[i], label=strategy_name)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "trajectory_comparison.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_trajectory_statistics(self, waypoints: List[Waypoint],
                                      quality_metrics: Dict[str, Any],
                                      save_path: Optional[Path] = None) -> Path:
        """
        轨迹统计信息可视化
        
        Args:
            waypoints: 路径点列表
            quality_metrics: 质量指标
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 路径长度分布
        if len(waypoints) > 1:
            distances = []
            for i in range(1, len(waypoints)):
                distances.append(waypoints[i].distance_to(waypoints[i-1]))
            
            ax1.plot(distances, 'b-', linewidth=2)
            ax1.set_xlabel('路径段索引')
            ax1.set_ylabel('距离 (m)')
            ax1.set_title('路径段长度分布')
            ax1.grid(True, alpha=0.3)
        
        # 2. 朝向变化
        if len(waypoints) > 1:
            yaw_changes = []
            for i in range(1, len(waypoints)):
                yaw_change = abs(waypoints[i].yaw - waypoints[i-1].yaw)
                if yaw_change > np.pi:
                    yaw_change = 2 * np.pi - yaw_change
                yaw_changes.append(yaw_change)
            
            ax2.plot(yaw_changes, 'g-', linewidth=2)
            ax2.set_xlabel('路径段索引')
            ax2.set_ylabel('朝向变化 (弧度)')
            ax2.set_title('朝向变化分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 质量指标
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        
        ax3.bar(metrics_names, metrics_values, color='orange', alpha=0.7)
        ax3.set_ylabel('指标值')
        ax3.set_title('轨迹质量指标')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 路径点分布
        positions = np.array([[w.x, w.y] for w in waypoints])
        ax4.scatter(positions[:, 0], positions[:, 1], c=range(len(waypoints)), 
                   cmap='viridis', s=50, alpha=0.7)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('路径点分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "trajectory_statistics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _draw_room_bounds(self, ax, room_bounds: Dict[str, float]):
        """绘制房间边界"""
        x_min, x_max = room_bounds['x_min'], room_bounds['x_max']
        y_min, y_max = room_bounds['y_min'], room_bounds['y_max']
        
        # 绘制房间边界
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                'r--', linewidth=2, alpha=0.7, label='房间边界')
    
    def _draw_room_bounds_3d(self, ax, room_bounds: Dict[str, float]):
        """绘制3D房间边界"""
        x_min, x_max = room_bounds['x_min'], room_bounds['x_max']
        y_min, y_max = room_bounds['y_min'], room_bounds['y_max']
        z_min, z_max = room_bounds['z_min'], room_bounds['z_max']
        
        # 绘制房间边界框
        # 底面
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_min, z_min, z_min, z_min, z_min], 'r--', alpha=0.7)
        
        # 顶面
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_max, z_max, z_max, z_max, z_max], 'r--', alpha=0.7)
        
        # 垂直边
        for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            ax.plot([x, x], [y, y], [z_min, z_max], 'r--', alpha=0.7)
    
    def _draw_furniture_2d(self, ax, furniture_list: List):
        """绘制2D家具"""
        for furniture in furniture_list:
            bounds = furniture.get_bounds()
            x_min, x_max = bounds['x_min'], bounds['x_max']
            y_min, y_max = bounds['y_min'], bounds['y_max']
            
            # 绘制家具矩形
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=True, alpha=0.3, color='gray', 
                                     label=furniture.name if hasattr(furniture, 'name') else '家具'))
    
    def _draw_furniture_3d(self, ax, furniture_list: List):
        """绘制3D家具"""
        for furniture in furniture_list:
            bounds = furniture.get_bounds()
            x_min, x_max = bounds['x_min'], bounds['x_max']
            y_min, y_max = bounds['y_min'], bounds['y_max']
            z_min, z_max = bounds['z_min'], bounds['z_max']
            
            # 绘制家具立方体
            self._draw_cube_3d(ax, x_min, x_max, y_min, y_max, z_min, z_max, 
                             color='gray', alpha=0.3)
    
    def _draw_cube_3d(self, ax, x_min, x_max, y_min, y_max, z_min, z_max, 
                     color='gray', alpha=0.3):
        """绘制3D立方体"""
        # 定义立方体的8个顶点
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # 定义立方体的6个面
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        
        # 绘制每个面
        for face in faces:
            face_vertices = [vertices[i] for i in face]
            face_vertices.append(face_vertices[0])  # 闭合面
            
            xs, ys, zs = zip(*face_vertices)
            ax.plot(xs, ys, zs, color=color, alpha=alpha)
    
    def _draw_trajectory_2d(self, ax, waypoints: List[Waypoint], 
                          color='blue', label='轨迹'):
        """绘制2D轨迹"""
        if not waypoints:
            return
        
        # 提取坐标
        x_coords = [w.x for w in waypoints]
        y_coords = [w.y for w in waypoints]
        
        # 绘制轨迹线
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=label)
        
        # 绘制路径点
        ax.scatter(x_coords, y_coords, color=color, s=30, alpha=0.7)
        
        # 标记起始点和终点
        if len(waypoints) > 0:
            ax.scatter(x_coords[0], y_coords[0], color='green', s=100, 
                      marker='o', label='起点')
            ax.scatter(x_coords[-1], y_coords[-1], color='red', s=100, 
                      marker='s', label='终点')
        
        # 绘制朝向箭头
        for i in range(0, len(waypoints), max(1, len(waypoints) // 10)):
            waypoint = waypoints[i]
            dx = 0.2 * np.cos(waypoint.yaw)
            dy = 0.2 * np.sin(waypoint.yaw)
            ax.arrow(waypoint.x, waypoint.y, dx, dy, 
                    head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.7)
    
    def _draw_trajectory_3d(self, ax, waypoints: List[Waypoint]):
        """绘制3D轨迹"""
        if not waypoints:
            return
        
        # 提取坐标
        x_coords = [w.x for w in waypoints]
        y_coords = [w.y for w in waypoints]
        z_coords = [w.z for w in waypoints]
        
        # 绘制轨迹线
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='轨迹')
        
        # 绘制路径点
        ax.scatter(x_coords, y_coords, z_coords, c=range(len(waypoints)), 
                  cmap='viridis', s=30, alpha=0.7)
        
        # 标记起始点和终点
        if len(waypoints) > 0:
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, 
                      marker='o', label='起点')
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, 
                      marker='s', label='终点')
