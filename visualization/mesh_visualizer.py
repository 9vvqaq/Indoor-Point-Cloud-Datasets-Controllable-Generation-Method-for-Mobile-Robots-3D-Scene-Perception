"""
Mesh可视化器
提供房间mesh和家具的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Any, Optional


class MeshVisualizer:
    """
    Mesh可视化器
    提供房间mesh和家具的可视化功能
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("visualization_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_room_mesh(self, mesh: o3d.geometry.TriangleMesh,
                           room_bounds: Dict[str, float],
                           title: str = "房间Mesh",
                           save_path: Optional[Path] = None) -> Path:
        """
        可视化房间mesh
        
        Args:
            mesh: 房间mesh
            room_bounds: 房间边界
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取mesh数据
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            # 绘制mesh
            mesh_collection = Poly3DCollection(vertices[faces], alpha=0.6, facecolor='lightblue', edgecolor='black')
            ax.add_collection3d(mesh_collection)
            
            # 设置视角范围
            x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
            y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
            z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
            
            margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)
        
        # 绘制房间边界
        self._draw_room_bounds_3d(ax, room_bounds)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "room_mesh.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_mesh_with_trajectory(self, mesh: o3d.geometry.TriangleMesh,
                                     waypoints: List,
                                     room_bounds: Dict[str, float],
                                     title: str = "房间Mesh与轨迹",
                                     save_path: Optional[Path] = None) -> Path:
        """
        可视化房间mesh和机器人轨迹
        
        Args:
            mesh: 房间mesh
            waypoints: 路径点列表
            room_bounds: 房间边界
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制mesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            mesh_collection = Poly3DCollection(vertices[faces], alpha=0.3, facecolor='lightblue', edgecolor='black')
            ax.add_collection3d(mesh_collection)
        
        # 绘制轨迹
        if waypoints:
            x_coords = [w.x for w in waypoints]
            y_coords = [w.y for w in waypoints]
            z_coords = [w.z for w in waypoints]
            
            ax.plot(x_coords, y_coords, z_coords, 'r-', linewidth=3, label='机器人轨迹')
            ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, alpha=0.8)
            
            # 标记起始点和终点
            if len(waypoints) > 0:
                ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, 
                          marker='o', label='起点')
                ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='blue', s=100, 
                          marker='s', label='终点')
        
        # 绘制房间边界
        self._draw_room_bounds_3d(ax, room_bounds)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "mesh_with_trajectory.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_mesh_with_furniture(self, room_mesh: o3d.geometry.TriangleMesh,
                                    furniture_list: List,
                                    room_bounds: Dict[str, float],
                                    title: str = "房间与家具",
                                    save_path: Optional[Path] = None) -> Path:
        """
        可视化房间mesh和家具
        
        Args:
            room_mesh: 房间mesh
            furniture_list: 家具列表
            room_bounds: 房间边界
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制房间mesh
        vertices = np.asarray(room_mesh.vertices)
        faces = np.asarray(room_mesh.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            mesh_collection = Poly3DCollection(vertices[faces], alpha=0.2, facecolor='lightblue', edgecolor='black')
            ax.add_collection3d(mesh_collection)
        
        # 绘制家具
        colors = plt.cm.Set1(np.linspace(0, 1, len(furniture_list)))
        for i, furniture in enumerate(furniture_list):
            self._draw_furniture_3d(ax, furniture, color=colors[i], alpha=0.7)
        
        # 绘制房间边界
        self._draw_room_bounds_3d(ax, room_bounds)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "mesh_with_furniture.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_mesh_statistics(self, mesh: o3d.geometry.TriangleMesh,
                                 room_bounds: Dict[str, float],
                                 save_path: Optional[Path] = None) -> Path:
        """
        可视化mesh统计信息
        
        Args:
            mesh: 房间mesh
            room_bounds: 房间边界
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # 1. 顶点分布
        if len(vertices) > 0:
            ax1.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                       cmap='viridis', s=1, alpha=0.6)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('顶点分布 (颜色表示Z坐标)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Z坐标分布
        if len(vertices) > 0:
            ax2.hist(vertices[:, 2], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Z坐标 (m)')
            ax2.set_ylabel('频次')
            ax2.set_title('Z坐标分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 面片面积分布
        if len(faces) > 0:
            face_areas = []
            for face in faces:
                if len(face) == 3:
                    v1, v2, v3 = vertices[face]
                    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                    face_areas.append(area)
            
            ax3.hist(face_areas, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('面片面积 (m²)')
            ax3.set_ylabel('频次')
            ax3.set_title('面片面积分布')
            ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息
        stats_text = f"""
        Mesh统计信息
        ============
        
        顶点数: {len(vertices):,}
        面片数: {len(faces):,}
        
        房间尺寸:
        X: {room_bounds['x_max'] - room_bounds['x_min']:.2f} m
        Y: {room_bounds['y_max'] - room_bounds['y_min']:.2f} m
        Z: {room_bounds['z_max'] - room_bounds['z_min']:.2f} m
        
        房间体积: {(room_bounds['x_max'] - room_bounds['x_min']) * 
                  (room_bounds['y_max'] - room_bounds['y_min']) * 
                  (room_bounds['z_max'] - room_bounds['z_min']):.2f} m³
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "mesh_statistics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _draw_room_bounds_3d(self, ax, room_bounds: Dict[str, float]):
        """绘制3D房间边界"""
        x_min, x_max = room_bounds['x_min'], room_bounds['x_max']
        y_min, y_max = room_bounds['y_min'], room_bounds['y_max']
        z_min, z_max = room_bounds['z_min'], room_bounds['z_max']
        
        # 绘制房间边界框
        # 底面
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_min, z_min, z_min, z_min, z_min], 'r--', alpha=0.7, linewidth=2)
        
        # 顶面
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [z_max, z_max, z_max, z_max, z_max], 'r--', alpha=0.7, linewidth=2)
        
        # 垂直边
        for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            ax.plot([x, x], [y, y], [z_min, z_max], 'r--', alpha=0.7, linewidth=2)
    
    def _draw_furniture_3d(self, ax, furniture, color='gray', alpha=0.7):
        """绘制3D家具"""
        bounds = furniture.get_bounds()
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        z_min, z_max = bounds['z_min'], bounds['z_max']
        
        # 绘制家具立方体
        self._draw_cube_3d(ax, x_min, x_max, y_min, y_max, z_min, z_max, 
                          color=color, alpha=alpha)
    
    def _draw_cube_3d(self, ax, x_min, x_max, y_min, y_max, z_min, z_max, 
                     color='gray', alpha=0.7):
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
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1)
    
    def save_mesh_to_ply(self, mesh: o3d.geometry.TriangleMesh,
                        save_path: Path):
        """保存mesh到PLY文件"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(save_path), mesh)
        print(f"Mesh已保存到: {save_path}")
    
    def load_mesh_from_ply(self, load_path: Path) -> o3d.geometry.TriangleMesh:
        """从PLY文件加载mesh"""
        mesh = o3d.io.read_triangle_mesh(str(load_path))
        if len(mesh.vertices) == 0:
            raise ValueError(f"无法加载mesh文件: {load_path}")
        return mesh
