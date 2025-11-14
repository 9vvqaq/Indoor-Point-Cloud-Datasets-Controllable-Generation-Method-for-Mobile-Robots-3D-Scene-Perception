#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3DIS仿真可视化模块
支持3D视角、BEV视角和点云导出功能
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

try:
    from ..containers.s3dis_sim_scene import S3DISSimScene, S3DISSimFrame
    from ..containers.s3dis_scene import S3DISScene
except ImportError:
    from containers.s3dis_sim_scene import S3DISSimScene, S3DISSimFrame
    from containers.s3dis_scene import S3DISScene


class S3DISVisualizer:
    """S3DIS仿真可视化器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.vis_dir = output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        
        # 颜色配置
        self.colors = {
            'mesh': [0.7, 0.7, 0.7],      # 灰色 - 房间mesh
            'trajectory': [1.0, 0.0, 0.0], # 红色 - 轨迹
            'lidar_poses': [0.0, 1.0, 0.0], # 绿色 - LiDAR位姿
            'pointcloud': [0.0, 0.0, 1.0],  # 蓝色 - 点云
            'sampled_frames': [1.0, 1.0, 0.0] # 黄色 - 采样帧
        }
    
    
    def visualize_bev_scene(self, sim_scene: S3DISSimScene, 
                           scene: S3DISScene,
                           sampled_frames: Optional[List[int]] = None) -> str:
        """
        BEV (Bird's Eye View) 场景可视化
        
        Args:
            sim_scene: 仿真场景
            scene: 原始场景
            sampled_frames: 采样的帧索引列表
            
        Returns:
            保存的图片路径
        """
        print("[可视化] 生成BEV场景可视化...")
        
        # 设置matplotlib为非交互模式
        plt.ioff()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 绘制房间边界
        bounds = scene.room_bounds
        room_rect = plt.Rectangle(
            (bounds.x_min, bounds.y_min),
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            fill=False, edgecolor='black', linewidth=2, alpha=0.7
        )
        ax.add_patch(room_rect)
        
        # 绘制轨迹
        if len(sim_scene.frames) > 0:
            trajectory_x = []
            trajectory_y = []
            for frame in sim_scene.frames:
                if hasattr(frame, 'waypoint') and frame.waypoint is not None:
                    trajectory_x.append(frame.waypoint.position[0])
                    trajectory_y.append(frame.waypoint.position[1])
            
            if trajectory_x:
                ax.plot(trajectory_x, trajectory_y, 'r-', linewidth=3, 
                       label='机器人轨迹', alpha=0.8)
                ax.scatter(trajectory_x[0], trajectory_y[0], c='green', 
                          s=100, marker='o', label='起点', zorder=5)
                ax.scatter(trajectory_x[-1], trajectory_y[-1], c='red', 
                          s=100, marker='s', label='终点', zorder=5)
        
        # 绘制采样的帧点云
        if sampled_frames:
            for i, frame_idx in enumerate(sampled_frames):
                if frame_idx < len(sim_scene.frames):
                    frame = sim_scene.frames[frame_idx]
                    if len(frame.points) > 0:
                        # 只显示XY坐标
                        points_2d = frame.points[:, :2]
                        color = plt.cm.viridis(i / len(sampled_frames))
                        ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                                 c=[color], s=1, alpha=0.6, 
                                 label=f'帧 {frame_idx}' if i < 5 else "")
        
        # 设置图形属性
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_title('S3DIS BEV场景可视化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        # 保存图片
        output_path = self.vis_dir / "bev_scene_visualization.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 明确关闭图形
        
        print(f"[可视化] BEV场景可视化已保存: {output_path}")
        return str(output_path)
    
    def export_combined_pointcloud(self, sim_scene: S3DISSimScene, 
                                 format: str = "ply") -> str:
        """
        导出所有扫描点的组合点云
        
        Args:
            sim_scene: 仿真场景
            format: 导出格式 ("ply" 或 "pcd")
            
        Returns:
            保存的文件路径
        """
        print("[可视化] 导出组合点云...")
        
        # 收集所有点
        all_points = []
        all_colors = []
        
        for i, frame in enumerate(sim_scene.frames):
            if len(frame.points) > 0:
                all_points.append(frame.points)
                
                # 为每个帧使用不同颜色
                color = plt.cm.viridis(i / len(sim_scene.frames))[:3]
                frame_colors = np.tile(color, (len(frame.points), 1))
                all_colors.append(frame_colors)
        
        if not all_points:
            print("[警告] 没有找到有效的点云数据")
            return ""
        
        # 合并所有点
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # 创建点云
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # 保存点云
        if format.lower() == "ply":
            output_path = self.vis_dir / "combined_pointcloud.ply"
            o3d.io.write_point_cloud(str(output_path), combined_pcd)
        elif format.lower() == "pcd":
            output_path = self.vis_dir / "combined_pointcloud.pcd"
            o3d.io.write_point_cloud(str(output_path), combined_pcd)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"[可视化] 组合点云已保存: {output_path}")
        print(f"  总点数: {len(combined_points):,}")
        print(f"  点云范围:")
        print(f"    X: [{combined_points[:, 0].min():.2f}, {combined_points[:, 0].max():.2f}]")
        print(f"    Y: [{combined_points[:, 1].min():.2f}, {combined_points[:, 1].max():.2f}]")
        print(f"    Z: [{combined_points[:, 2].min():.2f}, {combined_points[:, 2].max():.2f}]")
        
        return str(output_path)
    
    def sample_frames(self, sim_scene: S3DISSimScene, num_samples: int = 5) -> List[int]:
        """
        随机采样帧进行可视化
        
        Args:
            sim_scene: 仿真场景
            num_samples: 采样数量
            
        Returns:
            采样的帧索引列表
        """
        if len(sim_scene.frames) == 0:
            return []
        
        # 确保采样数量不超过总帧数
        num_samples = min(num_samples, len(sim_scene.frames))
        
        # 随机采样
        sampled_indices = random.sample(range(len(sim_scene.frames)), num_samples)
        sampled_indices.sort()  # 按顺序排列
        
        print(f"[可视化] 随机采样了 {num_samples} 帧: {sampled_indices}")
        return sampled_indices
    
    def generate_all_visualizations(self, sim_scene: S3DISSimScene, 
                                   scene: S3DISScene,
                                   num_sample_frames: int = 5) -> Dict[str, str]:
        """
        生成所有可视化结果（简化版本，直接使用matplotlib）
        
        Args:
            sim_scene: 仿真场景
            scene: 原始场景
            num_sample_frames: 采样帧数量
            
        Returns:
            生成的文件路径字典
        """
        print("[可视化] 开始生成所有可视化结果...")
        
        results = {}
        
        # 采样帧
        sampled_frames = self.sample_frames(sim_scene, num_sample_frames)
        
        # 直接使用matplotlib生成3D可视化（避免Open3D问题）
        print("[可视化] 使用matplotlib生成3D可视化...")
        results['3d_visualization'] = self._create_matplotlib_3d_visualization(
            sim_scene, scene, sampled_frames, True, True
        )
        
        # 生成BEV可视化
        results['bev_visualization'] = self.visualize_bev_scene(
            sim_scene, scene, sampled_frames
        )
        
        # 导出组合点云
        results['combined_pointcloud'] = self.export_combined_pointcloud(sim_scene)
        
        print("[可视化] 所有可视化结果生成完成!")
        return results
    
    def _create_matplotlib_3d_visualization(self, sim_scene: S3DISSimScene, 
                                          scene: S3DISScene, 
                                          sampled_frames: Optional[List[int]] = None,
                                          show_trajectory: bool = True,
                                          show_mesh: bool = True) -> str:
        """使用matplotlib创建3D可视化（简化版本，参考LiT项目）"""
        print("[可视化] 使用matplotlib生成3D可视化...")
        
        # 设置matplotlib为非交互模式
        plt.ioff()
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 收集所有点云数据
        all_points = []
        all_colors = []
        
        # 添加采样的帧点云
        if sampled_frames:
            for i, frame_idx in enumerate(sampled_frames):
                if frame_idx < len(sim_scene.frames):
                    frame = sim_scene.frames[frame_idx]
                    if len(frame.points) > 0:
                        # 降采样以提高性能
                        if len(frame.points) > 5000:
                            indices = np.random.choice(len(frame.points), 5000, replace=False)
                            points = frame.points[indices]
                        else:
                            points = frame.points
                        
                        all_points.append(points)
                        
                        # 为每个帧使用不同颜色
                        color = plt.cm.viridis(i / len(sampled_frames))[:3]
                        frame_colors = np.tile(color, (len(points), 1))
                        all_colors.append(frame_colors)
        
        # 绘制点云
        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], 
                      s=1, c=combined_colors, alpha=0.6, label='扫描点云')
        
        # 添加轨迹
        if show_trajectory and len(sim_scene.frames) > 0:
            trajectory_x = []
            trajectory_y = []
            trajectory_z = []
            
            for frame in sim_scene.frames:
                if hasattr(frame, 'waypoint') and frame.waypoint is not None:
                    pos = frame.waypoint.position
                    trajectory_x.append(pos[0])
                    trajectory_y.append(pos[1])
                    trajectory_z.append(pos[2])
            
            if trajectory_x:
                ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2, 
                       label='机器人轨迹', alpha=0.8)
                ax.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0], 
                          c='green', s=50, marker='o', label='起点')
                ax.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], 
                          c='red', s=50, marker='s', label='终点')
        
        # 自动设置视角范围
        if all_points:
            all_xyz = np.vstack(all_points)
            x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
            y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
            z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()
            margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('S3DIS 3D场景可视化')
        ax.view_init(elev=20, azim=45)
        
        # 保存图片
        output_path = self.vis_dir / "3d_scene_visualization.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[可视化] 3D场景可视化已保存: {output_path}")
        return str(output_path)


def create_visualization_summary(results: Dict[str, str], output_dir: Path):
    """创建可视化结果摘要"""
    summary_path = output_dir / "visualization_summary.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# S3DIS仿真可视化结果摘要\n\n")
        f.write("## 生成的文件\n\n")
        
        for key, path in results.items():
            if path:
                f.write(f"- **{key}**: `{Path(path).name}`\n")
        
        f.write("\n## 文件说明\n\n")
        f.write("- **3D可视化**: 3D场景视图，包含mesh、轨迹和采样帧点云\n")
        f.write("- **BEV可视化**: 鸟瞰图，显示轨迹和点云分布\n")
        f.write("- **组合点云**: 所有扫描点的合并点云文件\n\n")
        
        f.write("## 使用方法\n\n")
        f.write("1. 使用Open3D查看PLY点云文件\n")
        f.write("2. 使用图片查看器查看PNG可视化图片\n")
        f.write("3. 使用点云处理软件进一步分析组合点云\n")
    
        print(f"[可视化] 结果摘要已保存: {summary_path}")
    
    def _create_matplotlib_3d_visualization(self, sim_scene: S3DISSimScene, 
                                          scene: S3DISScene, 
                                          sampled_frames: Optional[List[int]] = None,
                                          show_trajectory: bool = True,
                                          show_mesh: bool = True) -> str:
        """使用matplotlib创建3D可视化（简化版本，参考LiT项目）"""
        print("[可视化] 使用matplotlib生成3D可视化...")
        
        # 设置matplotlib为非交互模式
        plt.ioff()
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 收集所有点云数据
        all_points = []
        all_colors = []
        
        # 添加采样的帧点云
        if sampled_frames:
            for i, frame_idx in enumerate(sampled_frames):
                if frame_idx < len(sim_scene.frames):
                    frame = sim_scene.frames[frame_idx]
                    if len(frame.points) > 0:
                        # 降采样以提高性能
                        if len(frame.points) > 5000:
                            indices = np.random.choice(len(frame.points), 5000, replace=False)
                            points = frame.points[indices]
                        else:
                            points = frame.points
                        
                        all_points.append(points)
                        
                        # 为每个帧使用不同颜色
                        color = plt.cm.viridis(i / len(sampled_frames))[:3]
                        frame_colors = np.tile(color, (len(points), 1))
                        all_colors.append(frame_colors)
        
        # 绘制点云
        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], 
                      s=1, c=combined_colors, alpha=0.6, label='扫描点云')
        
        # 添加轨迹
        if show_trajectory and len(sim_scene.frames) > 0:
            trajectory_x = []
            trajectory_y = []
            trajectory_z = []
            
            for frame in sim_scene.frames:
                if hasattr(frame, 'waypoint') and frame.waypoint is not None:
                    pos = frame.waypoint.position
                    trajectory_x.append(pos[0])
                    trajectory_y.append(pos[1])
                    trajectory_z.append(pos[2])
            
            if trajectory_x:
                ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2, 
                       label='机器人轨迹', alpha=0.8)
                ax.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0], 
                          c='green', s=50, marker='o', label='起点')
                ax.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], 
                          c='red', s=50, marker='s', label='终点')
        
        # 自动设置视角范围
        if all_points:
            all_xyz = np.vstack(all_points)
            x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
            y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
            z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()
            margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('S3DIS 3D场景可视化')
        ax.view_init(elev=20, azim=45)
        
        # 保存图片
        output_path = self.vis_dir / "3d_scene_visualization.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[可视化] 3D场景可视化已保存: {output_path}")
        return str(output_path)
