"""
扫描结果可视化器
提供仿真扫描结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    from ..containers.s3dis_sim_frame import S3DISSimFrame
    from ..containers.s3dis_sim_scene import S3DISSimScene
except ImportError:
    from containers.s3dis_sim_frame import S3DISSimFrame
    from containers.s3dis_sim_scene import S3DISSimScene


class ScanResultVisualizer:
    """
    扫描结果可视化器
    提供扫描结果的各种可视化方式
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("visualization_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_scan_statistics(self, sim_scene: S3DISSimScene,
                                 save_path: Optional[Path] = None) -> Path:
        """
        扫描统计信息可视化
        
        Args:
            sim_scene: 仿真场景
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取统计数据
        frame_stats = sim_scene.get_frame_statistics()
        
        # 1. 每帧点数
        ax1.plot(frame_stats['frame_indices'], frame_stats['point_counts'], 
                'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('帧索引')
        ax1.set_ylabel('点数')
        ax1.set_title('每帧扫描点数')
        ax1.grid(True, alpha=0.3)
        
        # 2. 覆盖率
        ax2.plot(frame_stats['frame_indices'], frame_stats['coverage_ratios'], 
                'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('帧索引')
        ax2.set_ylabel('覆盖率')
        ax2.set_title('每帧扫描覆盖率')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. 点数分布直方图
        ax3.hist(frame_stats['point_counts'], bins=20, alpha=0.7, 
                color='blue', edgecolor='black')
        ax3.set_xlabel('点数')
        ax3.set_ylabel('频次')
        ax3.set_title('点数分布直方图')
        ax3.grid(True, alpha=0.3)
        
        # 4. 覆盖率分布直方图
        ax4.hist(frame_stats['coverage_ratios'], bins=20, alpha=0.7, 
                color='green', edgecolor='black')
        ax4.set_xlabel('覆盖率')
        ax4.set_ylabel('频次')
        ax4.set_title('覆盖率分布直方图')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "scan_statistics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_pointcloud_3d(self, sim_frame: S3DISSimFrame,
                               max_points: int = 10000,
                               save_path: Optional[Path] = None) -> Path:
        """
        3D点云可视化
        
        Args:
            sim_frame: 仿真帧
            max_points: 最大显示点数
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = sim_frame.points
        incident_angles = sim_frame.incident_angles
        
        # 降采样
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            incident_angles = incident_angles[indices]
        
        # 根据入射角生成颜色
        if len(incident_angles) > 0:
            norm_angles = (incident_angles - incident_angles.min()) / (incident_angles.ptp() + 1e-6)
            colors = plt.cm.jet(norm_angles)
        else:
            colors = 'blue'
        
        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'点云可视化 (帧 {sim_frame.frame_index}, 点数: {len(points)})')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / f"pointcloud_frame_{sim_frame.frame_index:04d}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_pointcloud_bev(self, sim_frame: S3DISSimFrame,
                                max_points: int = 10000,
                                save_path: Optional[Path] = None) -> Path:
        """
        点云鸟瞰图可视化
        
        Args:
            sim_frame: 仿真帧
            max_points: 最大显示点数
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        points = sim_frame.points
        incident_angles = sim_frame.incident_angles
        
        # 降采样
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            incident_angles = incident_angles[indices]
        
        # 根据入射角生成颜色
        if len(incident_angles) > 0:
            norm_angles = (incident_angles - incident_angles.min()) / (incident_angles.ptp() + 1e-6)
            colors = plt.cm.jet(norm_angles)
        else:
            colors = 'blue'
        
        # 绘制鸟瞰图
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.5, alpha=0.6)
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'点云鸟瞰图 (帧 {sim_frame.frame_index}, 点数: {len(points)})')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / f"pointcloud_bev_frame_{sim_frame.frame_index:04d}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_incident_angles(self, sim_frame: S3DISSimFrame,
                                 save_path: Optional[Path] = None) -> Path:
        """
        入射角分布可视化
        
        Args:
            sim_frame: 仿真帧
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        incident_angles = sim_frame.incident_angles
        
        # 1. 入射角分布直方图
        ax1.hist(incident_angles, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('入射角 (弧度)')
        ax1.set_ylabel('频次')
        ax1.set_title('入射角分布直方图')
        ax1.grid(True, alpha=0.3)
        
        # 2. 入射角统计信息
        stats_text = f"""
        平均入射角: {np.mean(incident_angles):.3f} 弧度
        标准差: {np.std(incident_angles):.3f} 弧度
        最小值: {np.min(incident_angles):.3f} 弧度
        最大值: {np.max(incident_angles):.3f} 弧度
        中位数: {np.median(incident_angles):.3f} 弧度
        """
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('入射角统计信息')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / f"incident_angles_frame_{sim_frame.frame_index:04d}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_scan_quality_evolution(self, sim_scene: S3DISSimScene,
                                        save_path: Optional[Path] = None) -> Path:
        """
        扫描质量演化可视化
        
        Args:
            sim_scene: 仿真场景
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        frame_stats = sim_scene.get_frame_statistics()
        
        # 1. 扫描密度演化
        ax1.plot(frame_stats['frame_indices'], frame_stats['scan_densities'], 
                'purple', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('帧索引')
        ax1.set_ylabel('扫描密度')
        ax1.set_title('扫描密度演化')
        ax1.grid(True, alpha=0.3)
        
        # 2. 入射角演化
        ax2.plot(frame_stats['frame_indices'], frame_stats['incident_angles'], 
                'orange', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('帧索引')
        ax2.set_ylabel('平均入射角 (弧度)')
        ax2.set_title('平均入射角演化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 距离演化
        ax3.plot(frame_stats['frame_indices'], frame_stats['ranges'], 
                'red', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('帧索引')
        ax3.set_ylabel('平均距离 (m)')
        ax3.set_title('平均距离演化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 质量指标雷达图
        quality_dist = sim_scene.get_quality_distribution()
        
        metrics = ['coverage', 'point_count', 'incident_angle']
        values = [
            quality_dist['coverage_distribution']['mean'],
            quality_dist['point_count_distribution']['mean'] / 1000,  # 归一化
            quality_dist['incident_angle_distribution']['mean'] / np.pi  # 归一化
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax4.fill(angles, values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('质量指标雷达图')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "scan_quality_evolution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_scan_summary_report(self, sim_scene: S3DISSimScene,
                                  save_path: Optional[Path] = None) -> Path:
        """
        创建扫描总结报告
        
        Args:
            sim_scene: 仿真场景
            save_path: 保存路径
        
        Returns:
            save_path: 保存的文件路径
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 总体统计信息
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        stats_text = f"""
        S3DIS扫描仿真总结报告
        =====================
        
        场景名称: {sim_scene.scene_name}
        总帧数: {sim_scene.get_total_frames()}
        总点数: {sim_scene.get_total_points():,}
        平均覆盖率: {sim_scene.get_average_coverage():.3f}
        平均扫描密度: {sim_scene.get_average_scan_density():.3f}
        平均入射角: {sim_scene.get_average_incident_angle():.3f} 弧度
        平均距离: {sim_scene.get_average_range():.3f} 米
        """
        
        ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. 点数演化
        frame_stats = sim_scene.get_frame_statistics()
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(frame_stats['frame_indices'], frame_stats['point_counts'], 'b-', linewidth=2)
        ax2.set_xlabel('帧索引')
        ax2.set_ylabel('点数')
        ax2.set_title('点数演化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 覆盖率演化
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(frame_stats['frame_indices'], frame_stats['coverage_ratios'], 'g-', linewidth=2)
        ax3.set_xlabel('帧索引')
        ax3.set_ylabel('覆盖率')
        ax3.set_title('覆盖率演化')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. 入射角演化
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(frame_stats['frame_indices'], frame_stats['incident_angles'], 'r-', linewidth=2)
        ax4.set_xlabel('帧索引')
        ax4.set_ylabel('入射角 (弧度)')
        ax4.set_title('入射角演化')
        ax4.grid(True, alpha=0.3)
        
        # 5. 点数分布
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(frame_stats['point_counts'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax5.set_xlabel('点数')
        ax5.set_ylabel('频次')
        ax5.set_title('点数分布')
        ax5.grid(True, alpha=0.3)
        
        # 6. 覆盖率分布
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(frame_stats['coverage_ratios'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax6.set_xlabel('覆盖率')
        ax6.set_ylabel('频次')
        ax6.set_title('覆盖率分布')
        ax6.grid(True, alpha=0.3)
        
        # 7. 质量指标
        ax7 = fig.add_subplot(gs[2, 2])
        quality_dist = sim_scene.get_quality_distribution()
        
        metrics = ['覆盖率', '点数', '入射角']
        means = [
            quality_dist['coverage_distribution']['mean'],
            quality_dist['point_count_distribution']['mean'] / 1000,
            quality_dist['incident_angle_distribution']['mean'] / np.pi
        ]
        
        ax7.bar(metrics, means, color=['blue', 'green', 'red'], alpha=0.7)
        ax7.set_ylabel('归一化值')
        ax7.set_title('质量指标')
        ax7.tick_params(axis='x', rotation=45)
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "scan_summary_report.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
