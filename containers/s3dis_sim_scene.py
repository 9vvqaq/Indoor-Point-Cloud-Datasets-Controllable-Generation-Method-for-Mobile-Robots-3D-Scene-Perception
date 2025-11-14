"""
S3DIS simulation scene data container.
Contains all simulation frames, statistics, result export, and other functionality.
"""

import numpy as np
import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .s3dis_sim_frame import S3DISSimFrame, ScanQuality


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class SimulationStats:
    """
    Simulation statistics.
    Contains overall statistics, quality metrics, performance metrics, etc.
    """
    total_frames: int
    total_points: int
    average_coverage: float
    average_scan_density: float
    average_incident_angle: float
    average_range: float
    simulation_time: float
    frames_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_frames': self.total_frames,
            'total_points': self.total_points,
            'average_coverage': self.average_coverage,
            'average_scan_density': self.average_scan_density,
            'average_incident_angle': self.average_incident_angle,
            'average_range': self.average_range,
            'simulation_time': self.simulation_time,
            'frames_per_second': self.frames_per_second
        }


class ResultExporter:
    """
    Result exporter.
    Supports exporting simulation results in multiple formats.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_frames(self, frames: List[S3DISSimFrame], format: str = "pkl"):
        """Export all frame data."""
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for frame in frames:
            if format == "pkl":
                frame_path = frames_dir / f"frame_{frame.frame_index:04d}.pkl"
                with open(frame_path, 'wb') as f:
                    pickle.dump(frame.to_dict(), f)
            elif format == "json":
                frame_path = frames_dir / f"frame_{frame.frame_index:04d}.json"
                with open(frame_path, 'w') as f:
                    json.dump(frame.to_dict(), f, indent=2, cls=NumpyEncoder)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def export_statistics(self, stats: SimulationStats, format: str = "json"):
        """Export statistics."""
        if format == "json":
            stats_path = self.output_dir / "simulation_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2, cls=NumpyEncoder)
        elif format == "txt":
            stats_path = self.output_dir / "simulation_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Simulation Statistics\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Total frames: {stats.total_frames}\n")
                f.write(f"Total points: {stats.total_points}\n")
                f.write(f"Average coverage: {stats.average_coverage:.3f}\n")
                f.write(f"Average scan density: {stats.average_scan_density:.3f}\n")
                f.write(f"Average incident angle: {stats.average_incident_angle:.3f}\n")
                f.write(f"Average range: {stats.average_range:.3f}\n")
                f.write(f"Simulation time: {stats.simulation_time:.3f}s\n")
                f.write(f"Frames per second: {stats.frames_per_second:.3f} FPS\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_summary(self, sim_scene: 'S3DISSimScene', format: str = "json"):
        """Export simulation summary."""
        summary = {
            'scene_name': sim_scene.scene_name,
            'simulation_config': sim_scene.simulation_config,
            'statistics': sim_scene.statistics.to_dict(),
            'frame_summary': {
                'frame_indices': [frame.frame_index for frame in sim_scene.frames],
                'point_counts': [frame.get_num_points() for frame in sim_scene.frames],
                'coverage_ratios': [frame.get_coverage_ratio() for frame in sim_scene.frames]
            }
        }
        
        if format == "json":
            summary_path = self.output_dir / "simulation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
        else:
            raise ValueError(f"Unsupported format: {format}")


class S3DISSimScene:
    """
    S3DIS simulation scene data container.
    Contains all simulation frames, statistics, result export, and other functionality.
    """
    
    def __init__(self, scene_name: str, simulation_config: Optional[Dict[str, Any]] = None, 
                 mesh: Optional[object] = None, s3dis_data_root: Optional[str] = None,
                 area: Optional[str] = None, room: Optional[str] = None):
        self.scene_name = scene_name
        self.simulation_config = simulation_config or {}
        self.frames: List[S3DISSimFrame] = []
        self.statistics: Optional[SimulationStats] = None
        self.exporter: Optional[ResultExporter] = None
        self.mesh = mesh  # Store mesh reference for color assignment
        self.s3dis_data_root = s3dis_data_root  # S3DIS data root directory
        self.area = area  # Area name
        self.room = room  # Room name
        
        # S3DIS data cache
        self._s3dis_cache = None
    
    def append_frame(self, frame: S3DISSimFrame):
        """Add simulation frame."""
        self.frames.append(frame)
    
    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return len(self.frames)
    
    def get_total_points(self) -> int:
        """Get total number of points."""
        return sum(frame.get_num_points() for frame in self.frames)
    
    def get_average_coverage(self) -> float:
        """Get average coverage ratio."""
        if not self.frames:
            return 0.0
        return np.mean([frame.get_coverage_ratio() for frame in self.frames])
    
    def get_average_scan_density(self) -> float:
        """Get average scan density."""
        if not self.frames:
            return 0.0
        return np.mean([frame.get_scan_density() for frame in self.frames])
    
    def get_average_incident_angle(self) -> float:
        """Get average incident angle."""
        if not self.frames:
            return 0.0
        return np.mean([frame.get_mean_incident_angle() for frame in self.frames])
    
    def get_average_range(self) -> float:
        """Get average range."""
        if not self.frames:
            return 0.0
        return np.mean([frame.get_mean_range() for frame in self.frames])
    
    def get_frame_statistics(self) -> Dict[str, List[float]]:
        """Get per-frame statistics."""
        if not self.frames:
            return {}
        
        return {
            'frame_indices': [frame.frame_index for frame in self.frames],
            'point_counts': [frame.get_num_points() for frame in self.frames],
            'coverage_ratios': [frame.get_coverage_ratio() for frame in self.frames],
            'scan_densities': [frame.get_scan_density() for frame in self.frames],
            'incident_angles': [frame.get_mean_incident_angle() for frame in self.frames],
            'ranges': [frame.get_mean_range() for frame in self.frames]
        }
    
    def get_quality_distribution(self) -> Dict[str, Any]:
        """Get quality distribution statistics."""
        if not self.frames:
            return {}
        
        frame_stats = self.get_frame_statistics()
        
        return {
            'coverage_distribution': {
                'mean': np.mean(frame_stats['coverage_ratios']),
                'std': np.std(frame_stats['coverage_ratios']),
                'min': np.min(frame_stats['coverage_ratios']),
                'max': np.max(frame_stats['coverage_ratios'])
            },
            'point_count_distribution': {
                'mean': np.mean(frame_stats['point_counts']),
                'std': np.std(frame_stats['point_counts']),
                'min': np.min(frame_stats['point_counts']),
                'max': np.max(frame_stats['point_counts'])
            },
            'incident_angle_distribution': {
                'mean': np.mean(frame_stats['incident_angles']),
                'std': np.std(frame_stats['incident_angles']),
                'min': np.min(frame_stats['incident_angles']),
                'max': np.max(frame_stats['incident_angles'])
            }
        }
    
    def compute_statistics(self, simulation_time: float = 0.0):
        """Compute simulation statistics."""
        if not self.frames:
            self.statistics = SimulationStats(
                total_frames=0, total_points=0, average_coverage=0.0,
                average_scan_density=0.0, average_incident_angle=0.0,
                average_range=0.0, simulation_time=0.0, frames_per_second=0.0
            )
            return
        
        self.statistics = SimulationStats(
            total_frames=self.get_total_frames(),
            total_points=self.get_total_points(),
            average_coverage=self.get_average_coverage(),
            average_scan_density=self.get_average_scan_density(),
            average_incident_angle=self.get_average_incident_angle(),
            average_range=self.get_average_range(),
            simulation_time=simulation_time,
            frames_per_second=self.get_total_frames() / simulation_time if simulation_time > 0 else 0.0
        )
    
    def save_results(self, output_dir: Path, formats: List[str] = ["pkl", "txt"]):
        """Save simulation results (default uses pkl and txt formats to avoid JSON issues)."""
        self.exporter = ResultExporter(output_dir)
        
        # Compute statistics
        self.compute_statistics()
        
        # Export frame data
        # for format in formats:
        #     if format in ["pkl", "json"]:
        #         self.exporter.export_frames(self.frames, format)
        
        # Export statistics
        for format in formats:
            if format in ["json", "txt"]:
                self.exporter.export_statistics(self.statistics, format)
        
        # Export summary
        if "json" in formats:
            self.exporter.export_summary(self, "json")
        elif "txt" in formats:
            self._save_simple_summary(output_dir)
        
        # Export combined point cloud (original format)
        self._export_combined_pointcloud(output_dir)
        
        # Export combined point cloud with labels (new format)
        self._export_combined_pointcloud_with_labels(output_dir)
    
    def _save_simple_summary(self, output_dir: Path):
        """Save simplified summary information (txt format)."""
        summary_path = output_dir / "simulation_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("S3DIS Simulation Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scene name: {self.scene_name}\n")
            f.write(f"Total frames: {len(self.frames)}\n")
            f.write(f"Total points: {self.get_total_points():,}\n")
            f.write(f"Average coverage: {self.get_average_coverage():.3f}\n")
            f.write(f"Average scan density: {self.get_average_scan_density():.3f}\n")
            f.write(f"Average incident angle: {self.get_average_incident_angle():.1f}°\n")
            f.write(f"Average range: {self.get_average_range():.2f}m\n")
            
            if self.statistics:
                f.write(f"\nSimulation Statistics:\n")
                f.write(f"  Simulation time: {self.statistics.simulation_time:.2f}s\n")
                f.write(f"  Frame rate: {self.statistics.frames_per_second:.1f} FPS\n")
            
            f.write(f"\nFrame Details:\n")
            f.write("-" * 30 + "\n")
            for i, frame in enumerate(self.frames):
                f.write(f"Frame {i+1:2d}: {frame.get_num_points():5d} points, "
                       f"coverage {frame.get_coverage_ratio():.3f}, "
                       f"density {frame.get_scan_density():.3f}\n")
    
    def _export_combined_pointcloud(self, output_dir: Path):
        """Export combined point cloud (original format)."""
        # Collect all point cloud data
        all_points = []
        all_colors = []
        
        for i, frame in enumerate(self.frames):
            if len(frame.points) > 0:
                all_points.append(frame.points)
                
                # Use different colors for each frame (to distinguish frames)
                import matplotlib.pyplot as plt
                color = plt.cm.viridis(i / len(self.frames))[:3]
                frame_colors = np.tile(color, (len(frame.points), 1))
                all_colors.append(frame_colors)
        
        if not all_points:
            return
        
        # Merge all point clouds
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Create Open3D point cloud
        import open3d as o3d
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Save as PLY file
        output_path = output_dir / "combined_pointcloud.ply"
        o3d.io.write_point_cloud(str(output_path), combined_pcd)
    
    def _export_combined_pointcloud_with_labels(self, output_dir: Path):
        """Export combined point cloud with labels (new format, contains 8 attributes)."""
        # Collect all point cloud data
        all_points = []
        all_colors = []
        all_semantic_labels = []
        all_instance_labels = []
        
        for frame in self.frames:
            if len(frame.points) > 0:
                all_points.append(frame.points)
                
                # Get colors and labels from S3DIS original data
                colors, semantic_labels, instance_labels = self._get_colors_and_labels_from_s3dis(frame.points)
                
                all_colors.append(colors)
                all_semantic_labels.append(semantic_labels)
                all_instance_labels.append(instance_labels)
        
        if not all_points:
            return
        
        # Merge all data
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        combined_semantic = np.concatenate(all_semantic_labels)
        combined_instance = np.concatenate(all_instance_labels)
        
        # Convert colors to 0-255 range
        colors_255 = (combined_colors * 255).astype(np.uint8)
        
        # Create labeled point cloud data
        self._save_labeled_ply(
            output_dir / "combined_pointcloud_with_label.ply",
            combined_points,
            colors_255,
            combined_semantic,
            combined_instance
        )
    
    def _get_colors_and_labels_from_s3dis(self, points: np.ndarray) -> tuple:
        """
        Get colors and labels from S3DIS original data (simplified approach).
        
        Args:
            points: Laser points [N, 3]
            
        Returns:
            (colors, semantic_labels, instance_labels): Colors and labels
        """
        # Check if S3DIS data path information is available
        if not self.s3dis_data_root or not self.area or not self.room:
            return self._get_default_colors_and_labels(len(points))
        
        # Check cache
        if self._s3dis_cache is None:
            try:
                # 1. Load S3DIS annotation data (contains point cloud, colors, labels)
                s3dis_points, s3dis_colors, s3dis_labels, s3dis_instances = self._load_s3dis_annotations_with_colors()
                
                if s3dis_points is None or len(s3dis_points) == 0:
                    return self._get_default_colors_and_labels(len(points))
                
                # Cache data
                self._s3dis_cache = {
                    'points': s3dis_points,
                    'colors': s3dis_colors,
                    'labels': s3dis_labels,
                    'instances': s3dis_instances
                }
                
            except Exception as e:
                return self._get_default_colors_and_labels(len(points))
        
        try:
            # 2. Nearest neighbor matching
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self._s3dis_cache['points'])
            distances, indices = nbrs.kneighbors(points)
            
            # 3. Assign RGB, labels, instances
            colors = self._s3dis_cache['colors'][indices.flatten()]
            semantic_labels = self._s3dis_cache['labels'][indices.flatten()]
            instance_labels = self._s3dis_cache['instances'][indices.flatten()]
            
            return colors, semantic_labels, instance_labels
            
        except Exception as e:
            return self._get_default_colors_and_labels(len(points))
    
    def _load_s3dis_original_data(self) -> tuple:
        """Load S3DIS original point cloud data (RGB)."""
        if not self.s3dis_data_root or not self.area or not self.room:
            return None, None
        
        try:
            import open3d as o3d
            
            # Build S3DIS original point cloud file path
            # According to data structure, point cloud file is in {room}.txt format
            pointcloud_path = f"{self.s3dis_data_root}/{self.area}/{self.room}/{self.room}.txt"
            
            # Try to load point cloud file
            if not os.path.exists(pointcloud_path):
                # If standard path doesn't exist, try other possible paths
                alternative_paths = [
                    f"{self.s3dis_data_root}/{self.area}/{self.room}/pointcloud.ply",
                    f"{self.s3dis_data_root}/{self.area}/{self.room}/Area_{self.area}_{self.room}.ply",
                    f"{self.s3dis_data_root}/{self.area}/{self.room}/Area_{self.area}_{self.room}_inst_nostring.ply",
                    f"{self.s3dis_data_root}/{self.area}/{self.room}/Area_{self.area}_{self.room}_inst_nostring.txt"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        pointcloud_path = alt_path
                        break
                else:
                    return None, None
            
            # Load point cloud
            if pointcloud_path.endswith('.txt'):
                # Load S3DIS .txt format point cloud file
                points, colors = self._load_s3dis_txt_pointcloud(pointcloud_path)
            else:
                # Load PLY format point cloud file
                pcd = o3d.io.read_point_cloud(pointcloud_path)
                if len(pcd.points) == 0:
                    return None, None
                
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
            if points is None or len(points) == 0:
                return None, None
            
            if colors is None:
                # Use default colors
                colors = np.ones((len(points), 3), dtype=np.float32) * 0.5
            
            return points, colors
            
        except Exception as e:
            return None, None
    
    def _load_s3dis_txt_pointcloud(self, file_path: str) -> tuple:
        """Load S3DIS TXT format point cloud file."""
        try:
            # S3DIS TXT format is usually: x y z r g b or x y z r g b label
            data = np.loadtxt(file_path)
            
            if data.shape[1] < 6:
                return None, None
            
            # Extract coordinates and colors
            points = data[:, :3]  # x, y, z
            colors = data[:, 3:6]  # r, g, b
            
            # Ensure color values are in [0,1] range
            if colors.max() > 1.0:
                colors = colors / 255.0
            
            return points, colors
            
        except Exception as e:
            return None, None
    
    def _load_s3dis_annotations_with_colors(self) -> tuple:
        """Parse labels and instances from annotation folder, and get colors at the same time."""
        if not self.s3dis_data_root or not self.area or not self.room:
            return None, None, None, None
        
        try:
            # Use previously written annotation loader
            from s3dis_annotation_loader import S3DISAnnotationLoader
            
            # Create annotation loader
            loader = S3DISAnnotationLoader(self.s3dis_data_root)
            
            # Load room annotations
            room_annotations = loader.load_room_annotations(self.area, self.room)
            
            if not room_annotations:
                return None, None, None, None
            
            # Create labeled point cloud
            points, semantic_labels, instance_labels = loader.create_labeled_pointcloud_with_instances(room_annotations)
            
            if len(points) == 0:
                return None, None, None, None
            
            # Get colors from original point cloud
            original_points, original_colors = self._load_s3dis_original_data()
            
            if original_points is None or original_colors is None:
                colors = np.ones((len(points), 3), dtype=np.float32) * 0.5
            else:
                # Get colors through nearest neighbor matching
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(original_points)
                distances, indices = nbrs.kneighbors(points)
                colors = original_colors[indices.flatten()]
            
            return points, colors, semantic_labels, instance_labels
            
        except Exception as e:
            return None, None, None, None
    
    def _load_s3dis_annotations(self) -> tuple:
        """Parse labels and instances from annotation folder."""
        if not self.s3dis_data_root or not self.area or not self.room:
            return None, None
        
        try:
            # Use previously written annotation loader
            from s3dis_annotation_loader import S3DISAnnotationLoader
            
            # Create annotation loader
            loader = S3DISAnnotationLoader(self.s3dis_data_root)
            
            # Load room annotations
            room_annotations = loader.load_room_annotations(self.area, self.room)
            
            if not room_annotations:
                return None, None
            
            # Create labeled point cloud
            points, semantic_labels, instance_labels = loader.create_labeled_pointcloud_with_instances(room_annotations)
            
            if len(points) == 0:
                return None, None
            
            return semantic_labels, instance_labels
            
        except Exception as e:
            return None, None
    
    def _get_default_colors_and_labels(self, num_points: int) -> tuple:
        """Get default colors and labels."""
        # Generate gradient colors based on spatial position
        colors = np.ones((num_points, 3), dtype=np.float32) * 0.5
        
        # Default labels: all points are unlabeled (label 0)
        semantic_labels = np.zeros(num_points, dtype=np.uint16)
        instance_labels = np.zeros(num_points, dtype=np.uint16)
        
        return colors, semantic_labels, instance_labels
    
    def _decode_colors_to_labels(self, colors: np.ndarray) -> tuple:
        """
        Decode colors to semantic labels and instance labels.
        
        Args:
            colors: Color array [N, 3] (RGB, 0-1 range)
            
        Returns:
            (semantic_labels, instance_labels): Semantic labels and instance labels
        """
        try:
            # Import color decoder
            from s3dis_annotation_loader import S3DISColorEncoder
            
            # Create decoder
            color_encoder = S3DISColorEncoder()
            
            # Decode colors
            semantic_labels, instance_labels = color_encoder.decode_colors_to_labels_and_instances(colors)
            
            return semantic_labels, instance_labels
            
        except Exception as e:
            # Return default labels
            semantic_labels = np.zeros(len(colors), dtype=np.uint16)
            instance_labels = np.zeros(len(colors), dtype=np.uint16)
            return semantic_labels, instance_labels
    
    def _save_labeled_ply(self, output_path: Path, points: np.ndarray, colors: np.ndarray, 
                         semantic_labels: np.ndarray, instance_labels: np.ndarray):
        """Save labeled PLY file (8 attribute format)."""
        import struct
        
        with open(output_path, 'wb') as f:
            # PLY file header
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(b'element vertex %d\n' % len(points))
            f.write(b'property float x\n')
            f.write(b'property float y\n')
            f.write(b'property float z\n')
            f.write(b'property uchar red\n')
            f.write(b'property uchar green\n')
            f.write(b'property uchar blue\n')
            f.write(b'property ushort sem\n')
            f.write(b'property ushort ins\n')
            f.write(b'end_header\n')
            
            # Write data
            for i in range(len(points)):
                # x, y, z (float32)
                f.write(struct.pack('<fff', points[i, 0], points[i, 1], points[i, 2]))
                # red, green, blue (uint8)
                f.write(struct.pack('<BBB', colors[i, 0], colors[i, 1], colors[i, 2]))
                # sem, ins (uint16)
                f.write(struct.pack('<HH', semantic_labels[i], instance_labels[i]))
    
    def filter_frames_by_quality(self, min_coverage: float = 0.0, 
                                max_coverage: float = 1.0) -> 'S3DISSimScene':
        """Filter frames by quality."""
        filtered_frames = []
        for frame in self.frames:
            if min_coverage <= frame.get_coverage_ratio() <= max_coverage:
                filtered_frames.append(frame)
        
        filtered_scene = S3DISSimScene(self.scene_name, self.simulation_config)
        filtered_scene.frames = filtered_frames
        return filtered_scene
    
    def get_best_frames(self, num_frames: int = 10, 
                       quality_metric: str = "coverage") -> List[S3DISSimFrame]:
        """Get best quality frames."""
        if quality_metric == "coverage":
            sorted_frames = sorted(self.frames, key=lambda f: f.get_coverage_ratio(), reverse=True)
        elif quality_metric == "points":
            sorted_frames = sorted(self.frames, key=lambda f: f.get_num_points(), reverse=True)
        elif quality_metric == "density":
            sorted_frames = sorted(self.frames, key=lambda f: f.get_scan_density(), reverse=True)
        else:
            raise ValueError(f"Unsupported quality metric: {quality_metric}")
        
        return sorted_frames[:num_frames]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scene_name': self.scene_name,
            'simulation_config': self.simulation_config,
            'frames': [frame.to_dict() for frame in self.frames],
            'statistics': self.statistics.to_dict() if self.statistics else None
        }
    
    @classmethod
    def from_dict(cls, scene_dict: Dict[str, Any]) -> 'S3DISSimScene':
        """Create from dictionary."""
        sim_scene = cls(
            scene_name=scene_dict['scene_name'],
            simulation_config=scene_dict.get('simulation_config', {})
        )
        
        for frame_dict in scene_dict['frames']:
            frame = S3DISSimFrame.from_dict(frame_dict)
            sim_scene.append_frame(frame)
        
        if scene_dict.get('statistics'):
            sim_scene.statistics = SimulationStats(**scene_dict['statistics'])
        
        return sim_scene
    
    def __repr__(self) -> str:
        return (f"S3DISSimScene(name='{self.scene_name}', "
                f"frames={self.get_total_frames()}, "
                f"points={self.get_total_points()}, "
                f"avg_coverage={self.get_average_coverage():.3f})")
