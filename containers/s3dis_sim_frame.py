"""
S3DIS simulation frame data container.
Contains simulated point cloud, incident angles, scan quality, and other information.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScanQuality:
    """
    Scan quality metrics.
    Contains coverage ratio, point count, incident angle statistics, and other information.
    """
    coverage_ratio: float  # Coverage ratio
    num_points: int  # Number of points
    incident_angle_mean: float  # Mean incident angle
    incident_angle_std: float  # Incident angle standard deviation
    scan_density: float  # Scan density
    range_mean: float  # Mean range
    range_std: float  # Range standard deviation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'coverage_ratio': self.coverage_ratio,
            'num_points': self.num_points,
            'incident_angle_mean': self.incident_angle_mean,
            'incident_angle_std': self.incident_angle_std,
            'scan_density': self.scan_density,
            'range_mean': self.range_mean,
            'range_std': self.range_std
        }
    
    @classmethod
    def from_dict(cls, quality_dict: Dict[str, Any]) -> 'ScanQuality':
        """Create from dictionary."""
        return cls(**quality_dict)


@dataclass
class IncidentAngles:
    """
    Incident angle information.
    Contains incident angles, surface normals, and other information for each point.
    """
    angles: np.ndarray  # (N,) Incident angle array
    surface_normals: Optional[np.ndarray] = None  # (N, 3) Surface normals
    ray_directions: Optional[np.ndarray] = None  # (N, 3) Ray directions
    
    def get_mean_angle(self) -> float:
        """Get mean incident angle."""
        return np.mean(self.angles)
    
    def get_std_angle(self) -> float:
        """Get incident angle standard deviation."""
        return np.std(self.angles)
    
    def get_angle_distribution(self, num_bins: int = 20) -> tuple:
        """Get incident angle distribution."""
        hist, bins = np.histogram(self.angles, bins=num_bins)
        return hist, bins
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'angles': self.angles.tolist(),
            'surface_normals': self.surface_normals.tolist() if self.surface_normals is not None else None,
            'ray_directions': self.ray_directions.tolist() if self.ray_directions is not None else None
        }
    
    @classmethod
    def from_dict(cls, angles_dict: Dict[str, Any]) -> 'IncidentAngles':
        """Create from dictionary."""
        return cls(
            angles=np.array(angles_dict['angles']),
            surface_normals=np.array(angles_dict['surface_normals']) if angles_dict['surface_normals'] else None,
            ray_directions=np.array(angles_dict['ray_directions']) if angles_dict['ray_directions'] else None
        )


class S3DISSimFrame:
    """
    S3DIS simulation frame data container.
    Contains simulated point cloud, incident angles, scan quality, and other information.
    """
    
    def __init__(self, frame_index: int, points: np.ndarray, 
                 incident_angles: np.ndarray, scan_quality: ScanQuality,
                 frame_metadata: Optional[Dict[str, Any]] = None):
        self.frame_index = frame_index
        self.points = points  # (N, 3) Point cloud
        self.incident_angles = incident_angles  # (N,) Incident angles
        self.scan_quality = scan_quality
        self.frame_metadata = frame_metadata or {}
        
        # Validate data consistency
        if len(points) != len(incident_angles):
            raise ValueError(f"Point cloud count ({len(points)}) does not match incident angle count ({len(incident_angles)})")
    
    def get_num_points(self) -> int:
        """Get number of points."""
        return len(self.points)
    
    def get_coverage_ratio(self) -> float:
        """Get coverage ratio."""
        return self.scan_quality.coverage_ratio
    
    def get_scan_density(self) -> float:
        """Get scan density."""
        return self.scan_quality.scan_density
    
    def get_mean_incident_angle(self) -> float:
        """Get mean incident angle."""
        return self.scan_quality.incident_angle_mean
    
    def get_incident_angle_std(self) -> float:
        """Get incident angle standard deviation."""
        return self.scan_quality.incident_angle_std
    
    def get_mean_range(self) -> float:
        """Get mean range."""
        return self.scan_quality.range_mean
    
    def get_range_std(self) -> float:
        """Get range standard deviation."""
        return self.scan_quality.range_std
    
    def get_point_cloud_bounds(self) -> Dict[str, float]:
        """Get point cloud bounds."""
        if len(self.points) == 0:
            return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0}
        
        return {
            'x_min': float(self.points[:, 0].min()),
            'x_max': float(self.points[:, 0].max()),
            'y_min': float(self.points[:, 1].min()),
            'y_max': float(self.points[:, 1].max()),
            'z_min': float(self.points[:, 2].min()),
            'z_max': float(self.points[:, 2].max())
        }
    
    def get_point_cloud_center(self) -> np.ndarray:
        """Get point cloud center."""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.mean(self.points, axis=0)
    
    def get_point_cloud_std(self) -> np.ndarray:
        """Get point cloud standard deviation."""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.std(self.points, axis=0)
    
    def filter_points_by_angle(self, min_angle: float = 0, max_angle: float = np.pi/2) -> 'S3DISSimFrame':
        """Filter point cloud by incident angle."""
        mask = (self.incident_angles >= min_angle) & (self.incident_angles <= max_angle)
        filtered_points = self.points[mask]
        filtered_angles = self.incident_angles[mask]
        
        # Recalculate scan quality
        filtered_quality = ScanQuality(
            coverage_ratio=self.scan_quality.coverage_ratio * (len(filtered_points) / len(self.points)),
            num_points=len(filtered_points),
            incident_angle_mean=np.mean(filtered_angles) if len(filtered_angles) > 0 else 0,
            incident_angle_std=np.std(filtered_angles) if len(filtered_angles) > 0 else 0,
            scan_density=self.scan_quality.scan_density * (len(filtered_points) / len(self.points)),
            range_mean=np.mean(np.linalg.norm(filtered_points, axis=1)) if len(filtered_points) > 0 else 0,
            range_std=np.std(np.linalg.norm(filtered_points, axis=1)) if len(filtered_points) > 0 else 0
        )
        
        return S3DISSimFrame(
            frame_index=self.frame_index,
            points=filtered_points,
            incident_angles=filtered_angles,
            scan_quality=filtered_quality,
            frame_metadata=self.frame_metadata.copy()
        )
    
    def filter_points_by_range(self, min_range: float = 0, max_range: float = float('inf')) -> 'S3DISSimFrame':
        """Filter point cloud by range."""
        ranges = np.linalg.norm(self.points, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        filtered_points = self.points[mask]
        filtered_angles = self.incident_angles[mask]
        
        # Recalculate scan quality
        filtered_quality = ScanQuality(
            coverage_ratio=self.scan_quality.coverage_ratio * (len(filtered_points) / len(self.points)),
            num_points=len(filtered_points),
            incident_angle_mean=np.mean(filtered_angles) if len(filtered_angles) > 0 else 0,
            incident_angle_std=np.std(filtered_angles) if len(filtered_angles) > 0 else 0,
            scan_density=self.scan_quality.scan_density * (len(filtered_points) / len(self.points)),
            range_mean=np.mean(np.linalg.norm(filtered_points, axis=1)) if len(filtered_points) > 0 else 0,
            range_std=np.std(np.linalg.norm(filtered_points, axis=1)) if len(filtered_points) > 0 else 0
        )
        
        return S3DISSimFrame(
            frame_index=self.frame_index,
            points=filtered_points,
            incident_angles=filtered_angles,
            scan_quality=filtered_quality,
            frame_metadata=self.frame_metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame_index': self.frame_index,
            'points': self.points.tolist(),
            'incident_angles': self.incident_angles.tolist(),
            'scan_quality': self.scan_quality.to_dict(),
            'frame_metadata': self.frame_metadata
        }
    
    @classmethod
    def from_dict(cls, frame_dict: Dict[str, Any]) -> 'S3DISSimFrame':
        """Create from dictionary."""
        return cls(
            frame_index=frame_dict['frame_index'],
            points=np.array(frame_dict['points']),
            incident_angles=np.array(frame_dict['incident_angles']),
            scan_quality=ScanQuality.from_dict(frame_dict['scan_quality']),
            frame_metadata=frame_dict.get('frame_metadata', {})
        )
    
    def __repr__(self) -> str:
        return (f"S3DISSimFrame(index={self.frame_index}, "
                f"points={self.get_num_points()}, "
                f"coverage={self.get_coverage_ratio():.3f}, "
                f"mean_angle={self.get_mean_incident_angle():.3f})")
