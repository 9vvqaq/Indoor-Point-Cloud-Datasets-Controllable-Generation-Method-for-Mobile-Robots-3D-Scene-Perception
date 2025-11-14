"""
Trajectory generator base class.
Defines basic interfaces and data structures for trajectory generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Waypoint:
    """
    Waypoint data structure.
    Contains position, orientation, timestamp, and other information.
    """
    x: float
    y: float
    z: float
    yaw: float  # Yaw angle (radians)
    timestamp: float = 0.0
    velocity: Optional[float] = None  # Linear velocity
    angular_velocity: Optional[float] = None  # Angular velocity
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, yaw]."""
        return np.array([self.x, self.y, self.z, self.yaw])
    
    def to_pose_matrix(self) -> np.ndarray:
        """Convert to 4x4 pose matrix."""
        matrix = np.eye(4)
        matrix[0, 3] = self.x
        matrix[1, 3] = self.y
        matrix[2, 3] = self.z
        
        # Rotation matrix (rotate around Z-axis by yaw angle)
        cos_yaw, sin_yaw = np.cos(self.yaw), np.sin(self.yaw)
        matrix[0, 0] = cos_yaw
        matrix[0, 1] = -sin_yaw
        matrix[1, 0] = sin_yaw
        matrix[1, 1] = cos_yaw
        
        return matrix
    
    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate distance to another waypoint."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def angle_to(self, other: 'Waypoint') -> float:
        """Calculate angle to another waypoint."""
        dx = other.x - self.x
        dy = other.y - self.y
        return np.arctan2(dy, dx)
    
    def __repr__(self) -> str:
        return f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, yaw={self.yaw:.2f})"


@dataclass
class TrajectoryQuality:
    """
    Trajectory quality assessment results.
    """
    coverage_ratio: float  # Coverage ratio
    path_length: float  # Path length
    turn_count: int  # Turn count
    efficiency: float  # Scanning efficiency
    collision_count: int  # Collision count
    smoothness: float  # Smoothness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'coverage_ratio': self.coverage_ratio,
            'path_length': self.path_length,
            'turn_count': self.turn_count,
            'efficiency': self.efficiency,
            'collision_count': self.collision_count,
            'smoothness': self.smoothness
        }


class TrajectoryGeneratorBase(ABC):
    """
    Trajectory generator base class.
    Defines basic interfaces for trajectory generation.
    """
    
    def __init__(self, room_bounds: Dict[str, float], robot_height: float = 1.0):
        self.room_bounds = room_bounds
        self.robot_height = robot_height
        self.robot_radius = 0.3  # Robot radius (meters)
    
    @abstractmethod
    def generate_trajectory(self, **kwargs) -> Tuple[List[Waypoint], TrajectoryQuality]:
        """
        Generate trajectory.
        
        Returns:
            waypoints: List of waypoints
            quality: Trajectory quality assessment
        """
        pass
    
    def waypoints_to_poses(self, waypoints: List[Waypoint]) -> List[np.ndarray]:
        """
        Convert waypoints to pose matrices.
        
        Args:
            waypoints: List of waypoints
        
        Returns:
            poses: List of pose matrices
        """
        return [waypoint.to_pose_matrix() for waypoint in waypoints]
    
    def calculate_path_length(self, waypoints: List[Waypoint]) -> float:
        """Calculate total path length."""
        if len(waypoints) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(waypoints)):
            total_length += waypoints[i].distance_to(waypoints[i-1])
        
        return total_length
    
    def count_turns(self, waypoints: List[Waypoint], angle_threshold: float = 0.1) -> int:
        """Calculate turn count."""
        if len(waypoints) < 3:
            return 0
        
        turn_count = 0
        for i in range(1, len(waypoints) - 1):
            # Calculate angle change
            angle_change = abs(waypoints[i+1].yaw - waypoints[i].yaw)
            # Handle angle wrap-around
            if angle_change > np.pi:
                angle_change = 2 * np.pi - angle_change
            
            if angle_change > angle_threshold:
                turn_count += 1
        
        return turn_count
    
    def calculate_smoothness(self, waypoints: List[Waypoint]) -> float:
        """Calculate trajectory smoothness."""
        if len(waypoints) < 3:
            return 1.0
        
        angle_changes = []
        for i in range(1, len(waypoints) - 1):
            angle_change = abs(waypoints[i+1].yaw - waypoints[i].yaw)
            if angle_change > np.pi:
                angle_change = 2 * np.pi - angle_change
            angle_changes.append(angle_change)
        
        # Smoothness = 1 / (1 + standard deviation of angle changes)
        if not angle_changes:
            return 1.0
        
        smoothness = 1.0 / (1.0 + np.std(angle_changes))
        return smoothness
    
    def is_point_in_room(self, waypoint: Waypoint) -> bool:
        """Check if point is within room."""
        return (self.room_bounds['x_min'] <= waypoint.x <= self.room_bounds['x_max'] and
                self.room_bounds['y_min'] <= waypoint.y <= self.room_bounds['y_max'] and
                self.room_bounds['z_min'] <= waypoint.z <= self.room_bounds['z_max'])
    
    def clip_to_room_bounds(self, waypoint: Waypoint) -> Waypoint:
        """Clip waypoint to room boundaries."""
        clipped_x = np.clip(waypoint.x, self.room_bounds['x_min'], self.room_bounds['x_max'])
        clipped_y = np.clip(waypoint.y, self.room_bounds['y_min'], self.room_bounds['y_max'])
        clipped_z = np.clip(waypoint.z, self.room_bounds['z_min'], self.room_bounds['z_max'])
        
        return Waypoint(
            x=clipped_x, y=clipped_y, z=clipped_z,
            yaw=waypoint.yaw, timestamp=waypoint.timestamp,
            velocity=waypoint.velocity, angular_velocity=waypoint.angular_velocity
        )
    
    def evaluate_trajectory_quality(self, waypoints: List[Waypoint], 
                                  collision_count: int = 0) -> TrajectoryQuality:
        """Evaluate trajectory quality."""
        path_length = self.calculate_path_length(waypoints)
        turn_count = self.count_turns(waypoints)
        smoothness = self.calculate_smoothness(waypoints)
        
        # Calculate coverage ratio (simplified version)
        coverage_ratio = self._calculate_coverage_ratio(waypoints)
        
        # Calculate efficiency
        efficiency = coverage_ratio / path_length if path_length > 0 else 0
        
        return TrajectoryQuality(
            coverage_ratio=coverage_ratio,
            path_length=path_length,
            turn_count=turn_count,
            efficiency=efficiency,
            collision_count=collision_count,
            smoothness=smoothness
        )
    
    def _calculate_coverage_ratio(self, waypoints: List[Waypoint]) -> float:
        """Calculate coverage ratio (simplified version)."""
        if not waypoints:
            return 0.0
        
        # Simplified coverage calculation: based on waypoint distribution
        positions = np.array([[w.x, w.y] for w in waypoints])
        
        # Calculate covered area
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        covered_area = x_range * y_range
        
        # Calculate room area
        room_area = ((self.room_bounds['x_max'] - self.room_bounds['x_min']) * 
                    (self.room_bounds['y_max'] - self.room_bounds['y_min']))
        
        return min(covered_area / room_area, 1.0)
