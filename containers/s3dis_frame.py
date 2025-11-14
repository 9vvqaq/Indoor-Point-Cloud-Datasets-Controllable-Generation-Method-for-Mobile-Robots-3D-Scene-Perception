"""
S3DIS frame data container.
Contains robot pose, LiDAR pose, timestamp, and other information.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RobotPose:
    """
    Robot pose.
    Contains position, orientation, timestamp, and other information.
    """
    position: np.ndarray  # (3,) Position
    orientation: np.ndarray  # (3, 3) Rotation matrix
    timestamp: float = 0.0  # Timestamp
    velocity: Optional[np.ndarray] = None  # (3,) Velocity
    angular_velocity: Optional[np.ndarray] = None  # (3,) Angular velocity
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.orientation
        matrix[:3, 3] = self.position
        return matrix
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, timestamp: float = 0.0) -> 'RobotPose':
        """Create from 4x4 transformation matrix."""
        return cls(
            position=matrix[:3, 3],
            orientation=matrix[:3, :3],
            timestamp=timestamp
        )
    
    def get_yaw(self) -> float:
        """Get yaw angle (rotation around Z-axis)."""
        return np.arctan2(self.orientation[1, 0], self.orientation[0, 0])
    
    def get_pitch(self) -> float:
        """Get pitch angle (rotation around Y-axis)."""
        return np.arctan2(-self.orientation[2, 0], 
                         np.sqrt(self.orientation[2, 1]**2 + self.orientation[2, 2]**2))
    
    def get_roll(self) -> float:
        """Get roll angle (rotation around X-axis)."""
        return np.arctan2(self.orientation[2, 1], self.orientation[2, 2])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'timestamp': self.timestamp,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'angular_velocity': self.angular_velocity.tolist() if self.angular_velocity is not None else None
        }


@dataclass
class LidarPose:
    """
    LiDAR pose.
    LiDAR sensor pose relative to robot.
    """
    position: np.ndarray  # (3,) Position
    orientation: np.ndarray  # (3, 3) Rotation matrix
    sensor_id: str = "lidar_0"  # Sensor ID
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.orientation
        matrix[:3, 3] = self.position
        return matrix
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, sensor_id: str = "lidar_0") -> 'LidarPose':
        """Create from 4x4 transformation matrix."""
        return cls(
            position=matrix[:3, 3],
            orientation=matrix[:3, :3],
            sensor_id=sensor_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'sensor_id': self.sensor_id
        }


class S3DISFrame:
    """
    S3DIS frame data container.
    Contains robot pose, LiDAR pose, timestamp, and other information.
    """
    
    def __init__(self, frame_index: int, robot_pose: RobotPose, 
                 lidar_poses: Optional[Dict[str, LidarPose]] = None,
                 frame_metadata: Optional[Dict[str, Any]] = None):
        self.frame_index = frame_index
        self.robot_pose = robot_pose
        self.lidar_poses = lidar_poses or {"lidar_0": LidarPose(
            position=np.array([0, 0, 0]),
            orientation=np.eye(3)
        )}
        self.frame_metadata = frame_metadata or {}
    
    def get_robot_pose_matrix(self) -> np.ndarray:
        """Get robot pose matrix."""
        return self.robot_pose.to_matrix()
    
    def get_lidar_pose_matrix(self, sensor_id: str = "lidar_0") -> np.ndarray:
        """Get LiDAR pose matrix."""
        if sensor_id not in self.lidar_poses:
            raise ValueError(f"LiDAR sensor {sensor_id} does not exist")
        return self.lidar_poses[sensor_id].to_matrix()
    
    def get_global_lidar_pose(self, sensor_id: str = "lidar_0") -> np.ndarray:
        """Get global LiDAR pose matrix."""
        robot_pose = self.get_robot_pose_matrix()
        lidar_pose = self.get_lidar_pose_matrix(sensor_id)
        return robot_pose @ lidar_pose
    
    def get_timestamp(self) -> float:
        """Get timestamp."""
        return self.robot_pose.timestamp
    
    def get_robot_position(self) -> np.ndarray:
        """Get robot position."""
        return self.robot_pose.position
    
    def get_robot_orientation(self) -> np.ndarray:
        """Get robot orientation."""
        return self.robot_pose.orientation
    
    def get_lidar_position(self, sensor_id: str = "lidar_0") -> np.ndarray:
        """Get LiDAR position."""
        if sensor_id not in self.lidar_poses:
            raise ValueError(f"LiDAR sensor {sensor_id} does not exist")
        return self.lidar_poses[sensor_id].position
    
    def get_lidar_orientation(self, sensor_id: str = "lidar_0") -> np.ndarray:
        """Get LiDAR orientation."""
        if sensor_id not in self.lidar_poses:
            raise ValueError(f"LiDAR sensor {sensor_id} does not exist")
        return self.lidar_poses[sensor_id].orientation
    
    def add_lidar_pose(self, sensor_id: str, lidar_pose: LidarPose):
        """Add LiDAR pose."""
        self.lidar_poses[sensor_id] = lidar_pose
    
    def remove_lidar_pose(self, sensor_id: str):
        """Remove LiDAR pose."""
        if sensor_id in self.lidar_poses:
            del self.lidar_poses[sensor_id]
    
    def get_available_sensors(self) -> list:
        """Get available sensor list."""
        return list(self.lidar_poses.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame_index': self.frame_index,
            'robot_pose': self.robot_pose.to_dict(),
            'lidar_poses': {k: v.to_dict() for k, v in self.lidar_poses.items()},
            'frame_metadata': self.frame_metadata
        }
    
    @classmethod
    def from_dict(cls, frame_dict: Dict[str, Any]) -> 'S3DISFrame':
        """Create from dictionary."""
        robot_pose = RobotPose(
            position=np.array(frame_dict['robot_pose']['position']),
            orientation=np.array(frame_dict['robot_pose']['orientation']),
            timestamp=frame_dict['robot_pose']['timestamp'],
            velocity=np.array(frame_dict['robot_pose']['velocity']) if frame_dict['robot_pose']['velocity'] else None,
            angular_velocity=np.array(frame_dict['robot_pose']['angular_velocity']) if frame_dict['robot_pose']['angular_velocity'] else None
        )
        
        lidar_poses = {}
        for sensor_id, lidar_data in frame_dict['lidar_poses'].items():
            lidar_poses[sensor_id] = LidarPose(
                position=np.array(lidar_data['position']),
                orientation=np.array(lidar_data['orientation']),
                sensor_id=sensor_id
            )
        
        return cls(
            frame_index=frame_dict['frame_index'],
            robot_pose=robot_pose,
            lidar_poses=lidar_poses,
            frame_metadata=frame_dict.get('frame_metadata', {})
        )
    
    def __repr__(self) -> str:
        return (f"S3DISFrame(index={self.frame_index}, "
                f"timestamp={self.get_timestamp():.3f}, "
                f"sensors={self.get_available_sensors()})")
