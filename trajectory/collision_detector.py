"""
Simplified collision detector.
Furniture in mesh itself is occlusion, only need to avoid furniture areas during trajectory planning.
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .trajectory_generator import Waypoint


@dataclass
class FurnitureInfo:
    """
    Furniture information (simplified version).
    Furniture in mesh itself is occlusion, here only records position and size for trajectory planning.
    """
    name: str
    position: np.ndarray  # (3,) Position
    size: np.ndarray  # (3,) Size [length, width, height]
    category: str  # Furniture type
    
    def get_bounds(self) -> Dict[str, float]:
        """Get furniture bounding box (for trajectory planning)."""
        half_size = self.size / 2
        return {
            'x_min': self.position[0] - half_size[0],
            'x_max': self.position[0] + half_size[0],
            'y_min': self.position[1] - half_size[1],
            'y_max': self.position[1] + half_size[1],
            'z_min': self.position[2] - half_size[2],
            'z_max': self.position[2] + half_size[2]
        }
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if point is inside furniture (for trajectory planning)."""
        bounds = self.get_bounds()
        return (bounds['x_min'] <= point[0] <= bounds['x_max'] and
                bounds['y_min'] <= point[1] <= bounds['y_max'] and
                bounds['z_min'] <= point[2] <= bounds['z_max'])


class CollisionDetector:
    """
    Simplified collision detector.
    Furniture in mesh itself is occlusion, here only used for trajectory planning obstacle avoidance.
    """
    
    def __init__(self, robot_radius: float = 0.3):
        self.robot_radius = robot_radius
        self.furniture_list: List[FurnitureInfo] = []
    
    def add_furniture(self, furniture: FurnitureInfo):
        """Add furniture."""
        self.furniture_list.append(furniture)
    
    def add_furniture_from_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                               name: str, category: str = "unknown"):
        """Add furniture from mesh (simplified version, only for trajectory planning)."""
        # Calculate mesh bounding box
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            return
        
        # Calculate position and size
        position = np.mean(vertices, axis=0)
        size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        
        # Create furniture information (simplified version)
        furniture = FurnitureInfo(
            name=name,
            position=position,
            size=size,
            category=category
        )
        
        self.add_furniture(furniture)
    
    def detect_collision(self, waypoint: Waypoint) -> Tuple[bool, Optional[FurnitureInfo]]:
        """
        Detect collision for a single waypoint.
        
        Args:
            waypoint: Waypoint
        
        Returns:
            has_collision: Whether collision occurred
            collided_furniture: Collided furniture (if any)
        """
        robot_pos = np.array([waypoint.x, waypoint.y, waypoint.z])
        
        for furniture in self.furniture_list:
            if self._check_robot_furniture_collision(robot_pos, furniture):
                return True, furniture
        
        return False, None
    
    def detect_path_collision(self, waypoints: List[Waypoint]) -> List[Tuple[int, FurnitureInfo]]:
        """
        Detect path collisions.
        
        Args:
            waypoints: List of waypoints
        
        Returns:
            collisions: List of collisions [(waypoint_index, furniture), ...]
        """
        collisions = []
        
        for i, waypoint in enumerate(waypoints):
            has_collision, furniture = self.detect_collision(waypoint)
            if has_collision:
                collisions.append((i, furniture))
        
        return collisions
    
    def _check_robot_furniture_collision(self, robot_pos: np.ndarray, 
                                       furniture: FurnitureInfo) -> bool:
        """Check collision between robot and furniture."""
        # Method 1: Bounding box based collision detection
        if self._check_bbox_collision(robot_pos, furniture):
            return True
        
        # Method 2: Mesh-based precise collision detection
        if furniture.mesh is not None:
            return self._check_mesh_collision(robot_pos, furniture)
        
        return False
    
    def _check_bbox_collision(self, robot_pos: np.ndarray, furniture: FurnitureInfo) -> bool:
        """Bounding box based collision detection."""
        bounds = furniture.get_bounds()
        
        # Expand bounding box to account for robot radius
        expanded_bounds = {
            'x_min': bounds['x_min'] - self.robot_radius,
            'x_max': bounds['x_max'] + self.robot_radius,
            'y_min': bounds['y_min'] - self.robot_radius,
            'y_max': bounds['y_max'] + self.robot_radius,
            'z_min': bounds['z_min'] - self.robot_radius,
            'z_max': bounds['z_max'] + self.robot_radius
        }
        
        return (expanded_bounds['x_min'] <= robot_pos[0] <= expanded_bounds['x_max'] and
                expanded_bounds['y_min'] <= robot_pos[1] <= expanded_bounds['y_max'] and
                expanded_bounds['z_min'] <= robot_pos[2] <= expanded_bounds['z_max'])
    
    def _check_mesh_collision(self, robot_pos: np.ndarray, furniture: FurnitureInfo) -> bool:
        """Mesh-based precise collision detection."""
        if furniture.mesh is None:
            return False
        
        # Create robot spherical mesh
        robot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.robot_radius)
        robot_sphere.translate(robot_pos)
        
        # Check if two meshes intersect
        # Here use simplified method: check if robot center is inside furniture mesh
        # More precise methods can use Open3D collision detection functionality
        
        vertices = np.asarray(furniture.mesh.vertices)
        if len(vertices) == 0:
            return False
        
        # Simplified collision detection: check if robot center is within mesh bounding box
        return furniture.is_point_inside(robot_pos)
    
    def suggest_avoidance_path(self, waypoint: Waypoint, 
                             collided_furniture: FurnitureInfo) -> List[Waypoint]:
        """
        Suggest avoidance path.
        
        Args:
            waypoint: Collided waypoint
            collided_furniture: Collided furniture
        
        Returns:
            avoidance_waypoints: List of avoidance waypoints
        """
        avoidance_waypoints = []
        
        # Get furniture boundaries
        bounds = collided_furniture.get_bounds()
        furniture_center = collided_furniture.position
        
        # Calculate avoidance direction
        robot_pos = np.array([waypoint.x, waypoint.y, waypoint.z])
        direction_to_furniture = furniture_center - robot_pos
        direction_to_furniture[2] = 0  # Ignore Z-axis
        
        if np.linalg.norm(direction_to_furniture) > 0:
            direction_to_furniture = direction_to_furniture / np.linalg.norm(direction_to_furniture)
        
        # Generate avoidance waypoints
        avoidance_distance = self.robot_radius + 0.5  # Avoidance distance
        
        # Option 1: Bypass
        for angle_offset in [-np.pi/2, np.pi/2]:  # Left and right bypass
            avoidance_direction = self._rotate_vector(direction_to_furniture, angle_offset)
            avoidance_pos = robot_pos + avoidance_direction * avoidance_distance
            
            avoidance_waypoint = Waypoint(
                x=avoidance_pos[0],
                y=avoidance_pos[1],
                z=avoidance_pos[2],
                yaw=waypoint.yaw + angle_offset
            )
            avoidance_waypoints.append(avoidance_waypoint)
        
        # Option 2: Backward
        back_direction = -direction_to_furniture
        back_pos = robot_pos + back_direction * avoidance_distance
        
        back_waypoint = Waypoint(
            x=back_pos[0],
            y=back_pos[1],
            z=back_pos[2],
            yaw=waypoint.yaw
        )
        avoidance_waypoints.append(back_waypoint)
        
        return avoidance_waypoints
    
    def _rotate_vector(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """Rotate vector."""
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        return rotation_matrix @ vector
    
    def get_furniture_list(self) -> List[FurnitureInfo]:
        """Get furniture list."""
        return self.furniture_list.copy()
    
    def clear_furniture(self):
        """Clear furniture list."""
        self.furniture_list.clear()
    
    def get_collision_statistics(self, waypoints: List[Waypoint]) -> Dict[str, Any]:
        """Get collision statistics."""
        collisions = self.detect_path_collision(waypoints)
        
        collision_count = len(collisions)
        collision_furniture = {}
        
        for _, furniture in collisions:
            if furniture.name not in collision_furniture:
                collision_furniture[furniture.name] = 0
            collision_furniture[furniture.name] += 1
        
        return {
            'total_collisions': collision_count,
            'collision_rate': collision_count / len(waypoints) if waypoints else 0,
            'collision_furniture': collision_furniture
        }
