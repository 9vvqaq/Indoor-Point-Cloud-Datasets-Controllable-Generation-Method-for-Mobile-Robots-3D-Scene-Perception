"""
S3DIS scene data container.
Defines scene container following LiT syntax structure.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path


class RoomBounds:
    """
    Room boundary information.
    Defines 3D bounding box for a room.
    """
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, 
                 z_min: float, z_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    
    def get_center(self) -> np.ndarray:
        """Get room center point."""
        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        ])
    
    def get_size(self) -> np.ndarray:
        """Get room size."""
        return np.array([
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        ])
    
    def get_volume(self) -> float:
        """Get room volume."""
        size = self.get_size()
        return size[0] * size[1] * size[2]
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if point is inside room."""
        return (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max and
                self.z_min <= point[2] <= self.z_max)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'x_min': self.x_min, 'x_max': self.x_max,
            'y_min': self.y_min, 'y_max': self.y_max,
            'z_min': self.z_min, 'z_max': self.z_max
        }
    
    @classmethod
    def from_dict(cls, bounds_dict: Dict[str, float]) -> 'RoomBounds':
        """Create from dictionary."""
        return cls(**bounds_dict)
    
    @classmethod
    def from_mesh(cls, mesh: o3d.geometry.TriangleMesh) -> 'RoomBounds':
        """Automatically calculate bounds from mesh."""
        vertices = np.asarray(mesh.vertices)
        return cls(
            x_min=vertices[:, 0].min(),
            x_max=vertices[:, 0].max(),
            y_min=vertices[:, 1].min(),
            y_max=vertices[:, 1].max(),
            z_min=vertices[:, 2].min(),
            z_max=vertices[:, 2].max()
        )


class SemanticInfo:
    """
    Semantic information.
    Contains semantic labels, object information, etc. for a room.
    """
    
    def __init__(self, room_type: str = "unknown", 
                 furniture_info: Optional[Dict[str, Any]] = None,
                 semantic_labels: Optional[Dict[str, int]] = None):
        self.room_type = room_type
        self.furniture_info = furniture_info or {}
        self.semantic_labels = semantic_labels or {}
    
    def add_furniture(self, name: str, position: np.ndarray, 
                     size: np.ndarray, category: str = "unknown"):
        """Add furniture information."""
        self.furniture_info[name] = {
            'position': position.tolist(),
            'size': size.tolist(),
            'category': category
        }
    
    def get_furniture_count(self) -> int:
        """Get furniture count."""
        return len(self.furniture_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'room_type': self.room_type,
            'furniture_info': self.furniture_info,
            'semantic_labels': self.semantic_labels
        }


@dataclass
class S3DISScene:
    """
    S3DIS scene container.
    Follows LiT syntax structure.
    """
    
    scene_name: str
    room_mesh: o3d.geometry.TriangleMesh
    room_bounds: RoomBounds = field(default_factory=lambda: None)
    semantic_info: SemanticInfo = field(default_factory=lambda: None)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.room_bounds is None:
            self.room_bounds = RoomBounds.from_mesh(self.room_mesh)
        if self.semantic_info is None:
            self.semantic_info = SemanticInfo()
        
        # Calculate mesh statistics
        self.num_vertices = len(self.room_mesh.vertices)
        self.num_triangles = len(self.room_mesh.triangles)
        self.mesh_volume = self._calculate_mesh_volume()
    
    def _calculate_mesh_volume(self) -> float:
        """Calculate mesh volume (simplified version)."""
        # Use bounding box volume as approximation
        return self.room_bounds.get_volume()
    
    def get_bounds_center(self) -> np.ndarray:
        """Get room center point."""
        return self.room_bounds.get_center()
    
    def get_bounds_size(self) -> np.ndarray:
        """Get room size."""
        return self.room_bounds.get_size()
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if point is inside room."""
        return self.room_bounds.is_point_inside(point)
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        return {
            'num_vertices': self.num_vertices,
            'num_triangles': self.num_triangles,
            'volume': self.mesh_volume,
            'bounds': self.room_bounds.to_dict()
        }
    
    def save_mesh(self, output_path: Path):
        """Save mesh to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_path), self.room_mesh)
    
    def load_mesh(self, mesh_path: Path) -> bool:
        """Load mesh from file."""
        try:
            self.room_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(self.room_mesh.vertices) == 0:
                return False
            
            # Update statistics
            self.num_vertices = len(self.room_mesh.vertices)
            self.num_triangles = len(self.room_mesh.triangles)
            self.room_bounds = RoomBounds.from_mesh(self.room_mesh)
            self.mesh_volume = self._calculate_mesh_volume()
            
            return True
        except Exception as e:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scene_name': self.scene_name,
            'room_bounds': self.room_bounds.to_dict(),
            'semantic_info': self.semantic_info.to_dict(),
            'mesh_statistics': self.get_mesh_statistics()
        }
    
    @classmethod
    def from_mesh_file(cls, scene_name: str, mesh_path: Path, 
                      semantic_info: Optional[SemanticInfo] = None) -> 'S3DISScene':
        """Create scene from mesh file."""
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise ValueError(f"Cannot load mesh file: {mesh_path}")
        
        return cls(scene_name, mesh, semantic_info=semantic_info)
    
    def __repr__(self) -> str:
        return (f"S3DISScene(name='{self.scene_name}', "
                f"vertices={self.num_vertices}, triangles={self.num_triangles}, "
                f"bounds={self.room_bounds.get_size()})")
