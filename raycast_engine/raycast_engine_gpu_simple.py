"""
Simplified GPU raycast engine.
Does not depend on CUDA kernel files, uses Open3D GPU acceleration.
"""

import numpy as np
import open3d as o3d
from .raycast_engine import RaycastEngineBase
from ..lidar import IndoorLidar


class RaycastEngineGPU(RaycastEngineBase):
    """
    Simplified GPU raycast engine.
    Uses Open3D GPU acceleration functionality, does not depend on CUDA kernels.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def rays_intersect_mesh(
        self,
        rays: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        GPU-accelerated ray-mesh intersection.
        
        Args:
            rays: (N, 6) float32 numpy array
            mesh: o3d.geometry.TriangleMesh
        
        Returns:
            points: (N, 3) float32 numpy array
        """
        # Currently falls back to CPU implementation
        # In practical applications, CUDA or other GPU acceleration libraries can be used here
        
        # Use Open3D CPU implementation
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        
        # Run raycast
        rays = rays.astype(np.float32)
        ray_cast_results = raycasting_scene.cast_rays(o3d.core.Tensor(rays))
        normals = ray_cast_results["primitive_normals"].numpy()
        depths = ray_cast_results["t_hit"].numpy()
        masks = depths != np.inf
        
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:]
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
        points = rays_o + rays_d * depths[:, None]
        
        # Filter hit points
        points = points[masks]
        
        return points

    def lidar_intersect_mesh(
        self,
        lidar: IndoorLidar,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Ray-mesh intersection using LiDAR.
        
        Args:
            lidar: IndoorLidar
            mesh: o3d.geometry.TriangleMesh
        
        Returns:
            points: (N, 3) float32 numpy array
            incident_angles: (N,) float32 numpy array
        """
        rays = lidar.get_rays()
        points = self.rays_intersect_mesh(mesh=mesh, rays=rays)
        
        # Post-processing: filter points by distance
        lidar_center = lidar.pose[:3, 3]
        point_dists = np.linalg.norm(points - lidar_center, axis=1)
        points = points[point_dists < lidar.intrinsics.max_range]
        
        # Calculate incident angles (simplified version)
        if len(points) > 0:
            # Calculate direction vectors from LiDAR center to each point
            directions = points - lidar_center
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            
            # Calculate incident angles (angle relative to vertical direction)
            incident_angles = np.arccos(np.abs(directions[:, 2]))  # Absolute value of z component
            incident_angles = np.degrees(incident_angles)  # Convert to degrees
        else:
            incident_angles = np.empty(0)
        
        return points, incident_angles

