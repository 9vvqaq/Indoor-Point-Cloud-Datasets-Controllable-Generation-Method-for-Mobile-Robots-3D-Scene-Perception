"""
CPU Raycast engine for mesh-ray intersection.
"""

import numpy as np
import open3d as o3d

try:
    from ..lidar import IndoorLidar
    from .raycast_engine import RaycastEngineBase
except ImportError:
    from lidar import IndoorLidar
    from raycast_engine import RaycastEngineBase


class RaycastEngineCPU(RaycastEngineBase):
    """
    CPU implementation of raycast engine based on Open3D and Embree.
    """

    def __init__(self):
        super().__init__()

    def rays_intersect_mesh(
        self,
        rays: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the given rays.

        Args:
            rays: (N, 6) float32 numpy array
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
        """
        # Sanity checks.
        if not isinstance(rays, np.ndarray):
            raise TypeError("rays must be a numpy array.")
        if rays.ndim != 2 or rays.shape[1] != 6:
            raise ValueError("rays must be a (N, 6) array.")

        # Convert mesh to raycasting_scene.
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        # Run ray cast.
        rays = rays.astype(np.float32)
        ray_cast_results = raycasting_scene.cast_rays(o3d.core.Tensor(rays))
        normals = ray_cast_results["primitive_normals"].numpy()
        depths = ray_cast_results["t_hit"].numpy()
        masks = depths != np.inf
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:]
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
        
        # Safely calculate point positions, avoid inf values
        valid_depths = np.isfinite(depths)
        points = np.zeros_like(rays_o)
        points[valid_depths] = rays_o[valid_depths] + rays_d[valid_depths] * depths[valid_depths, None]

        # Filter by hit masks.
        hit_dict = {
            "masks": masks,
            "depths": depths,
            "points": points,
            "normals": normals,
        }
        points = hit_dict["points"][hit_dict["masks"]]

        return points

    def lidar_intersect_mesh(
        self,
        lidar: IndoorLidar,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the lidar rays.

        Args:
            lidar: IndoorLidar
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
            incident_angles: (N,) float32 numpy array
        """
        rays = lidar.get_rays()
        points = self.rays_intersect_mesh(mesh=mesh, rays=rays)

        # Post-processing: filter points by range.
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
