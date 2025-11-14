"""
Indoor LiDAR class.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from .lidar_intrinsics import Indoor8LineLidarIntrinsics, DualAxisLidarIntrinsics


@dataclass
class IndoorLidar:
    """
    Indoor LiDAR class.
    Follows LiT Lidar class structure.
    """
    
    intrinsics: Indoor8LineLidarIntrinsics
    pose: np.ndarray  # (4, 4) LiDAR pose matrix
    
    def __post_init__(self):
        """Post-initialization processing."""
        assert isinstance(self.intrinsics, Indoor8LineLidarIntrinsics)
        assert isinstance(self.pose, np.ndarray)
        assert self.pose.shape == (4, 4)
    
    def get_rays(self) -> np.ndarray:
        """
        Get LiDAR rays in world coordinate system.
        
        Returns:
            rays: (N, 6) float32.
                - rays[:, :3]: Ray origins in world coordinate system
                - rays[:, 3:]: Ray directions in world coordinate system
                - Ray directions are normalized (norm = 1)
        """
        if self.intrinsics.vertical_degrees is None:
            rays_o, rays_d = self._gen_lidar_rays(
                pose=self.pose,
                fov_up=self.intrinsics.fov_up,
                fov_down=self.intrinsics.fov_down,
                H=self.intrinsics.vertical_res,
                W=self.intrinsics.horizontal_res,
            )
        else:
            rays_o, rays_d = self._gen_lidar_rays_with_vertical_degrees(
                pose=self.pose,
                vertical_degrees=self.intrinsics.vertical_degrees,
                W=self.intrinsics.horizontal_res,
            )
        
        rays = np.concatenate([rays_o, rays_d], axis=-1)
        return rays
    
    @staticmethod
    def _gen_lidar_rays(pose, fov_up, fov_down, H, W):
        """
        Generate LiDAR rays (uniform distribution).
        """
        # Ensure H and W are integers and greater than 0
        H = max(1, int(H))
        W = max(1, int(W))
        
        # Generate vertical angles
        fov_up_rad = np.deg2rad(fov_up)
        fov_down_rad = np.deg2rad(fov_down)
        vertical_angles = np.linspace(fov_up_rad, -fov_down_rad, H)
        
        # Generate horizontal angles
        horizontal_angles = np.linspace(0, 2 * np.pi, W, endpoint=False)
        
        # Generate ray directions
        rays_d = []
        for v_angle in vertical_angles:
            for h_angle in horizontal_angles:
                # Spherical to Cartesian coordinates
                x = np.cos(v_angle) * np.cos(h_angle)
                y = np.cos(v_angle) * np.sin(h_angle)
                z = np.sin(v_angle)
                rays_d.append([x, y, z])
        
        rays_d = np.array(rays_d, dtype=np.float32)
        
        # Ray origins (LiDAR position)
        rays_o = np.tile(pose[:3, 3], (len(rays_d), 1)).astype(np.float32)
        
        # Transform ray directions to world coordinate system
        rays_d_world = (pose[:3, :3] @ rays_d.T).T
        rays_d_world = rays_d_world.astype(np.float32)
        
        return rays_o, rays_d_world
    
    @staticmethod
    def _gen_lidar_rays_with_vertical_degrees(pose, vertical_degrees, W):
        """
        Generate LiDAR rays using specified vertical angles (consistent with LiT).
        """
        # Ensure W is integer and greater than 0
        W = max(1, int(W))
        H = len(vertical_degrees)
        
        # Check if vertical_degrees is empty
        if not vertical_degrees or len(vertical_degrees) == 0:
            vertical_degrees = [0.0]
            H = 1
        
        # Create grid indices (consistent with LiT)
        j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        i = i.reshape([H * W])
        j = j.reshape([H * W])
        
        # Calculate horizontal angles (consistent with LiT)
        beta = -(i - W / 2) / W * 2 * np.pi
        
        # Calculate vertical angles
        alpha = np.array([np.deg2rad(deg) for deg in vertical_degrees])
        alpha = alpha[j]  # Select angle based on vertical index
        
        # Calculate ray directions (consistent with LiT)
        directions = np.stack([
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ], axis=-1)
        
        # Transform to world coordinate system (consistent with LiT)
        rays_d = np.dot(directions, pose[:3, :3].T)
        rays_o = pose[:3, 3]  # Single origin
        rays_o = np.expand_dims(rays_o, axis=0).repeat(H * W, axis=0)
        
        return rays_o.astype(np.float32), rays_d.astype(np.float32)
    
    def get_total_rays(self) -> int:
        """Get total number of rays."""
        return len(self.get_rays())
    
    def get_scan_frequency(self) -> float:
        """Get scan frequency."""
        return self.intrinsics.get_scan_frequency()
    
    def get_range_limits(self) -> tuple:
        """Get range limits."""
        return self.intrinsics.get_range_limits()


@dataclass
class DualAxisLidar:
    """
    Dual-axis LiDAR class.
    Based on linear spiral scanning model, implements true dual-axis continuous scanning.
    """
    
    intrinsics: DualAxisLidarIntrinsics
    pose: np.ndarray  # (4, 4) LiDAR pose matrix
    
    def __post_init__(self):
        """Post-initialization processing."""
        assert isinstance(self.intrinsics, DualAxisLidarIntrinsics)
        assert isinstance(self.pose, np.ndarray)
        assert self.pose.shape == (4, 4)
    
    def get_rays_at_time(self, t: float) -> np.ndarray:
        """
        Get LiDAR rays at specified time.
        
        Args:
            t: Time (s)
            
        Returns:
            rays: (1, 6) float32. Single ray
                - rays[0, :3]: Ray origin
                - rays[0, 3:]: Ray direction
        """
        # Use unified angle calculation method
        phi, theta = self.intrinsics.calculate_angles_at_time(t, line_idx=0)
        
        # Generate ray direction
        direction = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            np.sin(theta)
        ], dtype=np.float32)
        
        # Transform to world coordinate system
        direction_world = (self.pose[:3, :3] @ direction).astype(np.float32)
        origin_world = self.pose[:3, 3].astype(np.float32)
        
        # Combine ray
        ray = np.concatenate([origin_world, direction_world])
        return ray.reshape(1, 6)
    
    def get_rays_sequence(self, time_sequence: np.ndarray) -> np.ndarray:
        """
        Get LiDAR rays for time sequence (batch processing for efficiency).
        
        Args:
            time_sequence: Time sequence array
            
        Returns:
            rays: (N, 6) float32. Ray array
        """
        # Use unified angle calculation method
        all_rays = []
        for t in time_sequence:
            phi, theta = self.intrinsics.calculate_angles_at_time(t, line_idx=0)
            
            # Generate ray direction
            direction = np.array([
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                np.sin(theta)
            ])
            
            # Transform to world coordinate system
            direction_world = (self.pose[:3, :3] @ direction).astype(np.float32)
            origin_world = self.pose[:3, 3].astype(np.float32)
            
            # Combine ray
            ray = np.concatenate([origin_world, direction_world])
            all_rays.append(ray)
        
        return np.array(all_rays, dtype=np.float32)
    
    def get_multi_line_rays(self, num_points: int = None) -> np.ndarray:
        """
        Get 32-line dual-axis scanning rays (borrowing from original 32-line approach).
        
        Principle:
        - 32 lines, each line independently rotates 360 degrees horizontally
        - Each line swings ±5° at base position
        - Each line has independent horizontal angle sequence
        - Forms 32 independent swinging spiral lines
        
        Args:
            num_points: Number of rays, defaults to point_rate * scan_duration
            
        Returns:
            rays: (N, 6) float32. Ray array
        """
        if num_points is None:
            num_points = int(self.intrinsics.point_rate * self.intrinsics.scan_duration)
        
        # Calculate points per line
        points_per_line = num_points // self.intrinsics.num_vertical_lines
        
        # Generate 32 fixed vertical angles (base position for each line)
        theta_start = self.intrinsics.theta_range[1]  # 15 degrees (radians)
        theta_end = self.intrinsics.theta_range[0]     # -20 degrees (radians)
        base_theta_angles = np.linspace(theta_start, theta_end, self.intrinsics.num_vertical_lines)
        
        # Generate horizontal angle sequence (each line independently rotates 360 degrees)
        horizontal_angles = np.linspace(0, 2 * np.pi, points_per_line, endpoint=False)
        
        # Generate all rays
        all_rays = []
        
        for line_idx, base_theta in enumerate(base_theta_angles):
            # Each line has different swing phase, use more uniform distribution
            phase_offset = line_idx * np.pi / self.intrinsics.num_vertical_lines  # Reduce phase offset
            
            for phi in horizontal_angles:
                # Calculate swing (based on horizontal angle, not time)
                swing = self.intrinsics.swing_amplitude * np.sin(self.intrinsics.swing_frequency * phi + phase_offset)
                theta = base_theta + swing
                
                # Angle range limitation
                theta = np.clip(theta, self.intrinsics.theta_range[0], self.intrinsics.theta_range[1])
                
                # Apply noise
                if self.intrinsics.angle_noise_std > 0:
                    phi += np.random.normal(0, self.intrinsics.angle_noise_std)
                    theta += np.random.normal(0, self.intrinsics.angle_noise_std)
                
                # Generate ray direction
                direction = np.array([
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    np.sin(theta)
                ])
                
                # Transform to world coordinate system
                direction_world = (self.pose[:3, :3] @ direction).astype(np.float32)
                origin_world = self.pose[:3, 3].astype(np.float32)
                
                # Combine ray
                ray = np.concatenate([origin_world, direction_world])
                all_rays.append(ray)
        
        rays = np.array(all_rays, dtype=np.float32)
        
        # Apply dropout logic
        if self.intrinsics.dropout_probability > 0:
            keep_mask = np.random.random(len(rays)) > self.intrinsics.dropout_probability
            rays = rays[keep_mask]
        
        return rays
    
    def get_rays_frame(self, frame_duration: float = None) -> np.ndarray:
        """
        Get LiDAR rays for one frame.
        
        Args:
            frame_duration: Frame duration, defaults to intrinsics setting
            
        Returns:
            rays: (N, 6) float32. Ray array
        """
        time_sequence = self.intrinsics.generate_time_sequence(frame_duration)
        return self.get_rays_sequence(time_sequence)
    
    def get_rays(self) -> np.ndarray:
        """
        Get complete scan LiDAR rays (compatible interface).
        
        Returns:
            rays: (N, 6) float32. Ray array
        """
        # 32-line dual-axis scanning mode
        return self.get_multi_line_rays()
    
    def get_spiral_scan_rays(self, num_points: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get spiral scan rays and timestamps.
        
        Args:
            num_points: Number of rays, defaults to point_rate * scan_duration
            
        Returns:
            (rays, timestamps): Ray array and timestamp array
        """
        if num_points is None:
            num_points = int(self.intrinsics.point_rate * self.intrinsics.scan_duration)
        
        # Generate time sequence
        timestamps = np.linspace(0, self.intrinsics.scan_duration, num_points)
        
        # Generate rays
        rays = self.get_rays_sequence(timestamps)
        
        return rays, timestamps
    
    def get_total_rays(self) -> int:
        """Get total number of rays."""
        return int(self.intrinsics.point_rate * self.intrinsics.scan_duration)
    
    def get_scan_frequency(self) -> float:
        """Get scan frequency."""
        return 1.0 / self.intrinsics.scan_duration
    
    def get_range_limits(self) -> tuple:
        """Get range limits."""
        return (0.5, self.intrinsics.max_range)  # BLK2GO minimum range 0.5m
    
    def add_noise_to_rays(self, rays: np.ndarray) -> np.ndarray:
        """
        Add noise to rays.
        
        Args:
            rays: Original ray array
            
        Returns:
            noisy_rays: Ray array with noise added
        """
        if self.intrinsics.dropout_probability > 0:
            # Random dropout
            keep_mask = np.random.random(len(rays)) > self.intrinsics.dropout_probability
            rays = rays[keep_mask]
        
        return rays


# Unified LiDAR interface
LidarType = IndoorLidar | DualAxisLidar
IntrinsicsType = Indoor8LineLidarIntrinsics | DualAxisLidarIntrinsics


def create_lidar(intrinsics: IntrinsicsType, pose: np.ndarray) -> LidarType:
    """
    Factory function to create LiDAR instance.
    
    Args:
        intrinsics: LiDAR intrinsics configuration
        pose: LiDAR pose matrix
        
    Returns:
        lidar: Corresponding LiDAR instance
    """
    if isinstance(intrinsics, DualAxisLidarIntrinsics):
        return DualAxisLidar(intrinsics=intrinsics, pose=pose)
    elif isinstance(intrinsics, Indoor8LineLidarIntrinsics):
        return IndoorLidar(intrinsics=intrinsics, pose=pose)
    else:
        raise ValueError(f"Unsupported LiDAR intrinsics type: {type(intrinsics)}")


def get_lidar_type(intrinsics: IntrinsicsType) -> str:
    """
    Get LiDAR type string.
    
    Args:
        intrinsics: LiDAR intrinsics configuration
        
    Returns:
        lidar_type: LiDAR type string
    """
    if isinstance(intrinsics, DualAxisLidarIntrinsics):
        return "Dual-axis spiral scanning"
    elif isinstance(intrinsics, Indoor8LineLidarIntrinsics):
        if hasattr(intrinsics, 'dual_axis') and intrinsics.dual_axis:
            return "Single-axis simulated dual-axis"
        else:
            return f"{intrinsics.vertical_res}-line single-axis scanning"
    else:
        return "Unknown type"
