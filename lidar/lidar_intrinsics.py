"""
LiDAR intrinsics base class.
Defines LiDAR sensor parameters following LiT syntax structure.
"""

import numpy as np
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LidarIntrinsics(ABC):
    """
    Abstract class for lidar intrinsics.
    
    Ref: https://www.mathworks.com/help/lidar/ref/lidarparameters.html
    """
    
    fov_up: float  # Positive value.
    fov_down: float  # Positive value.
    vertical_res: int
    horizontal_res: int
    max_range: float
    vertical_degrees: List[float] = None


@dataclass
class DualAxisLidarIntrinsics(LidarIntrinsics):
    """
    Dual-axis LiDAR intrinsics configuration.
    Based on linear spiral scanning model, simulates dual-axis continuous scanning devices like BLK2GO.
    """
    
    # Base class required parameters
    fov_up: float = 15.0  # Upward 15 degrees
    fov_down: float = 20.0  # Downward 20 degrees
    vertical_res: int = 1  # Dual-axis scanning does not require fixed vertical line count
    horizontal_res: int = 1  # Dual-axis scanning does not require fixed horizontal resolution
    max_range: float = 25.0  # Maximum range
    vertical_degrees: List[float] = None  # Dual-axis scanning does not use fixed angles
    
    # Dual-axis scanning core parameters
    phi_0: float = 0.0  # Initial horizontal angle (azimuth)
    omega_phi: float = 2.0 * np.pi  # Horizontal angular velocity (rad/s)
    
    # Scanning time parameters
    scan_duration: float = 1.0  # Single scan duration (s)
    point_rate: int = 420000  # Point cloud acquisition rate (points/s)
    
    # Scanning range
    phi_range: tuple = (0.0, 2.0 * np.pi)  # Horizontal angle range
    theta_range: tuple = (-20.0 * np.pi/180, 15.0 * np.pi/180)  # Vertical angle range
    
    # Noise and error models
    angle_noise_std: float = 0.001  # Angle noise (rad)
    timing_jitter_std: float = 0.0001  # Timing jitter (s)
    dropout_probability: float = 0.02  # Dropout probability
    
    # Scanning mode control
    frame_duration: float = 0.1  # Output frame time interval (s)
    num_vertical_lines: int = 32  # Number of vertical lines
    
    # Swing parameters
    swing_amplitude: float = 5.0 * np.pi/180  # Swing amplitude (±5 degrees)
    swing_frequency: float = 1.0  # Swing frequency (1 swing per rotation, more uniform)
    
    def get_scan_parameters(self) -> dict:
        """Get scan parameters."""
        return {
            'phi_0': self.phi_0,
            'omega_phi': self.omega_phi,
            'scan_duration': self.scan_duration,
            'point_rate': self.point_rate,
            'phi_range': self.phi_range,
            'theta_range': self.theta_range,
            'swing_amplitude': self.swing_amplitude,
            'swing_frequency': self.swing_frequency
        }
    
    def calculate_angles_at_time(self, t: float, line_idx: int = 0) -> tuple:
        """
        Calculate dual-axis angles at specified time.
        
        Args:
            t: Time (s)
            line_idx: Line index (used in 32-line mode)
            
        Returns:
            (phi, theta): Horizontal and vertical angles (rad)
        """
        # Horizontal angle: continuous rotation
        phi = self.phi_0 + self.omega_phi * t
        phi = phi % (2 * np.pi)
        
        # 32-line dual-axis scanning: each line swings
        # Calculate 32 base angles
        theta_start = self.theta_range[1]  # 15 degrees
        theta_end = self.theta_range[0]     # -20 degrees
        base_theta_angles = np.linspace(theta_start, theta_end, self.num_vertical_lines)
        base_theta = base_theta_angles[line_idx % self.num_vertical_lines]
        
        # Each line has different swing phase
        phase_offset = line_idx * 2 * np.pi / self.num_vertical_lines
        swing = self.swing_amplitude * np.sin(self.swing_frequency * t + phase_offset)
        theta = base_theta + swing
        
        # Angle range limitation
        theta = np.clip(theta, self.theta_range[0], self.theta_range[1])
        
        # Apply noise
        if self.angle_noise_std > 0:
            phi += np.random.normal(0, self.angle_noise_std)
            theta += np.random.normal(0, self.angle_noise_std)
        
        return phi, theta
    
    def generate_time_sequence(self, frame_duration: float = None) -> np.ndarray:
        """
        Generate time sequence.
        
        Args:
            frame_duration: Frame duration, defaults to self.frame_duration
            
        Returns:
            time_sequence: Time sequence array
        """
        if frame_duration is None:
            frame_duration = self.frame_duration
            
        # Calculate number of points in this frame
        points_per_frame = int(self.point_rate * frame_duration)
        
        # Generate uniform time sequence
        dt = frame_duration / points_per_frame
        time_sequence = np.arange(0, frame_duration, dt)
        
        return time_sequence
    
    def get_total_points_per_scan(self) -> int:
        """Get total points per scan."""
        return int(self.point_rate * self.scan_duration)
    
    def get_scan_frequency(self) -> float:
        """Get scan frequency."""
        return 1.0 / self.scan_duration
    
    def get_range_limits(self) -> tuple:
        """Get range limits."""
        return (0.5, self.max_range)  # BLK2GO minimum range 0.5m
    
    @classmethod
    def create_blk2go_dual_axis(cls) -> 'DualAxisLidarIntrinsics':
        """Create Leica BLK2GO dual-axis LiDAR configuration (true dual-axis spiral scanning)."""
        return cls(
            # Base class parameters
            fov_up=15.0,
            fov_down=20.0,
            vertical_res=1,
            horizontal_res=1,
            max_range=25.0,
            vertical_degrees=None,
            
            # Dual-axis scanning parameters
            phi_0=0.0,  # Initial horizontal angle
            omega_phi=2.0 * np.pi,  # Horizontal angular velocity: 1 rotation per second
            
            # Scanning time parameters
            scan_duration=0.1,  # Complete one scan in 0.1s (10Hz)
            point_rate=640000,  # Increase point cloud density: 640,000 points/s
            
            # Scanning range
            phi_range=(0.0, 2.0 * np.pi),  # Horizontal 360 degrees
            theta_range=(-20.0 * np.pi/180, 15.0 * np.pi/180),  # Vertical -20° to +15°
            
            # Noise model
            angle_noise_std=0.001,  # Angle noise
            timing_jitter_std=0.0001,  # Timing jitter
            dropout_probability=0.02,  # 2% dropout rate
            
            # Scanning mode
            frame_duration=0.1,  # Output one frame every 100ms
            num_vertical_lines=32,  # 32-line scanning
            swing_amplitude=5.0 * np.pi/180,  # ±5 degree swing
            swing_frequency=1.0  # 1 swing per rotation, more uniform
        )
    
    @classmethod
    def create_custom_dual_axis(cls, 
                               phi_0: float = 0.0,
                               theta_0: float = 15.0,
                               omega_phi: float = 2.0 * np.pi,
                               omega_theta: float = -0.1,
                               point_rate: int = 420000,
                               scan_duration: float = 1.0) -> 'DualAxisLidarIntrinsics':
        """Create custom dual-axis LiDAR configuration."""
        return cls(
            phi_0=phi_0,
            theta_0=theta_0 * np.pi/180,  # Convert to radians
            omega_phi=omega_phi,
            omega_theta=omega_theta,
            scan_duration=scan_duration,
            point_rate=point_rate,
            use_spiral_scan=True,
            frame_duration=0.1,
            fov_up=15.0,
            fov_down=20.0,
            vertical_res=1,
            horizontal_res=1,
            max_range=25.0
        )


@dataclass
class Indoor8LineLidarIntrinsics(LidarIntrinsics):
    """
    8-line indoor LiDAR intrinsics configuration.
    Based on real indoor LiDAR sensor characteristics.
    """
    
    # 8-line LiDAR vertical angle configuration
    fov_up: float = 15.0  # Upward 15 degrees
    fov_down: float = 20.0  # Downward 20 degrees
    vertical_res: int = 8  # 8 lines
    horizontal_res: int = 2000  # Horizontal resolution
    max_range: float = 20.0  # Indoor maximum range 20 meters
    vertical_degrees: List[float] = field(default_factory=lambda: [15, 10, 5, 0, -5, -10, -15, -20])
    
    # Indoor LiDAR specific parameters
    min_range: float = 0.1  # Minimum range
    range_resolution: float = 0.01  # Range resolution
    scan_frequency: float = 10.0  # Scan frequency
    points_per_beam: int = 2000  # Points per laser beam
    
    # Noise model
    range_noise_std: float = 0.02  # Range noise standard deviation
    angle_noise_std: float = 0.01  # Angle noise standard deviation
    
    # BLK2GO dual-axis LiDAR specific parameters
    dual_axis: bool = False  # Whether dual-axis scanning
    capture_rate: int = 200000  # Point cloud acquisition rate (points/s)
    intensity_noise_std: float = 0.1  # Intensity noise standard deviation
    dropout_probability: float = 0.05  # Dropout probability
    
    @classmethod
    def create_standard_8line(cls) -> 'Indoor8LineLidarIntrinsics':
        """Create standard 8-line indoor LiDAR configuration."""
        return cls()
    
    @classmethod
    def create_high_resolution_8line(cls) -> 'Indoor8LineLidarIntrinsics':
        """Create high-resolution 8-line indoor LiDAR configuration."""
        return cls(
            horizontal_res=4000,
            points_per_beam=4000,
            range_resolution=0.005
        )
    
    @classmethod
    def create_low_cost_8line(cls) -> 'Indoor8LineLidarIntrinsics':
        """Create low-cost 8-line indoor LiDAR configuration."""
        return cls(
            horizontal_res=1000,
            points_per_beam=1000,
            range_resolution=0.02,
            range_noise_std=0.05
        )
    
    @classmethod
    def create_dense_32line(cls) -> 'Indoor8LineLidarIntrinsics':
        """Create high-density 32-line indoor LiDAR configuration."""
        # 32-line vertical angle configuration (denser vertical distribution)
        vertical_degrees = []
        for i in range(32):
            angle = 15.0 - (i * 35.0 / 31.0)  # Uniform distribution from 15 degrees to -20 degrees
            vertical_degrees.append(round(angle, 1))
        
        return cls(
            fov_up=15.0,
            fov_down=20.0,
            vertical_res=32,  # 32 lines
            horizontal_res=4000,  # High resolution
            max_range=25.0,  # Increased range
            vertical_degrees=vertical_degrees,
            points_per_beam=3000,  # More points per beam
            range_resolution=0.005,  # Higher precision
            range_noise_std=0.01,  # Reduced noise
            angle_noise_std=0.005
        )
    
    @classmethod
    def create_leica_blk2go(cls) -> 'Indoor8LineLidarIntrinsics':
        """Create Leica BLK2GO dual-axis LiDAR configuration (single-axis simulation version)."""
        # BLK2GO dual-axis LiDAR characteristics
        # Dual-axis scanning, 360-degree horizontal + vertical coverage
        vertical_degrees = []
        for i in range(64):  # Increase vertical line count to simulate dual-axis
            angle = 15.0 - (i * 35.0 / 63.0)  # From 15 degrees to -20 degrees
            vertical_degrees.append(round(angle, 1))
        
        return cls(
            fov_up=15.0,
            fov_down=20.0,
            vertical_res=64,  # Dual-axis scanning, more vertical lines
            horizontal_res=8000,  # 360-degree horizontal scanning
            max_range=25.0,  # BLK2GO range
            vertical_degrees=vertical_degrees,
            points_per_beam=5000,  # 420,000 points/s, higher density
            range_resolution=0.003,  # ±3mm@10m precision
            range_noise_std=0.003,  # Higher precision, lower noise
            angle_noise_std=0.002,
            # BLK2GO specific parameters
            min_range=0.5,  # BLK2GO minimum range 0.5m
            scan_frequency=20.0,  # Higher scan frequency
            dual_axis=True,  # Dual-axis scanning flag
            capture_rate=420000  # 420,000 points/s
        )
    
    @classmethod
    def create_custom_lidar(cls, num_beams: int = 8, 
                           beam_angles: Optional[List[float]] = None,
                           horizontal_resolution: float = 0.1,
                           max_range: float = 20.0,
                           points_per_beam: int = 2000) -> 'Indoor8LineLidarIntrinsics':
        """Create custom LiDAR configuration."""
        # Calculate vertical FOV
        if beam_angles:
            fov_up = max(beam_angles)
            fov_down = abs(min(beam_angles))
            vertical_degrees = beam_angles
        else:
            # Default uniform distribution
            fov_up = 15.0
            fov_down = 20.0
            vertical_degrees = [15, 10, 5, 0, -5, -10, -15, -20]
        
        # Limit maximum horizontal_res to avoid integer overflow
        horizontal_res = int(360.0 / horizontal_resolution)
        if horizontal_res > 10000:  # Limit maximum resolution
            horizontal_res = 10000
        
        return cls(
            fov_up=fov_up,
            fov_down=fov_down,
            vertical_res=num_beams,
            horizontal_res=horizontal_res,
            max_range=max_range,
            vertical_degrees=vertical_degrees,
            points_per_beam=points_per_beam
        )
    
    def get_total_points_per_scan(self) -> int:
        """Get total points per scan."""
        return self.vertical_res * self.horizontal_res
    
    def get_scan_frequency(self) -> float:
        """Get scan frequency."""
        return self.scan_frequency
    
    def get_range_limits(self) -> tuple:
        """Get range limits."""
        return (self.min_range, self.max_range)
    
    def add_noise(self, points: np.ndarray, ranges: np.ndarray, 
                  angles: np.ndarray, intensities: np.ndarray) -> tuple:
        """Add noise."""
        # Range noise
        noisy_ranges = ranges + np.random.normal(0, self.range_noise_std, ranges.shape)
        
        # Angle noise
        noisy_angles = angles + np.random.normal(0, np.deg2rad(self.angle_noise_std), angles.shape)
        
        # Intensity noise
        noisy_intensities = np.clip(
            intensities + np.random.normal(0, self.intensity_noise_std, intensities.shape), 
            0, 1
        )
        
        # Dropout
        if self.dropout_probability > 0:
            keep_mask = np.random.random(len(points)) > self.dropout_probability
            noisy_points = points[keep_mask]
            noisy_ranges = noisy_ranges[keep_mask]
            noisy_angles = noisy_angles[keep_mask]
            noisy_intensities = noisy_intensities[keep_mask]
        else:
            noisy_points = points
        
        return noisy_points, noisy_ranges, noisy_angles, noisy_intensities
