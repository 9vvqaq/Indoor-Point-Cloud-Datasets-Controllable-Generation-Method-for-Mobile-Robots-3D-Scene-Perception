"""
S3DIS trajectory generation module.
Provides intelligent and controllable trajectory generation and collision detection functionality.
"""

from .trajectory_generator import TrajectoryGeneratorBase, Waypoint, TrajectoryQuality
from .collision_detector import CollisionDetector, FurnitureInfo
from .auto_trajectory_generator import AutoTrajectoryGenerator, RoomAnalysis, TrajectoryCandidate

__all__ = [
    # Base classes
    'TrajectoryGeneratorBase', 'Waypoint', 'TrajectoryQuality',
    
    # Collision detection
    'CollisionDetector', 'FurnitureInfo',
    
    # Auto trajectory generation
    'AutoTrajectoryGenerator', 'RoomAnalysis', 'TrajectoryCandidate',
]
