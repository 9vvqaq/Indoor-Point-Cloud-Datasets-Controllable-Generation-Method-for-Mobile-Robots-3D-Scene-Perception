"""
S3DIS data container module.
Provides data container classes for S3DIS scenes, frames, and simulation results.
"""

from .s3dis_scene import S3DISScene, RoomBounds, SemanticInfo
from .s3dis_frame import S3DISFrame, RobotPose, LidarPose
from .s3dis_sim_frame import S3DISSimFrame, ScanQuality, IncidentAngles
from .s3dis_sim_scene import S3DISSimScene, SimulationStats, ResultExporter

__all__ = [
    # Scene containers
    'S3DISScene', 'RoomBounds', 'SemanticInfo',
    
    # Frame containers
    'S3DISFrame', 'RobotPose', 'LidarPose',
    
    # Simulation frame containers
    'S3DISSimFrame', 'ScanQuality', 'IncidentAngles',
    
    # Simulation scene containers
    'S3DISSimScene', 'SimulationStats', 'ResultExporter'
]
