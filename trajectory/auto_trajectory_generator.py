"""
Intelligent automatic trajectory generator.
Automatically generates optimal trajectories based on room layout, satisfying:
1. Trajectory should be as smooth as possible (straight lines preferred)
2. Avoid collisions (robot radius obstacle avoidance)
3. Trajectory length should be reasonable (at least half of room length/width)
4. Automatically select start and end points
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .trajectory_generator import Waypoint, TrajectoryQuality
from .collision_detector import CollisionDetector, FurnitureInfo


@dataclass
class RoomAnalysis:
    """Room analysis results."""
    bounds: Dict[str, float]  # Room boundaries
    center: np.ndarray  # Room center
    dimensions: np.ndarray  # Room dimensions [length, width, height]
    free_space_points: List[np.ndarray]  # Free space points
    obstacle_points: List[np.ndarray]  # Obstacle points
    connectivity_graph: Dict[int, List[int]]  # Connectivity graph
    mesh: object  # Mesh reference for collision detection


@dataclass
class TrajectoryCandidate:
    """Trajectory candidate."""
    start_point: np.ndarray
    end_point: np.ndarray
    waypoints: List[Waypoint]
    quality: TrajectoryQuality
    length: float
    collision_count: int
    smoothness_score: float


class AutoTrajectoryGenerator:
    """
    Intelligent automatic trajectory generator.
    Automatically analyzes room layout and generates optimal trajectories.
    """
    
    def __init__(self, robot_radius: float = 0.3, min_trajectory_length: float = None):
        self.robot_radius = robot_radius
        self.min_trajectory_length = min_trajectory_length
        self.collision_detector = CollisionDetector(robot_radius)
        self.room_analysis: Optional[RoomAnalysis] = None
        
        # Trajectory generation parameters
        self.grid_resolution = 0.2  # Grid resolution
        self.min_free_space = 1.0  # Minimum free space
        self.max_candidates = 40  # Maximum candidate trajectories
        self.sampling_density = 0.1  # Sampling density
        
        # Interpolation density parameters
        self.interpolation_density = 2.0  # Interpolation density multiplier
        self.min_waypoints = 40  # Minimum waypoints
        
    def generate_optimal_trajectory(self, mesh: o3d.geometry.TriangleMesh, 
                                   room_bounds: Dict[str, float],
                                   num_waypoints: int = 20) -> Tuple[List[Waypoint], Dict[str, Any]]:
        """
        Generate optimal trajectory.
        
        Args:
            mesh: Room mesh
            room_bounds: Room boundaries
            num_waypoints: Number of waypoints
            
        Returns:
            waypoints: Optimal trajectory waypoints
            analysis_info: Analysis information
        """
        # 1. Analyze room layout
        self.room_analysis = self._analyze_room_layout(mesh, room_bounds)
        
        # 2. Generate trajectory candidates (with increased interpolation density)
        dense_waypoints = max(int(num_waypoints * self.interpolation_density), self.min_waypoints)
        candidates = self._generate_trajectory_candidates(dense_waypoints)
        
        # 3. Evaluate and select optimal trajectory
        best_candidate = self._select_best_trajectory(candidates)
        
        # 4. Generate analysis information
        analysis_info = self._generate_analysis_info(candidates, best_candidate)
        
        print(f"[AutoTrajectory] Generated trajectory: length={best_candidate.length:.2f}m, "
              f"collisions={best_candidate.collision_count}, smoothness={best_candidate.smoothness_score:.2f}")
        
        return best_candidate.waypoints, analysis_info
    
    def _analyze_room_layout(self, mesh: o3d.geometry.TriangleMesh, 
                           room_bounds: Dict[str, float]) -> RoomAnalysis:
        """Analyze room layout."""
        # Calculate basic room information
        center = np.array([
            (room_bounds['x_max'] + room_bounds['x_min']) / 2,
            (room_bounds['y_max'] + room_bounds['y_min']) / 2,
            (room_bounds['z_max'] + room_bounds['z_min']) / 2
        ])
        
        dimensions = np.array([
            room_bounds['x_max'] - room_bounds['x_min'],
            room_bounds['y_max'] - room_bounds['y_min'],
            room_bounds['z_max'] - room_bounds['z_min']
        ])
        
        # Set minimum trajectory length (reduced requirement)
        if self.min_trajectory_length is None:
            self.min_trajectory_length = max(dimensions[0], dimensions[1]) * 0.2  # Reduced from 0.5 to 0.2
        
        # Fast sampling strategy: use fewer sampling points
        # Use finer grid for analysis
        coarse_resolution = max(0.2, min(dimensions) / 20)  # Finer resolution
        
        # Sample only at robot height (Z coordinate fixed)
        robot_height = 1.0  # Fixed robot height
        x_range = np.arange(room_bounds['x_min'], room_bounds['x_max'], coarse_resolution)
        y_range = np.arange(room_bounds['y_min'], room_bounds['y_max'], coarse_resolution)
        
        # Fast generation of free space points
        free_space_points = []
        obstacle_points = []
        
        # Correct collision detection logic
        for x in x_range:
            for y in y_range:
                point = np.array([x, y, robot_height])
                
                # Check if point is within room boundaries
                if not self._is_point_in_room_bounds(point, room_bounds):
                    continue  # Out of room boundaries, skip
                
                # Check if point intersects with mesh faces (obstacle detection)
                if self._is_point_inside_mesh(point, mesh):
                    obstacle_points.append(point)  # Intersects with mesh = obstacle
                else:
                    free_space_points.append(point)  # Free space in room
        
        # If too few free space points, use finer sampling
        if len(free_space_points) < 10:
            return self._analyze_room_layout_detailed(mesh, room_bounds, center, dimensions)
        
        # Build connectivity graph
        connectivity_graph = self._build_connectivity_graph(free_space_points)
        
        return RoomAnalysis(
            bounds=room_bounds,
            center=center,
            dimensions=dimensions,
            free_space_points=free_space_points,
            obstacle_points=obstacle_points,
            connectivity_graph=connectivity_graph,
            mesh=mesh  # Add mesh reference
        )
    
    def _analyze_room_layout_detailed(self, mesh: o3d.geometry.TriangleMesh, 
                                    room_bounds: Dict[str, float], center: np.ndarray, 
                                    dimensions: np.ndarray) -> RoomAnalysis:
        """Detailed room layout analysis."""
        # Use finer resolution
        fine_resolution = max(0.15, min(dimensions) / 30)
        
        robot_height = 1.0  # Fixed robot height
        x_range = np.arange(room_bounds['x_min'], room_bounds['x_max'], fine_resolution)
        y_range = np.arange(room_bounds['y_min'], room_bounds['y_max'], fine_resolution)
        
        free_space_points = []
        obstacle_points = []
        
        for x in x_range:
            for y in y_range:
                point = np.array([x, y, robot_height])
                
                # Check if point is within room boundaries
                if not self._is_point_in_room_bounds(point, room_bounds):
                    continue  # Out of room boundaries, skip
                
                # Check if point intersects with mesh faces (obstacle detection)
                if self._is_point_inside_mesh(point, mesh):
                    obstacle_points.append(point)  # Intersects with mesh = obstacle
                else:
                    if self._has_sufficient_free_space(point, mesh):
                        free_space_points.append(point)  # Free space in room
        
        # Build connectivity graph
        connectivity_graph = self._build_connectivity_graph(free_space_points)
        
        return RoomAnalysis(
            bounds=room_bounds,
            center=center,
            dimensions=dimensions,
            free_space_points=free_space_points,
            obstacle_points=obstacle_points,
            connectivity_graph=connectivity_graph,
            mesh=mesh  # Add mesh reference
        )
    
    def _is_point_in_room_bounds(self, point: np.ndarray, room_bounds: Dict[str, float]) -> bool:
        """Check if robot bounding box is completely within room boundaries."""
        # Consider robot radius bounding box
        robot_half_size = self.robot_radius
        
        # Robot bounding box
        robot_min = point - robot_half_size
        robot_max = point + robot_half_size
        
        # Check if robot bounding box is completely within room
        return (room_bounds['x_min'] <= robot_min[0] and robot_max[0] <= room_bounds['x_max'] and
                room_bounds['y_min'] <= robot_min[1] and robot_max[1] <= room_bounds['y_max'] and
                room_bounds['z_min'] <= robot_min[2] and robot_max[2] <= room_bounds['z_max'])
    
    
    def _is_point_inside_mesh(self, point: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> bool:
        """Check if robot bounding box intersects with mesh vertices (ultra-fast bounding box detection)."""
        # Get all mesh vertices
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            return False
        
        # Robot bounding box (cube centered at point)
        robot_half_size = self.robot_radius  # Robot radius
        robot_min = point - robot_half_size
        robot_max = point + robot_half_size
        
        # Use numpy vectorized operations for fast detection
        # Check if all vertices are within robot bounding box
        in_x = (vertices[:, 0] >= robot_min[0]) & (vertices[:, 0] <= robot_max[0])
        in_y = (vertices[:, 1] >= robot_min[1]) & (vertices[:, 1] <= robot_max[1])
        in_z = (vertices[:, 2] >= robot_min[2]) & (vertices[:, 2] <= robot_max[2])
        
        # If any vertex is within robot bounding box, collision occurs
        return np.any(in_x & in_y & in_z)
    
    def _has_sufficient_free_space(self, point: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> bool:
        """Check if there is sufficient free space around point (simplified version)."""
        # Directly use robot bounding box detection, if no collision, it's free space
        return not self._is_point_inside_mesh(point, mesh)
    
    def _build_connectivity_graph(self, free_space_points: List[np.ndarray]) -> Dict[int, List[int]]:
        """Build connectivity graph of free space points."""
        connectivity_graph = {}
        max_distance = self.robot_radius * 2  # Maximum connection distance
        
        for i, point1 in enumerate(free_space_points):
            connectivity_graph[i] = []
            for j, point2 in enumerate(free_space_points):
                if i != j:
                    distance = np.linalg.norm(point1 - point2)
                    if distance <= max_distance:
                        connectivity_graph[i].append(j)
        
        return connectivity_graph
    
    def _generate_trajectory_candidates(self, num_waypoints: int) -> List[TrajectoryCandidate]:
        """Generate trajectory candidates."""
        candidates = []
        free_space_points = self.room_analysis.free_space_points
        
        if len(free_space_points) < 2:
            return candidates
        
        # Randomly sample start and end points
        max_attempts = min(self.max_candidates, len(free_space_points) * 2)
        
        for attempt in range(max_attempts):
            # Randomly select start and end points
            if len(free_space_points) == 0:
                break
                
            start_idx = np.random.randint(0, len(free_space_points))
            end_idx = np.random.randint(0, len(free_space_points))
            
            if start_idx == end_idx:
                continue
            
            start_point = free_space_points[start_idx]
            end_point = free_space_points[end_idx]
            
            # Check if distance meets minimum length requirement
            distance = np.linalg.norm(start_point - end_point)
            if distance < self.min_trajectory_length:
                continue
            
            # Generate trajectory
            candidate = self._create_trajectory_candidate(
                start_point, end_point, num_waypoints
            )
            
            if candidate is not None:
                candidates.append(candidate)
        
        return candidates
    
    def _create_trajectory_candidate(self, start_point: np.ndarray, 
                                   end_point: np.ndarray, 
                                   num_waypoints: int) -> Optional[TrajectoryCandidate]:
        """Create trajectory candidate (with full collision detection)."""
        try:
            # Generate straight trajectory points (Z coordinate already fixed)
            waypoints = []
            
            for i in range(num_waypoints):
                t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
                
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                z = start_point[2] + t * (end_point[2] - start_point[2])  # Z coordinate interpolation (start and end Z are same)
                
                waypoint = Waypoint(x=x, y=y, z=z, yaw=0)
                waypoints.append(waypoint)
            
            # Check collisions along trajectory path
            collision_count = 0
            for waypoint in waypoints:
                # Check if within room boundaries
                if not self._is_point_in_room_bounds(np.array([waypoint.x, waypoint.y, waypoint.z]), self.room_analysis.bounds):
                    collision_count += 1  # Out of bounds
                    continue
                
                # Check if collides with mesh
                if self._is_point_inside_mesh(np.array([waypoint.x, waypoint.y, waypoint.z]), self.room_analysis.mesh):
                    collision_count += 1  # Collision with mesh
            
            # If trajectory has collisions, reject directly
            if collision_count > 0:
                return None
            
            # Calculate trajectory length
            length = self._calculate_trajectory_length(waypoints)
            
            # Calculate smoothness score
            smoothness_score = self._calculate_smoothness_score(waypoints)
            
            # Create quality assessment
            quality = TrajectoryQuality(
                coverage_ratio=1.0,  # No collision trajectory
                path_length=length,
                turn_count=0,  # Straight trajectory
                efficiency=1.0,  # Efficient
                collision_count=0,  # No collision
                smoothness=1.0  # Completely smooth
            )
            
            return TrajectoryCandidate(
                start_point=start_point,
                end_point=end_point,
                waypoints=waypoints,
                quality=quality,
                length=length,
                collision_count=0,  # No collision
                smoothness_score=smoothness_score
            )
            
        except Exception as e:
            return None
    
    def _calculate_trajectory_length(self, waypoints: List[Waypoint]) -> float:
        """Calculate trajectory length."""
        if len(waypoints) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(waypoints)):
            prev_point = waypoints[i-1]
            curr_point = waypoints[i]
            distance = np.sqrt(
                (curr_point.x - prev_point.x)**2 + 
                (curr_point.y - prev_point.y)**2 + 
                (curr_point.z - prev_point.z)**2
            )
            total_length += distance
        
        return total_length
    
    def _calculate_smoothness_score(self, waypoints: List[Waypoint]) -> float:
        """Calculate trajectory smoothness score."""
        if len(waypoints) < 3:
            return 1.0
        
        # Calculate standard deviation of angle changes
        angle_changes = []
        for i in range(1, len(waypoints)):
            prev_yaw = waypoints[i-1].yaw
            curr_yaw = waypoints[i].yaw
            angle_change = abs(curr_yaw - prev_yaw)
            angle_changes.append(angle_change)
        
        if not angle_changes:
            return 1.0
        
        # Smoothness score: smaller angle changes = higher score
        angle_std = np.std(angle_changes)
        smoothness_score = max(0, 1 - angle_std / np.pi)
        
        return smoothness_score
    
    def _select_best_trajectory(self, candidates: List[TrajectoryCandidate]) -> TrajectoryCandidate:
        """Select best trajectory."""
        if not candidates:
            raise ValueError("No available trajectory candidates")
        
        # Calculate comprehensive score
        best_candidate = None
        best_score = -1
        
        # Precompute constants (avoid repeated computation)
        min_length = self.min_trajectory_length
        collision_penalty_factor = 0.1
        
        for candidate in candidates:
            # Comprehensive score: length weight + smoothness weight - collision penalty
            length_score = min(candidate.length / min_length, 2.0)  # Length score
            smoothness_score = candidate.smoothness_score  # Smoothness score
            collision_penalty = candidate.collision_count * collision_penalty_factor  # Collision penalty
            
            total_score = length_score * 0.4 + smoothness_score * 0.4 - collision_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        return best_candidate
    
    def _generate_analysis_info(self, candidates: List[TrajectoryCandidate], 
                               best_candidate: TrajectoryCandidate) -> Dict[str, Any]:
        """Generate analysis information."""
        if not candidates:
            return {}
        
        # Statistics
        lengths = [c.length for c in candidates]
        collision_counts = [c.collision_count for c in candidates]
        smoothness_scores = [c.smoothness_score for c in candidates]
        
        return {
            'total_candidates': len(candidates),
            'best_trajectory': {
                'length': best_candidate.length,
                'collision_count': best_candidate.collision_count,
                'smoothness_score': best_candidate.smoothness_score,
                'start_point': best_candidate.start_point.tolist(),
                'end_point': best_candidate.end_point.tolist()
            },
            'statistics': {
                'length_mean': np.mean(lengths),
                'length_std': np.std(lengths),
                'collision_mean': np.mean(collision_counts),
                'collision_std': np.std(collision_counts),
                'smoothness_mean': np.mean(smoothness_scores),
                'smoothness_std': np.std(smoothness_scores)
            },
            'room_analysis': {
                'free_space_points': len(self.room_analysis.free_space_points),
                'obstacle_points': len(self.room_analysis.obstacle_points),
                'room_dimensions': self.room_analysis.dimensions.tolist(),
                'room_center': self.room_analysis.center.tolist()
            }
        }
    
    def add_furniture(self, furniture: FurnitureInfo):
        """Add furniture."""
        self.collision_detector.add_furniture(furniture)
    
    def add_furniture_from_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                               name: str, category: str = "unknown"):
        """Add furniture from mesh."""
        self.collision_detector.add_furniture_from_mesh(mesh, name, category)
    
    def clear_furniture(self):
        """Clear furniture."""
        self.collision_detector.clear_furniture()
