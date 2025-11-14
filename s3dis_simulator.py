#!/usr/bin/env python3
"""
S3DIS simulation entry point.
Integrates all components to provide the full simulation workflow.
"""

import numpy as np
import open3d as o3d
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

# Import S3DIS modules
try:
    from containers import S3DISScene, S3DISSimScene, S3DISSimFrame, ScanQuality
    from lidar import Indoor8LineLidarIntrinsics, IndoorLidar, DualAxisLidarIntrinsics, DualAxisLidar, create_lidar
    from trajectory import SmartTrajectoryGenerator, PathType, Waypoint, CollisionDetector
    from trajectory.auto_trajectory_generator import AutoTrajectoryGenerator
    from raycast_engine import RaycastEngineCPU, RaycastEngineGPU
    from visualization import TrajectoryVisualizer, ScanResultVisualizer, MeshVisualizer, S3DISVisualizer
except ImportError:
    # Fallback to absolute import when relative import fails
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from containers import S3DISScene, S3DISSimScene, S3DISSimFrame, ScanQuality
    from lidar import Indoor8LineLidarIntrinsics, IndoorLidar, DualAxisLidarIntrinsics, DualAxisLidar, create_lidar
    from trajectory import SmartTrajectoryGenerator, PathType, Waypoint, CollisionDetector
    from trajectory.auto_trajectory_generator import AutoTrajectoryGenerator
    from raycast_engine import RaycastEngineCPU, RaycastEngineGPU
    from visualization import TrajectoryVisualizer, ScanResultVisualizer, MeshVisualizer, S3DISVisualizer


class S3DISSimulator:
    """High-level simulator that orchestrates the full S3DIS scanning workflow."""
    
    def __init__(self, config: Dict[str, Any], use_dense_lidar: bool = False, use_blk2go: bool = False):
        self.config = config
        self.use_dense_lidar = use_dense_lidar
        self.use_blk2go = use_blk2go
        self.scene: Optional[S3DISScene] = None
        self.lidar_config = None  # LiDAR configuration selected at runtime
        self.raycast_engine = None
        self.trajectory_generator: Optional[SmartTrajectoryGenerator] = None
        self.auto_trajectory_generator: Optional[AutoTrajectoryGenerator] = None
        self.collision_detector: Optional[CollisionDetector] = None
        
        # Visualization utilities
        self.trajectory_visualizer = TrajectoryVisualizer()
        self.scan_visualizer = ScanResultVisualizer()
        self.mesh_visualizer = MeshVisualizer()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LiDAR, raycast engine, and collision detector."""
        if self.use_blk2go:
            self.lidar_config = DualAxisLidarIntrinsics.create_blk2go_dual_axis()
        elif self.use_dense_lidar:
            self.lidar_config = Indoor8LineLidarIntrinsics.create_dense_32line()
        else:
            self.lidar_config = Indoor8LineLidarIntrinsics.create_standard_8line()
        
        # Initialize raycast engine
        use_gpu = self.config.get('raycast_engine', {}).get('use_gpu', False)
        if use_gpu:
            try:
                self.raycast_engine = RaycastEngineGPU()
            except Exception:
                self.raycast_engine = RaycastEngineCPU()
        else:
            self.raycast_engine = RaycastEngineCPU()
        
        # Initialize collision detector
        self.collision_detector = CollisionDetector()
        
    
    def load_scene(self, scene_path: str, scene_name: Optional[str] = None) -> S3DISScene:
        """
        Load a reconstructed scene mesh and prepare trajectory generators.
        
        Args:
            scene_path: Path to the mesh file.
            scene_name: Optional name override.
        
        Returns:
            The instantiated `S3DISScene`.
        """
        mesh = o3d.io.read_triangle_mesh(scene_path)
        if len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh file: {scene_path}")
        
        # Compute room bounds
        vertices = np.asarray(mesh.vertices)
        room_bounds = {
            'x_min': float(vertices[:, 0].min()),
            'x_max': float(vertices[:, 0].max()),
            'y_min': float(vertices[:, 1].min()),
            'y_max': float(vertices[:, 1].max()),
            'z_min': float(vertices[:, 2].min()),
            'z_max': float(vertices[:, 2].max())
        }
        
        # Derive scene name if not provided
        if scene_name is None:
            scene_name = Path(scene_path).stem
        
        # Build RoomBounds helper
        try:
            from containers.s3dis_scene import RoomBounds
        except ImportError:
            from containers.s3dis_scene import RoomBounds
        room_bounds_obj = RoomBounds(
            x_min=room_bounds['x_min'], x_max=room_bounds['x_max'],
            y_min=room_bounds['y_min'], y_max=room_bounds['y_max'],
            z_min=room_bounds['z_min'], z_max=room_bounds['z_max']
        )
        
        self.scene = S3DISScene(scene_name, mesh, room_bounds=room_bounds_obj)
        
        # Initialize trajectory generators
        self.trajectory_generator = SmartTrajectoryGenerator(
            room_bounds, 
            robot_height=self.config.get('trajectory', {}).get('robot_height', 1.0)
        )
        self.auto_trajectory_generator = AutoTrajectoryGenerator(robot_radius=0.15)  # Reduced robot radius for narrow spaces
        
        return self.scene
    
    def generate_auto_trajectory(self, num_waypoints: int = 20) -> Tuple[List[Waypoint], Dict[str, Any]]:
        """
        Generate an automated trajectory that maximizes coverage.
        
        Args:
            num_waypoints: Number of waypoints in the candidate paths.
        
        Returns:
            waypoints: The selected waypoint sequence.
            analysis_info: Diagnostic information returned by the generator.
        """
        if self.auto_trajectory_generator is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        
        if self.scene is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        
        # Gather mesh and bounds
        mesh = self.scene.room_mesh
        room_bounds = {
            'x_min': self.scene.room_bounds.x_min,
            'x_max': self.scene.room_bounds.x_max,
            'y_min': self.scene.room_bounds.y_min,
            'y_max': self.scene.room_bounds.y_max,
            'z_min': self.scene.room_bounds.z_min,
            'z_max': self.scene.room_bounds.z_max
        }
        
        # Compute optimal trajectory
        waypoints, analysis_info = self.auto_trajectory_generator.generate_optimal_trajectory(
            mesh=mesh,
            room_bounds=room_bounds,
            num_waypoints=num_waypoints
        )
        
        return waypoints, analysis_info
    
    def add_furniture(self, furniture_mesh: o3d.geometry.TriangleMesh, 
                     name: str, category: str = "unknown"):
        """
        Register furniture for collision checking during trajectory planning.
        Mesh geometry already creates occlusion during ray casting; this step
        only informs the trajectory planner.
        """
        if self.collision_detector is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        
        self.collision_detector.add_furniture_from_mesh(furniture_mesh, name, category)
    def generate_trajectory(self, start_point: Tuple[float, float, float],
                          end_point: Tuple[float, float, float],
                          path_type: PathType = PathType.STRAIGHT,
                          num_waypoints: int = 20) -> Tuple[List[Waypoint], Dict[str, Any]]:
        """
        Generate a trajectory between two poses.
        
        Args:
            start_point: Start pose (x, y, z).
            end_point: End pose (x, y, z).
            path_type: Desired trajectory type.
            num_waypoints: Number of waypoints to sample.
        
        Returns:
            waypoints: Ordered list of waypoints.
            quality: Trajectory quality metrics.
        """
        if self.trajectory_generator is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        
        # Generate the trajectory
        waypoints, quality = self.trajectory_generator.generate_trajectory(
            start_point=start_point,
            end_point=end_point,
            path_type=path_type,
            num_waypoints=num_waypoints
        )
        
        # Re-run trajectory generation with collision detector if furniture is present
        if self.collision_detector and self.collision_detector.furniture_list:
            self.trajectory_generator.collision_detector = self.collision_detector
            waypoints, quality = self.trajectory_generator.generate_trajectory(
                start_point=start_point,
                end_point=end_point,
                path_type=path_type,
                num_waypoints=num_waypoints
            )
        
        return waypoints, quality.to_dict()
    
    def run_simulation(self, waypoints: List[Waypoint]) -> S3DISSimScene:
        """
        Execute the full simulation given a trajectory.
        
        Args:
            waypoints: Trajectory waypoint list.
        
        Returns:
            sim_scene: Simulation results container.
        """
        if self.scene is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        
        if self.raycast_engine is None:
            raise ValueError("Raycast engine is not initialized.")
        
        # Create simulation scene (uses mesh and optional S3DIS data for colors)
        sim_scene = S3DISSimScene(
            scene_name=self.scene.scene_name, 
            simulation_config=self.config, 
            mesh=self.scene.room_mesh,
            s3dis_data_root=self.config.get('s3dis_data_root', None),
            area=self.config.get('area', None),
            room=self.config.get('room', None)
        )
        
        # Track runtime
        start_time = time.time()
        
        # Pre-compute constants
        total_points_per_scan = self.lidar_config.get_total_points_per_scan()
        room_volume = self.scene.room_bounds.get_volume()
        
        # Simulate each frame
        for i, waypoint in enumerate(waypoints):
            # Create LiDAR instance for current pose
            lidar_pose = waypoint.to_pose_matrix()
            lidar = create_lidar(self.lidar_config, lidar_pose)
            
            # Perform ray casting
            try:
                points, incident_angles = self.raycast_engine.lidar_intersect_mesh(
                    lidar, self.scene.room_mesh
                )
                
                # Normalize return format
                if isinstance(points, tuple):
                    points, incident_angles = points
                else:
                    incident_angles = np.zeros(len(points))
                
            except Exception:
                points = np.empty((0, 3))
                incident_angles = np.empty(0)
            
            # Compute scan quality metrics
            scan_quality = ScanQuality(
                coverage_ratio=len(points) / total_points_per_scan,
                num_points=len(points),
                incident_angle_mean=np.mean(incident_angles) if len(incident_angles) > 0 else 0,
                incident_angle_std=np.std(incident_angles) if len(incident_angles) > 0 else 0,
                scan_density=len(points) / room_volume,
                range_mean=np.mean(np.linalg.norm(points, axis=1)) if len(points) > 0 else 0,
                range_std=np.std(np.linalg.norm(points, axis=1)) if len(points) > 0 else 0
            )
            
            # Store frame
            sim_frame = S3DISSimFrame(i, points, incident_angles, scan_quality)
            sim_scene.append_frame(sim_frame)
            
        # Record total runtime
        simulation_time = time.time() - start_time
        
        # Aggregate statistics
        sim_scene.compute_statistics(simulation_time)
        
        return sim_scene
    
    def save_results(self, sim_scene: S3DISSimScene, output_dir: Path,
                    waypoints: Optional[List[Waypoint]] = None,
                    save_visualizations: bool = True):
        """
        Persist simulation outputs and optional visualizations.
        
        Args:
            sim_scene: Simulation scene to save.
            output_dir: Directory for serialized outputs.
            waypoints: Optional trajectory to visualize.
            save_visualizations: Whether to generate plots.
        """
        sim_scene.save_results(output_dir)
        
        # Export visualization assets
        if save_visualizations:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Propagate output directory to visualizers
            self.trajectory_visualizer.output_dir = vis_dir
            self.scan_visualizer.output_dir = vis_dir
            self.mesh_visualizer.output_dir = vis_dir
            
            # Instantiate helper visualizer
            s3dis_visualizer = S3DISVisualizer(output_dir)
            
            if waypoints and self.scene:
                self.trajectory_visualizer.visualize_trajectory_2d(
                    waypoints, self.scene.room_bounds.to_dict(),
                    title=f"Robot trajectory - {self.scene.scene_name}"
                )
                self.trajectory_visualizer.visualize_trajectory_3d(
                    waypoints, self.scene.room_bounds.to_dict(),
                    title=f"Robot trajectory 3D - {self.scene.scene_name}"
                )
            
            self.scan_visualizer.visualize_scan_statistics(sim_scene)
            self.scan_visualizer.visualize_scan_quality_evolution(sim_scene)
            self.scan_visualizer.create_scan_summary_report(sim_scene)
            '''
            # Optional mesh visualization (disabled by default to save time)
            if self.scene and self.config.get('enable_mesh_visualization', False):
                self.mesh_visualizer.visualize_room_mesh(
                    self.scene.room_mesh, self.scene.room_bounds.to_dict(),
                    title=f"Room mesh - {self.scene.scene_name}"
                )
                self.mesh_visualizer.visualize_mesh_statistics(
                    self.scene.room_mesh, self.scene.room_bounds.to_dict()
                )
            '''
            if self.scene:
                try:
                    from visualization.s3dis_visualizer import create_visualization_summary
                    viz_results = s3dis_visualizer.generate_all_visualizations(
                        sim_scene, self.scene, num_sample_frames=5
                    )
                    create_visualization_summary(viz_results, output_dir)
                except Exception:
                    pass
                '''
                if waypoints:
                    self.mesh_visualizer.visualize_mesh_with_trajectory(
                        self.scene.room_mesh, waypoints, self.scene.room_bounds.to_dict(),
                        title=f"Room and trajectory - {self.scene.scene_name}"
                    )
                '''
        
    def run_complete_simulation(self, scene_path: str,
                              start_point: Tuple[float, float, float],
                              end_point: Tuple[float, float, float],
                              path_type: PathType = PathType.STRAIGHT,
                              num_waypoints: int = 20,
                              output_dir: Optional[Path] = None,
                              scene_name: Optional[str] = None) -> S3DISSimScene:
        """
        Load a scene, generate a trajectory, run the simulation, and save outputs.
        
        Args:
            scene_path: Path to the reconstructed mesh file.
            start_point: Start pose.
            end_point: End pose.
            path_type: Desired trajectory pattern.
            num_waypoints: Number of waypoints for the trajectory.
            output_dir: Directory to store results.
            scene_name: Optional scene name override.
        
        Returns:
            sim_scene: Completed simulation results.
        """
        # 1. Load scene assets
        self.load_scene(scene_path, scene_name)
        
        # 2. Generate trajectory
        waypoints, trajectory_quality = self.generate_trajectory(
            start_point, end_point, path_type, num_waypoints
        )
        
        # 3. Run simulation
        sim_scene = self.run_simulation(waypoints)
        
        # 4. Persist results
        if output_dir is None:
            output_dir = Path("s3dis_simulation_results")
        
        self.save_results(sim_scene, output_dir, waypoints)
        
        return sim_scene
    
    def run_auto_simulation(self, scene_path: str,
                           num_waypoints: int = 20,
                           output_dir: Optional[Path] = None,
                           scene_name: Optional[str] = None) -> S3DISSimScene:
        """
        Full workflow using automatically generated trajectory.
        
        Args:
            scene_path: Path to the reconstructed mesh file.
            num_waypoints: Number of candidate waypoints.
            output_dir: Directory to store results.
            scene_name: Optional scene name override.
        
        Returns:
            sim_scene: Completed simulation results.
        """
        # 1. Load scene assets
        self.load_scene(scene_path, scene_name)
        
        # 2. Generate automated trajectory
        waypoints, analysis_info = self.generate_auto_trajectory(num_waypoints)
        
        # 3. Run simulation
        sim_scene = self.run_simulation(waypoints)
        
        # 4. Persist results
        if output_dir is None:
            output_dir = Path("s3dis_auto_simulation_results")
        
        self.save_results(sim_scene, output_dir, waypoints)
        
        # Save analysis metadata
        analysis_file = output_dir / "trajectory_analysis.json"
        import json
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_info, f, indent=2, ensure_ascii=False)
        
        return sim_scene


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_default_config() -> Dict[str, Any]:
    """Load the default simulator configuration."""
    config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    return load_config(str(config_path))


def create_simulator_from_config(config_path: Optional[str] = None) -> S3DISSimulator:
    """
    Create a simulator instance from a configuration file.
    
    Args:
        config_path: Optional path to a configuration file. Uses the default
            configuration when omitted.
    
    Returns:
        simulator: Configured simulator.
    """
    if config_path is None:
        config = load_default_config()
    else:
        config = load_config(config_path)
    
    return S3DISSimulator(config)


def run_single_scene_simulation(scene_path: str, scene_name: str = None, 
                               num_waypoints: int = 20, output_base_dir: str = 'simulation_results',
                               use_gpu: bool = False, robot_height: float = 1.0,
                               use_dense_lidar: bool = False, use_blk2go: bool = True,
                               enable_mesh_visualization: bool = False,
                               use_auto_trajectory: bool = True,
                               s3dis_data_root: str = "S3DIS/raw/S3DIS/data/Stanford3dDataset_v1.2_Aligned_Version",
                               area: str = None, room: str = None) -> S3DISSimScene:
    """
    Convenience wrapper that processes a single scene end-to-end.
    
    Args:
        scene_path: Path to the reconstructed mesh file.
        scene_name: Optional scene identifier override.
        num_waypoints: Number of waypoints.
        output_base_dir: Root directory for outputs.
        use_gpu: Enable GPU raycast engine.
        robot_height: Robot sensor height.
        use_dense_lidar: Use dense 32-line LiDAR.
        use_blk2go: Use Leica BLK2GO dual-axis LiDAR.
        enable_mesh_visualization: Toggle mesh visualization.
        use_auto_trajectory: Use automatic trajectory generation.
        s3dis_data_root: Root directory for S3DIS annotations.
        area: Area name for annotation lookup.
        room: Room name for annotation lookup.
    
    Returns:
        sim_scene: Simulation results.
    """
    # Derive scene name when not provided
    if scene_name is None:
        scene_name = Path(scene_path).parent.name
    
    # Create scene-specific output directory
    output_dir = Path(output_base_dir) / scene_name
    
    # Build minimal configuration
    config = {
        'raycast_engine': {'use_gpu': use_gpu},
        'trajectory': {'robot_height': robot_height},
        'enable_mesh_visualization': enable_mesh_visualization,
        's3dis_data_root': s3dis_data_root,
        'area': area,
        'room': room
    }
    
    simulator = S3DISSimulator(config, use_dense_lidar=use_dense_lidar, use_blk2go=use_blk2go)
    
    if use_auto_trajectory:
        print(f"[Simulation] Using auto-generated trajectory for {scene_name}.")
        sim_scene = simulator.run_auto_simulation(
            scene_path=scene_path,
            num_waypoints=num_waypoints,
            output_dir=output_dir
        )
    else:
        print(f"[Simulation] Using manual trajectory for {scene_name}.")
        start_point = (-16.0, 35.0, 0.5)
        end_point = (-20.0, 35.0, 0.5)
        path_type = PathType.STRAIGHT
        
    sim_scene = simulator.run_complete_simulation(
        scene_path=scene_path,
        start_point=start_point,
        end_point=end_point,
        path_type=path_type,
        num_waypoints=num_waypoints,
        output_dir=output_dir
    )
    
    print(f"[Simulation] Scene {scene_name} completed.")
    return sim_scene


def find_available_scenes(reconstruction_dir: str = "outputs/s3dis_reconstruction/reconstruction_results") -> List[Tuple[str, str, str]]:
    """
    Discover reconstructed scenes ready for simulation.
    
    Args:
        reconstruction_dir: Directory containing reconstruction outputs.
    
    Returns:
        List of tuples (mesh_path, scene_name, area_room_id).
    """
    reconstruction_path = Path(reconstruction_dir)
    if not reconstruction_path.exists():
        print(f"[Discovery] Reconstruction directory not found: {reconstruction_dir}")
        return []
    
    available_scenes = []
    
    # Traverse sub-directories for meshes
    for scene_dir in reconstruction_path.iterdir():
        if scene_dir.is_dir():
            # Find mesh files
            mesh_files = list(scene_dir.glob("mesh_*.ply"))
            if mesh_files:
                # Prefer dense mesh when available
                dense_mesh = scene_dir / "mesh_dense.ply"
                if dense_mesh.exists():
                    mesh_path = str(dense_mesh)
                else:
                    mesh_path = str(mesh_files[0])
                
                scene_name = scene_dir.name
                available_scenes.append((mesh_path, scene_name, scene_name))
                print(f"[Discovery] Found scene {scene_name}: {mesh_path}")
    
    print(f"[Discovery] Total scenes discovered: {len(available_scenes)}")
    return available_scenes


def main():
    """Batch processing entry point for multiple scenes."""
    # Hard-coded configuration
    reconstruction_dir = "outputs/s3dis_reconstruction/reconstruction_results"
    num_waypoints = 20
    output_base_dir = 'simulation_results'
    
    # Simulator options
    use_gpu = False
    robot_height = 1.0
    use_dense_lidar = False
    use_blk2go = True
    enable_mesh_visualization = False
    
    # Trajectory mode
    use_auto_trajectory = True
    
    # S3DIS data root
    s3dis_data_root = "S3DIS/raw/S3DIS/data/Stanford3dDataset_v1.2_Aligned_Version"
    
    print("=== S3DIS batch simulation ===")
    print(f"Reconstruction directory: {reconstruction_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Trajectory mode: {'auto' if use_auto_trajectory else 'manual'}")
    print(f"Number of waypoints: {num_waypoints}")
    print(f"Raycast engine: {'GPU' if use_gpu else 'CPU'}")
    print(f"LiDAR profile: {'BLK2GO dual-axis' if use_blk2go else 'dense 32-line' if use_dense_lidar else 'standard 8-line'}")
    
    available_scenes = find_available_scenes(reconstruction_dir)
    
    if not available_scenes:
        print("[Batch] No scenes found. Verify the reconstruction directory.")
        return
    
    successful_scenes = []
    failed_scenes = []
    skipped_scenes = []
    
    pbar = tqdm(available_scenes, desc="Simulation progress", unit="scene", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    start_time = time.time()
    
    for i, (scene_path, scene_name, area_room) in enumerate(pbar, 1):
        pbar.set_description(f"Processing {scene_name} ({i}/{len(available_scenes)})")
        print(f"\n{'='*60}")
        print(f"[Batch] Processing scene {i}/{len(available_scenes)}: {scene_name}")
        print(f"{'='*60}")
        
        scene_output_dir = Path(output_base_dir) / scene_name
        ply_file = scene_output_dir / "combined_pointcloud_with_label.ply"
        stats_file = scene_output_dir / "simulation_statistics.txt"
        
        if ply_file.exists() and stats_file.exists():
            print(f"[Batch] Scene {scene_name} already processed. Skipping.")
            print(f"    PLY: {ply_file}")
            print(f"    Stats: {stats_file}")
            skipped_scenes.append(scene_name)
            continue
        elif ply_file.exists() or stats_file.exists():
            print(f"[Batch] Scene {scene_name} partially processed. Re-running.")
            print(f"    Existing file: {ply_file if ply_file.exists() else stats_file}")
        
        try:
            # Extract area and room details
            if '_' in area_room:
                parts = area_room.split('_')
                if len(parts) >= 3 and parts[0] == 'Area':
                    area = f"{parts[0]}_{parts[1]}"  # Area_1
                    room = '_'.join(parts[2:])       # conferenceRoom_2
                else:
                    area, room = area_room.split('_', 1)
            else:
                area, room = area_room, area_room
            
            sim_scene = run_single_scene_simulation(
                scene_path=scene_path,
                scene_name=scene_name,
                num_waypoints=num_waypoints,
                output_base_dir=output_base_dir,
                use_gpu=use_gpu,
                robot_height=robot_height,
                use_dense_lidar=use_dense_lidar,
                use_blk2go=use_blk2go,
                enable_mesh_visualization=enable_mesh_visualization,
                use_auto_trajectory=use_auto_trajectory,
                s3dis_data_root=s3dis_data_root,
                area=area,
                room=room
            )
            
            successful_scenes.append((scene_name, sim_scene))
            print(f"[Batch] Scene {scene_name} succeeded.")
            pbar.update(1)
            
        except Exception as e:
            failed_scenes.append((scene_name, str(e)))
            print(f"[Batch] Scene {scene_name} failed: {e}")
            pbar.update(1)
    
    pbar.close()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("[Batch] Summary")
    print(f"{'='*60}")
    print(f"Total scenes: {len(available_scenes)}")
    print(f"Completed: {len(successful_scenes)}")
    print(f"Skipped: {len(skipped_scenes)}")
    print(f"Failed: {len(failed_scenes)}")
    print(f"Success rate: {len(successful_scenes)/len(available_scenes)*100:.1f}%")
    print(f"Skip rate: {len(skipped_scenes)/len(available_scenes)*100:.1f}%")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average per scene: {total_time/len(available_scenes):.1f}s")
    
    if successful_scenes:
        print("\n[Batch] Completed scenes:")
        for scene_name, sim_scene in successful_scenes:
            total_points = sum(len(frame.points) for frame in sim_scene.frames)
            print(f"  - {scene_name}: {len(sim_scene.frames)} frames, {total_points:,} points")
    
    if skipped_scenes:
        print("\n[Batch] Skipped scenes:")
        for scene_name in skipped_scenes:
            print(f"  - {scene_name}")
    
    if failed_scenes:
        print("\n[Batch] Failed scenes:")
        for scene_name, error in failed_scenes:
            print(f"  - {scene_name}: {error}")
    
    print(f"\n[Batch] Completed. Results saved to {output_base_dir}.")


def main_single():
    """Helper entry point for a single, hard-coded scene."""
    # Hard-coded configuration
    scene_path = "outputs/s3dis_reconstruction/reconstruction_results/Area_1_office_1/mesh_dense.ply"
    num_waypoints = 20
    output_dir = Path('simulation_results')
    
    # Simulator options
    use_gpu = False
    robot_height = 1.0
    use_dense_lidar = False
    use_blk2go = True
    enable_mesh_visualization = False
    
    # Trajectory mode
    use_auto_trajectory = True
    
    # S3DIS data configuration
    s3dis_data_root = "S3DIS/raw/S3DIS/data/Stanford3dDataset_v1.2_Aligned_Version"
    area = "Area_1"
    room = "office_1"
    
    sim_scene = run_single_scene_simulation(
        scene_path=scene_path,
        scene_name="Area_1_office_1",
        num_waypoints=num_waypoints,
        output_base_dir='simulation_results',
        use_gpu=use_gpu,
        robot_height=robot_height,
        use_dense_lidar=use_dense_lidar,
        use_blk2go=use_blk2go,
        enable_mesh_visualization=enable_mesh_visualization,
        use_auto_trajectory=use_auto_trajectory,
        s3dis_data_root=s3dis_data_root,
        area=area,
        room=room
    )
    
    print("[Single] Simulation completed.")
    
    # Print configuration summary
    print("\n=== Configuration ===")
    print(f"Scene path: {scene_path}")
    print(f"Trajectory mode: {'auto' if use_auto_trajectory else 'manual'}")
    print(f"Waypoint count: {num_waypoints}")
    print(f"Output directory: {output_dir}")
    print(f"Raycast engine: {'GPU' if use_gpu else 'CPU'}")
    print(f"LiDAR profile: {'BLK2GO dual-axis' if use_blk2go else 'dense 32-line' if use_dense_lidar else 'standard 8-line'}")
    print(f"Robot height: {robot_height}m")
    print(f"S3DIS root: {s3dis_data_root}")
    print(f"S3DIS area: {area}")
    print(f"S3DIS room: {room}")
    
    if sim_scene and len(sim_scene.frames) > 0:
        print("\n=== Simulation Stats ===")
        total_scanned_points = sum(len(frame.points) for frame in sim_scene.frames)
        print(f"Frames: {len(sim_scene.frames)}")
        print(f"Total points: {total_scanned_points:,}")
        print(f"Average points per frame: {total_scanned_points // len(sim_scene.frames):,}")
        
        if hasattr(sim_scene, 'trajectory_quality'):
            quality = sim_scene.trajectory_quality
            print("Trajectory quality:")
            print(f"  - Coverage: {quality.coverage_ratio:.2%}")
            print(f"  - Path length: {quality.path_length:.2f}m")
            print(f"  - Turns: {quality.turn_count}")
            print(f"  - Efficiency: {quality.efficiency:.2%}")
            print(f"  - Collisions: {quality.collision_count}")
            print(f"  - Smoothness: {quality.smoothness:.2f}")
    
    print("\n=== Output Files ===")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        print(f"Directory: {output_dir}")
        print(f"File count: {len(files)}")
        for file in files[:10]:
            print(f"  - {file.name}")
        if len(files) > 10:
            print(f"  ... {len(files) - 10} more files")


if __name__ == "__main__":
    main()
