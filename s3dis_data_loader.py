#!/usr/bin/env python3
"""
S3DIS data loading and preprocessing utilities.
Provides a standalone workflow configurable via YAML.
"""

import argparse
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class S3DISLoader:
    """Loader for the S3DIS dataset."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.area_names = [f"Area_{i}" for i in range(1, 7)]
    
    def load_room_data(self, area_name: str, room_name: str):
        """
        Load point cloud data for a single room.
        
        Args:
            area_name: Area identifier (for example, "Area_1").
            room_name: Room identifier (for example, "office_1").
            
        Returns:
            points: (N, 3) point coordinates.
            colors: (N, 3) RGB colors in [0, 1].
            labels: (N,) semantic labels if available.
        """
        print(f"[Loader] Loading {area_name}/{room_name} ...")
        
        room_path = self.data_root / area_name / room_name
        
        if not room_path.exists():
            raise FileNotFoundError(f"Room data not found: {room_path}")
        
        # Try multiple formats
        txt_files = list(room_path.glob("*.txt"))
        if txt_files:
            data_file = txt_files[0]
            print(f"[Loader] Loading TXT file: {data_file}")
            data = np.loadtxt(data_file)
            points = data[:, :3]
            colors = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else np.ones((len(data), 3))
            labels = data[:, 6].astype(int) if data.shape[1] > 6 else None
        else:
            # Fallback to .npy
            npy_file = room_path / "points.npy"
            if npy_file.exists():
                data = np.load(npy_file)
                points = data[:, :3]
                colors = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else np.ones((len(data), 3))
                labels = data[:, 6].astype(int) if data.shape[1] > 6 else None
            else:
                raise FileNotFoundError(f"Point cloud data missing: {room_path}")
        
        print(f"[Loader] Loaded {len(points)} points.")
        return points, colors, labels
    
    def get_available_rooms(self, area_name: str):
        """Return sorted list of available rooms in the given area."""
        area_path = self.data_root / area_name
        if not area_path.exists():
            return []
        rooms = [d.name for d in area_path.iterdir() if d.is_dir()]
        return sorted(rooms)


class S3DISPreprocessor:
    """Preprocess S3DIS point clouds using configurable operations."""
    
    def __init__(self, config_dict):
        self.config = config_dict['preprocessing']
        # Force headless backend for matplotlib
        plt.switch_backend('Agg')
    
    def preprocess_pointcloud(self, points, colors=None, labels=None):
        """Run the configured preprocessing pipeline on a point cloud."""
        print(f"[Preprocess] Starting with {len(points)} points.")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if self.config['remove_outliers']:
            print("[Preprocess] Removing statistical outliers ...")
            pcd_clean, outlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.config['outlier_nb_neighbors'],
                std_ratio=self.config['outlier_std_ratio']
            )
            n_outliers = len(outlier_indices)
            n_inliers = len(pcd_clean.points)
            print(f"[Preprocess] Removed {n_outliers} outliers, kept {n_inliers} points.")
            
            if labels is not None:
                inlier_mask = np.ones(len(points), dtype=bool)
                inlier_mask[outlier_indices] = False
                labels = labels[inlier_mask]
            
            pcd = pcd_clean
        
        processed_points = np.asarray(pcd.points)
        processed_colors = np.asarray(pcd.colors) if pcd.has_colors() else colors
        
        normals = None
        if self.config['estimate_normals']:
            if self.config.get('use_shs_net', False):
                print("[Preprocess] Estimating normals with SHS-Net ...")
                try:
                    from shs_net_normal_estimator import create_shs_estimator
                    shs_estimator = create_shs_estimator(self.config)
                    if shs_estimator:
                        normals = shs_estimator.estimate_normals(processed_points)
                        print("[Preprocess] SHS-Net normal estimation complete.")
                    else:
                        print("[Preprocess] SHS-Net unavailable. Falling back to Open3D PCA normals.")
                        pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=self.config['normal_search_radius'],
                                max_nn=self.config['normal_max_nn']
                            )
                        )
                        normals = np.asarray(pcd.normals)
                        print("[Preprocess] Open3D PCA normal estimation complete.")
                except Exception as e:
                    print(f"[Preprocess] SHS-Net estimation failed: {e}")
                    print("[Preprocess] Falling back to Open3D PCA normals.")
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=self.config['normal_search_radius'],
                            max_nn=self.config['normal_max_nn']
                        )
                    )
                    normals = np.asarray(pcd.normals)
                    print("[Preprocess] Open3D PCA normal estimation complete.")
            else:
                print("[Preprocess] Estimating normals with Open3D PCA ...")
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.config['normal_search_radius'],
                        max_nn=self.config['normal_max_nn']
                    )
                )
                normals = np.asarray(pcd.normals)
                print("[Preprocess] Open3D PCA normal estimation complete.")
        
        print(f"[Preprocess] Finished preprocessing: {len(points)} → {len(processed_points)} points.")
        
        return processed_points, processed_colors, labels, normals
    
    def simulate_robot_sparsity(self, points, colors=None, labels=None, sparsity_config=None):
        """Apply sparsity and occlusion heuristics to mimic robotic scanning."""
        if sparsity_config is None:
            return points, colors, labels
        
        print(f"[Sparsity] Applying strategy: {sparsity_config['name']}")
        
        if sparsity_config["keep_ratio"] < 1.0:
            n_keep = int(len(points) * sparsity_config["keep_ratio"])
            indices = np.random.choice(len(points), n_keep, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
            if labels is not None:
                labels = labels[indices]
            print(f"[Sparsity] Retained {len(points)} points after random sampling.")
        
        if sparsity_config.get("occlusion", False):
            points, colors, labels = self._simulate_occlusion(points, colors, labels)
        
        return points, colors, labels
    
    def _simulate_occlusion(self, points, colors=None, labels=None, occlusion_ratio=0.3):
        """Simulate occlusion during robotic scanning."""
        print(f"[Occlusion] Simulating occlusion with ratio {occlusion_ratio}.")
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        room_size = max_coords - min_coords
        
        robot_pos = np.array([
            (min_coords[0] + max_coords[0]) / 2,
            (min_coords[1] + max_coords[1]) / 2,
            min_coords[2] + 0.5
        ])
        
        rel_pos = points - robot_pos
        distances = np.linalg.norm(rel_pos, axis=1)
        
        elevation_angles = np.arctan2(rel_pos[:, 2], np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2))
        vertical_fov_mask = np.abs(elevation_angles) < np.radians(30)
        
        distance_mask = distances < 8.0
        
        corner_mask = np.ones(len(points), dtype=bool)
        for i in [0, 1]:
            corner_regions = [
                points[:, i] < min_coords[i] + room_size[i] * 0.1,
                points[:, i] > max_coords[i] - room_size[i] * 0.1
            ]
            for corner_region in corner_regions:
                corner_mask[corner_region] &= np.random.random(np.sum(corner_region)) > 0.6
        
        final_mask = vertical_fov_mask & distance_mask & corner_mask
        
        occluded_points = points[final_mask]
        occluded_colors = colors[final_mask] if colors is not None else None
        occluded_labels = labels[final_mask] if labels is not None else None
        
        print(f"[Occlusion] Retained {len(occluded_points)} points after occlusion simulation.")
        
        return occluded_points, occluded_colors, occluded_labels
    
    def visualize_normals(self, points, normals, colors=None, output_path=None, 
                         max_points=5000, scale=0.1, title="Point Cloud Normals"):
        """Legacy matplotlib visualization of point clouds and normals."""
        if normals is None:
            print("[Visualization] Warning: no normal data, skipping render.")
            return
        
        print(f"[Visualization] Generating normals plot (matplotlib): {title}")
        
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            vis_points = points[indices]
            vis_normals = normals[indices]
            vis_colors = colors[indices] if colors is not None else None
        else:
            vis_points = points
            vis_normals = normals
            vis_colors = colors
        
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        if vis_colors is not None:
            ax1.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_colors, s=1, alpha=0.6)
        else:
            ax1.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_points[:, 2], s=1, alpha=0.6, cmap='viridis')
        ax1.set_title(f'{title} - Point Cloud')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        ax2 = fig.add_subplot(132, projection='3d')
        if vis_colors is not None:
            ax2.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_colors, s=1, alpha=0.6)
        else:
            ax2.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_points[:, 2], s=1, alpha=0.6, cmap='viridis')
        
        step = max(1, len(vis_points) // 100)
        for i in range(0, len(vis_points), step):
            start = vis_points[i]
            end = start + vis_normals[i] * scale
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    'r-', linewidth=0.5, alpha=0.7)
        
        ax2.set_title(f'{title} - Normals (XY view)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        ax3 = fig.add_subplot(133, projection='3d')
        if vis_colors is not None:
            ax3.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_colors, s=1, alpha=0.6)
        else:
            ax3.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                       c=vis_points[:, 2], s=1, alpha=0.6, cmap='viridis')
        
        for i in range(0, len(vis_points), step):
            start = vis_points[i]
            end = start + vis_normals[i] * scale
            ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    'r-', linewidth=0.5, alpha=0.7)
        
        ax3.set_title(f'{title} - Normals (XZ view)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[Visualization] Saved normals plot to {output_path}")
        
        plt.close()
    
    def visualize_normals_high_quality(self, points, normals, colors=None, 
                                       output_dir=None, base_name="pointcloud",
                                       use_surface_reconstruction=True,
                                       point_size=3.0, dpi=300,
                                       image_format="auto", jpg_quality=95,
                                       save_original_files=True):
        """
        Render a high-quality Open3D visualization for points and normals.
        
        Args:
            points: (N, 3) point coordinates.
            normals: (N, 3) normal vectors.
            colors: Optional (N, 3) RGB colors. If omitted, normals are visualized as colors.
            output_dir: Directory to write rendered images and meshes.
            base_name: Base filename for generated assets.
            use_surface_reconstruction: Enable Poisson reconstruction for dense surfaces.
            point_size: Point size for Open3D rendering.
            dpi: Target resolution for saved images.
            image_format: "png", "jpg", or "auto" (auto switches to JPG for large files).
            jpg_quality: Quality setting when saving JPG images.
            save_original_files: Persist PLY outputs for points/meshes.
        """
        if normals is None:
            print("[Visualization] Warning: no normal data; skipping high-quality render.")
            return
        
        print(f"[Visualization] Rendering high-quality normals view for {base_name} ({len(points)} points).")
        
        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Encode normals as RGB colors
        # Normalize to [-1, 1] and map to [0, 1]
        normals_normalized = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        normal_colors = (normals_normalized + 1.0) / 2.0
        normal_colors = np.clip(normal_colors, 0, 1)
        
        # Use provided colors when available
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
            use_normal_colors = False
        else:
            pcd.colors = o3d.utility.Vector3dVector(normal_colors)
            use_normal_colors = True
        
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Compute scene statistics for camera placement
        center = np.mean(points, axis=0)
        bounds = np.max(points, axis=0) - np.min(points, axis=0)
        max_bound = np.max(bounds)
        
        # Attempt to create an interactive Visualizer
        use_visualizer = False
        vis = None
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1600, height=1200)
            use_visualizer = True
            print("[Visualization] Using Open3D Visualizer (GUI).")
        except Exception as e:
            print(f"[Visualization] Visualizer unavailable; switching to offscreen renderer: {e}")
            use_visualizer = False
        
        # Collect viewpoints
        views = []
        if use_visualizer:
            vis.add_geometry(pcd)
            
            render_option = vis.get_render_option()
            if render_option is not None:
                render_option.point_size = point_size
                render_option.background_color = np.array([0.95, 0.95, 0.95])
            else:
                print("[Visualization] Warning: render_option is None.")
            
            view_control = vis.get_view_control()
            
            view_control.set_zoom(0.7)
            view_control.set_front([0.5, -0.5, -0.7])
            view_control.set_lookat(center)
            view_control.set_up([0, 0, 1])
            views.append(("isometric", view_control.convert_to_pinhole_camera_parameters()))
            
            view_control.set_zoom(0.8)
            view_control.set_front([0, 0, -1])
            view_control.set_lookat(center)
            view_control.set_up([0, 1, 0])
            views.append(("bev", view_control.convert_to_pinhole_camera_parameters()))
            
            view_control.set_zoom(0.7)
            view_control.set_front([0, -1, 0])
            view_control.set_lookat(center)
            view_control.set_up([0, 0, 1])
            views.append(("front", view_control.convert_to_pinhole_camera_parameters()))
            
            view_control.set_zoom(0.7)
            view_control.set_front([1, 0, 0])
            view_control.set_lookat(center)
            view_control.set_up([0, 0, 1])
            views.append(("side", view_control.convert_to_pinhole_camera_parameters()))
        else:
            # Use offscreen renderer in headless environments
            print("[Visualization] Using OffscreenRenderer.")
            try:
                from o3d.visualization import rendering
                renderer = rendering.OffscreenRenderer(1600, 1200)
                
                scene = renderer.scene
                scene.set_background([0.95, 0.95, 0.95, 1.0])
                
                mat = rendering.MaterialRecord()
                mat.point_size = point_size
                mat.shader = "defaultUnlit"
                
                scene.add_geometry("pointcloud", pcd, mat)
                
                camera_params = {
                    "isometric": {
                        "zoom": 0.7,
                        "front": [0.5, -0.5, -0.7],
                        "lookat": center.tolist(),
                        "up": [0, 0, 1]
                    },
                    "bev": {
                        "zoom": 0.8,
                        "front": [0, 0, -1],
                        "lookat": center.tolist(),
                        "up": [0, 1, 0]
                    },
                    "front": {
                        "zoom": 0.7,
                        "front": [0, -1, 0],
                        "lookat": center.tolist(),
                        "up": [0, 0, 1]
                    },
                    "side": {
                        "zoom": 0.7,
                        "front": [1, 0, 0],
                        "lookat": center.tolist(),
                        "up": [0, 0, 1]
                    }
                }
                
                for view_name, params in camera_params.items():
                    views.append((view_name, params))
                
                print("[Visualization] OffscreenRenderer initialized.")
            except ImportError:
                print("[Visualization] Error: OffscreenRenderer requires Open3D >= 0.13.0. Skipping image export.")
                views = []
        
        # Optionally run Poisson surface reconstruction
        mesh = None
        if use_surface_reconstruction and len(points) > 100:
            print("[Visualization] Running Poisson surface reconstruction.")
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9
                )
                vertices_to_remove = densities < np.quantile(densities, 0.1)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                mesh.compute_vertex_normals()
                print(f"[Visualization] Poisson reconstruction succeeded with {len(mesh.vertices)} vertices.")
            except Exception as e:
                print(f"[Visualization] Poisson reconstruction failed: {e}. Continuing with point cloud only.")
                mesh = None
        
        # Render configured views when using the interactive visualizer
        saved_images = []
        if use_visualizer and views:
            try:
                if mesh is not None:
                    vis.clear_geometries()
                    vis.add_geometry(mesh)
                    view_control = vis.get_view_control()
                    for view_name, camera_param in views:
                        view_control.convert_from_pinhole_camera_parameters(camera_param)
                        vis.poll_events()
                        vis.update_renderer()
                
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    view_control = vis.get_view_control()
                    for view_name, camera_param in views:
                        view_control.convert_from_pinhole_camera_parameters(camera_param)
                        vis.poll_events()
                        vis.update_renderer()
                        
                        temp_png_path = output_dir / f"{base_name}_{view_name}_temp.png"
                        vis.capture_screen_image(str(temp_png_path), do_render=True)
                        
                        file_size_mb = temp_png_path.stat().st_size / (1024 * 1024)
                        
                        if image_format == "auto":
                            if file_size_mb > 10:
                                final_format = "jpg"
                                print(f"[Visualization] PNG file is large ({file_size_mb:.2f} MB). Converting to JPG.")
                            else:
                                final_format = "png"
                        else:
                            final_format = image_format.lower()
                        
                        if final_format == "jpg":
                            try:
                                from PIL import Image
                                img = Image.open(temp_png_path)
                                if img.mode in ('RGBA', 'LA'):
                                    background = Image.new('RGB', img.size, (242, 242, 242))
                                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                                    img = background
                                elif img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                final_path = output_dir / f"{base_name}_{view_name}.jpg"
                                img.save(final_path, 'JPEG', quality=jpg_quality, optimize=True)
                                temp_png_path.unlink()
                                print(f"[Visualization] Saved view as JPG ({file_size_mb:.2f} MB → {final_path.stat().st_size/(1024*1024):.2f} MB): {final_path}")
                            except ImportError:
                                print("[Visualization] Warning: PIL not installed; keeping PNG output.")
                                final_path = output_dir / f"{base_name}_{view_name}.png"
                                temp_png_path.rename(final_path)
                                print(f"[Visualization] Saved view as PNG ({file_size_mb:.2f} MB): {final_path}")
                            except Exception as e:
                                print(f"[Visualization] JPG conversion failed ({e}); keeping PNG.")
                                final_path = output_dir / f"{base_name}_{view_name}.png"
                                temp_png_path.rename(final_path)
                        else:
                            final_path = output_dir / f"{base_name}_{view_name}.png"
                            temp_png_path.rename(final_path)
                            print(f"[Visualization] Saved view as PNG ({file_size_mb:.2f} MB): {final_path}")
                        
                        saved_images.append(final_path)
                
                if vis is not None:
                    vis.destroy_window()
            except Exception as e:
                print(f"[Visualization] Rendering failed: {e}. Only PLY artifacts will be saved.")
                if vis is not None:
                    try:
                        vis.destroy_window()
                    except:
                        pass
        else:
            print("[Visualization] Skipping image rendering; running in headless mode.")
        
        # Optionally persist original PLY assets
        if save_original_files and output_dir:
            output_dir = Path(output_dir)
            
            pcd_path = output_dir / f"{base_name}_with_normals.ply"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            file_size_mb = pcd_path.stat().st_size / (1024 * 1024)
            print(f"[Visualization] Saved point cloud ({file_size_mb:.2f} MB): {pcd_path}")
            
            if mesh is not None:
                mesh_path = output_dir / f"{base_name}_reconstructed_mesh.ply"
                o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                file_size_mb = mesh_path.stat().st_size / (1024 * 1024)
                print(f"[Visualization] Saved reconstructed mesh ({file_size_mb:.2f} MB): {mesh_path}")
        
        print("[Visualization] High-quality rendering complete.")
        print(f"  Rendered images: {len(saved_images)}")
        if save_original_files:
            print("  Source PLY files saved for offline inspection.")
        
        return saved_images
    
    def visualize_normal_statistics(self, normals, output_path=None, title="Normal Statistics"):
        """
        Plot statistics for normal vectors using matplotlib.
        
        Args:
            normals: (N, 3) normal vectors.
            output_path: Optional path to save the figure.
            title: Title for the visualization.
        """
        if normals is None:
            print("[Visualization] Warning: no normal data; skipping statistics plot.")
            return
        
        print(f"[Visualization] Rendering normal statistics: {title}")
        
        normal_lengths = np.linalg.norm(normals, axis=1)
        normal_x = normals[:, 0]
        normal_y = normals[:, 1]
        normal_z = normals[:, 2]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].hist(normal_lengths, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Normal Length Distribution')
        axes[0, 0].set_xlabel('Length')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(1.0, color='red', linestyle='--', label='Unit Length')
        axes[0, 0].legend()
        
        axes[0, 1].hist(normal_x, bins=50, alpha=0.7, color='red')
        axes[0, 1].set_title('Normal X Component Distribution')
        axes[0, 1].set_xlabel('X Component')
        axes[0, 1].set_ylabel('Count')
        
        axes[0, 2].hist(normal_y, bins=50, alpha=0.7, color='green')
        axes[0, 2].set_title('Normal Y Component Distribution')
        axes[0, 2].set_xlabel('Y Component')
        axes[0, 2].set_ylabel('Count')
        
        axes[1, 0].hist(normal_z, bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Normal Z Component Distribution')
        axes[1, 0].set_xlabel('Z Component')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].scatter(normal_x, normal_y, s=1, alpha=0.5, c=normal_z, cmap='viridis')
        axes[1, 1].set_title('Normal Orientation (XY Projection)')
        axes[1, 1].set_xlabel('X Component')
        axes[1, 1].set_ylabel('Y Component')
        axes[1, 1].set_aspect('equal')
        
        axes[1, 2].scatter(normal_x, normal_z, s=1, alpha=0.5, c=normal_y, cmap='plasma')
        axes[1, 2].set_title('Normal Orientation (XZ Projection)')
        axes[1, 2].set_xlabel('X Component')
        axes[1, 2].set_ylabel('Z Component')
        axes[1, 2].set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[Visualization] Saved normal statistics plot to {output_path}")
        
        plt.close()
        
        print("[Statistics] Normal vector summary:")
        print(f"  Count: {len(normals)}")
        print(f"  Mean length: {np.mean(normal_lengths):.4f}")
        print(f"  Length std: {np.std(normal_lengths):.4f}")
        print(f"  Length range: [{np.min(normal_lengths):.4f}, {np.max(normal_lengths):.4f}]")
        print(f"  X range: [{np.min(normal_x):.4f}, {np.max(normal_x):.4f}]")
        print(f"  Y range: [{np.min(normal_y):.4f}, {np.max(normal_y):.4f}]")
        print(f"  Z range: [{np.min(normal_z):.4f}, {np.max(normal_z):.4f}]")


def save_processed_data(data_dict, output_path):
    """Persist preprocessed data to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"[Output] Saved preprocessed data to: {output_path}")


def save_pointcloud_variants(room_data, output_dir, config):
    """Export point cloud variants with different sparsity levels."""
    output_dir = Path(output_dir)
    pointcloud_dir = output_dir / "pointclouds"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    
    area_name = room_data['room_info']['area_name']
    room_name = room_data['room_info']['room_name']
    
    print(f"[PointCloud] Writing outputs to: {pointcloud_dir}")
    
    original_data = room_data['original_data']
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_data['points'])
    if original_data['colors'] is not None:
        original_pcd.colors = o3d.utility.Vector3dVector(original_data['colors'])
    
    original_path = pointcloud_dir / f"{area_name}_{room_name}_original.ply"
    o3d.io.write_point_cloud(str(original_path), original_pcd)
    print(f"[PointCloud] Original cloud: {original_path}")
    
    processed_data = room_data['processed_data']
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(processed_data['points'])
    if processed_data['colors'] is not None:
        processed_pcd.colors = o3d.utility.Vector3dVector(processed_data['colors'])
    if processed_data['normals'] is not None:
        processed_pcd.normals = o3d.utility.Vector3dVector(processed_data['normals'])
    
    processed_path = pointcloud_dir / f"{area_name}_{room_name}_processed.ply"
    o3d.io.write_point_cloud(str(processed_path), processed_pcd)
    print(f"[PointCloud] Processed cloud: {processed_path}")
    
    for variant_name, variant_data in room_data['sparsity_variants'].items():
        variant_pcd = o3d.geometry.PointCloud()
        variant_pcd.points = o3d.utility.Vector3dVector(variant_data['points'])
        if variant_data['colors'] is not None:
            variant_pcd.colors = o3d.utility.Vector3dVector(variant_data['colors'])
        if variant_data['normals'] is not None:
            variant_pcd.normals = o3d.utility.Vector3dVector(variant_data['normals'])
        
        variant_path = pointcloud_dir / f"{area_name}_{room_name}_{variant_name}.ply"
        o3d.io.write_point_cloud(str(variant_path), variant_pcd)
        print(f"[PointCloud] Variant '{variant_name}': {variant_path}")
    
    summary_path = pointcloud_dir / f"{area_name}_{room_name}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"S3DIS point cloud summary - {area_name}/{room_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Original cloud: {len(original_data['points'])} points\n")
        f.write(f"Processed cloud: {len(processed_data['points'])} points\n\n")
        
        f.write("Sparsity variants:\n")
        f.write("-"*40 + "\n")
        for variant_name, variant_data in room_data['sparsity_variants'].items():
            config = variant_data['config']
            f.write(f"{variant_name}:\n")
            f.write(f"  Points: {len(variant_data['points'])}\n")
            f.write(f"  Keep ratio: {config.get('keep_ratio', 1.0)}\n")
            f.write(f"  Occlusion simulation: {config.get('occlusion', False)}\n")
            f.write(f"  File: {area_name}_{room_name}_{variant_name}.ply\n\n")
    
    print(f"[PointCloud] Summary file: {summary_path}")
    print(f"[PointCloud] Total point cloud files: {len(room_data['sparsity_variants']) + 2}")


def load_config(config_path):
    """Load YAML configuration from disk."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Preprocessing entry point."""
    parser = argparse.ArgumentParser(description="S3DIS preprocessing utility")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--area_name", type=str, help="Override area name from config")
    parser.add_argument("--room_name", type=str, help="Override room name from config")
    parser.add_argument("--save_pointclouds", action="store_true", help="Export individual point clouds to PLY")
    parser.add_argument("--save_normal_visualizations", action="store_true", help="Save normal visualizations")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override configuration with CLI arguments when provided
    if args.area_name:
        config['data']['area_name'] = args.area_name
    if args.room_name:
        config['data']['room_name'] = args.room_name
    if args.save_normal_visualizations:
        if 'visualization' not in config:
            config['visualization'] = {}
        config['visualization']['save_normal_visualizations'] = True
    
    print("S3DIS preprocessing CLI")
    print("="*50)
    print(f"Config file: {args.config}")
    print(f"Data root: {config['data']['data_root']}")
    print(f"Area: {config['data']['area_name']}")
    print(f"Room: {config['data']['room_name'] or 'ALL'}")
    print("="*50)
    
    loader = S3DISLoader(config['data']['data_root'])
    preprocessor = S3DISPreprocessor(config)
    
    output_dir = Path(config['data']['output_root']) / "preprocessed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    area_name = config['data']['area_name']
    room_name = config['data']['room_name']
    
    if room_name:
        rooms_to_process = [room_name]
    else:
        rooms_to_process = loader.get_available_rooms(area_name)
        if not rooms_to_process:
            print(f"[Error] No rooms found for area {area_name}.")
            return
        print(f"[Info] Discovered {len(rooms_to_process)} rooms: {rooms_to_process}")
    
    for room in rooms_to_process:
        print(f"\n{'-'*40}")
        print(f"[Info] Processing room: {area_name}/{room}")
        print(f"{'-'*40}")
        
        try:
            start_time = time.time()
            original_points, original_colors, original_labels = loader.load_room_data(area_name, room)
            
            processed_points, processed_colors, processed_labels, normals = preprocessor.preprocess_pointcloud(
                original_points, original_colors, original_labels
            )
            
            if config.get('visualization', {}).get('save_normal_visualizations', True):
                vis_dir = output_dir / "normal_visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                vis_config = config.get('visualization', {})
                use_high_quality = vis_config.get('use_high_quality_visualization', True)
                
                if use_high_quality:
                    preprocessor.visualize_normals_high_quality(
                        processed_points, normals, processed_colors,
                        output_dir=vis_dir,
                        base_name=f"{area_name}_{room}_processed",
                        use_surface_reconstruction=vis_config.get('use_surface_reconstruction', True),
                        point_size=vis_config.get('point_size', 3.0),
                        image_format=vis_config.get('image_format', 'auto'),
                        jpg_quality=vis_config.get('jpg_quality', 95),
                        save_original_files=vis_config.get('save_original_files', True)
                    )
                else:
                    normal_vis_path = vis_dir / f"{area_name}_{room}_normals.png"
                    preprocessor.visualize_normals(
                        processed_points, normals, processed_colors, 
                        normal_vis_path, 
                        title=f"{area_name}/{room} - Processed normals"
                    )
                
                normal_stats_path = vis_dir / f"{area_name}_{room}_normal_stats.png"
                preprocessor.visualize_normal_statistics(
                    normals, normal_stats_path,
                    title=f"{area_name}/{room} - Normal statistics"
                )
            
            room_data = {
                'room_info': {
                    'area_name': area_name,
                    'room_name': room,
                    'processing_time': time.time() - start_time
                },
                'original_data': {
                    'points': original_points,
                    'colors': original_colors,
                    'labels': original_labels,
                    'point_count': len(original_points)
                },
                'processed_data': {
                    'points': processed_points,
                    'colors': processed_colors,
                    'labels': processed_labels,
                    'normals': normals,
                    'point_count': len(processed_points)
                },
                'sparsity_variants': {}
            }
            
            for sparsity_config in config['sparsity_experiments']:
                print(f"\n[Sparsity] Generating variant: {sparsity_config['name']}")
                
                if sparsity_config['name'] == 'dense':
                    sparse_points = processed_points
                    sparse_colors = processed_colors
                    sparse_labels = processed_labels
                else:
                    sparse_points, sparse_colors, sparse_labels = preprocessor.simulate_robot_sparsity(
                        processed_points, processed_colors, processed_labels, sparsity_config
                    )
                
                if normals is not None and len(sparse_points) != len(processed_points):
                    if config['preprocessing'].get('use_shs_net', False):
                        print("[Preprocess] Re-estimating normals with SHS-Net for sparse variant...")
                        try:
                            from shs_net_normal_estimator import create_shs_estimator
                            shs_estimator = create_shs_estimator(config['preprocessing'])
                            if shs_estimator:
                                sparse_normals = shs_estimator.estimate_normals(sparse_points)
                                print("[Preprocess] SHS-Net normal estimation complete.")
                            else:
                                # Fall back to Open3D PCA
                                pcd_temp = o3d.geometry.PointCloud()
                                pcd_temp.points = o3d.utility.Vector3dVector(sparse_points)
                                pcd_temp.estimate_normals(
                                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                        radius=config['preprocessing']['normal_search_radius'],
                                        max_nn=config['preprocessing']['normal_max_nn']
                                    )
                                )
                                sparse_normals = np.asarray(pcd_temp.normals)
                                print("[Preprocess] Open3D PCA normal estimation complete.")
                        except Exception as e:
                            print(f"[Preprocess] SHS-Net estimation failed: {e}")
                            # Fall back to Open3D PCA
                            pcd_temp = o3d.geometry.PointCloud()
                            pcd_temp.points = o3d.utility.Vector3dVector(sparse_points)
                            pcd_temp.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                    radius=config['preprocessing']['normal_search_radius'],
                                    max_nn=config['preprocessing']['normal_max_nn']
                                )
                            )
                            sparse_normals = np.asarray(pcd_temp.normals)
                            print("[Preprocess] Open3D PCA normal estimation complete.")
                    else:
                        print("[Preprocess] Re-estimating normals with Open3D PCA for sparse variant...")
                        pcd_temp = o3d.geometry.PointCloud()
                        pcd_temp.points = o3d.utility.Vector3dVector(sparse_points)
                        pcd_temp.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=config['preprocessing']['normal_search_radius'],
                                max_nn=config['preprocessing']['normal_max_nn']
                            )
                        )
                        sparse_normals = np.asarray(pcd_temp.normals)
                        print("[Preprocess] Open3D PCA normal estimation complete.")
                else:
                    sparse_normals = normals
                
                room_data['sparsity_variants'][sparsity_config['name']] = {
                    'points': sparse_points,
                    'colors': sparse_colors,
                    'labels': sparse_labels,
                    'normals': sparse_normals,
                    'point_count': len(sparse_points),
                    'config': sparsity_config
                }
                
                if config.get('visualization', {}).get('save_normal_visualizations', True):
                    vis_dir = output_dir / "normal_visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    vis_config = config.get('visualization', {})
                    use_high_quality = vis_config.get('use_high_quality_visualization', True)
                    
                    if use_high_quality:
                        preprocessor.visualize_normals_high_quality(
                            sparse_points, sparse_normals, sparse_colors,
                            output_dir=vis_dir,
                            base_name=f"{area_name}_{room}_{sparsity_config['name']}",
                            use_surface_reconstruction=vis_config.get('use_surface_reconstruction', True),
                            point_size=vis_config.get('point_size', 3.0),
                            image_format=vis_config.get('image_format', 'auto'),
                            jpg_quality=vis_config.get('jpg_quality', 95),
                            save_original_files=vis_config.get('save_original_files', True)
                        )
                    else:
                        sparse_normal_vis_path = vis_dir / f"{area_name}_{room}_{sparsity_config['name']}_normals.png"
                        preprocessor.visualize_normals(
                            sparse_points, sparse_normals, sparse_colors,
                            sparse_normal_vis_path,
                            title=f"{area_name}/{room} - {sparsity_config['name']} normals"
                        )
                    
                    sparse_normal_stats_path = vis_dir / f"{area_name}_{room}_{sparsity_config['name']}_normal_stats.png"
                    preprocessor.visualize_normal_statistics(
                        sparse_normals, sparse_normal_stats_path,
                        title=f"{area_name}/{room} - {sparsity_config['name']} normal stats"
                    )
            
            output_path = output_dir / f"{area_name}_{room}_preprocessed.pkl"
            save_processed_data(room_data, output_path)
            
            if args.save_pointclouds:
                save_pointcloud_variants(room_data, output_dir, config)
            
            print(f"[Success] Room {area_name}/{room} preprocessed.")
            print(f"  Original points: {len(original_points)}")
            print(f"  Processed points: {len(processed_points)}")
            print(f"  Sparsity variants: {len(room_data['sparsity_variants'])}")
            
        except Exception as e:
            print(f"[Error] Failed to preprocess {area_name}/{room}: {e}")
    
    print("\n[Done] Preprocessing complete.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
