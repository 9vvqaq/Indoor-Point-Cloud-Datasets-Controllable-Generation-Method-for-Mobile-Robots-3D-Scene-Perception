#!/usr/bin/env python3
"""
LiDAR-Net bounding box generation and visualization utilities.
Generates 3D bounding boxes from per-point semantic and instance labels.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import argparse


class LiDARNetBBoxVisualizer:
    """LiDAR-Net bounding box generator and visualizer."""
    
    def __init__(self):
        # LiDAR-Net detection classes
        self.detection_classes = {
            'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs'
        }
        
        # Semantic label mapping for LiDAR-Net
        self.semantic_mapping = {
            'window': 15,      # Window
            'table': 18,       # Table  
            'chair': 19,       # Chair
            'sofa': 20,        # Sofa
            'bookcase': 23,    # Bookshelf
            'board': 21,       # Blackboard
            'stairs': 10       # Stair
        }
        
        # Class colors for visualization (RGB)
        self.class_colors = {
            'window': [1.0, 0.0, 0.0],    # Red
            'table': [0.0, 0.0, 1.0],     # Blue
            'chair': [1.0, 1.0, 0.0],     # Yellow
            'sofa': [1.0, 0.0, 1.0],      # Magenta
            'bookcase': [0.0, 1.0, 1.0],  # Cyan
            'board': [1.0, 0.5, 0.0],     # Orange
            'stairs': [0.5, 0.0, 1.0]     # Violet
        }
        
        # Bounding box generation parameters
        self.bbox_params = {
            'min_points': 10,
            'min_volume': 0.001,
            'max_aspect_ratio': 10.0,
            'max_vis_points': 1000,
        }
    
    def load_ply_file(self, ply_path):
        """Load a LiDAR-Net PLY file that includes semantic and instance labels."""
        try:
            # Load PLY via Open3D
            pcd = o3d.io.read_point_cloud(str(ply_path))
            if len(pcd.points) == 0:
                print(f"        [Warning] Empty PLY file: {ply_path.name}")
                return None
            
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
            # Read binary PLY payload to recover semantic and instance labels
            semantic_labels = None
            instance_labels = None
            
            try:
                import struct
                with open(ply_path, 'rb') as f:
                    # Read PLY header
                    header_lines = []
                    while True:
                        line = f.readline().decode('utf-8').strip()
                        header_lines.append(line)
                        if line == 'end_header':
                            break
                    
                    # Summarize PLY header information
                    property_lines = [line for line in header_lines if line.startswith('property')]
                    print(f"        PLY property count: {len(property_lines)}")
                    print("        PLY header:")
                    for line in property_lines:
                        print(f"          {line}")
                    
                    # Check for semantic/instance properties
                    has_semantic = any('sem' in line for line in property_lines)
                    has_instance = any('ins' in line for line in property_lines)
                    
                    if not (has_semantic and has_instance):
                        print("        [Error] Semantic or instance attributes missing.")
                        print(f"        Semantic labels present: {has_semantic}")
                        print(f"        Instance labels present: {has_instance}")
                        print("        Skipping file (expected x,y,z,r,g,b,sem,ins).")
                        return None
                    
                    # Locate vertex count
                    vertex_count = 0
                    for line in header_lines:
                        if line.startswith('element vertex'):
                            vertex_count = int(line.split()[-1])
                    
                    print(f"        Vertex count: {vertex_count}")
                    
                    # Read vertex payload
                    semantic_labels = []
                    instance_labels = []
                    
                    for i in range(vertex_count):
                        # Skip x, y, z (float32) and r, g, b (uint8)
                        f.read(12 + 3)
                        
                        # Read semantic and instance IDs (uint16)
                        sem, ins = struct.unpack('HH', f.read(4))
                        semantic_labels.append(sem)
                        instance_labels.append(ins)
                        
                        # Show first few samples for debugging
                        if i < 5:
                            print(f"          Point {i}: sem={sem}, ins={ins}")
                    
                    semantic_labels = np.array(semantic_labels, dtype=np.uint16)
                    instance_labels = np.array(instance_labels, dtype=np.uint16)
                    
                    print("        [Info] Semantic/instance labels parsed successfully.")
                    print(f"        Semantic label range: {semantic_labels.min()} - {semantic_labels.max()}")
                    print(f"        Instance label range: {instance_labels.min()} - {instance_labels.max()}")
                    
            except Exception as e:
                print(f"        [Warning] Unable to read semantic/instance labels: {e}")
                semantic_labels = np.zeros(len(points), dtype=np.uint16)
                instance_labels = np.zeros(len(points), dtype=np.uint16)
            
            print(f"        Points loaded: {len(points)}")
            print(f"        Coordinate range: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                  f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                  f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            unique_semantic = np.unique(semantic_labels)
            unique_instance = np.unique(instance_labels)
            print(f"        Semantic labels: range {unique_semantic.min()} - {unique_semantic.max()} ({len(unique_semantic)} unique)")
            print(f"        Instance labels: range {unique_instance.min()} - {unique_instance.max()} ({len(unique_instance)} unique)")
            
            for sem_id in unique_semantic:
                count = np.sum(semantic_labels == sem_id)
                print(f"        Semantic ID {sem_id}: {count} points")
            
            return {
                'points': points,
                'colors': colors,
                'semantic_labels': semantic_labels,
                'instance_labels': instance_labels,
                'file_name': ply_path.name
            }
            
        except Exception as e:
            print(f"        [Error] Failed to load PLY {ply_path}: {e}")
            return None
    
    def extract_instances_by_semantic(self, room_data, target_semantic_id):
        """Extract object instances for a given semantic ID."""
        points = room_data['points']
        colors = room_data['colors']
        semantic_labels = room_data['semantic_labels']
        instance_labels = room_data['instance_labels']
        
        # Gather target semantic points
        semantic_mask = semantic_labels == target_semantic_id
        if not np.any(semantic_mask):
            return []
        
        target_points = points[semantic_mask]
        target_colors = colors[semantic_mask] if colors is not None else None
        target_instance_labels = instance_labels[semantic_mask]
        
        # Split by instance id
        unique_instances = np.unique(target_instance_labels)
        instances = []
        
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip unlabeled points
                continue
                
            instance_mask = target_instance_labels == instance_id
            if np.sum(instance_mask) < self.bbox_params['min_points']:
                continue
                
            instance_points = target_points[instance_mask]
            instance_colors = target_colors[instance_mask] if target_colors is not None else None
            
            instances.append({
                'points': instance_points,
                'colors': instance_colors,
                'instance_id': instance_id,
                'semantic_id': target_semantic_id
            })
        
        return instances
    
    def remove_outliers(self, points, colors=None, k=20, std_ratio=2.0):
        """Remove statistical outliers from a point set."""
        if len(points) < k:
            return points, colors
        
        # Remove outliers using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        
        # Extract filtered point cloud
        filtered_pcd = pcd.select_by_index(ind)
        filtered_points = np.asarray(filtered_pcd.points)
        filtered_colors = np.asarray(filtered_pcd.colors) if colors is not None else None
        
        removed_count = len(points) - len(filtered_points)
        print(f"        Outlier removal: {len(points)} → {len(filtered_points)} points (removed {removed_count}).")
        
        return filtered_points, filtered_colors
    
    def calculate_bbox(self, points):
        """Compute an axis-aligned 3D bounding box."""
        if len(points) == 0:
            return None
        
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        volume = float(np.prod(size))
        sorted_size = np.sort(size)
        aspect_ratio = sorted_size[2] / sorted_size[0] if sorted_size[0] > 0 else float('inf')
        
        return {
            'center': center.tolist(),
            'size': size.tolist(),
            'min_coords': min_coords.tolist(),
            'max_coords': max_coords.tolist(),
            'volume': volume,
            'aspect_ratio': aspect_ratio,
            'point_count': len(points)
        }
    
    def evaluate_bbox_quality(self, bbox, object_name):
        """Evaluate basic quality metrics for a bounding box."""
        if bbox['volume'] < self.bbox_params['min_volume']:
            return False, f"Volume too small: {bbox['volume']:.6f} < {self.bbox_params['min_volume']}"
        
        if bbox['aspect_ratio'] > self.bbox_params['max_aspect_ratio']:
            return False, f"Aspect ratio too large: {bbox['aspect_ratio']:.2f} > {self.bbox_params['max_aspect_ratio']}"
        
        return True, "Quality check passed."
    
    def process_room_instances(self, room_ply_path, output_dir=None):
        """Process all instances in a single room PLY."""
        print(f"Processing room: {room_ply_path}")
        
        if not room_ply_path.exists():
            print(f"  [Error] Room file does not exist: {room_ply_path}")
            return None
        
        # Load PLY data
        room_data = self.load_ply_file(room_ply_path)
        if room_data is None:
            return None
        
        all_bboxes = []
        
        # Process each target class
        for class_name, semantic_id in self.semantic_mapping.items():
            print(f"  Processing class: {class_name} (semantic ID: {semantic_id})")
            
            # Extract instances for the semantic class
            instances = self.extract_instances_by_semantic(room_data, semantic_id)
            print(f"    Instances found: {len(instances)}")
            
            for i, instance in enumerate(instances):
                print(f"    Processing instance {i+1}/{len(instances)} (instance ID: {instance['instance_id']})")
                
                # Remove outliers
                filtered_points, filtered_colors = self.remove_outliers(
                    instance['points'], instance['colors']
                )
                
                if len(filtered_points) == 0:
                    print("      [Skip] No points remain after outlier removal.")
                    continue
                
                # Compute bounding box
                bbox_info = self.calculate_bbox(filtered_points)
                if bbox_info is None:
                    print("      [Skip] Unable to compute bounding box.")
                    continue
                
                # Evaluate quality
                is_suitable, reason = self.evaluate_bbox_quality(bbox_info, class_name)
                if not is_suitable:
                    print(f"      [Skip] Quality check failed: {reason}")
                    continue
                
                # Append metadata
                bbox_info['object_name'] = class_name
                bbox_info['instance_id'] = instance['instance_id']
                bbox_info['semantic_id'] = instance['semantic_id']
                bbox_info['filtered_points'] = filtered_points
                bbox_info['filtered_colors'] = filtered_colors
                
                print(f"      [OK] {class_name}: {len(instance['points'])} → {len(filtered_points)} points, volume {bbox_info['volume']:.4f}")
                
                all_bboxes.append(bbox_info)
        
        print(f"  Total valid bounding boxes: {len(all_bboxes)}")
        
        if not all_bboxes:
            print("  No valid bounding boxes generated.")
            return None
        
        # Persist per-room JSON annotations within simulation_results
        if all_bboxes:
            simulation_results_dir = Path("simulation_results")
            simulation_results_dir.mkdir(parents=True, exist_ok=True)
            
            room_name = room_ply_path.stem
            scene_name = room_name
            
            scene_dir = simulation_results_dir / scene_name
            if scene_dir.exists():
                json_path = scene_dir / f"{scene_name}_detection_annotations.json"
                self.generate_detection_annotations(all_bboxes, json_path)
                print(f"  Detection JSON written to existing scene directory: {json_path}")
            else:
                print(f"  [Info] Creating scene directory: {scene_dir}")
                scene_dir.mkdir(parents=True, exist_ok=True)
                json_path = scene_dir / f"{scene_name}_detection_annotations.json"
                self.generate_detection_annotations(all_bboxes, json_path)
                print(f"  Detection JSON written to: {json_path}")
            
            try:
                import shutil
                target_ply_path = scene_dir / room_ply_path.name
                if not target_ply_path.exists():
                    shutil.copy2(room_ply_path, target_ply_path)
                    print(f"  Copied original PLY to: {target_ply_path}")
                else:
                    print(f"  PLY already present: {target_ply_path}")
            except Exception as e:
                print(f"  [Warning] Failed to copy PLY file: {e}")
        
        return all_bboxes
    
    def generate_detection_annotations(self, bboxes, output_path):
        """Generate detection annotations compatible with Group-Free-3D."""
        # Class mapping compatible with Group-Free-3D
        class_name_mapping = {
            'window': 'window',
            'table': 'table',
            'chair': 'chair',
            'sofa': 'sofa',
            'bookcase': 'bookshelf',
            'board': 'picture',
            'stairs': 'counter'
        }
        
        detection_annotations = []
        
        for i, bbox in enumerate(bboxes):
            # Resolve class name expected by Group-Free-3D
            original_class = bbox['object_name']
            groupfree_class = class_name_mapping.get(original_class, original_class)
            
            # Group-Free-3D compliant annotation
            ann = {
                'instance_id': i + 1,
                'class_name': groupfree_class,
                'original_class_name': original_class,
                'bbox_3d': {
                    'center': bbox['center'],
                    'size': bbox['size'],
                    'rotation': [0, 0, 0],
                    'min_coords': bbox['min_coords'],
                    'max_coords': bbox['max_coords']
                },
                'point_count': int(bbox['point_count']),
                'volume': float(bbox['volume']),
                'aspect_ratio': float(bbox['aspect_ratio']),
                'confidence': 1.0,
                'bbox_format': 'AABB',
                'coordinate_system': 'world',
                'units': 'meters',
                'framework': 'Group-Free-3D'
            }
            detection_annotations.append(ann)
        
        # Compose annotation payload
        annotation_file = {
            'metadata': {
                'dataset': 'LiDAR-Net',
                'annotation_type': '3D_object_detection',
                'framework': 'Group-Free-3D',
                'classes': list(class_name_mapping.values()),
                'original_classes': list(class_name_mapping.keys()),
                'class_mapping': class_name_mapping,
                'bbox_format': 'AABB',
                'coordinate_system': 'world',
                'units': 'meters',
                'total_objects': len(detection_annotations),
                'compatible_with': ['Group-Free-3D', 'VoteNet', 'ScanNet']
            },
            'annotations': detection_annotations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_file, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(detection_annotations)} detection annotations to: {output_path}")
        print("Annotation format: Group-Free-3D compatible")
        print(f"Class mapping: {class_name_mapping}")
        return detection_annotations
    
    def process_area(self, area_path, output_dir=None):
        """Process an entire area; sample rooms for visualization while saving JSON for all."""
        print(f"Processing area: {area_path}")
        
        if not area_path.exists():
            print(f"  [Error] Area path not found: {area_path}")
            return None
        
        all_ply_files = list(area_path.glob("*.ply"))
        print(f"  PLY files discovered: {len(all_ply_files)}")
        
        if not all_ply_files:
            print("  No PLY files detected in area.")
            return None
        
        if len(all_ply_files) > 5:
            selected_ply_files = np.random.choice(all_ply_files, 5, replace=False).tolist()
        else:
            selected_ply_files = all_ply_files
        
        print(f"  Selected {len(selected_ply_files)} rooms for visualization subset.")
        print(f"  All {len(all_ply_files)} rooms will produce JSON annotations.")
        
        all_room_bboxes = []
        for ply_file in all_ply_files:
            print(f"\n  Processing room: {ply_file.name}")
            room_bboxes = self.process_room_instances(ply_file, output_dir)
            if room_bboxes:
                all_room_bboxes.extend(room_bboxes)
                print(f"    [Info] {ply_file.name}: {len(room_bboxes)} bounding boxes")
            else:
                print(f"    [Warning] {ply_file.name}: no valid bounding boxes")
        
        # Create visualization data only for selected rooms
        selected_bboxes = []
        selected_rooms_data = []
        
        for ply_file in selected_ply_files:
            print(f"\n  Processing room for visualization: {ply_file.name}")
            room_bboxes = self.process_room_instances(ply_file, None)
            if room_bboxes:
                representative_bbox = max(room_bboxes, key=lambda x: x['volume'])
                selected_bboxes.append(representative_bbox)
                selected_rooms_data.append({
                    'room_name': ply_file.stem,
                    'bboxes': room_bboxes,
                    'representative_bbox': representative_bbox
                })
                print(f"    [Info] Selected {representative_bbox['object_name']} (volume {representative_bbox['volume']:.4f})")
            else:
                print(f"    [Warning] {ply_file.name}: no valid bounding boxes for visualization.")
        
        print(f"\nArea {area_path.name} processed:")
        print(f"  Rooms processed: {len(all_ply_files)}")
        print(f"  Rooms selected for visualization: {len(selected_ply_files)}")
        print(f"  Total bounding boxes: {len(all_room_bboxes)}")
        print(f"  Representative objects: {len(selected_bboxes)}")
        
        if selected_bboxes:
            self.create_area_summary_visualization(selected_bboxes, selected_rooms_data, area_path.name, output_dir)
        
        return selected_bboxes
    
    def create_area_summary_visualization(self, selected_bboxes, selected_rooms, area_name, output_dir):
        """Create area-level visualization assets."""
        print(f"  Creating area summary visualization: {area_name}")
        
        if len(selected_bboxes) == 0:
            print("    No bounding boxes available for visualization.")
            return
        
        print(f"    Displaying {len(selected_bboxes)} representative boxes from {len(selected_rooms)} rooms.")
        
        # Configure matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        if output_dir:
            # 1. 3D view
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_area_bboxes_3d(ax, selected_bboxes, area_name)
            plt.tight_layout()
            vis_3d_path = output_dir / f"{area_name}_area_3d_view.png"
            plt.savefig(vis_3d_path, dpi=300, bbox_inches='tight')
            print(f"    Saved 3D view to: {vis_3d_path}")
            plt.close()
            
            # 2. XY projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'XY')
            plt.tight_layout()
            vis_xy_path = output_dir / f"{area_name}_area_xy_view.png"
            plt.savefig(vis_xy_path, dpi=300, bbox_inches='tight')
            print(f"    Saved XY projection to: {vis_xy_path}")
            plt.close()
            
            # 3. XZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'XZ')
            plt.tight_layout()
            vis_xz_path = output_dir / f"{area_name}_area_xz_view.png"
            plt.savefig(vis_xz_path, dpi=300, bbox_inches='tight')
            print(f"    Saved XZ projection to: {vis_xz_path}")
            plt.close()
            
            # 4. YZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'YZ')
            plt.tight_layout()
            vis_yz_path = output_dir / f"{area_name}_area_yz_view.png"
            plt.savefig(vis_yz_path, dpi=300, bbox_inches='tight')
            print(f"    Saved YZ projection to: {vis_yz_path}")
            plt.close()
            
            # 5. Statistics figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_statistics(ax, selected_bboxes, selected_rooms, area_name)
            plt.tight_layout()
            stats_path = output_dir / f"{area_name}_area_statistics.png"
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            print(f"    Saved statistics figure to: {stats_path}")
            plt.close()
    
    def plot_area_bboxes_3d(self, ax, bboxes, area_name):
        """Render 3D view of area bounding boxes."""
        for i, bbox in enumerate(bboxes):
            color = self.class_colors.get(bbox['object_name'], [0.5, 0.5, 0.5])
            
            self.draw_bbox_3d(ax, bbox, color)
            
            center = bbox['center']
            ax.text(center[0], center[1], center[2], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Area 3D view - {area_name}')
    
    def draw_bbox_3d(self, ax, bbox_info, color):
        """Draw a 3D axis-aligned bounding box."""
        center = np.array(bbox_info['center'])
        size = np.array(bbox_info['size'])
        
        # Compute vertices
        half_size = size / 2
        vertices = np.array([
            [center[0] - half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
            [center[0] + half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
            [center[0] + half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
            [center[0] - half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
            [center[0] - half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
            [center[0] + half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
            [center[0] + half_size[0], center[1] + half_size[1], center[2] + half_size[2]],
            [center[0] - half_size[0], center[1] + half_size[1], center[2] + half_size[2]]
        ])
        
        # Edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Draw box edges
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                     color=color, linewidth=2, alpha=0.8)
    
    def draw_bbox_2d(self, ax, bbox_info, color, x_idx, y_idx):
        """Draw a 2D projection of a bounding box."""
        center = np.array(bbox_info['center'])
        size = np.array(bbox_info['size'])
        
        # Compute projected rectangle
        center_2d = center[[x_idx, y_idx]]
        size_2d = size[[x_idx, y_idx]]
        
        # Lower-left corner and size
        bottom_left = center_2d - size_2d / 2
        width, height = size_2d
        
        # Draw rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle(bottom_left, width, height, 
                       fill=False, color=color, linewidth=2, alpha=0.8)
        ax.add_patch(rect)
    
    def plot_area_bboxes_2d(self, ax, bboxes, area_name, projection):
        """Render 2D projections of bounding boxes for an area."""
        # Select projection axes
        if projection == 'XY':
            x_idx, y_idx = 0, 1
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        elif projection == 'XZ':
            x_idx, y_idx = 0, 2
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        elif projection == 'YZ':
            x_idx, y_idx = 1, 2
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
        
        for i, bbox in enumerate(bboxes):
            color = self.class_colors.get(bbox['object_name'], [0.5, 0.5, 0.5])
            self.draw_bbox_2d(ax, bbox, color, x_idx, y_idx)
            
            center = bbox['center']
            ax.text(center[x_idx], center[y_idx], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_title(f'{projection} projection - {area_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_area_statistics(self, ax, selected_bboxes, selected_rooms, area_name):
        """Render textual summary for area-level statistics."""
        ax.axis('off')
        
        # Count instances per class
        class_counts = {}
        for bbox in selected_bboxes:
            class_name = bbox['object_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Compose statistics text
        stats_text = f"""
Area summary - {area_name}:

Selected representative boxes:
• Total: {len(selected_bboxes)}
• Rooms represented: {len(selected_rooms)}

Per-class counts:
"""
        for class_name, count in sorted(class_counts.items()):
            stats_text += f"• {class_name}: {count}\n"
        
        stats_text += "\nSelected rooms:\n"
        for room_data in selected_rooms:
            room_name = room_data['room_name']
            room_bbox_count = len(room_data['bboxes'])
            stats_text += f"• {room_name}: {room_bbox_count} bounding boxes\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def create_room_visualization(self, bboxes, room_name, output_dir):
        """Create visualization assets for a single room."""
        print(f"  Creating room visualization: {room_name}")
        
        if len(bboxes) == 0:
            print("    No bounding boxes to display.")
            return
        
        print(f"    Displaying {len(bboxes)} bounding boxes.")
        
        # Configure matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        if output_dir:
            # 1. 3D view
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_room_bboxes_3d(ax, bboxes, room_name)
            plt.tight_layout()
            vis_3d_path = output_dir / f"{room_name}_room_3d_view.png"
            plt.savefig(vis_3d_path, dpi=300, bbox_inches='tight')
            print(f"    Saved room 3D view to: {vis_3d_path}")
            plt.close()
            
            # 2. XY projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_room_bboxes_2d(ax, bboxes, room_name, 'XY')
            plt.tight_layout()
            vis_xy_path = output_dir / f"{room_name}_room_xy_view.png"
            plt.savefig(vis_xy_path, dpi=300, bbox_inches='tight')
            print(f"    Saved room XY projection to: {vis_xy_path}")
            plt.close()
            
            # 3. XZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_room_bboxes_2d(ax, bboxes, room_name, 'XZ')
            plt.tight_layout()
            vis_xz_path = output_dir / f"{room_name}_room_xz_view.png"
            plt.savefig(vis_xz_path, dpi=300, bbox_inches='tight')
            print(f"    Saved room XZ projection to: {vis_xz_path}")
            plt.close()
            
            # 4. YZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_room_bboxes_2d(ax, bboxes, room_name, 'YZ')
            plt.tight_layout()
            vis_yz_path = output_dir / f"{room_name}_room_yz_view.png"
            plt.savefig(vis_yz_path, dpi=300, bbox_inches='tight')
            print(f"    Saved room YZ projection to: {vis_yz_path}")
            plt.close()
            
            # 5. Statistics figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_room_statistics(ax, bboxes, room_name)
            plt.tight_layout()
            stats_path = output_dir / f"{room_name}_room_statistics.png"
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            print(f"    Saved room statistics to: {stats_path}")
            plt.close()
    
    def plot_room_bboxes_3d(self, ax, bboxes, room_name):
        """Render room-level bounding boxes in 3D."""
        for i, bbox in enumerate(bboxes):
            color = self.class_colors.get(bbox['object_name'], [0.5, 0.5, 0.5])
            
            self.draw_bbox_3d(ax, bbox, color)
            
            center = bbox['center']
            ax.text(center[0], center[1], center[2], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Room 3D view - {room_name}')
    
    def plot_room_bboxes_2d(self, ax, bboxes, room_name, projection):
        """Render 2D projections of room bounding boxes."""
        # Select projection axes
        if projection == 'XY':
            x_idx, y_idx = 0, 1
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        elif projection == 'XZ':
            x_idx, y_idx = 0, 2
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        elif projection == 'YZ':
            x_idx, y_idx = 1, 2
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
        
        for i, bbox in enumerate(bboxes):
            color = self.class_colors.get(bbox['object_name'], [0.5, 0.5, 0.5])
            self.draw_bbox_2d(ax, bbox, color, x_idx, y_idx)
            
            center = bbox['center']
            ax.text(center[x_idx], center[y_idx], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_title(f'{projection} projection - {room_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_room_statistics(self, ax, bboxes, room_name):
        """Render textual statistics for a room."""
        ax.axis('off')
        
        # Count per-class detections
        class_counts = {}
        for bbox in bboxes:
            class_name = bbox['object_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Compose statistics text
        stats_text = f"""
Room statistics - {room_name}:

Bounding box summary:
• Total boxes: {len(bboxes)}

Class counts:
"""
        for class_name, count in sorted(class_counts.items()):
            stats_text += f"• {class_name}: {count}\n"
        
        volumes = [bbox['volume'] for bbox in bboxes]
        if volumes:
            stats_text += "\nVolume statistics:\n"
            stats_text += f"• Mean volume: {np.mean(volumes):.4f} m³\n"
            stats_text += f"• Max volume: {np.max(volumes):.4f} m³\n"
            stats_text += f"• Min volume: {np.min(volumes):.4f} m³\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


def main():
    """Command-line entry point for LiDAR-Net bounding boxes."""
    parser = argparse.ArgumentParser(description="LiDAR-Net bounding box generator")
    parser.add_argument("--data_root", type=str, required=True, help="LiDAR-Net dataset root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    print("LiDAR-Net Bounding Box Generator")
    print("="*50)
    print(f"Data root: {args.data_root}")
    print(f"Detection classes: {', '.join(['window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs'])}")
    print("="*50)
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[Error] Data root does not exist: {data_root}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / "lidar_net_bbox_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = LiDARNetBBoxVisualizer()
    
    all_ply_files = list(data_root.glob("*.ply"))
    print(f"Found {len(all_ply_files)} PLY file(s) in root directory.")
    
    if not all_ply_files:
        print("[Error] No PLY files found.")
        return
    
    all_bboxes = []
    processed_rooms = []
    failed_rooms = []
    
    print(f"\nProcessing {len(all_ply_files)} PLY file(s)...")
    
    for i, ply_file in enumerate(all_ply_files, 1):
        print(f"\n[{i}/{len(all_ply_files)}] Processing room: {ply_file.name}")
        room_bboxes = visualizer.process_room_instances(ply_file, None)
        if room_bboxes:
            all_bboxes.extend(room_bboxes)
            processed_rooms.append({
                'room_name': ply_file.stem,
                'bboxes': room_bboxes,
                'ply_path': ply_file
            })
            print(f"  [Info] {ply_file.name}: {len(room_bboxes)} bounding boxes")
        else:
            failed_rooms.append(ply_file.name)
            print(f"  [Warning] {ply_file.name}: no valid bounding boxes")
    
    print("\nAll rooms processed.")
    print("  Summary:")
    print(f"    - Total PLY files: {len(all_ply_files)}")
    print(f"    - Successful rooms: {len(processed_rooms)}")
    print(f"    - Failed rooms: {len(failed_rooms)}")
    print(f"    - Total bounding boxes: {len(all_bboxes)}")
    success_rate = len(processed_rooms) / len(all_ply_files) * 100 if all_ply_files else 0.0
    print(f"    - Success rate: {success_rate:.1f}%")
    
    if failed_rooms:
        print(f"  Failed rooms: {', '.join(failed_rooms)}")
    
    if processed_rooms:
        print(f"  Successful rooms: {', '.join([room['room_name'] for room in processed_rooms])}")
    
    # Visualization routines disabled to reduce memory usage.
    # Example snippet kept for reference:
    # if processed_rooms:
    #     best_room = max(processed_rooms, key=lambda x: len(x['bboxes']))
    #     print(f"\nSelected room {best_room['room_name']} for visualization ({len(best_room['bboxes'])} boxes)")
    #     
    #     # Generate visualization
    #     visualizer.create_room_visualization(best_room['bboxes'], best_room['room_name'], output_dir)
    
    print("\nProcessing complete.")
    print("All room detection JSON files are stored in the simulation_results directory.")


if __name__ == "__main__":
    main()
