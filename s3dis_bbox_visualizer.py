#!/usr/bin/env python3
"""
S3DIS bounding box generation and visualization tool.
Generates bounding boxes for specified object classes and provides visualization.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import argparse


class S3DISBBoxVisualizer:
    """S3DIS bounding box generator and visualizer."""
    
    def __init__(self):
        # Detection classes (excluding door)
        self.detection_classes = {
            'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs'
        }
        
        # Class color mapping for visualization
        self.class_colors = {
            'window': [1.0, 0.0, 0.0],    # Red
            'table': [0.0, 0.0, 1.0],     # Blue
            'chair': [1.0, 1.0, 0.0],     # Yellow
            'sofa': [1.0, 0.0, 1.0],      # Magenta
            'bookcase': [0.0, 1.0, 1.0],  # Cyan
            'board': [1.0, 0.5, 0.0],     # Orange
            'stairs': [0.5, 0.0, 1.0]     # Purple
        }
        
        # Bounding box generation parameters
        self.bbox_params = {
            'min_points': 10,
            'min_volume': 0.001,
            'max_aspect_ratio': 10.0,
            'max_vis_points': 1000,  # Maximum points for visualization
        }
    
    def load_annotation_file(self, annotation_file):
        """Load a single annotation file."""
        try:
            data = np.loadtxt(annotation_file)
            if len(data) == 0:
                return None
            
            # S3DIS annotation format: each file contains point cloud for a specific class
            # Format: [x, y, z, r, g, b] or [x, y, z, r, g, b, label]
            points = data[:, :3]  # XYZ coordinates
            colors = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else None
            labels = data[:, 6].astype(int) if data.shape[1] > 6 else None
            
            object_name = annotation_file.stem.split('_')[0]
            
            return {
                'points': points,
                'colors': colors,
                'labels': labels,
                'object_name': object_name,
                'file_name': annotation_file.name
            }
            
        except Exception as e:
            return None
    
    def remove_outliers(self, points, colors=None, k=20, std_ratio=2.0):
        """Remove outliers using statistical methods."""
        if len(points) < k:
            return points, colors
        
        # Use Open3D for outlier removal
        import open3d as o3d
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        
        # Get filtered point cloud
        filtered_pcd = pcd.select_by_index(ind)
        filtered_points = np.asarray(filtered_pcd.points)
        filtered_colors = np.asarray(filtered_pcd.colors) if colors is not None else None
        
        return filtered_points, filtered_colors
    
    def calculate_bbox(self, points):
        """Calculate 3D bounding box."""
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
            'center': center,
            'size': size,
            'min_coords': min_coords,
            'max_coords': max_coords,
            'volume': volume,
            'aspect_ratio': aspect_ratio,
            'point_count': len(points)
        }
    
    def evaluate_bbox_quality(self, bbox, object_name):
        """Evaluate bounding box quality."""
        # Check if class is in detection set
        if object_name not in self.detection_classes:
            return False, f"Class {object_name} not in detection set"
        
        # Check point count
        if bbox['point_count'] < self.bbox_params['min_points']:
            return False, f"Insufficient points: {bbox['point_count']} < {self.bbox_params['min_points']}"
        
        # Check volume
        if bbox['volume'] < self.bbox_params['min_volume']:
            return False, f"Volume too small: {bbox['volume']:.6f} < {self.bbox_params['min_volume']}"
        
        # Check aspect ratio
        if bbox['aspect_ratio'] > self.bbox_params['max_aspect_ratio']:
            return False, f"Aspect ratio too large: {bbox['aspect_ratio']:.2f} > {self.bbox_params['max_aspect_ratio']}"
        
        return True, "Quality OK"
    
    def create_bbox_mesh(self, bbox_info, color=None):
        """Create a mesh for the bounding box."""
        center = bbox_info['center']
        size = bbox_info['size']
        
        # Create axis-aligned bounding box
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = center
        bbox.extent = size
        
        # Set color
        if color is None:
            color = [0.5, 0.5, 0.5]  # Default gray
        bbox.color = color
        
        return bbox
    
    def load_room_point_cloud(self, room_path):
        """Load point cloud data for an entire room."""
        # Try to load main point cloud file
        room_files = list(room_path.glob("*.txt"))
        if not room_files:
            return None
        
        # Use first file as room point cloud
        room_file = room_files[0]
        
        try:
            data = np.loadtxt(room_file)
            if len(data) == 0:
                return None
            
            points = data[:, :3]  # XYZ coordinates
            colors = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else None
            labels = data[:, 6].astype(int) if data.shape[1] > 6 else None
            
            return {
                'points': points,
                'colors': colors,
                'labels': labels,
                'file_name': room_file.name
            }
            
        except Exception as e:
            return None
    
    def visualize_room_bboxes(self, room_path, output_dir=None):
        """Visualize bounding boxes for a single room."""
        annotations_dir = room_path / "Annotations"
        if not annotations_dir.exists():
            return None
        
        # Collect all bounding boxes
        bboxes = []
        
        txt_files = list(annotations_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            data = self.load_annotation_file(txt_file)
            if data is None:
                continue
            
            object_name = data['object_name']
            
            # Only process specified detection classes
            if object_name not in self.detection_classes:
                continue
            
            # Remove outliers
            filtered_points, filtered_colors = self.remove_outliers(data['points'], data['colors'])
            if len(filtered_points) == 0:
                continue
            
            # Calculate bounding box
            bbox_info = self.calculate_bbox(filtered_points)
            if bbox_info is None:
                continue
            
            # Evaluate quality
            is_suitable, reason = self.evaluate_bbox_quality(bbox_info, object_name)
            if not is_suitable:
                continue
            
            # Add object information
            bbox_info['object_name'] = object_name
            bbox_info['file_name'] = data['file_name']
            bbox_info['room_name'] = room_path.name  # Add room name
            
            bboxes.append(bbox_info)
        
        if not bboxes:
            return None
        
        # Generate JSON file for each room (save to simulation_results directory)
        if bboxes:
            # Save directly to simulation_results directory
            simulation_results_dir = Path("simulation_results")
            simulation_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Build complete scene name (Area_X_room_Y format)
            area_name = room_path.parent.name  # Area_1
            room_name = room_path.name         # conferenceRoom_1
            scene_name = f"{area_name}_{room_name}"  # Area_1_conferenceRoom_1
            
            # Check if corresponding scene directory exists
            scene_dir = simulation_results_dir / scene_name
            if not scene_dir.exists():
                scene_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = scene_dir / f"{scene_name}_detection_annotations.json"
            self.generate_detection_annotations(bboxes, json_path)
        
        return bboxes
    
    def process_area(self, area_path, output_dir=None):
        """Process entire area, randomly select 5 rooms for visualization, but generate JSON for all rooms."""
        if not area_path.exists():
            return None
        
        # Get all rooms
        all_rooms = [d for d in area_path.iterdir() if d.is_dir()]
        
        # Randomly select 5 rooms for visualization
        if len(all_rooms) > 5:
            selected_rooms_for_vis = np.random.choice(all_rooms, 5, replace=False).tolist()
        else:
            selected_rooms_for_vis = all_rooms
        
        # Process all rooms to generate JSON
        all_room_bboxes = []
        for room in all_rooms:
            # Don't pass output_dir, let JSON files save directly to room directory
            room_bboxes = self.visualize_room_bboxes(room, None)
            if room_bboxes:
                all_room_bboxes.extend(room_bboxes)
        
        # Visualization data selection commented out to save memory
        # Only create visualization data for selected rooms (from processed rooms)
        # selected_bboxes = []
        # selected_rooms_data = []
        # 
        # # Select representative bounding boxes from processed rooms
        # for room in selected_rooms_for_vis:
        #     # Find corresponding bounding boxes from processed rooms
        #     room_bboxes = [bbox for bbox in all_room_bboxes if bbox.get('room_name') == room.name]
        #     if room_bboxes:
        #         # Select bounding box with largest volume as representative
        #         representative_bbox = max(room_bboxes, key=lambda x: x['volume'])
        #         selected_bboxes.append(representative_bbox)
        #         selected_rooms_data.append({
        #             'room_name': room.name,
        #             'bboxes': room_bboxes,
        #             'representative_bbox': representative_bbox
        #         })
        
        # Simplified statistics
        selected_bboxes = []
        selected_rooms_data = []
        
        # Visualization functionality commented out to save memory and avoid process termination
        # if selected_bboxes:
        #     # Create area summary visualization (only show 5 representative objects)
        #     self.create_area_summary_visualization(selected_bboxes, selected_rooms_data, area_path.name, output_dir)
        
        return selected_bboxes
    
    def create_area_summary_visualization(self, selected_bboxes, selected_rooms, area_name, output_dir):
        """Create area summary visualization (saved as multiple subplots)."""
        if len(selected_bboxes) == 0:
            return
        
        # Set matplotlib backend
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
            plt.close()
            
            # 2. XY projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'XY')
            plt.tight_layout()
            vis_xy_path = output_dir / f"{area_name}_area_xy_view.png"
            plt.savefig(vis_xy_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. XZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'XZ')
            plt.tight_layout()
            vis_xz_path = output_dir / f"{area_name}_area_xz_view.png"
            plt.savefig(vis_xz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. YZ projection
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_bboxes_2d(ax, selected_bboxes, area_name, 'YZ')
            plt.tight_layout()
            vis_yz_path = output_dir / f"{area_name}_area_yz_view.png"
            plt.savefig(vis_yz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Statistics
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_area_statistics(ax, selected_bboxes, selected_rooms, area_name)
            plt.tight_layout()
            stats_path = output_dir / f"{area_name}_area_statistics.png"
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_area_bboxes_3d(self, ax, bboxes, area_name):
        """Plot 3D view of area bounding boxes."""
        for i, bbox in enumerate(bboxes):
            color = self.class_colors.get(bbox['object_name'], [0.5, 0.5, 0.5])
            
            # Draw bounding box
            self.draw_bbox_3d(ax, bbox, color)
            
            # Add label
            center = bbox['center']
            ax.text(center[0], center[1], center[2], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Area View - {area_name}')
        # Remove legend as there are no labels
    
    def draw_bbox_3d(self, ax, bbox_info, color):
        """Draw bounding box in 3D plot."""
        center = bbox_info['center']
        size = bbox_info['size']
        
        # Calculate 8 vertices of bounding box
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
        
        # Define 12 edges of bounding box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Draw bounding box
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                     color=color, linewidth=2, alpha=0.8)
    
    def draw_bbox_2d(self, ax, bbox_info, color, x_idx, y_idx):
        """Draw bounding box projection in 2D plot."""
        center = bbox_info['center']
        size = bbox_info['size']
        
        # Calculate 8 vertices of bounding box
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
        
        # Project to 2D
        proj_vertices = vertices[:, [x_idx, y_idx]]
        
        # Draw bounding box
        from matplotlib.patches import Rectangle
        min_coords = proj_vertices.min(axis=0)
        max_coords = proj_vertices.max(axis=0)
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        
        rect = Rectangle(min_coords, width, height, 
                       fill=False, color=color, linewidth=2, alpha=0.8)
        ax.add_patch(rect)
    
    def plot_area_bboxes_2d(self, ax, bboxes, area_name, projection):
        """Plot 2D projection of area bounding boxes."""
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
            
            # Add label
            center = bbox['center']
            ax.text(center[x_idx], center[y_idx], 
                   f'{bbox["object_name"]}_{i+1}', fontsize=8, color=color)
        
        ax.set_title(f'{projection} Projection - {area_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_area_statistics(self, ax, selected_bboxes, selected_rooms, area_name):
        """Plot area statistics."""
        ax.axis('off')
        
        # Count classes
        class_counts = {}
        for bbox in selected_bboxes:
            class_name = bbox['object_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create statistics text
        stats_text = f"""
Area Summary Statistics - {area_name}:

Selected Representative Bounding Boxes:
• Total: {len(selected_bboxes)}
• From rooms: {len(selected_rooms)}

Class Statistics:
"""
        for class_name, count in sorted(class_counts.items()):
            stats_text += f"• {class_name}: {count}\n"
        
        stats_text += f"\nSelected Rooms:\n"
        for room_data in selected_rooms:
            room_name = room_data['room_name']
            room_bbox_count = len(room_data['bboxes'])
            stats_text += f"• {room_name}: {room_bbox_count} bounding boxes\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def generate_detection_annotations(self, bboxes, output_path):
        """Generate detection annotation file (compatible with Group-Free-3D)."""
        # Class name mapping (compatible with Group-Free-3D)
        class_name_mapping = {
            'window': 'window',
            'table': 'table',
            'chair': 'chair',
            'sofa': 'sofa',
            'bookcase': 'bookshelf',
            'board': 'picture',
            'stairs': 'counter'  # Stairs -> counter (closest category)
        }
        
        detection_annotations = []
        
        for i, bbox in enumerate(bboxes):
            # Get Group-Free-3D compatible class name
            original_class = bbox['object_name']
            groupfree_class = class_name_mapping.get(original_class, original_class)
            
            # Standard 3D object detection annotation format (Group-Free-3D compatible)
            ann = {
                # Basic information
                'instance_id': i + 1,
                'class_name': groupfree_class,  # Use Group-Free-3D compatible class name
                'original_class_name': original_class,  # Keep original class name
                
                # 3D bounding box information (Group-Free-3D standard format)
                'bbox_3d': {
                    'center': bbox['center'].tolist(),  # [x, y, z]
                    'size': bbox['size'].tolist(),      # [length, width, height]
                    'rotation': [0, 0, 0],              # [rx, ry, rz] axis-aligned bounding box
                    'min_coords': bbox['min_coords'].tolist(),
                    'max_coords': bbox['max_coords'].tolist()
                },
                
                # Quality information
                'point_count': bbox['point_count'],
                'volume': bbox['volume'],
                'aspect_ratio': bbox['aspect_ratio'],
                'confidence': 1.0,  # Manual annotation, confidence = 1
                
                # Compatibility information
                'bbox_format': 'AABB',  # Axis-Aligned Bounding Box
                'coordinate_system': 'world',  # World coordinate system
                'units': 'meters',
                'framework': 'Group-Free-3D'  # Specify framework
            }
            detection_annotations.append(ann)
        
        # Create complete annotation file structure (Group-Free-3D compatible)
        annotation_file = {
            'metadata': {
                'dataset': 'S3DIS',
                'annotation_type': '3D_object_detection',
                'framework': 'Group-Free-3D',
                'classes': list(class_name_mapping.values()),  # Group-Free-3D classes
                'original_classes': list(class_name_mapping.keys()),  # Original classes
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
        
        print(f"[Annotation] Saved {len(detection_annotations)} annotations to: {output_path}")
        return detection_annotations


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="S3DIS bounding box generation and visualization tool (headless version)")
    parser.add_argument("--data_root", type=str, required=True, help="S3DIS dataset root directory")
    parser.add_argument("--area_name", type=str, help="Specify area name (e.g., Area_1)")
    parser.add_argument("--room_name", type=str, help="Specify room name (e.g., office_1)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--process_all_areas", action="store_true", help="Process all areas")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[Error] Data root directory does not exist: {data_root}")
        return
    
    # Create visualizer
    visualizer = S3DISBBoxVisualizer()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / "bbox_visualization_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine processing mode
    if args.process_all_areas:
        # Process all areas
        print("[Main] Processing all areas...")
        area_names = [f"Area_{i}" for i in range(1, 7)]
        total_bboxes = 0
        
        for area_name in area_names:
            area_path = data_root / area_name
            if area_path.exists():
                area_bboxes = visualizer.process_area(area_path, output_dir)
                if area_bboxes:
                    total_bboxes += len(area_bboxes)
            else:
                print(f"[Main] Area {area_name} does not exist")
        
        print(f"[Main] All areas processed. Total bounding boxes: {total_bboxes}")
        
    elif args.area_name and args.room_name:
        # Process specified room
        room_path = data_root / args.area_name / args.room_name
        if room_path.exists():
            bboxes = visualizer.visualize_room_bboxes(room_path, None)
            if bboxes:
                print(f"[Main] Room processed: {len(bboxes)} bounding boxes")
            else:
                print("[Main] No valid bounding boxes in room")
        else:
            print(f"[Main] Room does not exist: {room_path}")
    
    elif args.area_name:
        # Process specified area
        area_path = data_root / args.area_name
        if area_path.exists():
            area_bboxes = visualizer.process_area(area_path, output_dir)
            if area_bboxes:
                print(f"[Main] Area processed: {len(area_bboxes)} bounding boxes")
            else:
                print("[Main] No valid bounding boxes in area")
        else:
            print(f"[Main] Area does not exist: {area_path}")
    
    else:
        # Process first found room
        area_names = [f"Area_{i}" for i in range(1, 7)]
        
        for area_name in area_names:
            area_path = data_root / area_name
            if area_path.exists():
                rooms = [d for d in area_path.iterdir() if d.is_dir()]
                if rooms:
                    test_room = rooms[0]
                    bboxes = visualizer.visualize_room_bboxes(test_room, output_dir)
                    if bboxes:
                        # Save detection annotations
                        ann_path = output_dir / f"{area_name}_{test_room.name}_detection_annotations.json"
                        visualizer.generate_detection_annotations(bboxes, ann_path)
                        print(f"[Main] Room processed: {len(bboxes)} bounding boxes")
                    else:
                        print("[Main] No valid bounding boxes in room")
                    break


if __name__ == "__main__":
    main()
