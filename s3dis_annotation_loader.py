"""
S3DIS annotation utilities.
Load semantic labels from S3DIS annotation folders and support color encoding.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import glob


class S3DISAnnotationLoader:
    """
    Loader for S3DIS point-level annotations.
    
    Capabilities:
    1. Read S3DIS annotation files.
    2. Support label mapping for selected classes.
    3. Convert labels into color encodings.
    """
    
    def __init__(self, data_root: str):
        """
        Initialize the loader.
        
        Args:
            data_root: Root path to the S3DIS dataset.
        """
        self.data_root = data_root
        
        # Mapping from S3DIS classes to LiDAR-Net classes
        self.class_mapping = {
            # S3DIS class -> LiDAR-Net class
            'floor': 'floor',
            'ceiling': 'ceiling', 
            'wall': 'wall',
            'window': 'window',
            'table': 'table',
            'chair': 'chair',
            'sofa': 'sofa',
            'bookcase': 'bookshelf',
            'board': 'blackboard',
            'stairs': 'stair'  # Special handling
        }
        
        # Valid S3DIS classes that will be processed
        self.valid_classes = list(self.class_mapping.keys())
        
        # Original S3DIS class ID mapping
        self.s3dis_class_ids = {
            'ceiling': 0,
            'floor': 1, 
            'wall': 2,
            'beam': 3,
            'column': 4,
            'window': 5,
            'door': 6,
            'table': 7,
            'chair': 8,
            'sofa': 9,
            'bookcase': 10,
            'board': 11,
            'clutter': 12
        }
        
        print("[AnnotationLoader] Initialized.")
        print(f"  - Data root: {data_root}")
        print(f"  - Valid classes: {len(self.valid_classes)}")
        print(f"  - Class mapping: {self.class_mapping}")
    
    def load_room_annotations(self, area: str, room: str) -> Dict[str, np.ndarray]:
        """
        Load annotation data for a given room.
        
        Args:
            area: Area identifier (e.g., "Area_1").
            room: Room identifier (e.g., "conferenceRoom_1").
            
        Returns:
            Dictionary mapping instance names to point clouds.
        """
        print(f"[Load] Room annotations: {area}/{room}")
        
        annotation_dir = os.path.join(self.data_root, area, room, "Annotations")
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"Annotation directory missing: {annotation_dir}")
        
        room_annotations = {}
        
        for class_name in self.valid_classes:
            pattern = os.path.join(annotation_dir, f"{class_name}_*.txt")
            annotation_files = glob.glob(pattern)
            
            if annotation_files:
                print(f"  - Found {len(annotation_files)} files for class '{class_name}'")
                
                for i, file_path in enumerate(annotation_files):
                    points = self._load_annotation_file(file_path)
                    if len(points) > 0:
                        instance_name = f"{class_name}_{i+1}"
                        room_annotations[instance_name] = points
                        print(f"    - {instance_name}: {len(points)} points")
                
                if not any(len(self._load_annotation_file(f)) > 0 for f in annotation_files):
                    print(f"    - Warning: no valid points for class '{class_name}'")
            else:
                print(f"  - No annotation files found for class '{class_name}'")
        
        return room_annotations
    
    def _load_annotation_file(self, file_path: str) -> np.ndarray:
        """
        Load a single annotation text file.
        
        Args:
            file_path: Path to the annotation file.
            
        Returns:
            Array of point coordinates [N, 3].
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            points = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    coords = line.split()
                    if len(coords) >= 3:
                        try:
                            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue
            
            return np.array(points) if points else np.array([]).reshape(0, 3)
            
        except Exception as e:
            print(f"    - Warning: failed to load {file_path}: {e}")
            return np.array([]).reshape(0, 3)
    
    def create_labeled_pointcloud(self, room_annotations: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a labeled point cloud from room annotations.
        
        Args:
            room_annotations: Annotation dictionary for the room.
            
        Returns:
            Tuple of (points, labels).
        """
        print("[PointCloud] Building labeled point cloud from annotations...")
        
        all_points = []
        all_labels = []
        
        for class_name, points in room_annotations.items():
            if len(points) > 0:
                all_points.append(points)
                class_id = self.s3dis_class_ids.get(class_name, -1)
                if class_id >= 0:
                    labels = np.full(len(points), class_id, dtype=np.int32)
                    all_labels.append(labels)
                    print(f"  - {class_name}: {len(points)} points (label ID {class_id})")
        
        if all_points:
            points = np.vstack(all_points)
            labels = np.concatenate(all_labels)
            print(f"  - Total points: {len(points)}")
            print(f"  - Label range: {labels.min()} - {labels.max()}")
            return points, labels
        else:
            print("  - Warning: no valid points found.")
            return np.array([]).reshape(0, 3), np.array([], dtype=np.int32)
    
    def create_labeled_pointcloud_with_instances(self, room_annotations: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a point cloud with semantic labels and instance ids.
        
        Args:
            room_annotations: Annotation dictionary for the room.
            
        Returns:
            Tuple of (points, labels, instances).
        """
        print("[PointCloud] Building labeled + instance point cloud...")
        
        all_points = []
        all_labels = []
        all_instances = []
        
        for instance_name, points in room_annotations.items():
            if len(points) > 0:
                all_points.append(points)
                
                # Derive class name from instance identifier (e.g., "chair_1" -> "chair")
                if '_' in instance_name:
                    class_name = instance_name.split('_')[0]
                else:
                    class_name = instance_name
                
                class_id = self.s3dis_class_ids.get(class_name, -1)
                if class_id >= 0:
                    labels = np.full(len(points), class_id, dtype=np.int32)
                    all_labels.append(labels)
                    
                    # Parse instance id suffix (defaults to 1 when missing)
                    if '_' in instance_name:
                        try:
                            instance_id = int(instance_name.split('_')[-1])
                        except ValueError:
                            instance_id = 1
                    else:
                        instance_id = 1
                    
                    instances = np.full(len(points), instance_id, dtype=np.int32)
                    all_instances.append(instances)
                    
                    print(f"  - {instance_name}: {len(points)} points (label {class_id}, instance {instance_id})")
        
        if all_points:
            print(f"  - Debug: all_points length = {len(all_points)}")
            print(f"  - Debug: all_labels length = {len(all_labels)}")
            print(f"  - Debug: all_instances length = {len(all_instances)}")
            try:
                points = np.vstack(all_points)
                labels = np.concatenate(all_labels)
                instances = np.concatenate(all_instances)
                print(f"  - Total points: {len(points)}")
                print(f"  - Label range: {labels.min()} - {labels.max()}")
                print(f"  - Instance range: {instances.min()} - {instances.max()}")
                return points, labels, instances
            except Exception as e:
                print(f"  - Error: failed to concatenate arrays: {e}")
                print(f"  - Debug: all_points shapes: {[p.shape for p in all_points]}")
                print(f"  - Debug: all_labels shapes: {[l.shape for l in all_labels]}")
                print(f"  - Debug: all_instances shapes: {[i.shape for i in all_instances]}")
                return np.array([]).reshape(0, 3), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        else:
            print("  - Warning: no valid points found.")
            print(f"  - Debug: all_points length = {len(all_points)}")
            print(f"  - Debug: room annotation keys = {list(room_annotations.keys())}")
            return np.array([]).reshape(0, 3), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    def filter_valid_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Filter labels, retaining only classes that have mappings.
        
        Args:
            labels: Original label array.
            
        Returns:
            Filtered label array.
        """
        valid_class_ids = [self.s3dis_class_ids[class_name] 
                          for class_name in self.valid_classes 
                          if class_name in self.s3dis_class_ids]
        
        valid_mask = np.isin(labels, valid_class_ids)
        
        filtered_labels = labels.copy()
        filtered_labels[~valid_mask] = -1
        
        print(f"[LabelFilter] Total labels: {len(labels)}")
        print(f"  - Valid labels: {np.sum(valid_mask)}")
        print(f"  - Invalid labels: {np.sum(~valid_mask)}")
        
        return filtered_labels


class S3DISColorEncoder:
    """
    Color encoder for S3DIS semantic and instance labels.
    Encodes both class and instance identifiers deterministically into RGB.
    """
    
    def __init__(self):
        """Initialize the color encoder."""
        # Base RGB colors per class (0-255)
        self.class_base_colors = {
            'floor': [100, 50, 25],        # Brown tone
            'ceiling': [200, 200, 200],    # White tone
            'wall': [150, 150, 150],       # Gray tone
            'window': [50, 150, 200],      # Blue tone
            'table': [100, 50, 25],        # Dark brown
            'chair': [200, 50, 50],        # Red tone
            'sofa': [150, 50, 150],        # Purple tone
            'bookcase': [50, 100, 50],     # Green tone
            'board': [25, 25, 25],         # Dark tone
            'stairs': [200, 150, 50]       # Yellow tone
        }
        
        # Mapping from S3DIS class IDs to names
        self.id_to_class = {
            1: 'floor',      # floor
            0: 'ceiling',    # ceiling
            2: 'wall',       # wall
            5: 'window',    # window
            7: 'table',     # table
            8: 'chair',     # chair
            9: 'sofa',      # sofa
            10: 'bookcase', # bookcase
            11: 'board',    # board
        }
        
        # Instance encoding parameters
        self.max_instances_per_class = 20
        self.instance_step = 1
        
        print("[ColorEncoder] Initialized.")
        print(f"  - Supported classes: {len(self.class_base_colors)}")
        print(f"  - Max instances per class: {self.max_instances_per_class}")
        print("  - Encoding: label + instance → RGB")
    
    def encode_labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode labels into RGB colors (semantic only).
        
        Args:
            labels: Label array [N].
            
        Returns:
            Color array [N, 3].
        """
        print("[ColorEncoder] Encoding labels...")
        
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        
        for i, label in enumerate(labels):
            if label in self.id_to_class:
                class_name = self.id_to_class[label]
                base_color = self.class_base_colors[class_name]
                colors[i] = [c/255.0 for c in base_color]
            else:
                colors[i] = [0.0, 0.0, 0.0]
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("  - Label distribution:")
        for label, count in zip(unique_labels, counts):
            if label in self.id_to_class:
                class_name = self.id_to_class[label]
                color = [c/255.0 for c in self.class_base_colors[class_name]]
                print(f"    Label {label} ({class_name}): {count} pts, color RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
            else:
                print(f"    Label {label} (invalid): {count} pts")
        
        return colors
    
    def encode_labels_and_instances_to_colors(self, labels: np.ndarray, instances: np.ndarray) -> np.ndarray:
        """
        Encode labels and instance ids into RGB colors.
        
        Args:
            labels: Label array [N].
            instances: Instance array [N].
            
        Returns:
            Color array [N, 3] in [0, 1].
        """
        print("[ColorEncoder] Encoding labels + instances...")
        
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        
        for i, (label, instance) in enumerate(zip(labels, instances)):
            if label in self.id_to_class:
                class_name = self.id_to_class[label]
                base_color = self.class_base_colors[class_name]
                
                # Encode instance id into blue channel
                instance_id = int(instance) if instance >= 0 else 0
                instance_id = min(instance_id, self.max_instances_per_class - 1)
                
                # Encode as [R, G, B + instance id]
                encoded_color = [
                    base_color[0] / 255.0,
                    base_color[1] / 255.0,
                    (base_color[2] + instance_id) / 255.0
                ]
                
                colors[i] = encoded_color
            else:
                colors[i] = [0.0, 0.0, 0.0]
        
        unique_combinations = {}
        for label, instance in zip(labels, instances):
            if label in self.id_to_class:
                class_name = self.id_to_class[label]
                key = f"{class_name}_{int(instance)}"
                unique_combinations[key] = unique_combinations.get(key, 0) + 1
        
        print("  - Label + instance distribution:")
        for combo, count in unique_combinations.items():
            class_name, instance_id = combo.split('_')
            base_color = self.class_base_colors[class_name]
            encoded_color = [
                base_color[0] / 255.0,
                base_color[1] / 255.0,
                (base_color[2] + int(instance_id)) / 255.0
            ]
            print(f"    {combo}: {count} pts, color RGB({encoded_color[0]:.3f}, {encoded_color[1]:.3f}, {encoded_color[2]:.3f})")
        
        return colors
    
    def decode_colors_to_labels_and_instances(self, colors: np.ndarray) -> tuple:
        """
        Decode colors back into labels and instance ids.
        
        Args:
            colors: Color array [N, 3] in [0, 1].
            
        Returns:
            Tuple of (labels, instances).
        """
        print("[ColorEncoder] Decoding colors...")
        
        labels = np.zeros(len(colors), dtype=np.int32)
        instances = np.zeros(len(colors), dtype=np.int32)
        
        # Convert back to 0-255 range
        colors_255 = (colors * 255).astype(np.int32)
        
        for i, (r, g, b) in enumerate(colors_255):
            # Find nearest base color by Manhattan distance
            best_class = None
            best_distance = float('inf')
            
            for class_name, base_color in self.class_base_colors.items():
                distance = abs(r - base_color[0]) + abs(g - base_color[1])
                if distance < best_distance:
                    best_distance = distance
                    best_class = class_name
            
            if best_class:
                # Resolve class ID
                for class_id, class_name in self.id_to_class.items():
                    if class_name == best_class:
                        labels[i] = class_id
                        break
                
                # Decode instance id from blue channel
                base_b = self.class_base_colors[best_class][2]
                instance_id = max(0, b - base_b)
                instances[i] = min(instance_id, self.max_instances_per_class - 1)
            else:
                labels[i] = -1
                instances[i] = -1
        
        print(f"  - Decoded points: {len(labels)}")
        print(f"  - Label range: {labels.min()} - {labels.max()}")
        print(f"  - Instance range: {instances.min()} - {instances.max()}")
        
        return labels, instances
    
    def _assign_colors_to_points(self, input_points, annotation_points, annotation_labels):
        """Assign colors to input points via nearest-neighbor matching."""
        from sklearn.neighbors import NearestNeighbors
        
        # Use nearest-neighbor mapping
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(annotation_points)
        distances, indices = nbrs.kneighbors(input_points)
        
        # Assign labels/colors from closest annotation point
        assigned_labels = annotation_labels[indices.flatten()]
        
        # Encode colors
        colors = self.encode_labels_to_colors(assigned_labels)
        
        return colors


def load_s3dis_room_labels(data_root: str, area: str, room: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load labels and colors for an S3DIS room.
    
    Args:
        data_root: Dataset root.
        area: Area identifier (e.g., "Area_1").
        room: Room identifier (e.g., "conferenceRoom_1").
        
    Returns:
        Tuple of (points, labels, colors).
    """
    print(f"[S3DIS] Loading room annotations: {area}/{room}")
    
    loader = S3DISAnnotationLoader(data_root)
    color_encoder = S3DISColorEncoder()
    
    try:
        room_annotations = loader.load_room_annotations(area, room)
        
        if not room_annotations:
            print("[S3DIS] Warning: no annotation data found.")
            return np.array([]).reshape(0, 3), np.array([], dtype=np.int32), np.array([]).reshape(0, 3)
        
        points, labels = loader.create_labeled_pointcloud(room_annotations)
        
        if len(points) == 0:
            print("[S3DIS] Warning: no valid point cloud.")
            return np.array([]).reshape(0, 3), np.array([], dtype=np.int32), np.array([]).reshape(0, 3)
        
        filtered_labels = loader.filter_valid_labels(labels)
        
        colors = color_encoder.encode_labels_to_colors(filtered_labels)
        
        print("[S3DIS] Load complete.")
        print(f"  - Points: {len(points)}")
        print(f"  - Labels: {len(filtered_labels)}")
        print(f"  - Colors: {len(colors)}")
        
        return points, filtered_labels, colors
        
    except Exception as e:
        print(f"[S3DIS] Failed to load annotations: {e}")
        return np.array([]).reshape(0, 3), np.array([], dtype=np.int32), np.array([]).reshape(0, 3)


def get_semantic_colors_from_points(points: np.ndarray, data_root: str, area: str, room: str) -> np.ndarray:
    """
    Assign semantic colors to input points using S3DIS annotations.
    
    Args:
        points: Input points [N, 3].
        data_root: S3DIS dataset root.
        area: Area identifier.
        room: Room identifier.
        
    Returns:
        colors: Semantic colors [N, 3].
    """
    print(f"[SemanticColor] Assigning semantic colors for {area}/{room}")
    
    color_encoder = S3DISColorEncoder()
    
    try:
        loader = S3DISAnnotationLoader(data_root)
        room_annotations = loader.load_room_annotations(area, room)
        
        if not room_annotations:
            print("[SemanticColor] Warning: no annotation data, using fallback colors.")
            return color_encoder._generate_basic_colors(points)
        
        annotation_points, labels = loader.create_labeled_pointcloud(room_annotations)
        
        if len(annotation_points) == 0:
            print("[SemanticColor] Warning: no valid annotation points, using fallback colors.")
            return color_encoder._generate_basic_colors(points)
        
        filtered_labels = loader.filter_valid_labels(labels)
        
        colors = color_encoder._assign_colors_to_points(points, annotation_points, filtered_labels)
        
        print("[SemanticColor] Color assignment complete.")
        print(f"  - Input points: {len(points)}")
        print(f"  - Annotation points: {len(annotation_points)}")
        print(f"  - Colors produced: {len(colors)}")
        
        return colors
        
    except Exception as e:
        print(f"[SemanticColor] Failed to assign colors: {e}")
        print("  - Falling back to basic colors.")
        return color_encoder._generate_basic_colors(points)


def get_semantic_colors_with_instances_from_points(points: np.ndarray, data_root: str, area: str, room: str) -> np.ndarray:
    """
    Assign semantic + instance colors to input points using S3DIS annotations.
    
    Args:
        points: Input points [N, 3].
        data_root: S3DIS dataset root.
        area: Area identifier.
        room: Room identifier.
        
    Returns:
        colors: Semantic + instance colors [N, 3].
    """
    print(f"[SemanticColor+Instance] Assigning semantic+instance colors for {area}/{room}")
    
    color_encoder = S3DISColorEncoder()
    
    try:
        loader = S3DISAnnotationLoader(data_root)
        room_annotations = loader.load_room_annotations(area, room)
        
        if not room_annotations:
            print("[SemanticColor+Instance] Warning: no annotation data, using fallback colors.")
            return color_encoder._generate_basic_colors(points)
        
        annotation_points, labels, instances = loader.create_labeled_pointcloud_with_instances(room_annotations)
        
        if len(annotation_points) == 0:
            print("[SemanticColor+Instance] Warning: no valid annotation points, using fallback colors.")
            return color_encoder._generate_basic_colors(points)
        
        filtered_labels = loader.filter_valid_labels(labels)
        filtered_instances = instances
        
        annotation_colors = color_encoder.encode_labels_and_instances_to_colors(filtered_labels, filtered_instances)
        
        colors = color_encoder._assign_colors_to_points(points, annotation_points, annotation_colors)
        
        print("[SemanticColor+Instance] Color assignment complete.")
        print(f"  - Input points: {len(points)}")
        print(f"  - Annotation points: {len(annotation_points)}")
        print(f"  - Colors produced: {len(colors)}")
        
        return colors
        
    except Exception as e:
        print(f"[SemanticColor+Instance] Failed to assign colors: {e}")
        print("  - Falling back to basic colors.")
        return color_encoder._generate_basic_colors(points)




if __name__ == "__main__":
    demo_s3dis_annotation_loading()
