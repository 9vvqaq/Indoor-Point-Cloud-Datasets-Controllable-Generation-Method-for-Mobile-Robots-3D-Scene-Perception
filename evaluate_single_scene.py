#!/usr/bin/env python3
"""
Single scene virtual LiDAR point cloud distribution evaluation script.
Given an S3DIS point cloud, automatically finds a LiDAR-Net point cloud with compatible volume ratio for comparison.
"""

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import random
import time

def load_point_cloud(ply_path):
    """Load PLY point cloud file."""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        return points
    except Exception as e:
        print(f"[Error] Failed to load point cloud {ply_path}: {e}")
        return None

def normalize_coordinates(points, method='center'):
    """Normalize coordinate system."""
    if method == 'center':
        # Method 1: Use room center as origin
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        normalized_points = points - center
    elif method == 'min':
        # Method 2: Use minimum coordinates as origin
        min_coords = points.min(axis=0)
        normalized_points = points - min_coords
    elif method == 'zero_center':
        # Method 3: Center normalization
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        # Set center point as origin
        normalized_points = points - center
    else:
        normalized_points = points
    
    return normalized_points

def sample_points(points, max_points=10000):
    """Sample point cloud to reduce computation."""
    if len(points) <= max_points:
        return points
    
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]

def compute_mmd_sampled(X, Y, max_points=10000, gamma=1.0):
    """Compute MMD - spatial distribution similarity."""
    # Sample point clouds
    X_sampled = sample_points(X, max_points)
    Y_sampled = sample_points(Y, max_points)
    
    # Compute MMD
    def rbf_kernel(X, Y, gamma):
        X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]
        pairwise_dists = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        pairwise_dists = np.maximum(pairwise_dists, 0)
        K = np.exp(-gamma * pairwise_dists)
        return K
    
    K_XX = rbf_kernel(X_sampled, X_sampled, gamma)
    K_YY = rbf_kernel(Y_sampled, Y_sampled, gamma)
    K_XY = rbf_kernel(X_sampled, Y_sampled, gamma)
    
    m, n = len(X_sampled), len(Y_sampled)
    mmd = (np.sum(K_XX) / (m * m) + 
           np.sum(K_YY) / (n * n) - 
           2 * np.sum(K_XY) / (m * n))
    
    return mmd

def compute_chamfer_distance(X, Y):
    """Compute Chamfer distance - overall aggregation."""
    # Sample point clouds
    X_sampled = sample_points(X, 5000)
    Y_sampled = sample_points(Y, 5000)
    
    # Compute distance matrix
    dist_matrix = np.linalg.norm(X_sampled[:, np.newaxis] - Y_sampled, axis=2)
    
    # Chamfer distance
    dist_XY = np.min(dist_matrix, axis=1)
    dist_YX = np.min(dist_matrix, axis=0)
    
    cd = np.mean(dist_XY) + np.mean(dist_YX)
    
    return cd

def compute_hausdorff_distance(X, Y):
    """Compute Hausdorff distance - local aggregation."""
    # Sample point clouds
    X_sampled = sample_points(X, 3000)
    Y_sampled = sample_points(Y, 3000)
    
    # Compute distance matrix
    dist_matrix = np.linalg.norm(X_sampled[:, np.newaxis] - Y_sampled, axis=2)
    
    # Hausdorff distance
    hd = max(np.max(np.min(dist_matrix, axis=1)), 
             np.max(np.min(dist_matrix, axis=0)))
    
    return hd

def analyze_point_cloud(points, name, normalize=True):
    """Analyze basic point cloud features."""
    # Coordinate normalization
    if normalize:
        normalized_points = normalize_coordinates(points, method='zero_center')
    else:
        normalized_points = points
    
    # Compute density
    x_range = normalized_points[:, 0].max() - normalized_points[:, 0].min()
    y_range = normalized_points[:, 1].max() - normalized_points[:, 1].min()
    z_range = normalized_points[:, 2].max() - normalized_points[:, 2].min()
    volume = x_range * y_range * z_range
    density = len(normalized_points) / volume if volume > 0 else 0
    
    return {
        'count': len(points),
        'volume': volume,
        'density': density,
        'normalized_points': normalized_points
    }

def check_volume_compatibility(volume1, volume2, threshold=0.3):
    """Check volume compatibility."""
    volume_diff = abs(volume1 - volume2) / max(volume1, volume2)
    is_compatible = volume_diff <= threshold
    
    return is_compatible, volume_diff

def find_lidar_net_scenes(data_root):
    """Find all LiDAR-Net scenes."""
    lidar_net_scenes = []
    
    if not os.path.exists(data_root):
        print(f"[Error] Data root directory does not exist: {data_root}")
        return lidar_net_scenes
    
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            # Skip directories containing 'Area' (these are S3DIS scenes)
            if 'Area' not in item:
                ply_file = os.path.join(item_path, f'{item}.ply')
                if os.path.exists(ply_file):
                    lidar_net_scenes.append({
                        'name': item,
                        'path': item_path,
                        'ply_file': ply_file
                    })
    
    return lidar_net_scenes

def evaluate_single_scene(s3dis_ply, lidar_net_ply, max_points=10000, volume_threshold=0.3):
    """Evaluate distribution similarity for a single scene."""
    s3dis_points = load_point_cloud(s3dis_ply)
    lidar_net_points = load_point_cloud(lidar_net_ply)
    
    if s3dis_points is None or lidar_net_points is None:
        return None
    
    # Analyze point clouds (including coordinate normalization)
    s3dis_stats = analyze_point_cloud(s3dis_points, "S3DIS", normalize=True)
    lidar_net_stats = analyze_point_cloud(lidar_net_points, "LiDAR-Net", normalize=True)
    
    # Check volume compatibility
    is_compatible, volume_diff = check_volume_compatibility(
        s3dis_stats['volume'], lidar_net_stats['volume'], volume_threshold)
    
    if not is_compatible:
        return None
    
    # Use normalized points to compute metrics
    s3dis_normalized = s3dis_stats['normalized_points']
    lidar_net_normalized = lidar_net_stats['normalized_points']
    
    # Compute various metrics
    mmd = compute_mmd_sampled(s3dis_normalized, lidar_net_normalized, max_points)
    cd = compute_chamfer_distance(s3dis_normalized, lidar_net_normalized)
    hd = compute_hausdorff_distance(s3dis_normalized, lidar_net_normalized)
    
    # Compute density ratio
    density_ratio = s3dis_stats['density'] / lidar_net_stats['density']
    
    return {
        'mmd': mmd,
        'cd': cd,
        'hd': hd,
        'density_ratio': density_ratio,
        's3dis_points': len(s3dis_points),
        'lidar_net_points': len(lidar_net_points),
        's3dis_density': s3dis_stats['density'],
        'lidar_net_density': lidar_net_stats['density'],
        's3dis_volume': s3dis_stats['volume'],
        'lidar_net_volume': lidar_net_stats['volume'],
        'volume_diff': volume_diff
    }

def find_best_match(s3dis_ply, data_root, max_points=10000, volume_threshold=0.3, max_candidates=50):
    """Find the best matching LiDAR-Net scene for the specified S3DIS point cloud."""
    # Load and analyze S3DIS point cloud
    s3dis_points = load_point_cloud(s3dis_ply)
    if s3dis_points is None:
        return None
    
    s3dis_stats = analyze_point_cloud(s3dis_points, "S3DIS", normalize=True)
    s3dis_volume = s3dis_stats['volume']
    
    # Find all LiDAR-Net scenes
    lidar_net_scenes = find_lidar_net_scenes(data_root)
    if not lidar_net_scenes:
        print("[Error] No LiDAR-Net scenes found")
        return None
    
    # Limit candidate count and shuffle randomly
    candidates = lidar_net_scenes[:max_candidates]
    random.shuffle(candidates)
    
    best_match = None
    best_volume_diff = float('inf')
    compatible_scenes = []
    
    # Iterate through candidate scenes to find volume-compatible ones
    for i, scene in enumerate(tqdm(candidates, desc="Finding matching scene")):
        # Quick volume check
        lidar_net_points = load_point_cloud(scene['ply_file'])
        if lidar_net_points is None:
            continue
        
        # Compute volume (without coordinate normalization to save time)
        x_range = lidar_net_points[:, 0].max() - lidar_net_points[:, 0].min()
        y_range = lidar_net_points[:, 1].max() - lidar_net_points[:, 1].min()
        z_range = lidar_net_points[:, 2].max() - lidar_net_points[:, 2].min()
        lidar_net_volume = x_range * y_range * z_range
        
        volume_diff = abs(s3dis_volume - lidar_net_volume) / max(s3dis_volume, lidar_net_volume)
        
        if volume_diff <= volume_threshold:
            # Perform detailed evaluation
            result = evaluate_single_scene(s3dis_ply, scene['ply_file'], max_points, volume_threshold)
            
            if result:
                result['s3dis_scene'] = os.path.basename(s3dis_ply)
                result['lidar_net_scene'] = scene['name']
                result['lidar_net_ply'] = scene['ply_file']
                compatible_scenes.append(result)
                
                # If this is the first compatible scene, return immediately
                if not best_match:
                    best_match = result
                    best_volume_diff = volume_diff
                    break
    
    if not best_match:
        print(f"[Warning] No volume-compatible LiDAR-Net scene found")
        print(f"  Tried {len(candidates)} candidate scenes")
        print(f"  Volume threshold: {volume_threshold:.1%}")
    
    return best_match

def main():
    parser = argparse.ArgumentParser(description='Single scene virtual LiDAR point cloud distribution evaluation - specify S3DIS, automatically find matching LiDAR-Net')
    parser.add_argument('--s3dis_ply', required=True, 
                       help='S3DIS point cloud file path')
    parser.add_argument('--data_root', default='simulation_results', 
                       help='LiDAR-Net data root directory')
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='Output directory')
    parser.add_argument('--max_points', type=int, default=10000, 
                       help='Maximum points per scene')
    parser.add_argument('--volume_threshold', type=float, default=0.3, 
                       help='Volume difference threshold')
    parser.add_argument('--max_candidates', type=int, default=50, 
                       help='Maximum number of candidate scenes')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[Evaluation] Starting single scene virtual LiDAR point cloud distribution evaluation...")
    print(f"  S3DIS file: {args.s3dis_ply}")
    print(f"  LiDAR-Net directory: {args.data_root}")
    print(f"  Volume threshold: {args.volume_threshold:.1%}")
    print(f"  Evaluation metrics: MMD + CD + HD + density analysis")
    
    start_time = time.time()
    
    # Check if S3DIS file exists
    if not os.path.exists(args.s3dis_ply):
        print(f"[Error] S3DIS file does not exist: {args.s3dis_ply}")
        return
    
    # Find best match
    result = find_best_match(
        args.s3dis_ply, 
        args.data_root, 
        args.max_points, 
        args.volume_threshold,
        args.max_candidates
    )
    
    if result is None:
        print("[Error] No suitable matching scene found")
        return
    
    # Save results
    output_file = os.path.join(args.output_dir, 'single_scene_evaluation.txt')
    with open(output_file, 'w') as f:
        f.write("Single Scene Virtual LiDAR Point Cloud Distribution Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Evaluation metrics: MMD + CD + HD + density analysis\n")
        f.write(f"Improvements: coordinate normalization + volume filtering (threshold: {args.volume_threshold:.1%})\n\n")
        
        f.write(f"S3DIS scene: {result['s3dis_scene']}\n")
        f.write(f"LiDAR-Net scene: {result['lidar_net_scene']}\n")
        f.write(f"LiDAR-Net file: {result['lidar_net_ply']}\n\n")
        
        f.write("Evaluation Results:\n")
        f.write(f"  MMD: {result['mmd']:.4f} (spatial distribution similarity)\n")
        f.write(f"  CD: {result['cd']:.4f} (overall aggregation)\n")
        f.write(f"  HD: {result['hd']:.4f} (local aggregation)\n")
        f.write(f"  Density ratio: {result['density_ratio']:.4f} (S3DIS density / LiDAR-Net density)\n")
        f.write(f"  Volume difference: {result['volume_diff']:.2%}\n\n")
        
        f.write("Point Cloud Statistics:\n")
        f.write(f"  S3DIS points: {result['s3dis_points']:,}\n")
        f.write(f"  LiDAR-Net points: {result['lidar_net_points']:,}\n")
        f.write(f"  S3DIS density: {result['s3dis_density']:.2f} points/m³\n")
        f.write(f"  LiDAR-Net density: {result['lidar_net_density']:.2f} points/m³\n")
        f.write(f"  S3DIS volume: {result['s3dis_volume']:.2f}\n")
        f.write(f"  LiDAR-Net volume: {result['lidar_net_volume']:.2f}\n")
    
    total_time = time.time() - start_time
    print(f"\n[Evaluation] Evaluation completed! Total time: {total_time:.2f} seconds")
    print(f"  Results saved to: {output_file}")
    
    # Output summary
    print(f"\n[Summary]")
    print(f"  S3DIS scene: {result['s3dis_scene']}")
    print(f"  Matched scene: {result['lidar_net_scene']}")
    print(f"  MMD: {result['mmd']:.4f}")
    print(f"  CD: {result['cd']:.4f}")
    print(f"  HD: {result['hd']:.4f}")
    print(f"  Density ratio: {result['density_ratio']:.4f}")
    print(f"  Volume difference: {result['volume_diff']:.2%}")

if __name__ == '__main__':
    main()
