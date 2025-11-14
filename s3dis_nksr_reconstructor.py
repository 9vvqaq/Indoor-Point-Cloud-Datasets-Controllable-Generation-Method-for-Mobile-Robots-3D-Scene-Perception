#!/usr/bin/env python3
"""
S3DIS NKSR reconstruction and visualization utilities.
Build meshes from preprocessed data and generate evaluation artifacts.
"""

import argparse
import numpy as np
import open3d as o3d
import torch
import nksr
import yaml
import pickle
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple

# Load S3DIS annotation helpers
from s3dis_annotation_loader import load_s3dis_room_labels


class S3DISReconstructor:
    """NKSR-based reconstructor for S3DIS rooms."""
    
    def __init__(self, config_dict):
        self.config = config_dict['nksr']
        
        # Select compute device
        if self.config['device'] == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config['device'])
        
        print(f"[Reconstructor] Using device: {self.device}")
        
        # Initialize NKSR reconstructor
        self.reconstructor = nksr.Reconstructor(self.device)
        if self.config['chunked']:
            self.reconstructor.chunk_tmp_device = torch.device("cpu")
            print(f"[Reconstructor] Chunked reconstruction enabled (chunk size: {self.config['chunk_size']}).")
    
    def reconstruct_mesh(self, points, normals=None, colors=None, use_semantic_colors=False, s3dis_data_root=None, area=None, room=None):
        """Reconstruct a mesh using NKSR from point cloud inputs."""
        print(f"[NKSR] Starting reconstruction with {len(points)} input points.")
        start_time = time.time()
        
        # Convert to torch tensors
        input_xyz = torch.from_numpy(points.astype(np.float32)).to(self.device)
        
        if normals is not None:
            input_normal = torch.from_numpy(normals.astype(np.float32)).to(self.device)
            print("[NKSR] Using provided normals.")
        else:
            print("[NKSR] Warning: normals not provided; reconstruction fidelity may degrade.")
            input_normal = None
        
        # Handle color attributes
        if use_semantic_colors and s3dis_data_root is not None and area is not None and room is not None:
            print("[NKSR] Applying semantic label colors.")
            from s3dis_annotation_loader import get_semantic_colors_from_points
            semantic_colors = get_semantic_colors_from_points(points, s3dis_data_root, area, room)
            input_color = torch.from_numpy(semantic_colors.astype(np.float32)).to(self.device)
        else:
            print("[NKSR] Using provided vertex colors.")
            input_color = torch.from_numpy(colors.astype(np.float32)).to(self.device)
        
        try:
            if self.config['chunked']:
                field = self.reconstructor.reconstruct(
                    input_xyz,
                    normal=input_normal,
                    voxel_size=0.02
                )
            else:
                field = self.reconstructor.reconstruct(
                    input_xyz,
                    normal=input_normal,
                    voxel_size=0.02
                )
            
            if colors is not None:
                print("[NKSR] Applying color texture.")
                input_color = torch.from_numpy(colors.astype(np.float32)).to(self.device)
                field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
            
            print("[NKSR] Extracting mesh via dual contouring.")
            mesh = field.extract_dual_mesh(mise_iter=2)
            
            reconstruction_time = time.time() - start_time
            stats = {
                "input_points": len(points),
                "output_vertices": len(mesh.v),
                "output_faces": len(mesh.f),
                "reconstruction_time": reconstruction_time,
                "has_colors": colors is not None and hasattr(mesh, 'c'),
                "device_used": str(self.device)
            }
            
            print("[NKSR] Reconstruction finished.")
            print(f"  - Input points: {stats['input_points']}")
            print(f"  - Output vertices: {stats['output_vertices']}")
            print(f"  - Output faces: {stats['output_faces']}")
            print(f"  - Runtime: {stats['reconstruction_time']:.2f}s")
            
            return mesh, stats
            
        except Exception as e:
            print(f"[NKSR] Reconstruction failed: {e}")
            raise
    
    def save_mesh(self, mesh, output_path, stats=None):
        """Persist reconstructed mesh to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[Mesh] Saving to: {output_path}")
        
        # Convert to Open3D format
        mesh_o3d = o3d.geometry.TriangleMesh()
        try:
            vertices = mesh.v.cpu().numpy()
        except:
            vertices = np.array(mesh.v)
        
        try:
            faces = mesh.f.cpu().numpy()
        except:
            faces = np.array(mesh.f)
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        
        if hasattr(mesh, 'c') and mesh.c is not None:
            try:
                colors = mesh.c.cpu().numpy()
            except:
                colors = np.array(mesh.c)
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
            print("[Mesh] Vertex colors embedded.")
        
        mesh_o3d.compute_vertex_normals()
        
        success = o3d.io.write_triangle_mesh(str(output_path), mesh_o3d)
        
        if success:
            print(f"[Mesh] Saved mesh: {output_path}")
            if stats is not None:
                stats_path = output_path.with_suffix('.json')
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"[Mesh] Metrics written to: {stats_path}")
        else:
            print(f"[Mesh] Failed to save mesh: {output_path}")


class S3DISVisualizer:
    """Visualization helper for S3DIS reconstruction outputs."""
    
    def __init__(self, config_dict):
        self.config = config_dict['visualization']
    
    def visualize_pointcloud(self, points, colors=None, output_path=None, title="Point Cloud"):
        """Render 2D projections of a point cloud."""
        if not self.config['save_visualizations']:
            return
        
        print(f"[Visualization] Rendering point cloud projections: {title}")
        
        max_points = self.config['max_vis_points']
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            vis_points = points[indices]
            vis_colors = colors[indices] if colors is not None else None
        else:
            vis_points = points
            vis_colors = colors
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if vis_colors is not None:
            scatter = axes[0].scatter(vis_points[:, 0], vis_points[:, 1], 
                                    c=vis_colors, s=0.5, alpha=0.6)
        else:
            scatter = axes[0].scatter(vis_points[:, 0], vis_points[:, 1], 
                                    c=vis_points[:, 2], s=0.5, alpha=0.6, cmap='viridis')
        axes[0].set_title(f'{title} - XY')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].axis('equal')
        
        if vis_colors is not None:
            axes[1].scatter(vis_points[:, 0], vis_points[:, 2], 
                          c=vis_colors, s=0.5, alpha=0.6)
        else:
            axes[1].scatter(vis_points[:, 0], vis_points[:, 2], 
                          c=vis_points[:, 1], s=0.5, alpha=0.6, cmap='plasma')
        axes[1].set_title(f'{title} - XZ')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].axis('equal')
        
        if vis_colors is not None:
            axes[2].scatter(vis_points[:, 1], vis_points[:, 2], 
                          c=vis_colors, s=0.5, alpha=0.6)
        else:
            axes[2].scatter(vis_points[:, 1], vis_points[:, 2], 
                          c=vis_points[:, 0], s=0.5, alpha=0.6, cmap='coolwarm')
        axes[2].set_title(f'{title} - YZ')
        axes[2].set_xlabel('Y (m)')
        axes[2].set_ylabel('Z (m)')
        axes[2].axis('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[Visualization] Point cloud projections written to: {output_path}")
        
        plt.close()
    
    def visualize_mesh_projection(self, mesh, output_path=None, title="Mesh"):
        """Render 2D projections of a mesh."""
        if not self.config['save_visualizations']:
            return
        
        print(f"[Visualization] Rendering mesh projections: {title}")
        
        try:
            vertices = mesh.v.cpu().numpy()
        except:
            vertices = np.array(mesh.v)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].scatter(vertices[:, 0], vertices[:, 1], 
                       c=vertices[:, 2], s=0.1, alpha=0.6, cmap='viridis')
        axes[0].set_title(f'{title} - XY')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].axis('equal')
        
        axes[1].scatter(vertices[:, 0], vertices[:, 2], 
                       c=vertices[:, 1], s=0.1, alpha=0.6, cmap='plasma')
        axes[1].set_title(f'{title} - XZ')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].axis('equal')
        
        axes[2].scatter(vertices[:, 1], vertices[:, 2], 
                       c=vertices[:, 0], s=0.1, alpha=0.6, cmap='coolwarm')
        axes[2].set_title(f'{title} - YZ')
        axes[2].set_xlabel('Y (m)')
        axes[2].set_ylabel('Z (m)')
        axes[2].axis('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[Visualization] Mesh projections written to: {output_path}")
        
        plt.close()


class S3DISEvaluator:
    """Evaluate quality metrics for reconstructed scenes."""
    
    def __init__(self, config_dict):
        self.config = config_dict['evaluation']
    
    def evaluate_reconstruction_quality(self, mesh, original_points, original_colors=None):
        """Evaluate reconstruction quality against the original point cloud."""
        print("[Eval] Computing reconstruction metrics...")
        
        metrics = {}
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        try:
            vertices = mesh.v.cpu().numpy()
        except:
            vertices = np.array(mesh.v)
        
        try:
            faces = mesh.f.cpu().numpy()
        except:
            faces = np.array(mesh.f)
        
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        
        n_sample = min(len(original_points), self.config['max_eval_points'])
        sampled_pcd = mesh_o3d.sample_points_poisson_disk(n_sample)
        sampled_points = np.asarray(sampled_pcd.points)
        
        if self.config['compute_chamfer']:
            metrics["chamfer_distance"] = self._compute_chamfer_distance(
                original_points[:n_sample], sampled_points
            )
        
        if self.config['compute_mesh_quality']:
            metrics["mesh_vertices"] = len(mesh.v)
            metrics["mesh_faces"] = len(mesh.f)
            metrics["surface_area"] = mesh_o3d.get_surface_area()
            metrics["is_watertight"] = mesh_o3d.is_watertight()
            
            if mesh_o3d.is_watertight():
                try:
                    metrics["volume"] = mesh_o3d.get_volume()
                except:
                    metrics["volume"] = -1
            else:
                metrics["volume"] = -1
        
        print("[Eval] Metrics ready.")
        if "chamfer_distance" in metrics:
            print(f"  - Chamfer distance: {metrics['chamfer_distance']:.6f}")
        if "surface_area" in metrics:
            print(f"  - Surface area: {metrics['surface_area']:.2f}")
        
        return metrics
    
    def _compute_chamfer_distance(self, points1, points2):
        """Compute the Chamfer distance between two point sets."""
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        
        dist1 = pcd1.compute_point_cloud_distance(pcd2)
        dist2 = pcd2.compute_point_cloud_distance(pcd1)
        
        chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2.0
        return chamfer_dist


def load_config(config_path):
    """Load YAML configuration from disk."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessed_data(data_path):
    """Load serialized preprocessed room data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_experiment_report(results, output_dir, room_info):
    """Write reconstruction summary artifacts."""
    report_path = output_dir / "reconstruction_report.json"
    
    report = {
        "room_info": room_info,
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[Report] Saved reconstruction report to: {report_path}")
    
    # Generate summary table
    summary_path = output_dir / "reconstruction_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("S3DIS NKSR Reconstruction Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Config':<15} {'InputPts':<10} {'Vertices':<10} {'Faces':<10} {'Time(s)':<10} {'EvalStatus':<12}\n")
        f.write("-"*80 + "\n")
        
        for config_name, result in results.items():
            if "error" in result:
                f.write(f"{config_name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}\n")
                continue
            
            recon_stats = result.get("reconstruction_stats", {})
            quality_metrics = result.get("quality_metrics", {})
            
            eval_status = "skipped" if not quality_metrics else "done"
            
            f.write(f"{config_name:<15} "
                   f"{recon_stats.get('input_points', 0):<10} "
                   f"{recon_stats.get('output_vertices', 0):<10} "
                   f"{recon_stats.get('output_faces', 0):<10} "
                   f"{recon_stats.get('reconstruction_time', 0):<10.2f} "
                   f"{eval_status:<12}\n")
    
    print(f"[Report] Summary table saved to: {summary_path}")


def main():
    """Command-line entry point for NKSR reconstruction."""
    parser = argparse.ArgumentParser(description="S3DIS NKSR reconstruction")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed data file")
    parser.add_argument("--sparsity", type=str, default=None, 
                       help="Reconstruct only the specified sparsity variant")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("S3DIS NKSR reconstruction tool")
    print("="*50)
    print(f"Config: {args.config}")
    print(f"Data file: {args.data_path}")
    print(f"NKSR options: detail_level={config['nksr']['detail_level']}, chunked={config['nksr']['chunked']}")
    print("="*50)
    
    print("[Load] Reading preprocessed data...")
    room_data = load_preprocessed_data(args.data_path)
    room_info = room_data['room_info']
    
    print(f"[Load] Room: {room_info['area_name']}/{room_info['room_name']}")
    print(f"[Load] Available variants: {list(room_data['sparsity_variants'].keys())}")
    
    reconstructor = S3DISReconstructor(config)
    visualizer = S3DISVisualizer(config)
    evaluator = S3DISEvaluator(config)
    
    output_dir = Path(config['data']['output_root']) / "reconstruction_results" / f"{room_info['area_name']}_{room_info['room_name']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_points = room_data['original_data']['points']
    original_colors = room_data['original_data']['colors']
    
    if args.sparsity:
        if args.sparsity not in room_data['sparsity_variants']:
        print(f"[Error] Sparsity variant '{args.sparsity}' not found.")
        print(f"[Error] Available variants: {list(room_data['sparsity_variants'].keys())}")
            return
        variants_to_process = [args.sparsity]
    else:
        variants_to_process = list(room_data['sparsity_variants'].keys())
    
    print(f"[Process] Reconstructing {len(variants_to_process)} variant(s).")
    
    experiment_results = {}
    
    for variant_name in tqdm(variants_to_process, desc="Reconstruction", unit="variant"):
        print(f"\n{'-'*40}")
        print(f"[Process] Variant: {variant_name}")
        print(f"{'-'*40}")
        
        try:
            variant_data = room_data['sparsity_variants'][variant_name]
            
            points = variant_data['points']
            colors = variant_data['colors']
            normals = variant_data['normals']
            
            print(f"[Reconstruct] Input points: {len(points)}")
            
            if config['visualization']['save_visualizations']:
                vis_path = output_dir / f"pointcloud_{variant_name}.png"
                visualizer.visualize_pointcloud(
                    points, colors, vis_path,
                    f"{room_info['area_name']}/{room_info['room_name']} - {variant_name}"
                )
            
            mesh, reconstruction_stats = reconstructor.reconstruct_mesh(
                points=points, 
                normals=normals, 
                colors=colors,
                use_semantic_colors=config.get('use_semantic_colors', False),
                s3dis_data_root=config.get('s3dis_data_root', None),
                area=room_info.get('area_name', None),
                room=room_info.get('room_name', None)
            )
            
            mesh_path = output_dir / f"mesh_{variant_name}.ply"
            reconstructor.save_mesh(mesh, mesh_path, reconstruction_stats)
            
            if config['visualization']['save_visualizations']:
                mesh_vis_path = output_dir / f"mesh_{variant_name}.png"
                visualizer.visualize_mesh_projection(
                    mesh, mesh_vis_path,
                    f"{room_info['area_name']}/{room_info['room_name']} - Mesh - {variant_name}"
                )
            
            # quality_metrics = evaluator.evaluate_reconstruction_quality(mesh, original_points, original_colors)
            quality_metrics = {}
            
            experiment_results[variant_name] = {
                "sparsity_config": variant_data['config'],
                "reconstruction_stats": reconstruction_stats,
                "quality_metrics": quality_metrics
            }
            
            print(f"[Success] {variant_name} reconstructed.")
            
        except Exception as e:
            print(f"[Error] Reconstruction failed for {variant_name}: {e}")
            experiment_results[variant_name] = {"error": str(e)}
    
    save_experiment_report(experiment_results, output_dir, room_info)
    
    print("\n[Done] Reconstruction complete.")
    print(f"Output directory: {output_dir}")
    print(f"Successful variants: {sum(1 for r in experiment_results.values() if 'error' not in r)}/{len(experiment_results)}")


if __name__ == "__main__":
    main()
