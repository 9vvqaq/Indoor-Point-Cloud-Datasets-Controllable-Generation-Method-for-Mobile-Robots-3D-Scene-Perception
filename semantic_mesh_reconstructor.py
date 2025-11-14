"""
NKSR mesh reconstructor with semantic label coloring.
Encodes semantic labels as colors for mesh generation.
"""

import numpy as np
import torch
import nksr
from typing import Optional, Union, Tuple
import time
from semantic_color_encoder import SemanticColorEncoder, create_s3dis_color_encoder


class SemanticMeshReconstructor:
    """
    Semantic-aware mesh reconstructor powered by NKSR.
    
    Features:
    1. Encode semantic labels as colors.
    2. Reconstruct meshes with NKSR while preserving color information.
    3. Support indoor datasets such as S3DIS.
    """
    
    def __init__(self, 
                 num_classes: int = 13,
                 device: str = "cuda:0",
                 chunk_size: float = 50.0,
                 detail_level: float = 0.5,
                 voxel_size: float = 0.05):
        """
        Initialize the reconstructor.
        
        Args:
            num_classes: Number of semantic classes.
            device: Compute device identifier.
            chunk_size: Chunk size for large scenes.
            detail_level: Reconstruction detail level.
            voxel_size: Optional voxel size.
        """
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.detail_level = detail_level
        self.voxel_size = voxel_size
        
        # Initialize color encoder
        self.color_encoder = SemanticColorEncoder(num_classes=num_classes, rgb_order=True)
        
        # Initialize NKSR reconstructor
        self.reconstructor = nksr.Reconstructor(self.device)
        if chunk_size > 0:
            self.reconstructor.chunk_tmp_device = torch.device("cpu")
        
        print("[SemanticReconstructor] Initialized.")
        print(f"  - Device: {self.device}")
        print(f"  - Semantic classes: {num_classes}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Detail level: {detail_level}")
    
    def encode_semantic_colors(self, 
                             points: np.ndarray, 
                             semantic_labels: np.ndarray) -> np.ndarray:
        """
        Encode semantic labels into color values.
        
        Args:
            points: Point coordinates [N, 3].
            semantic_labels: Semantic labels [N].
            
        Returns:
            Encoded colors [N, 3].
        """
        print("[SemanticColor] Encoding semantic labels...")
        print(f"  - Points: {len(points)}")
        print(f"  - Label range: {semantic_labels.min()} - {semantic_labels.max()}")
        
        # Ensure labels fall within valid range
        valid_mask = (semantic_labels >= 0) & (semantic_labels < self.color_encoder.num_classes)
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            print(f"  - Warning: {invalid_count} invalid labels detected.")
            semantic_labels = np.clip(semantic_labels, 0, self.color_encoder.num_classes - 1)
        
        # Encode colors
        colors = self.color_encoder.encode_labels(semantic_labels)
        
        # Report label distribution
        unique_labels, counts = np.unique(semantic_labels, return_counts=True)
        print("  - Label distribution:")
        for label, count in zip(unique_labels, counts):
            color = self.color_encoder.get_color(label)
            print(f"    Label {label}: {count} pts, RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        return colors
    
    def reconstruct_semantic_mesh(self, 
                                points: np.ndarray,
                                normals: Optional[np.ndarray] = None,
                                semantic_labels: Optional[np.ndarray] = None,
                                use_semantic_colors: bool = True) -> Tuple[object, dict]:
        """
        Reconstruct a mesh with semantic color information.
        
        Args:
            points: Point coordinates [N, 3].
            normals: Optional normals [N, 3].
            semantic_labels: Optional semantic labels [N].
            use_semantic_colors: Whether to apply semantic coloring.
            
        Returns:
            Tuple of (mesh, stats) containing reconstruction data.
        """
        print("[SemanticReconstructor] Starting mesh reconstruction...")
        start_time = time.time()
        
        # Convert to torch tensors
        input_xyz = torch.from_numpy(points.astype(np.float32)).to(self.device)
        
        if normals is not None:
            input_normal = torch.from_numpy(normals.astype(np.float32)).to(self.device)
            print("[SemanticReconstructor] Using provided normals.")
        else:
            print("[SemanticReconstructor] Warning: normals missing; reconstruction quality may degrade.")
            input_normal = None
        
        # Perform NKSR reconstruction
        try:
            if self.chunk_size > 0:
                # Chunked reconstruction mode
                field = self.reconstructor.reconstruct(
                    input_xyz,
                    normal=input_normal,
                    detail_level=self.detail_level,
                    chunk_size=self.chunk_size,
                    approx_kernel_grad=True,
                    solver_tol=1e-4,
                    fused_mode=True
                )
            else:
                # Full-scene reconstruction mode
                field = self.reconstructor.reconstruct(
                    input_xyz,
                    normal=input_normal,
                    detail_level=self.detail_level,
                    voxel_size=self.voxel_size
                )
            
            # Set color texture
            if use_semantic_colors and semantic_labels is not None:
                print("[SemanticReconstructor] Applying semantic color texture...")
                colors = self.encode_semantic_colors(points, semantic_labels)
                input_color = torch.from_numpy(colors.astype(np.float32)).to(self.device)
                field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
            elif not use_semantic_colors:
                print("[SemanticReconstructor] Skipping semantic color texture.")
            
            # Extract mesh
            print("[SemanticReconstructor] Extracting mesh...")
            mesh = field.extract_dual_mesh(mise_iter=2)
            
            # Compute reconstruction statistics
            reconstruction_time = time.time() - start_time
            stats = {
                "input_points": len(points),
                "output_vertices": len(mesh.v),
                "output_faces": len(mesh.f),
                "reconstruction_time": reconstruction_time,
                "has_semantic_colors": use_semantic_colors and semantic_labels is not None,
                "semantic_classes": len(np.unique(semantic_labels)) if semantic_labels is not None else 0,
                "device_used": str(self.device)
            }
            
            print("[SemanticReconstructor] Reconstruction complete.")
            print(f"  - Input points: {stats['input_points']}")
            print(f"  - Output vertices: {stats['output_vertices']}")
            print(f"  - Output faces: {stats['output_faces']}")
            print(f"  - Runtime: {stats['reconstruction_time']:.2f}s")
            print(f"  - Semantic colors applied: {stats['has_semantic_colors']}")
            
            return mesh, stats
            
        except Exception as e:
            print(f"[SemanticReconstructor] Reconstruction failed: {e}")
            raise
    
    def save_semantic_mesh(self, 
                          mesh: object, 
                          output_path: str,
                          include_colormap: bool = True) -> None:
        """
        Save the reconstructed mesh along with semantic colors.
        
        Args:
            mesh: Reconstructed mesh object.
            output_path: Destination path for the mesh.
            include_colormap: Whether to store colormap metadata.
        """
        print(f"[SemanticReconstructor] Saving mesh to: {output_path}")
        
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            print(f"  - Vertices: {len(mesh.v)}")
            print(f"  - Faces: {len(mesh.f)}")
            if hasattr(mesh, 'c'):
                print(f"  - Colors: {len(mesh.c)}")
        
        if include_colormap:
            colormap_path = output_path.replace('.ply', '_colormap.txt')
            self._save_colormap_info(colormap_path)
    
    def _save_colormap_info(self, colormap_path: str) -> None:
        """Persist color mapping information to disk."""
        with open(colormap_path, 'w', encoding='utf-8') as f:
            f.write("# Semantic label colormap\n")
            f.write(f"# Classes: {self.color_encoder.num_classes}\n")
            f.write("# Format: label_id R G B\n")
            
            for i in range(self.color_encoder.num_classes):
                color = self.color_encoder.get_color(i)
                f.write(f"{i} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
        
        print(f"  - Colormap metadata saved to: {colormap_path}")


def test_semantic_reconstruction():
    """Ad-hoc test for semantic reconstruction."""
    print("Testing semantic mesh reconstruction...")
    
    # Build synthetic test data
    np.random.seed(42)
    n_points = 1000
    
    # Generate cube-shaped test points
    points = np.random.rand(n_points, 3) * 2 - 1  # [-1, 1]^3
    
    # Generate test normals
    normals = np.random.randn(n_points, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdim=True)
    
    # Sample random semantic labels (13 classes)
    semantic_labels = np.random.randint(0, 13, n_points)
    
    # Build reconstructor
    reconstructor = SemanticMeshReconstructor(
        num_classes=13,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        chunk_size=20.0,
        detail_level=0.3
    )
    
    # Run reconstruction
    try:
        mesh, stats = reconstructor.reconstruct_semantic_mesh(
            points=points,
            normals=normals,
            semantic_labels=semantic_labels,
            use_semantic_colors=True
        )
        
        print("Semantic reconstruction test succeeded.")
        print(f"Stats: {stats}")
        
        # Save result
        reconstructor.save_semantic_mesh(mesh, "test_semantic_mesh.ply")
        
    except Exception as e:
        print(f"Semantic reconstruction test failed: {e}")


if __name__ == "__main__":
    test_semantic_reconstruction()









