"""
Semantic label color encoder.
Provides deterministic bit-based color mapping for NKSR mesh visualization.
"""

import numpy as np
import torch
from typing import Optional, Union, List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class SemanticColorEncoder:
    """
    Encode semantic labels into deterministic colors.
    
    Guarantees:
    1. Unique color per label id.
    2. Distinct colors for adjacent labels.
    3. Evenly distributed colors to avoid confusion.
    """
    
    def __init__(self, num_classes: int, rgb_order: bool = True):
        """
        Initialize the color encoder.
        
        Args:
            num_classes: Number of semantic classes.
            rgb_order: True for RGB, False for BGR ordering.
        """
        self.num_classes = num_classes
        self.rgb_order = rgb_order
        self.colormap = self._generate_colormap()
        
    def _bitget(self, val: int, bit_idx: int) -> int:
        """Return the bit at index bit_idx from val."""
        return (val >> bit_idx) & 1
    
    def _generate_colormap(self) -> np.ndarray:
        """
        Generate a bitwise color map.
        
        Reference: https://github.com/hhj1897/face_parsing/blob/master/ibug/face_parsing/utils.py
        """
        cmap = np.zeros((self.num_classes, 3), dtype=np.uint8)
        
        for i in range(self.num_classes):
            id_val = i
            r, g, b = 0, 0, 0
            
            # Build RGB value using 8 bits
            for j in range(8):
                # Extract three bits for R, G, B
                r = np.bitwise_or(r, (self._bitget(id_val, 0) << (7 - j)))
                g = np.bitwise_or(g, (self._bitget(id_val, 1) << (7 - j)))
                b = np.bitwise_or(b, (self._bitget(id_val, 2) << (7 - j)))
                
                # Shift three bits for next iteration
                id_val = id_val >> 3
            
            # Respect desired channel order
            if not self.rgb_order:  # BGR order
                cmap[i, 0] = b
                cmap[i, 1] = g
                cmap[i, 2] = r
            else:  # RGB order
                cmap[i, 0] = r
                cmap[i, 1] = g
                cmap[i, 2] = b
                
        return cmap
    
    def encode_labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode semantic labels into RGB colors.
        
        Args:
            labels: Semantic labels with shape [N] or [H, W].
            
        Returns:
            Color array with shape [N, 3] or [H, W, 3].
        """
        is_torch = isinstance(labels, torch.Tensor)
        
        if is_torch:
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels.copy()
        
        # Clamp labels to valid range
        labels_np = np.clip(labels_np, 0, self.num_classes - 1)
        
        # Gather colors
        if labels_np.ndim == 1:
            colors = self.colormap[labels_np]  # [N, 3]
        elif labels_np.ndim == 2:
            colors = self.colormap[labels_np]  # [H, W, 3]
        else:
            raise ValueError(f"Unsupported label dimension: {labels_np.ndim}")
        
        # Convert to float32 in [0, 1]
        colors = colors.astype(np.float32) / 255.0
        
        if is_torch:
            return torch.from_numpy(colors).to(labels.device)
        else:
            return colors
    
    def decode_colors(self, colors: Union[np.ndarray, torch.Tensor], 
                      threshold: float = 0.1) -> Union[np.ndarray, torch.Tensor]:
        """
        Decode colors back into semantic labels (approximate).
        
        Args:
            colors: Color array with shape [N, 3] or [H, W, 3].
            threshold: Color matching tolerance (unused placeholder).
            
        Returns:
            Decoded labels.
        """
        is_torch = isinstance(colors, torch.Tensor)
        
        if is_torch:
            colors_np = colors.cpu().numpy()
        else:
            colors_np = colors.copy()
        
        # Convert from [0, 1] to [0, 255]
        colors_np = (colors_np * 255).astype(np.uint8)
        
        if colors_np.ndim == 2:  # [N, 3]
            labels = np.zeros(colors_np.shape[0], dtype=np.int32)
            for i, color in enumerate(colors_np):
                # Find nearest color in colormap
                distances = np.sum((self.colormap - color) ** 2, axis=1)
                labels[i] = np.argmin(distances)
        elif colors_np.ndim == 3:  # [H, W, 3]
            labels = np.zeros(colors_np.shape[:2], dtype=np.int32)
            for i in range(colors_np.shape[0]):
                for j in range(colors_np.shape[1]):
                    color = colors_np[i, j]
                    distances = np.sum((self.colormap - color) ** 2, axis=1)
                    labels[i, j] = np.argmin(distances)
        else:
            raise ValueError(f"Unsupported color dimension: {colors_np.ndim}")
        
        if is_torch:
            return torch.from_numpy(labels).to(colors.device)
        else:
            return labels
    
    def get_color(self, label_id: int) -> np.ndarray:
        """Return the RGB color for a specific label id."""
        if 0 <= label_id < self.num_classes:
            return self.colormap[label_id] / 255.0
        else:
            raise ValueError(f"Label id {label_id} is out of range [0, {self.num_classes-1}].")
    
    def visualize_colormap(self, save_path: Optional[str] = None, 
                          figsize: tuple = (12, 8)) -> None:
        """
        Visualize the color map.
        
        Args:
            save_path: Optional path to save the figure; shows interactively when None.
            figsize: Matplotlib figure size.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a color bar
        colors_normalized = self.colormap / 255.0
        cmap = ListedColormap(colors_normalized)
        
        # Display gradient
        gradient = np.linspace(0, 1, self.num_classes)
        gradient = np.vstack((gradient, gradient))
        
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xlim(0, self.num_classes)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Label ID')
        ax.set_title(f'Semantic label colormap ({self.num_classes} classes)')
        
        # Annotate label ids
        for i in range(0, self.num_classes, max(1, self.num_classes // 20)):
            ax.text(i, 0, str(i), ha='center', va='center', 
                   color='white' if np.mean(colors_normalized[i]) < 0.5 else 'black',
                   fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Colormap image saved to: {save_path}")
        else:
            plt.show()
    
    def get_colormap_info(self) -> dict:
        """Return metadata about the color map."""
        return {
            'num_classes': self.num_classes,
            'rgb_order': self.rgb_order,
            'colormap_shape': self.colormap.shape,
            'color_range': (self.colormap.min(), self.colormap.max()),
            'unique_colors': len(np.unique(self.colormap.reshape(-1, 3), axis=0))
        }


def create_s3dis_color_encoder() -> SemanticColorEncoder:
    """
    Create a color encoder configured for the 13 S3DIS semantic classes.
    Classes: 0 ceiling, 1 floor, 2 wall, 3 beam, 4 column, 5 window, 6 door,
    7 table, 8 chair, 9 sofa, 10 bookcase, 11 board, 12 clutter.
    """
    return SemanticColorEncoder(num_classes=13, rgb_order=True)


def test_color_encoder():
    """Quick diagnostic for the color encoder."""
    print("Testing semantic label color encoder...")
    
    encoder = create_s3dis_color_encoder()
    
    test_labels = np.array([0, 1, 2, 5, 8, 12])
    colors = encoder.encode_labels_to_colors(test_labels)
    
    print(f"Test labels: {test_labels}")
    print(f"Encoded color shape: {colors.shape}")
    print(f"Encoded colors:\n{colors}")
    
    decoded_labels = encoder.decode_colors(colors)
    print(f"Decoded labels: {decoded_labels}")
    print(f"Decoding accuracy: {np.mean(test_labels == decoded_labels):.2%}")
    
    info = encoder.get_colormap_info()
    print("\nColormap info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    encoder.visualize_colormap(save_path="S3DIS/s3dis_colormap.png")
    
    print("Color encoder test finished.")


if __name__ == "__main__":
    test_color_encoder()
