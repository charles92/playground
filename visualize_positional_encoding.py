"""Visualize positional encoding patterns."""

import matplotlib.pyplot as plt
import numpy as np

from transformer import PositionalEncoding


def visualize_positional_encoding():
    """Create visualizations of the positional encoding patterns."""

    # Parameters
    seq_length = 100
    d_model = 512

    # Create positional encoding
    pe = PositionalEncoding(seq_length, d_model)

    # Get the positional encoding matrix
    pe_matrix = pe.pe.detach().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Positional Encoding Visualization", fontsize=16)

    # 1. Heatmap of the entire positional encoding matrix
    im1 = axes[0, 0].imshow(pe_matrix, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Full Positional Encoding Matrix")
    axes[0, 0].set_xlabel("Embedding Dimension")
    axes[0, 0].set_ylabel("Position")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. First 50 positions, first 100 dimensions
    im2 = axes[0, 1].imshow(pe_matrix[:50, :100], aspect="auto", cmap="viridis")
    axes[0, 1].set_title("First 50 Positions, First 100 Dimensions")
    axes[0, 1].set_xlabel("Embedding Dimension")
    axes[0, 1].set_ylabel("Position")
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Plot sine and cosine patterns for different frequencies
    positions = np.arange(seq_length)
    for i in range(0, min(20, d_model), 4):  # Plot every 4th dimension
        axes[1, 0].plot(
            positions, pe_matrix[:, i], label=f"Dim {i}", alpha=0.7, linewidth=1
        )
    axes[1, 0].set_title("Sine/Cosine Patterns (First 20 dimensions)")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Encoding Value")
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Show how encoding values change across positions for specific dimensions
    selected_dims = [0, 10, 50, 100, 200, 300]
    for dim in selected_dims:
        if dim < d_model:
            axes[1, 1].plot(
                positions, pe_matrix[:, dim], label=f"Dim {dim}", linewidth=2
            )
    axes[1, 1].set_title("Encoding Values Across Positions")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("Encoding Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("positional_encoding_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print some statistics
    print(f"Positional Encoding Statistics:")
    print(f"Matrix shape: {pe_matrix.shape}")
    print(f"Min value: {pe_matrix.min():.4f}")
    print(f"Max value: {pe_matrix.max():.4f}")
    print(f"Mean value: {pe_matrix.mean():.4f}")
    print(f"Std value: {pe_matrix.std():.4f}")

    # Show the first few values
    print(f"\nFirst 5 positions, first 10 dimensions:")
    print(pe_matrix[:5, :10])


if __name__ == "__main__":
    print("Visualizing positional encoding patterns...")
    visualize_positional_encoding()
    print("Visualization complete! Check the generated PNG file.")
