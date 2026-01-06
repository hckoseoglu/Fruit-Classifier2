import numpy as np
import matplotlib.pyplot as plt


def plot_matrix_heatmap(
    M, classes, label_encoder, title, cmap="viridis", fmt="{:.2f}", show_values=True
):
    """
    Visualize a matrix (distance, similarity, or confusion) as a heatmap.

    Args:
      M             : (C x C) matrix
      classes       : array of class ids
      label_encoder : fitted LabelEncoder
      title         : plot title
      cmap          : matplotlib colormap
      fmt           : value format for annotations
      show_values   : annotate each cell with its value
    """
    class_names = label_encoder.inverse_transform(classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap=cmap)

    # Axis ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_title(title)

    # Annotate values
    if show_values:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j,
                    i,
                    fmt.format(M[i, j]),
                    ha="center",
                    va="center",
                    color="white" if im.norm(M[i, j]) > 0.5 else "black",
                    fontsize=9,
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()
