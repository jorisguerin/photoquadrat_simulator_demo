import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from time import time

def compute_proportions(samples, n_classes=10):
    """
    Compute class proportions from samples, ensuring fixed output size

    Parameters:
    -----------
    samples : list of arrays
        List of sample arrays containing class labels
    n_classes : int
        Number of classes (excluding 0, which is treated as invalid)

    Returns:
    --------
    proportions : numpy.ndarray
        Array of length n_classes containing proportions for classes 1 to n_classes
    """
    all_pixels = np.concatenate([sample.flatten() for sample in samples])
    valid_pixels = all_pixels != 0
    n_valid = np.sum(valid_pixels)

    if n_valid > 0:  # Avoid division by zero if all pixels are invalid
        # Count occurrences of each class
        counts = np.bincount(all_pixels[valid_pixels], minlength=n_classes + 1)
        # Convert to proportions, excluding class 0
        return counts[1:] / n_valid

    return np.zeros(n_classes)


def display_map(map_image, class_params,
                sample_points=None, window_size_meters=1.0, pixel_size=0.01,
                figsize=(15, 10), save_path=None,
                fast_display=False, legend=True, downsample_factor=8):
    """
    Display a classified map with optional sampling points

    Parameters:
    -----------
    map_image : numpy.ndarray
        The classified map to display
    class_params : dict, optional
        Dictionary mapping class indices to their properties (color and name)
    sample_points : list of tuples, optional
        List of (y, x) coordinates for sampling points
    window_size_meters : float, optional
        Size of sampling windows in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)
    figsize : tuple, optional
        Figure size in inches (default: (15, 10))
    save_path : str, optional
        If provided, saves the figure to this path
    """
    if fast_display:
        map_image_display = map_image[::downsample_factor, ::downsample_factor]
        display_pixel_size = pixel_size * downsample_factor
    else:
        map_image_display = map_image
        display_pixel_size = pixel_size

    # Create color mapping
    colors = [class_params[i]['color'] for i in range(len(class_params))]
    classes = [class_params[i]['name'] for i in range(len(class_params))]
    values = np.unique(map_image_display)
    n_colors = len(colors)
    cmap = mcolors.ListedColormap(colors)

    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(map_image_display, cmap=cmap, vmin=0, vmax=n_colors)

    # Create legend patches
    handles = [Patch(facecolor=colors[i], edgecolor='black',
                     linewidth=0.5, label=classes[i]) for i in values]

    # Add scale bar
    scalebar = ScaleBar(
        dx=display_pixel_size,
        units='m',
        length_fraction=0.15,
        location='lower right',
        box_alpha=0.8,
        box_color='white',
        color='black',
        font_properties={'size': 12}
    )
    plt.gca().add_artist(scalebar)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    # Add sampling points if provided
    if sample_points is not None:
        quadrat_size_pixels = int(window_size_meters / display_pixel_size)

        if fast_display:
            for y, x in sample_points:
                half_size = quadrat_size_pixels // 2
                square = plt.Rectangle((x//downsample_factor - half_size, y//downsample_factor - half_size),
                                       quadrat_size_pixels, quadrat_size_pixels,
                                       fill=False, linewidth=2, edgecolor="black")
                plt.gca().add_patch(square)
        else:
            for y, x in sample_points:
                half_size = quadrat_size_pixels // 2
                square = plt.Rectangle((x - half_size, y - half_size),
                                       quadrat_size_pixels, quadrat_size_pixels,
                                       fill=False, linewidth=2, edgecolor="black")
                plt.gca().add_patch(square)

    # Add legend
    legend_loc = 'upper center'
    bbox_anchor = (0.5, 0.0)
    if legend:
        plt.legend(handles=handles, loc=legend_loc,
                   bbox_to_anchor=bbox_anchor, ncol=2, fontsize=15.4)

    plt.tight_layout()
    plt.gca().set_aspect('equal')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def display_map_streamlit(map_image, class_params,
                          sample_points=None, window_size_meters=1.0, pixel_size=0.01,
                          figsize=(15, 10), save_path=None,
                          fast_display=False, legend=True, downsample_factor=8):
    """
    Display a classified map with optional sampling points (Streamlit version)

    Parameters:
    -----------
    map_image : numpy.ndarray
        The classified map to display
    class_params : dict, optional
        Dictionary mapping class indices to their properties (color and name)
    sample_points : list of tuples, optional
        List of (y, x) coordinates for sampling points
    window_size_meters : float, optional
        Size of sampling windows in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)
    figsize : tuple, optional
        Figure size in inches (default: (15, 10))
    save_path : str, optional
        If provided, saves the figure to this path
    fast_display : bool, optional
        If True, downsample the image for faster display (default: False)
    legend : bool, optional
        If True, display legend (default: True)
    downsample_factor : int, optional
        Factor by which to downsample if fast_display is True (default: 8)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for Streamlit display
    """
    if fast_display:
        map_image_display = map_image[::downsample_factor, ::downsample_factor]
        display_pixel_size = pixel_size * downsample_factor
    else:
        map_image_display = map_image
        display_pixel_size = pixel_size

    # Create color mapping
    colors = [class_params[i]['color'] for i in range(len(class_params))]
    classes = [class_params[i]['name'] for i in range(len(class_params))]
    values = np.unique(map_image_display)
    n_colors = len(colors)
    cmap = mcolors.ListedColormap(colors)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(map_image_display, cmap=cmap, vmin=0, vmax=n_colors - 1)

    # Create legend patches
    handles = [Patch(facecolor=colors[i], edgecolor='black',
                     linewidth=0.5, label=classes[i]) for i in values]

    # Add scale bar
    scalebar = ScaleBar(
        dx=display_pixel_size,
        units='m',
        length_fraction=0.15,
        location='lower right',
        box_alpha=0.8,
        box_color='white',
        color='black',
        font_properties={'size': 12}
    )
    ax.add_artist(scalebar)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add sampling points if provided
    if sample_points is not None:
        quadrat_size_pixels = int(window_size_meters / display_pixel_size)

        if fast_display:
            for y, x in sample_points:
                half_size = quadrat_size_pixels // 2
                square = plt.Rectangle((x // downsample_factor - half_size, y // downsample_factor - half_size),
                                   quadrat_size_pixels, quadrat_size_pixels,
                                   fill=False, linewidth=2, edgecolor="black")
                ax.add_patch(square)
        else:
            for y, x in sample_points:
                half_size = quadrat_size_pixels // 2
                square = plt.Rectangle((x - half_size, y - half_size),
                                   quadrat_size_pixels, quadrat_size_pixels,
                                   fill=False, linewidth=2, edgecolor="black")
                ax.add_patch(square)

    # Add legend
    legend_loc = 'upper center'
    bbox_anchor = (0.5, 0.0)
    if legend:
        ax.legend(handles=handles, loc=legend_loc,
                  bbox_to_anchor=bbox_anchor, ncol=2, fontsize=15.4)

    plt.tight_layout()
    ax.set_aspect('equal')

    # Save if path provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig

def display_proportions(proportions_array, classes, column_labels, decimals=2):
    """
    Display multiple sets of class proportions in a formatted table

    Parameters:
    -----------
    proportions_array : list of numpy.ndarray
        List of proportion vectors to display as columns
    classes : list
        List of class names (including class 0)
    column_labels : list
        Labels for each column of proportions
    decimals : int, optional
        Number of decimal places to display (default: 1)
    """
    n_cols = len(proportions_array)
    col_width = 12  # Width for proportion columns
    name_width = 60


    # Create headers
    separator = "─" * (col_width)  # +1 for the │ between columns
    print(f"┌─{'─' * name_width}─┬{separator * (n_cols)}{'─' * (n_cols - 1)}┐")

    # Column headers
    header = f"│ {"Class":<{name_width}} │"
    for label in column_labels:
        header += f" {label:^{col_width - 2}} │"
    print(header)

    print(f"├─{'─' * name_width}─┼{separator * n_cols}{'─' * (n_cols - 1)}┤")

    # Display proportions for each class
    totals = np.zeros(n_cols)
    for cls, *props in zip(classes[1:], *proportions_array):
        row = f"│ {cls:<{name_width}} │"
        for i, prop in enumerate(props):
            totals[i] += prop
            row += f" {prop:>{col_width - 2}.{decimals}%} │"
        print(row)

    # Add totals
    print(f"├─{'─' * name_width}─┼{separator * n_cols}{'─' * (n_cols - 1)}┤")
    row = f"│ {'Total':<{name_width}} │"
    for total in totals:
        row += f" {total:>{col_width - 2}.{decimals}%} │"
    print(row)
    print(f"└─{'─' * name_width}─┴{separator * n_cols}{'─' * (n_cols - 1)}┘")

def compute_proportions_MC(samples_MC, n_classes=10):
    proportions = []
    for samples in samples_MC:
        proportions.append(compute_proportions(samples, n_classes))
    proportions = np.array(proportions)
    return proportions

def violin_plots(list_values_1, list_values_2, n_samples, labels, true_cover, class_name):
    color1 = 'tab:blue'
    color2 = 'tab:orange'

    plt.figure(figsize=(12, 6))

    plt.axhline(y=true_cover, color='black', linestyle='--', alpha=0.6)

    positions = np.arange(1, len(n_samples) + 1)

    violin_parts_1 = plt.violinplot(list_values_1, positions,
                                    showmeans=True,
                                    quantiles=len(list_values_1) * [[0.025, 0.975]],
                                    side='low')
    violin_parts_2 = plt.violinplot(list_values_2, positions,
                                    showmeans=True,
                                    quantiles=len(list_values_1) * [[0.025, 0.975]],
                                    side='high')
    # Customize appearance
    for pc in violin_parts_1['bodies']:
        pc.set_facecolor(color1)
        pc.set_alpha(0.6)
    violin_parts_1['cmeans'].set_color(color1)
    violin_parts_1['cquantiles'].set_color(color1)
    violin_parts_1['cmaxes'].set_color(color1)
    violin_parts_1['cmins'].set_color(color1)

    for pc in violin_parts_2['bodies']:
        pc.set_facecolor(color2)
        pc.set_alpha(0.6)
    violin_parts_2['cmeans'].set_color(color2)
    violin_parts_2['cquantiles'].set_color(color2)
    violin_parts_2['cmaxes'].set_color(color2)
    violin_parts_2['cmins'].set_color(color2)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color1, alpha=0.6, label=labels[0]),
        Patch(facecolor=color2, alpha=0.6, label=labels[1])
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Customize plot
    plt.xticks(positions, n_samples)
    plt.grid(True, alpha=0.3)
    plt.title(f'Cover Estimates Distributions for "{class_name}"')
    plt.xlabel('Number of samples')
    plt.ylabel('Cover estimate error')

    plt.tight_layout()
    plt.show()


def display_sample_quadrats_streamlit(samples, class_params, max_samples=24, points=None):
    """
    Display a grid of sample quadrats

    Parameters:
    -----------
    samples : list
        List of sample arrays from sampling
    class_params : dict
        Class parameters for colors
    max_samples : int
        Maximum number of samples to display (default: 20)

    Returns:
    --------
    fig : matplotlib figure
        Figure object for Streamlit display
    """

    if samples is None or len(samples) == 0:
        # Return empty figure with message
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, 'No samples available\nGenerate sampling first',
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    if not samples or len(samples) == 0:
        return None

    # Limit to first max_samples
    samples_to_show = samples[:max_samples]
    n_samples = len(samples_to_show)

    # Create color mapping
    colors = [class_params[i]['color'] for i in range(len(class_params))]
    cmap = mcolors.ListedColormap(colors)
    n_colors = len(colors)

    # Calculate grid dimensions
    if n_samples <= 8:
        ncols = n_samples
        nrows = 1
    elif n_samples <= 16:
        ncols = 8
        nrows = 2
    else:
        ncols = 8
        nrows = 3

    # Adjust figure size based on grid
    fig_width = min(10, ncols * 2)
    fig_height = min(8, nrows * 1.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    # Handle case where axes might not be 2D
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Display samples
    for i, sample in enumerate(samples_to_show):
        axes[i].imshow(sample, cmap=cmap, vmin=0, vmax=n_colors - 1)
        axes[i].tick_params(axis='both', which='both', length=0,
                            labelbottom=False, labelleft=False)

        if points is not None:
            # Plot sampling points as scatter
            pts = points[i]
            axes[i].scatter(pts[:, 1], pts[:, 0], c="tab:red", marker="x",
                            s=40, linewidths=1.5, alpha=1)

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig