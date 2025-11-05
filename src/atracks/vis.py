import numpy as np
from joblib import Parallel, delayed


def spatial_probability_3d(
    output_shape: tuple[int, int, int],
    centers: np.ndarray,
    radius: float,
    sigma: float,
    workers: int = -1,
    verbose: int = 0,
) -> np.ndarray:
    """Generates a 3D donut-shaped Gaussian distribution by applying a Gaussian function to the radial distance from the center for each plane in the stack.

    Args:
        shape (tuple[int, int, int]): The (height, width, depth) of the output array.
        centers (np.ndarray): The (y, x, z) coordinates of the centers.
        radius (float): The mean radius of the donut ring.
        sigma (float): The standard deviation (spread) of the ring.
    """
    shape_2d = output_shape[1:]

    # create list of centroids by plane (first axis)
    centroids = [centers[centers[:, 0] == t][:, 1:] for t in range(output_shape[0])]
    probs = Parallel(n_jobs=workers, verbose=verbose)(
        delayed(spatial_probability_2d)(shape_2d, centroid, radius, sigma)
        for centroid in centroids
    )
    probs_img = np.stack(probs, axis=0)
    print(f"target shape: {output_shape}, actual shape: {probs_img.shape}")
    return probs_img


def spatial_probability_2d(
    shape: tuple[int, int],
    center: np.ndarray,
    radius: float,
    sigma: float,
) -> np.ndarray:
    """Generates a 2D donut-shaped Gaussian distribution by applying a Gaussian function to the radial distance from the center.

    If center is a 2D array, creates masks for each center (each row) and adds them together.

    Args:
        shape (tuple[int, int]): The (height, width) of the output array.
        center (np.ndarray): The (y, x) coordinates of the center(s).
            If 1D (shape: [2]), treats as a single center.
            If 2D (shape: [N, 2]), creates masks for each row (center) and adds them together.
        radius (float): The mean radius of the donut ring.
        sigma (float): The standard deviation (spread) of the ring.

    Returns:
        np.ndarray: 2D array with donut-shaped Gaussian distribution(s).
    """
    # Convert to numpy array if needed
    center = np.asarray(center)

    # Handle 2D array (multiple centers)
    if center.ndim == 2:
        if center.shape[0] == 0:
            raise ValueError("center array cannot be empty")
        if center.shape[1] != 2:
            raise ValueError(f"center array must have shape (N, 2), got {center.shape}")

        # Create mask for first center
        result = spatial_probability_2d(shape, center[0], radius, sigma)

        # Add masks from remaining centers
        for i in range(1, center.shape[0]):
            mask = spatial_probability_2d(shape, center[i], radius, sigma)
            result = result + mask

        return result

    # Handle 1D array (single center)
    if center.ndim != 1 or center.shape[0] != 2:
        raise ValueError(
            f"center must have shape (2,) for single center, got {center.shape}"
        )

    y, x = np.ogrid[: shape[0], : shape[1]]
    y_center, x_center = center[0], center[1]
    # Calculate the radial distance from the center for each point
    radial_distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Apply a Gaussian function to the radial distances
    # The mean (radius) determines the position of the ring
    # The standard deviation (spread) determines its thickness
    donut = np.exp(-((radial_distance - radius) ** 2) / (2 * sigma**2))
    return donut


def export_animation(
    viewer,
    layers: list[int] | list[str],
    fname: str,
    fps: int = 30,
) -> None:
    """Export a napari viewer animation to a file.

    Sets the specified layers to visible and hides all other layers before
    capturing and exporting the animation.

    Args:
        viewer: Napari viewer instance to export animation from.
        layers (list[int] | list[str]): List of layer indices (int) or names (str)
            to include in the animation. These layers will be set to visible,
            while all other layers will be hidden.
        fname (str): Output filename for the animation.
        fps (int, optional): Frames per second for the animation. Defaults to 30.

    Raises:
        RuntimeError: If napari_animation is not installed.
    """
    try:
        from napari_animation import Animation  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "napari_animation is required to export animations. Please install napari-animation."
        ) from exc

    # Create list of layer names using list comprehension
    layer_names = [layer.name for layer in viewer.layers]

    # Hide all layers first
    for layer in viewer.layers:
        layer.visible = False

    # Show only the specified layers
    for layer_ref in layers:
        if isinstance(layer_ref, str) and layer_ref not in layer_names:
            raise ValueError(f"Layer '{layer_ref}' not found in viewer")
        viewer.layers[layer_ref].visible = True

    # Get last frame index from viewer dimensions
    # viewer.dims.range gives [(min, max, step), ...] for each dimension
    # First dimension is typically time/frames
    last_frame_idx = int(viewer.dims.range[0][1]) if len(viewer.dims.range) > 0 else 0

    animation = Animation(viewer)

    viewer.dims.current_step = (0, 0, 0)
    animation.capture_keyframe()
    viewer.dims.current_step = (last_frame_idx, 0, 0)
    animation.capture_keyframe(last_frame_idx)

    animation.animate(fname, fps=fps)
