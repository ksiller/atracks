import logging

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import Voronoi
from skimage.measure import label as sk_label, regionprops

logger = logging.getLogger(__name__)


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


def voronois_2d(
    plane_idx: int | tuple,
    binary: np.ndarray,
    stack_shape: tuple,
    connectivity: int = 1,
) -> tuple:
    """Process a single plane and return data for Voronoi tessellation.

    Args:
        plane_idx: Index of the plane. Use () for 2D case.
        binary: Binary mask for the plane.
        stack_shape: Shape of the full stack (used for boundary checking).
        connectivity: Connectivity for labeling objects.

    Returns:
        tuple: (polygons, points, features_dict) where features_dict contains
            all the feature lists for this plane.
    """
    plane_polygons = []
    plane_points = []
    plane_feat_y = []
    plane_feat_x = []
    plane_feat_area = []
    plane_feat_nverts = []
    plane_feat_plane = []
    plane_feat_vertex_dist_mean = []
    plane_feat_vertex_dist_std = []

    lbl = sk_label(binary, connectivity=connectivity)
    props = regionprops(lbl)
    if not props or len(props) < 2:
        return (
            plane_polygons,
            plane_points,
            {
                "y": plane_feat_y,
                "x": plane_feat_x,
                "area": plane_feat_area,
                "n_vertices": plane_feat_nverts,
                "plane": plane_feat_plane,
                "vertex_dist_mean": plane_feat_vertex_dist_mean,
                "vertex_dist_std": plane_feat_vertex_dist_std,
            },
        )
    # Collect centroids (x,y) for Voronoi
    all_centroids_xy = np.array(
        [[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float64
    )
    # Build Voronoi diagram
    vor = Voronoi(all_centroids_xy)
    is_2d = len(stack_shape) == 2

    for pi, prop in enumerate(props):
        reg_index = vor.point_region[pi]
        vert_index = vor.regions[reg_index]
        if -1 in vert_index or len(vert_index) == 0:
            continue
        else:
            if is_2d:
                vertices = np.array(
                    [
                        [vor.vertices[v_index][1], vor.vertices[v_index][0]]
                        for v_index in vert_index
                    ],
                    dtype=np.float32,
                )
                point = np.array([prop.centroid[0], prop.centroid[1]], dtype=np.float32)
            else:
                vertices = np.array(
                    [
                        [
                            float(plane_idx),
                            vor.vertices[v_index][1],
                            vor.vertices[v_index][0],
                        ]
                        for v_index in vert_index
                    ],
                    dtype=np.float32,
                )
                point = np.array(
                    [float(plane_idx), prop.centroid[0], prop.centroid[1]],
                    dtype=np.float32,
                )
            # only add points if all cell vertices are within the stack boundaries
            if (
                np.all(vertices[:, -2] >= 0.0)
                and np.all(vertices[:, -2] <= stack_shape[-2])
                and np.all(vertices[:, -1] >= 0.0)
                and np.all(vertices[:, -1] <= stack_shape[-1])
            ):
                # Calculate distances from centroid to each vertex
                # Extract spatial coordinates (y, x) from centroid
                if is_2d:
                    centroid_yx = np.array(
                        [prop.centroid[0], prop.centroid[1]], dtype=np.float64
                    )
                    vertices_yx = vertices
                else:
                    centroid_yx = np.array(
                        [prop.centroid[0], prop.centroid[1]], dtype=np.float64
                    )
                    vertices_yx = vertices[
                        :, -2:
                    ]  # Extract (y, x) from (N, 3) -> (N, 2)

                # Calculate Euclidean distance from centroid to each vertex
                distances = np.linalg.norm(vertices_yx - centroid_yx, axis=1)
                dist_mean = float(np.mean(distances))
                dist_std = float(np.std(distances, ddof=0))

                plane_polygons.append(vertices)
                plane_points.append(point)
                plane_feat_area.append(float(prop.area))
                plane_feat_nverts.append(int(len(vert_index)))
                plane_feat_plane.append(int(plane_idx if plane_idx != () else 0))
                plane_feat_y.append(float(prop.centroid[0]))
                plane_feat_x.append(float(prop.centroid[1]))
                plane_feat_vertex_dist_mean.append(dist_mean)
                plane_feat_vertex_dist_std.append(dist_std)

    return (
        plane_polygons,
        plane_points,
        {
            "y": plane_feat_y,
            "x": plane_feat_x,
            "area": plane_feat_area,
            "n_vertices": plane_feat_nverts,
            "plane": plane_feat_plane,
            "vertex_dist_mean": plane_feat_vertex_dist_mean,
            "vertex_dist_std": plane_feat_vertex_dist_std,
        },
    )


def voronois(
    stack: np.ndarray,
    connectivity: int = 1,
    workers: int = -1,
    verbose: int = 0,
) -> tuple:
    """Create napari layers for Voronoi tessellation per YX plane.

    Args:
        stack (np.ndarray): Binary array. Last two axes are (Y, X). Supports 2D (H,W)
            and 3D (T,H,W); for 3D, tessellation is computed per time-plane and
            layers contain coordinates with leading time dimension.
        connectivity (int): Connectivity for labeling objects.
        workers (int): Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose (int): Verbosity level for joblib.Parallel. Defaults to 0 (no output).
    Returns:
        tuple: (points_layer, polygons) from napari.layers, where the points layer
        contains object centroids and polygons contains Voronoi cell vertices.
        The points layer features include centroid coordinates (y, x), area,
        number of vertices, and vertex distance statistics.
    """
    logger.info("Creating Voronoi tessellation layers")

    try:
        from napari.layers import Shapes, Points  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "napari is required to create layers. Please install napari."
        ) from exc

    if stack.ndim not in (2, 3):
        raise ValueError("stack must be 2D (H,W) or 3D (T,H,W)")

    mask = stack.astype(bool, copy=False)

    # Prepare containers for polygons (list of (N, D) arrays) and points
    polygons: list[np.ndarray] = []
    points_list: list[np.ndarray] = []
    # Features stored as lists aligned with points_list
    feat_y: list[float] = []
    feat_x: list[float] = []
    feat_area: list[float] = []
    feat_nverts: list[int] = []
    feat_plane: list[int] = []
    feat_vertex_dist_mean: list[float] = []
    feat_vertex_dist_std: list[float] = []

    if stack.ndim == 2:
        planes = [()]
        planes_data = [(plane_idx, mask) for plane_idx in planes]
    else:
        planes = list(range(stack.shape[0]))
        planes_data = [(plane_idx, mask[plane_idx]) for plane_idx in planes]

    # Process planes in parallel and collect results
    results = Parallel(n_jobs=workers, verbose=verbose)(
        delayed(voronois_2d)(plane_idx, binary, stack.shape, connectivity)
        for plane_idx, binary in planes_data
    )

    # Merge results from all planes
    for plane_polygons, plane_points, plane_features in results:
        polygons.extend(plane_polygons)
        points_list.extend(plane_points)
        feat_y.extend(plane_features["y"])
        feat_x.extend(plane_features["x"])
        feat_area.extend(plane_features["area"])
        feat_nverts.extend(plane_features["n_vertices"])
        feat_plane.extend(plane_features["plane"])
        feat_vertex_dist_mean.extend(plane_features["vertex_dist_mean"])
        feat_vertex_dist_std.extend(plane_features["vertex_dist_std"])

    # Create napari layers
    pts = np.stack(points_list, axis=0)
    color_cycle = 20 * [
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "black",
    ]
    color_cycle = color_cycle[: np.max(feat_nverts)]
    print(f"color_cycle: {color_cycle}")
    features = {
        "plane": feat_plane,
        "centroid_y": feat_y,
        "centroid_x": feat_x,
        "area": feat_area,
        "n_vertices": feat_nverts,
        "vertex_dist_mean": feat_vertex_dist_mean,
        "vertex_dist_std": feat_vertex_dist_std,
    }
    points_layer = Points(
        data=pts,
        name="Atoms",
        face_color="n_vertices",
        face_color_cycle=color_cycle,
        features=features,
    )
    polygons_layer = Shapes(
        data=polygons,
        name="Voronoi",
        edge_color="blue",
        face_color="transparent",
        shape_type="polygon",
    )

    return points_layer, polygons
