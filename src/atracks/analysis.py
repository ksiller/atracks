import logging

import numpy as np

# from micro_sam.automatic_segmentation import (
#    get_predictor_and_segmenter,
#    automatic_tracking,
# )

from skimage.measure import label as sk_label, regionprops
from skimage.filters import (
    threshold_local,
    threshold_otsu,
    threshold_yen,
    threshold_li,
    threshold_isodata,
    threshold_triangle,
    threshold_mean,
    threshold_minimum,
    threshold_niblack,
    threshold_sauvola,
)
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    disk,
    binary_dilation,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial import cKDTree, Voronoi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion, generate_binary_structure

from joblib import Parallel, delayed

from .utils import auto_value_range, check_device
from .vis import spatial_probability_3d
from .preprocess import noise2stack_denoise, dog_gpu


logger = logging.getLogger(__name__)


def coords_from_stats(planes_dict: dict) -> np.ndarray:
    """Extract n-dimensional coordinates from plane statistics dictionary.

    Converts a dictionary of plane results (from object_stats) into a numpy array
    of n-dimensional coordinates. Each coordinate is formed by prepending the
    plane_id tuple to the centroid coordinates from each regionprops object.

    Args:
        planes_dict: Dictionary mapping plane_id (tuple[int, ...]) to lists of
            regionprops objects. Each regionprops object should have a centroid
            attribute.

    Returns:
        np.ndarray: Array of shape (N, D) where N is the total number of objects
            across all planes, and D is the dimensionality (len(plane_id) + 2).
            Each row is a coordinate: (plane_id[0], plane_id[1], ..., y, x).
            For 2D input (plane_id = ()), coordinates are (y, x).
            For 3D input (plane_id = (t,)), coordinates are (t, y, x).
            For 4D input (plane_id = (c, t)), coordinates are (c, t, y, x), etc.
    """
    all_coords = []

    for plane_id, props in planes_dict.items():
        # Extract centroids from regionprops
        centroids = np.array([prop.centroid for prop in props], dtype=np.float64)

        if len(centroids) == 0:
            continue

        # Prepend plane_id to each centroid
        if len(plane_id) == 0:
            # 2D case: plane_id is empty tuple, just use centroids as-is
            coords = centroids
        else:
            # Multi-dimensional case: prepend plane_id to each centroid
            plane_id_arr = np.array(plane_id, dtype=np.float64)
            # Broadcast plane_id to match number of centroids
            coords = np.hstack([np.tile(plane_id_arr, (len(centroids), 1)), centroids])

        all_coords.append(coords)

    if len(all_coords) == 0:
        # Return empty array with appropriate shape
        # Determine dimensionality from first plane_id (if any)
        if planes_dict:
            first_plane_id = next(iter(planes_dict.keys()))
            dim = len(first_plane_id) + 2
        else:
            dim = 2  # Default to 2D (y, x)
        return np.array([], dtype=np.float64).reshape(0, dim)

    # Concatenate all coordinates
    return np.vstack(all_coords)


def analyze(
    input: str | np.ndarray,
    sigma_low: float | tuple[float, ...] = 1.0,
    sigma_high: float | tuple[float, ...] = 5.0,
    verbose: int = 10,
) -> None:
    """Run analysis on the provided input and write results to output.

    Args:
        input (str | np.ndarray): Path to the input resource (e.g., MP4 file or directory)
            or numpy array (T, H, W, C).
        output (str): Path to the output file or directory for results.
        grayscale (bool): If True, convert loaded video frames to grayscale.
        start_frame (int | None): Optional starting frame index (0-based). If None, starts at 0.
        end_frame (int | None): Optional inclusive ending frame index. If None, reads until EOF.
        sigma_low (float | tuple[float, ...]): Sigma for the first (lower) Gaussian blur.
        sigma_high (float | tuple[float, ...]): Sigma for the second (higher) Gaussian blur.

    Returns:
        None
    """
    denoised = noise2stack_denoise(input)
    dog = dog_gpu(input, sigma_low=sigma_low, sigma_high=sigma_high)

    mask = segment_spots(dog, method="local", verbose=verbose)
    stats = object_stats(mask, summary_only=True, verbose=verbose)
    global_stats = stats["combined"]

    f_mask = filter_mask(
        mask,
        area=auto_value_range(global_stats["area_median"], tolerance=0.5),
        circularity=auto_value_range(
            global_stats["circularity_median"], tolerance=0.25
        ),
        solidity=(0.8, 1.0),
    )
    hole_mask = identify_lattice_holes(
        f_mask, radius=global_stats["nn_min_dist_mean"] // 2
    )

    final_stats = object_stats(f_mask, summary_only=False, verbose=verbose)
    centroids = coords_from_stats(final_stats["planes"])
    radius = final_stats["combined"]["nn_min_dist_median"]
    spread = final_stats["combined"]["nn_min_dist_std"]
    probs_img = spatial_probability_3d(
        input.shape, centroids, radius=radius, sigma=spread, verbose=verbose
    )

    return denoised, dog, mask, f_mask, hole_mask, final_stats, probs_img, centroids


def segment_micro_sam(
    stack: np.ndarray,
    channel: int | None = None,
    model_type: str = "vit_l_em_organelles",
    use_amg: bool = True,
    **generate_kwargs,
) -> np.ndarray:
    """Instance segment each frame with micro-sam and track over time.

    This implementation is inspired by the micro-sam example for automatic
    segmentation in live-cell data, see: [micro-sam example]
    (https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/automatic_segmentation.py).
    Credit: computational-cell-analytics/micro-sam authors.

    Args:
        stack (np.ndarray): Image stack with time as first axis (T, ..., H, W).
            If the stack has 4 dimensions, the channel axis is assumed to be the
            second axis (shape (T, C, H, W)) and the selected `channel` will be
            used for segmentation.
        channel (int | None): Optional channel index to use when `stack` has a
            channel axis. If provided for a 4D stack, the stack is sliced to
            `stack[:, channel, :, :]`. If not provided for a 4D stack, an error
            is raised. If the channel axis is last (T, H, W, C), the function will
            also accept `channel` and slice accordingly.

    Returns:
        np.ndarray: Labeled stack of shape (T, H, W) with 32-bit integer labels.
            0 denotes background; instance ids are propagated across time via IoU.

    Raises:
        RuntimeError: If micro_sam or required extras are unavailable.
        ValueError: If input dimensionality is unsupported.
    """
    try:
        from micro_sam import automatic_segmentation as ms_auto  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "micro_sam is required for segmentation. Please install micro-sam and its"
            " optional dependencies (e.g., vigra)."
        ) from exc

    if stack.ndim < 3:
        raise ValueError(
            "stack must be at least (T, H, W) or (T, C, H, W)/(T, H, W, C)"
        )

    t_dim = stack.shape[0]

    # Select channel if needed and reduce to (T, H, W)
    if stack.ndim == 3:
        gray_stack = stack
    elif stack.ndim == 4:
        if channel is None:
            raise ValueError("channel must be provided for 4D input stacks")
        # Prefer (T, C, H, W)
        if stack.shape[1] >= 1 and stack.shape[1] <= 16:
            gray_stack = stack[:, channel, :, :]
        # Also support (T, H, W, C)
        elif stack.shape[-1] >= 1 and stack.shape[-1] <= 16:
            gray_stack = stack[:, :, :, channel]
        else:
            # Fallback: assume last two dims are spatial
            gray_stack = stack.reshape((t_dim, *stack.shape[-2:]))
    else:
        # For higher-D inputs, collapse to (T, H, W)
        gray_stack = stack.reshape((t_dim, *stack.shape[-2:]))

    # Normalize to uint8
    if gray_stack.dtype != np.uint8:
        f32 = gray_stack.astype(np.float32)
        mx = float(f32.max()) if f32.size > 0 else 0.0
        gray_stack = (
            (255.0 * (f32 / mx)).astype(np.uint8) if mx > 0 else f32.astype(np.uint8)
        )

    device = check_device()

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        amg=use_amg,
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    # Use micro_sam's automatic_tracking for 3D time-lapse stacks
    segmentation, lineage = automatic_tracking(
        predictor=predictor,
        segmenter=segmenter,
        input_path=gray_stack,
        tile_shape=None,
        halo=None,
        **generate_kwargs,
    )

    print(type(segmentation), type(lineage))
    print(segmentation.shape, len(lineage))
    for i, item in enumerate(lineage):
        print(i, item)
    return segmentation.astype(np.int32, copy=False)


def segment_plane(
    plane: np.ndarray,
    block_size: int = 51,
    offset: float = 0.0,
    sigma: float = None,
    min_area: int = 20,
    opening_radius: int = 1,
    method: str = "local",
) -> np.ndarray:
    """Segment bright, approximately spherical objects using adaptive local thresholding.

    Args:
        plane (np.ndarray): Input array with spatial axes last (H, W).
        block_size (int): Block size for local thresholding methods.
        offset (float): Offset for local thresholding methods.
        sigma (float): Gaussian smoothing sigma.
        min_area (int): Minimum area for removing small objects.
        opening_radius (int): Opening radius for morphological operations.
        method (str): Thresholding method name.

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if sigma and sigma > 0:
        im = gaussian_filter(im, sigma=sigma)
    bw = _apply_threshold(plane, method, block_size, offset)
    mask = _postprocess_mask(bw, opening_radius, min_area)

    return mask


def segment_spots(
    stack: np.ndarray,
    block_size: int = 51,
    offset: float = 0.0,
    sigma: float = None,
    min_area: int = 20,
    opening_radius: int = 1,
    method: str = "local",
    workers: int = -1,
    verbose: int = 0,
) -> np.ndarray | list[np.ndarray]:
    """Segment bright, approximately spherical objects using adaptive local thresholding.

    Args:
        stack (np.ndarray): Input array with time as first axis and spatial axes last
            (T, ..., H, W). If 4D, a channel axis is assumed either at position 1
            or last.
        block_size (int): Block size for local thresholding methods.
        offset (float): Offset for local thresholding methods.
        sigma (float): Gaussian smoothing sigma.
        min_area (int): Minimum area for removing small objects.
        opening_radius (int): Opening radius for morphological operations.
        method (str): Thresholding method name.
        spot_size (tuple[float, float] | str): Spot size range or "auto" to calculate from data.
        circularity (tuple[float, float] | str | None): Circularit
        workers (int): Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose (int): Verbosity level for joblib.Parallel. Defaults to 0 (no output).
    Returns:
        np.ndarray | list[np.ndarray]: Binary mask stack (T, H, W) or list of masks (T, H, W) for each slice.
    """
    logger.info("Segmenting spots with method: %s", method)

    if stack.ndim < 2:
        raise ValueError("stack must be at least 2D with shape (H, W)")
    elif stack.ndim == 2:
        stack = [stack]
    # Normalize stack to float and prepare for processing
    img_f = stack.astype(np.float32, copy=False)
    if img_f.max() > 0:
        img_f = img_f / float(img_f.max())

    # Handle 3D case - loop over first axis
    masks = Parallel(n_jobs=workers, verbose=verbose)(
        delayed(segment_plane)(
            plane,
            block_size=block_size,
            offset=offset,
            sigma=sigma,
            min_area=min_area,
            opening_radius=opening_radius,
            method=method,
        )
        for plane in img_f
    )

    # Return result in same shape as input
    if stack.ndim == 2:
        return masks[0]
    else:
        return np.stack(masks, axis=0).astype(bool, copy=False)


def dilate_regions(
    mask: np.ndarray,
    max_area: float,
) -> np.ndarray:
    """Dilate regions in a 2D binary mask to a target maximum area.

    Identifies individual regions in the mask and dilates each region iteratively
    until its area reaches (but does not exceed) the specified maximum area.
    Ensures that dilation does not fuse adjacent regions by preventing overlap
    with other regions (both original and already-dilated).

    Args:
        mask (np.ndarray): 2D binary mask where True pixels represent regions.
        max_area (float): Maximum area for each region after dilation.
            Regions larger than this will be left unchanged.

    Returns:
        np.ndarray: 2D binary mask with dilated regions. Same shape as input.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    if not np.any(mask):
        return mask.copy()

    # Label regions
    labels = sk_label(mask, connectivity=1)
    props = regionprops(labels)

    # Track all regions that have been processed to prevent fusion
    # This mask contains all original regions plus any already-dilated regions
    occupied_mask = np.zeros_like(mask, dtype=bool)
    dilated_mask = np.zeros_like(mask, dtype=bool)

    for prop in props:
        if prop.area > max_area:
            # Already larger than target, add as-is
            region_mask = labels == prop.label
            dilated_mask |= region_mask
            occupied_mask |= region_mask
            continue

        # Create mask for this region
        region_mask = labels == prop.label
        current_area = prop.area

        # Define forbidden area: all other regions (original + already dilated)
        forbidden = occupied_mask & ~region_mask

        # Iteratively dilate with small structuring element until area reaches target
        dilated = region_mask.copy()
        selem = disk(1)  # Use small disk for fine-grained control
        max_iterations = int(np.ceil(max_area))  # Safety limit
        iteration = 0

        while current_area < max_area and iteration < max_iterations:
            try:
                # Dilate the current region
                new_dilated = binary_dilation(dilated, selem)

                # Remove any pixels that would overlap with other regions
                new_dilated = new_dilated & ~forbidden

                # Calculate area after excluding forbidden regions
                new_area = np.sum(new_dilated)

                if new_area > current_area and new_area <= max_area:
                    # Valid dilation step: area increased and still within limit
                    dilated = new_dilated
                    current_area = new_area
                    iteration += 1
                else:
                    # Would exceed target or no more valid dilation possible, stop
                    break
            except (ValueError, IndexError):
                break

        # Add dilated region to output mask and mark as occupied
        dilated_mask |= dilated
        occupied_mask |= dilated

    return dilated_mask


def iterative_threshold(
    img: np.ndarray,
    area: tuple[float, float],
    mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    circularity: tuple[float, float] | None = None,
    solidity: tuple[float, float] | None = (0.95, 1.0),
    aspect_ratio: tuple[float, float] | None = None,
    threshold: tuple[float, float] | None = (0.0, 1.0),
    iterations: int = 10,
    workers: int = -1,
    verbose: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate binary masks by iteratively thresholding an image at different intensity levels.

    Iterates over a range of threshold values from maximum to minimum intensity,
    creating a binary mask at each step where pixels exceed the threshold.
    Collects all regions found across all iterations into a single labeled array
    with unique labels.

    Args:
        img (np.ndarray): Input image to threshold.
        area (tuple[float, float]): Area range (currently unused, reserved for future filtering).
        mask (np.ndarray | None, optional): Optional mask to restrict intensity computation.
            If provided, maximum intensity is computed only within the masked region.
            Defaults to None.
        exclude_mask (np.ndarray | None, optional): Optional mask to exclude regions from
            intensity computation and final binary masks. Pixels where exclude_mask is True
            will be excluded. Defaults to None.
        circularity (tuple[float, float] | None, optional): Circularity range for filtering.
        solidity (tuple[float, float] | None, optional): Solidity range for filtering.
            Defaults to (0.95, 1.0).
        aspect_ratio (tuple[float, float] | None, optional): Aspect ratio range for filtering.
            Aspect ratio is calculated as minor_axis_length / major_axis_length.
            Defaults to None (no filtering).
        threshold (tuple[float, float] | None, optional): Threshold range (min, max).
            If None, defaults to (0.0, img_max). If first element is None, uses 0.0.
            If second element is None, uses img_max. Defaults to (0.0, 1.0).
        iterations (int, optional): Number of threshold iterations. Defaults to 10.
        workers (int): Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose (int): Verbosity level for joblib.Parallel. Defaults to 0 (no output).
    Returns:
        tuple[list[np.ndarray], np.ndarray]: A tuple containing:
            - List of binary masks, one for each threshold value.
            - Combined labeled regions array with unique labels across all iterations.
              Background pixels have value 0, regions have consecutive integer labels.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if exclude_mask is not None and exclude_mask.shape != img.shape:
        raise ValueError("exclude_mask must have the same shape as img")

    # Compute max intensity, optionally restricted by mask and excluding exclude_mask regions
    if mask is not None:
        if mask.shape != img.shape:
            raise ValueError("mask must have the same shape as img")
        # Combine mask and exclude_mask: use mask & ~exclude_mask
        valid_mask = mask.astype(bool)
        if exclude_mask is not None:
            valid_mask = valid_mask & ~exclude_mask.astype(bool)
        masked_img = img[valid_mask]
        if masked_img.size > 0:
            img_min = float(np.min(masked_img))
            img_max = float(np.max(masked_img))
        else:
            img_min = float(np.min(img))
            img_max = float(np.max(img))
    else:
        # If no mask provided, exclude exclude_mask regions from max computation
        if exclude_mask is not None:
            valid_mask = ~exclude_mask.astype(bool)
            masked_img = img[valid_mask]
            if masked_img.size > 0:
                img_min = float(np.min(masked_img))
                img_max = float(np.max(masked_img))
            else:
                img_min = float(np.min(img))
                img_max = float(np.max(img))
        else:
            img_min = float(np.min(img))
            img_max = float(np.max(img))

    # Handle threshold parameter: set defaults if None
    if threshold is None:
        threshold = (0.0, img_max)
    else:
        # Handle None values in threshold tuple
        th_min_val, th_max_val = threshold
        if th_min_val is None:
            th_min_val = 0.0
        if th_max_val is None:
            th_max_val = img_max
        threshold = (th_min_val, th_max_val)

    th_min = threshold[0]
    th_max = threshold[1]

    step_size = (th_max - th_min) / iterations

    # Generate threshold values from max to min (reverse the ascending array)
    # img_min = 0.0
    thresholds = np.arange(th_max, th_min, -step_size)

    # Calculate dilation target area (constant for all iterations)
    dilate_target_area = 0.5 * (area[1] + area[0]) if area is not None else None

    # Initialize exclude_mask if None
    if exclude_mask is None:
        exclude_mask = np.zeros_like(img, dtype=bool)
    else:
        exclude_mask = exclude_mask.copy().astype(bool)

    # Create binary mask for each threshold
    binary_masks = []

    # Initialize combined labeled regions array to track all regions across iterations
    combined_labeled_regions = np.zeros_like(img, dtype=np.int32)
    current_max_label = 0  # Track the highest label used so far

    for th in thresholds:
        logger.info(f"Thresholding with min={th}")  # and max={th_max}")
        binary_mask = img >= th  # & (img <= th_max).astype(bool)
        # Optionally apply mask to restrict output
        if mask is not None:
            binary_mask = binary_mask & mask.astype(bool)
        # Exclude regions already found in previous iterations
        binary_mask = binary_mask & ~exclude_mask
        filtered_binary_mask = filter_mask(
            binary_mask,
            area=area,
            circularity=circularity,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
        )

        # All regions found at this threshold are new (since we excluded previous ones)
        # Dilate new regions if needed, then add to exclude_mask
        new_regions_mask = filtered_binary_mask.copy()

        if dilate_target_area is not None and np.any(new_regions_mask):
            # Dilate new regions - handle both 2D and 3D arrays
            if new_regions_mask.ndim == 2:
                dilated_regions = dilate_regions(new_regions_mask, dilate_target_area)
            else:
                # 3D case: loop over first axis and dilate each plane
                dilated_planes = []
                for plane_idx in range(new_regions_mask.shape[0]):
                    dilated_plane = dilate_regions(
                        new_regions_mask[plane_idx], dilate_target_area
                    )
                    dilated_planes.append(dilated_plane)
                dilated_regions = np.stack(dilated_planes, axis=0)

            # Add dilated regions to exclude_mask
            exclude_mask = exclude_mask | dilated_regions
            # Update filtered_binary_mask to include dilated regions for this iteration's output
            filtered_binary_mask = filtered_binary_mask | dilated_regions
        else:
            # Add new regions directly to exclude_mask (no dilation)
            exclude_mask = exclude_mask | new_regions_mask

        binary_masks.append(filtered_binary_mask)

        # Label new regions and add to combined array with unique labels
        # Label based on new_regions_mask (before dilation) to preserve region identities,
        # then match each original region to its dilated version in filtered_binary_mask
        if np.any(new_regions_mask):
            if new_regions_mask.ndim == 2:
                # Label original regions before dilation
                labeled_original = sk_label(new_regions_mask, connectivity=1)
                n_new_regions = (
                    len(np.unique(labeled_original[labeled_original > 0]))
                    if np.any(labeled_original > 0)
                    else 0
                )

                # Offset labels to be unique across all iterations
                labeled_original[labeled_original > 0] += current_max_label

                # Label filtered_binary_mask to find connected components
                labeled_filtered = sk_label(filtered_binary_mask, connectivity=1)

                # Create labeled_final by matching original regions to their dilated versions
                labeled_final = np.zeros_like(filtered_binary_mask, dtype=np.int32)

                # For each original region, find the corresponding component in filtered_binary_mask
                unique_original_labels = np.unique(
                    labeled_original[labeled_original > 0]
                )
                for orig_label in unique_original_labels:
                    original_region = labeled_original == orig_label
                    # Find which component in filtered_binary_mask overlaps with this original region
                    overlapping_labels = np.unique(labeled_filtered[original_region])
                    overlapping_labels = overlapping_labels[overlapping_labels > 0]

                    if len(overlapping_labels) > 0:
                        # Assign the label to all pixels of the overlapping component(s)
                        # Since dilate_regions prevents fusion, there should be only one
                        for overlap_label in overlapping_labels:
                            labeled_final[labeled_filtered == overlap_label] = (
                                orig_label
                            )

                # Erode labeled_final by 1 pixel before adding to combined array
                if np.any(labeled_final):
                    structure = generate_binary_structure(
                        2, connectivity=1
                    )  # 4-connected
                    unique_labels = np.unique(labeled_final[labeled_final > 0])

                    # Erode each labeled region independently
                    eroded_final = np.zeros_like(labeled_final, dtype=np.int32)
                    for label in unique_labels:
                        label_mask = labeled_final == label
                        eroded_mask = binary_erosion(label_mask, structure=structure)
                        eroded_final[eroded_mask] = label

                    labeled_final = eroded_final

                # Add to combined array
                combined_labeled_regions[labeled_final > 0] = labeled_final[
                    labeled_final > 0
                ]
                # Update current_max_label
                if np.any(labeled_original):
                    current_max_label = int(np.max(labeled_original))
            else:
                # 3D case: process each plane separately
                total_new = 0
                for plane_idx in range(new_regions_mask.shape[0]):
                    labeled_original = sk_label(
                        new_regions_mask[plane_idx], connectivity=1
                    )
                    if np.any(labeled_original > 0):
                        total_new += len(
                            np.unique(labeled_original[labeled_original > 0])
                        )

                    # Offset labels to be unique across all iterations
                    labeled_original[labeled_original > 0] += current_max_label

                    # Label filtered_binary_mask for this plane
                    labeled_filtered = sk_label(
                        filtered_binary_mask[plane_idx], connectivity=1
                    )

                    # Create labeled_final by matching original regions to their dilated versions
                    labeled_final = np.zeros_like(
                        filtered_binary_mask[plane_idx], dtype=np.int32
                    )

                    unique_original_labels = np.unique(
                        labeled_original[labeled_original > 0]
                    )
                    for orig_label in unique_original_labels:
                        original_region = labeled_original == orig_label
                        overlapping_labels = np.unique(
                            labeled_filtered[original_region]
                        )
                        overlapping_labels = overlapping_labels[overlapping_labels > 0]

                        if len(overlapping_labels) > 0:
                            for overlap_label in overlapping_labels:
                                labeled_final[labeled_filtered == overlap_label] = (
                                    orig_label
                                )

                    # Erode labeled_final by 1 pixel before adding to combined array
                    if np.any(labeled_final):
                        structure = generate_binary_structure(
                            2, connectivity=1
                        )  # 4-connected
                        unique_labels = np.unique(labeled_final[labeled_final > 0])

                        # Erode each labeled region independently
                        eroded_final = np.zeros_like(labeled_final, dtype=np.int32)
                        for label in unique_labels:
                            label_mask = labeled_final == label
                            eroded_mask = binary_erosion(
                                label_mask, structure=structure
                            )
                            eroded_final[eroded_mask] = label

                        labeled_final = eroded_final

                    # Add to combined array
                    combined_labeled_regions[plane_idx][labeled_final > 0] = (
                        labeled_final[labeled_final > 0]
                    )
                    # Update current_max_label
                    if np.any(labeled_original):
                        current_max_label = int(np.max(labeled_original))

                n_new_regions = total_new

            # Log number of new regions found
            logger.info(f"Found {n_new_regions} new regions at threshold {th_min:.3f}")

    return binary_masks, combined_labeled_regions


def weighted_temporal_sum(
    stack: np.ndarray,
    n: int | tuple[int, int] = 1,
    decay: float = 0.5,  # decay factor for the weighted sum
) -> np.ndarray:
    """Compute distance-weighted sum across adjacent planes in a 3D array.

    For each plane k, computes a weighted sum of pixel values from adjacent planes,
    where the contribution of each adjacent plane decays by the decay factor per distance from k.

    Args:
        stack (np.ndarray): 3D array with shape (T, H, W) where T is the first axis.
        n (int | tuple[int, int], optional): Neighborhood size.
            If int: Symmetric range, sums planes from k-n to k+n. Defaults to 1.
            If tuple[int, int]: Asymmetric range, sums planes from k-n[0] to k+n[1].
                n[0] is the negative range (past planes) and n[1] is the positive range (future planes).
        decay (float, optional): Decay factor for the weighted sum. Weight = decay^distance.
            Defaults to 0.5.
    Returns:
        np.ndarray: 3D array of same shape as input, where each plane contains the
            weighted sum of itself and adjacent planes. Weight = decay^distance, where
            distance is |plane_index - k|.

    Raises:
        ValueError: If stack is not 3D or if n is invalid.
    """
    if stack.ndim != 3:
        raise ValueError("stack must be 3D with shape (T, H, W)")

    # Parse n parameter
    if isinstance(n, tuple):
        if len(n) != 2:
            raise ValueError(f"n tuple must have length 2, got {len(n)}")
        n_neg, n_pos = n
        if n_neg < 0 or n_pos < 0:
            raise ValueError(f"n tuple values must be non-negative, got {n}")
    else:
        # Symmetric range
        n_neg = n
        n_pos = n

    t_dim = stack.shape[0]
    result = np.zeros_like(stack, dtype=np.float64)

    # Loop through each plane k
    for k in range(t_dim):
        weighted_sum = np.zeros(stack.shape[1:], dtype=np.float64)

        # Sum contributions from planes k-n_neg to k+n_pos
        for d in range(-n_neg, n_pos + 1):
            plane_idx = k + d

            # Skip if out of bounds
            if plane_idx < 0 or plane_idx >= t_dim:
                continue

            # Calculate weight: decay^|distance|
            distance = abs(d)
            weight = decay**distance

            # Add weighted contribution
            weighted_sum += weight * stack[plane_idx].astype(np.float64)

        result[k] = weighted_sum

    return result  # .astype(stack.dtype, copy=False)


def fft_image(img: np.ndarray) -> np.ndarray:
    """Compute the 2D Fourier transform magnitude spectrum of an image.

    Performs a 2D FFT on the input image, shifts the zero-frequency component to the center,
    and returns the logarithmic magnitude spectrum for visualization.

    Args:
        img (np.ndarray): Input 2D image array.

    Returns:
        np.ndarray: Magnitude spectrum of the 2D FFT with zero-frequency at center.
            Values are in logarithmic scale (20 * log10) for better visualization.
            Same shape as input image.
    """
    f = np.fft.fft2(img)

    # Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum (amplitude) and apply a logarithmic scale
    # Adding 1 to avoid log(0)
    mag_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return mag_spectrum


def pattern_template(
    r: float,
    distance: float,
    coordination: int = 6,
    shape: tuple[int, int] | None = None,
    border: float | None = None,
) -> np.ndarray:
    """Create a binary 2D mask with a regular pattern of circular objects.

    Generates a pattern with one object at the center and coordination objects
    arranged in a circle around it. The coordination objects are equidistant from
    the center (at distance) and from each other.

    Args:
        r (float): Radius of each circular object.
        distance (float): Distance from center to each of the coordination objects.
            This is also the distance between adjacent objects on the circle.
        coordination (int, optional): Number of objects arranged around the center.
            Total number of objects will be coordination + 1 (center + ring).
            Must be in range 3-15. Defaults to 6.
        shape (tuple[int, int] | None, optional): Output mask shape (height, width).
            If None, automatically determined from distance, r, and border to create
            a square mask. Defaults to None.
        border (float | None, optional): Minimum distance of each object from the border.
            If None, defaults to 0.5 * distance. Defaults to None.

    Returns:
        np.ndarray: Binary 2D mask (boolean array) with circular objects.
            Objects are True, background is False. The hull is square.

    Raises:
        ValueError: If coordination number is not in range 3-15.
    """
    from skimage.draw import disk

    if coordination < 3 or coordination > 15:
        raise ValueError(
            f"Coordination number must be in range 3-15, got {coordination}"
        )

    # Set default border if not provided
    if border is None:
        border = 0.5 * distance

    # Total number of objects: center + coordination around it
    n = coordination + 1

    # Calculate mask size needed for square hull
    # Need to fit: center object + coordination objects at distance from center
    # Total span: distance (to outer objects) + r (object radius) + border
    if shape is None:
        # Size needed: 2 * (distance + r + border) to ensure square
        required_size = 2 * (distance + r + border) + 10  # Small margin
        size = int(np.ceil(required_size))
        width = height = size
        shape = (height, width)
    else:
        height, width = shape
        # Ensure square
        size = min(height, width)
        width = height = size
        shape = (height, width)

    # Create empty mask
    mask = np.zeros((height, width), dtype=bool)

    # Calculate center position
    center_x = width / 2
    center_y = height / 2

    # Place center object
    positions = [(center_y, center_x)]

    # Place coordination objects in a circle around the center
    # Calculate angles for coordination objects (equally spaced around a circle)
    angle_step = 2 * np.pi / coordination
    for i in range(coordination):
        angle = i * angle_step
        x = center_x + distance * np.cos(angle)
        y = center_y + distance * np.sin(angle)
        positions.append((y, x))

    # Draw circles at each position
    for y, x in positions:
        # Check bounds with border constraint
        if border <= x < (width - border) and border <= y < (height - border):
            try:
                rr, cc = disk((y, x), r, shape=(height, width))
                mask[rr, cc] = True
            except (ValueError, IndexError):
                # Skip if position is out of bounds
                continue

    return mask


def _apply_threshold(
    im: np.ndarray, method_name: str, block_size: int, offset: float
) -> np.ndarray:
    """Apply thresholding method to a single image plane.

    Args:
        im: Single image plane (H, W).
        method_name: Thresholding method name.
        block_size: Block size for local thresholding methods.
        offset: Offset for local thresholding.

    Returns:
        Binary mask.
    """
    from skimage.filters import (
        threshold_local,
        threshold_otsu,
        threshold_yen,
        threshold_li,
        threshold_isodata,
        threshold_triangle,
        threshold_mean,
        threshold_minimum,
        threshold_niblack,
        threshold_sauvola,
    )

    if method_name == "local":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        thr = threshold_local(im, block_size=bs, offset=offset, method="gaussian")
        bw = im > thr
    elif method_name == "niblack":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        thr = threshold_niblack(im, window_size=bs)
        bw = im > thr
    elif method_name == "sauvola":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        thr = threshold_sauvola(im, window_size=bs)
        bw = im > thr
    elif method_name == "otsu":
        thr = threshold_otsu(im)
        bw = im > thr
    elif method_name == "yen":
        thr = threshold_yen(im)
        bw = im > thr
    elif method_name == "li":
        thr = threshold_li(im)
        bw = im > thr
    elif method_name == "isodata":
        thr = threshold_isodata(im)
        bw = im > thr
    elif method_name == "triangle":
        thr = threshold_triangle(im)
        bw = im > thr
    elif method_name == "mean":
        thr = threshold_mean(im)
        bw = im > thr
    elif method_name == "minimum":
        thr = threshold_minimum(im)
        bw = im > thr
    else:
        raise ValueError(f"Unsupported method: {method_name}")
    return bw


def _postprocess_mask(bw: np.ndarray, opening_radius: int, min_area: int) -> np.ndarray:
    """Post-process binary mask with morphological operations and watershed.

    Args:
        bw: Binary mask (H, W).
        opening_radius: Radius for binary opening.
        min_area: Minimum area for removing small objects.

    Returns:
        Post-processed binary mask.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.feature import peak_local_max
    import scipy.ndimage as ndi
    from skimage.segmentation import watershed

    if opening_radius > 0:
        bw = binary_opening(bw, disk(max(1, int(opening_radius))))
    # Watershed to separate touching objects
    if np.any(bw):
        distance = distance_transform_edt(bw)
        peaks = peak_local_max(distance, labels=bw)
        dist_mask = np.zeros(distance.shape, dtype=bool)
        if peaks.size:
            dist_mask[tuple(peaks.T)] = True
        markers, _ = ndi.label(dist_mask)
        labels_ws = watershed(-distance, markers, mask=bw)
        bw = labels_ws > 0
    if min_area and min_area > 0:
        bw = remove_small_objects(bw, min_size=int(min_area))
    return bw.astype(bool, copy=False)


def _segment_with_method_stack(
    img_f: np.ndarray,
    method_name: str,
    block_size: int,
    offset: float,
    sigma: float,
    opening_radius: int,
    min_area: int,
) -> np.ndarray:
    """Segment stack using a specific thresholding method.

    Args:
        img_f: Normalized image stack (T, H, W).
        method_name: Thresholding method name.
        block_size: Block size for local methods.
        offset: Offset for local methods.
        sigma: Gaussian smoothing sigma.
        opening_radius: Opening radius.
        min_area: Minimum area.

    Returns:
        Binary mask stack (T, H, W).
    """
    from scipy.ndimage import gaussian_filter

    t_dim = img_f.shape[0]
    masks = []
    for t in range(t_dim):
        im = img_f[t]
        if sigma and sigma > 0:
            im = gaussian_filter(im, sigma=sigma)
        bw = _apply_threshold(im, method_name, block_size, offset)
        bw = _postprocess_mask(bw, opening_radius, min_area)
        masks.append(bw)
    return np.stack(masks, axis=0).astype(bool, copy=False)


def _calculate_area_thresholds(mask_stack: np.ndarray) -> tuple[float, float]:
    """Calculate area thresholds from median area.

    Args:
        mask_stack: Binary mask stack (T, H, W) or (H, W).

    Returns:
        Tuple of (min_area, max_area).
    """
    from skimage.measure import label as sk_label, regionprops

    all_areas = []
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]
    for t in range(mask_stack.shape[0]):
        if np.any(mask_stack[t]):
            lbl = sk_label(mask_stack[t], connectivity=1)
            props = regionprops(lbl)
            all_areas.extend([p.area for p in props])

    if len(all_areas) > 0:
        median_area = float(np.median(all_areas))
        area_min = 0.5 * median_area
        area_max = 1.5 * median_area
    else:
        area_min = 0.0
        area_max = float("inf")
    return area_min, area_max


def segment_spots_full(
    stack: np.ndarray,
    channel: int | None = None,
    block_size: int = 51,
    offset: float = 0.0,
    sigma: float = 1.0,
    min_area: int = 20,
    opening_radius: int = 1,
    method: str = "local",
    spot_size: tuple[float, float] | str = "auto",
    circularity: tuple[float, float] | str | None = "auto",
) -> np.ndarray | list[np.ndarray]:
    """Segment bright, approximately spherical objects using adaptive local thresholding.

    Args:
        stack (np.ndarray): Input array with time as first axis and spatial axes last
            (T, ..., H, W). If 4D, a channel axis is assumed either at position 1
            (T, C, H, W) or at the last position (T, H, W, C). Use `channel` to select.
        channel (int | None): Optional channel index to select for 4D input.
        block_size (int): Odd window size for local thresholding (skimage threshold_local).
        offset (float): Constant subtracted from local threshold.
        sigma (float): Gaussian smoothing (in pixels) applied before thresholding.
        min_area (int): Remove connected components smaller than this area (in pixels).
        opening_radius (int): Radius for binary opening with a disk structuring element.
        method (str): Select thresholding method. Supported:
            "local" (default), "niblack", "sauvola", "otsu", "yen", "li",
            "isodata", "triangle", "mean", "minimum", or "all". If "all",
            returns a list of results for all methods.
        spot_size (tuple[float, float] | str): Area filtering mode. If "auto" (default),
            calculates median area of all regions and filters to 0.5*median to 1.5*median.
            If a tuple, interpret as (min_area, max_area). Regions outside this range
            are set to black (removed from mask).
        circularity (tuple[float, float] | str | None): Circularity filtering mode.
            If "auto" (default), calculates median circularity and filters to
            0.75*median to max(1.0, 1.25*median). If a tuple, interpret as
            (min_circularity, max_circularity). If None, no circularity filtering
            is applied. Circularity is calculated as 4π*area/perimeter² (1.0 = perfect circle).

    Returns:
        np.ndarray | list[np.ndarray]: Boolean mask stack (T, H, W). If
            method=="all" a list of mask stacks in the order listed above.
    """
    # Prepare stack and normalize
    img_f, t_dim = _prepare_stack(stack, channel)

    methods_all = [
        "local",
        "niblack",
        "sauvola",
        "otsu",
        "yen",
        "li",
        "isodata",
        "triangle",
        "mean",
        "minimum",
    ]

    # Parse circularity parameter
    if circularity is None:
        circularity_min = 0.0
        circularity_max = 1.0
    elif circularity == "auto":
        circularity_min = None
        circularity_max = None
    elif isinstance(circularity, tuple):
        circularity_min = float(circularity[0])
        circularity_max = float(circularity[1])
    else:
        raise ValueError(
            f"circularity must be 'auto', None, or a tuple of (min, max), got: {circularity}"
        )

    # Determine area thresholds and perform segmentation
    if spot_size == "auto":
        # Get initial segmentation to compute median area
        if method == "all":
            temp_mask = _segment_with_method_stack(
                img_f,
                methods_all[0],
                block_size,
                offset,
                sigma,
                opening_radius,
                min_area,
            )
        else:
            res = _segment_with_method_stack(
                img_f, method, block_size, offset, sigma, opening_radius, min_area
            )
            temp_mask = res

        # Calculate area thresholds from median
        area_min, area_max = _calculate_area_thresholds(temp_mask)

        # Apply filtering
        if method == "all":
            results = []
            for m in methods_all:
                mask = _segment_with_method_stack(
                    img_f, m, block_size, offset, sigma, opening_radius, min_area
                )
                # Determine circularity thresholds if needed
                circ_min_final = None
                circ_max_final = None
                if circularity == "auto":
                    circ_min_final, circ_max_final = (
                        _calculate_circularity_thresholds_from_props(
                            mask, area_min, area_max
                        )
                    )
                elif circularity is not None:
                    circ_min_final = circularity_min
                    circ_max_final = circularity_max
                filtered = _apply_filters(
                    mask, area_min, area_max, circ_min_final, circ_max_final
                )
                results.append(filtered)
            return results
        else:
            # Determine circularity thresholds if needed
            circ_min_final = None
            circ_max_final = None
            if circularity == "auto":
                circ_min_final, circ_max_final = (
                    _calculate_circularity_thresholds_from_props(
                        res, area_min, area_max
                    )
                )
            elif circularity is not None:
                circ_min_final = circularity_min
                circ_max_final = circularity_max
            filtered_res = _apply_filters(
                res, area_min, area_max, circ_min_final, circ_max_final
            )
            return filtered_res if stack.ndim >= 3 else filtered_res[0]
    elif isinstance(spot_size, tuple):
        area_min = float(spot_size[0])
        area_max = float(spot_size[1])

        # Apply segmentation and filtering
        if method == "all":
            results = []
            for m in methods_all:
                mask = _segment_with_method_stack(
                    img_f, m, block_size, offset, sigma, opening_radius, min_area
                )
                # Determine circularity thresholds if needed
                circ_min_final = None
                circ_max_final = None
                if circularity == "auto":
                    circ_min_final, circ_max_final = (
                        _calculate_circularity_thresholds_from_props(
                            mask, area_min, area_max
                        )
                    )
                elif circularity is not None:
                    circ_min_final = circularity_min
                    circ_max_final = circularity_max
                filtered = _apply_filters(
                    mask, area_min, area_max, circ_min_final, circ_max_final
                )
                results.append(filtered)
            return results

        res = _segment_with_method_stack(
            img_f, method, block_size, offset, sigma, opening_radius, min_area
        )
        # Determine circularity thresholds if needed
        circ_min_final = None
        circ_max_final = None
        if circularity == "auto":
            circ_min_final, circ_max_final = (
                _calculate_circularity_thresholds_from_props(res, area_min, area_max)
            )
        elif circularity is not None:
            circ_min_final = circularity_min
            circ_max_final = circularity_max
        filtered_res = _apply_filters(
            res, area_min, area_max, circ_min_final, circ_max_final
        )
        return filtered_res if stack.ndim >= 3 else filtered_res[0]
    else:
        raise ValueError(
            f"spot_size must be 'auto' or a tuple of (min, max), got: {spot_size}"
        )


def _apply_filters_plane(
    bw: np.ndarray,
    lbl: np.ndarray,
    props: list,
    area_min: float | None = None,
    area_max: float | None = None,
    circularity_min: float | None = None,
    circularity_max: float | None = None,
) -> np.ndarray:
    """Apply area and circularity filters to a single plane using pre-computed regionprops.

    Args:
        bw: Binary mask plane of shape (H, W).
        lbl: Labeled image from skimage.measure.label.
        props: List of regionprops objects.
        area_min: Minimum allowed area. If None, no area filtering.
        area_max: Maximum allowed area. If None, no area filtering.
        circularity_min: Minimum allowed circularity. If None, no circularity filtering.
        circularity_max: Maximum allowed circularity. If None, no circularity filtering.

    Returns:
        Filtered binary mask plane with same shape.
    """
    filtered = np.zeros_like(bw, dtype=bool)
    for prop in props:
        # Check area filter
        area_ok = True
        if area_min is not None and area_max is not None:
            area_ok = area_min <= prop.area <= area_max

        # Check circularity filter
        circularity_ok = True
        if circularity_min is not None and circularity_max is not None:
            if prop.perimeter > 0:
                circ = 4.0 * np.pi * float(prop.area) / (float(prop.perimeter) ** 2)
                circularity_ok = circularity_min <= circ <= circularity_max
            else:
                circularity_ok = False

        # Keep region if both filters pass (or are disabled)
        if area_ok and circularity_ok:
            filtered[lbl == prop.label] = True

    return filtered


def _apply_filters(
    mask_stack: np.ndarray,
    area_min: float | None = None,
    area_max: float | None = None,
    circularity_min: float | None = None,
    circularity_max: float | None = None,
) -> np.ndarray:
    """Apply area and circularity filters to a mask stack, computing regionprops once per plane.

    Args:
        mask_stack: Binary mask stack of shape (T, H, W) or (H, W).
        area_min: Minimum allowed area. If None, no area filtering.
        area_max: Maximum allowed area. If None, no area filtering.
        circularity_min: Minimum allowed circularity. If None, no circularity filtering.
        circularity_max: Maximum allowed circularity. If None, no circularity filtering.

    Returns:
        Filtered binary mask stack with same shape.
    """
    from skimage.measure import label as sk_label, regionprops

    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]
        squeeze = True
    else:
        squeeze = False

    filtered_masks = []
    for t in range(mask_stack.shape[0]):
        bw = mask_stack[t]
        if not np.any(bw):
            filtered_masks.append(bw)
            continue

        # Compute regionprops once per plane
        lbl = sk_label(bw, connectivity=1)
        props = regionprops(lbl)

        # Apply both filters in a single pass
        filtered = _apply_filters_plane(
            bw, lbl, props, area_min, area_max, circularity_min, circularity_max
        )
        filtered_masks.append(filtered)

    result = np.stack(filtered_masks, axis=0)
    if squeeze:
        result = result[0]
    return result.astype(bool, copy=False)


def _calculate_circularity_thresholds_from_props(
    mask_stack: np.ndarray,
    area_min: float | None = None,
    area_max: float | None = None,
) -> tuple[float, float]:
    """Calculate min and max circularity thresholds from median circularity of area-filtered regions.

    Args:
        mask_stack: Binary mask stack to analyze.
        area_min: Optional minimum area filter to apply before calculating thresholds.
        area_max: Optional maximum area filter to apply before calculating thresholds.

    Returns:
        Tuple of (min_circularity, max_circularity).
    """
    from skimage.measure import label as sk_label, regionprops

    all_circularities = []
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]
    for t in range(mask_stack.shape[0]):
        if np.any(mask_stack[t]):
            lbl = sk_label(mask_stack[t], connectivity=1)
            props = regionprops(lbl)
            for prop in props:
                # Apply area filter if specified
                if area_min is not None and area_max is not None:
                    if not (area_min <= prop.area <= area_max):
                        continue
                # Calculate circularity
                if prop.perimeter > 0:
                    circ = 4.0 * np.pi * float(prop.area) / (float(prop.perimeter) ** 2)
                    all_circularities.append(circ)

    if len(all_circularities) > 0:
        median_circularity = float(np.median(all_circularities))
        min_circularity = 0.75 * median_circularity
        max_circularity = max(1.0, 1.25 * median_circularity)
    else:
        # Fallback if no regions found
        min_circularity = 0.0
        max_circularity = 1.0
    return min_circularity, max_circularity


def identify_lattice_holes(mask: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Identify holes in a lattice of regularly spaced white spots.

    A hole is defined as an empty area (black region in the mask) that is larger
    than a disk of radius r.

    Args:
        mask (np.ndarray): Binary mask where white (True) represents spots and
            black (False) represents background. Supports 2D (H, W) or 3D (T, H, W)
            arrays. For 3D arrays, processing is performed independently for each
            plane defined by the last two axes.
        radius (float): Radius of the disk-shaped structuring element. Defaults to 1.0.

    Returns:
        np.ndarray: Binary mask of the same shape as input, where True indicates
            holes in the lattice (empty regions that fit a rolling disk of the given radius).

    Raises:
        ValueError: If mask is not 2D or 3D.
    """
    logger.info(f"Identifying lattice holes using rolling disk with radius={radius}")
    from skimage import morphology

    # Handle 2D case - wrap in list for uniform loop
    if mask.ndim == 2:
        mask_list = [mask]
    else:
        # Handle 3D case - convert to list of planes
        mask_list = mask.copy()

    # Define a disk-shaped structuring element
    selem = morphology.disk(radius)

    # Process each plane: invert mask (holes become True), then binary opening to find holes that fit the disk
    hole_mask = np.array(
        [morphology.binary_opening(~plane.astype(bool), selem) for plane in mask_list]
    )

    # Return result in same shape as input
    if mask.ndim == 2:
        return hole_mask[0]
    else:
        return hole_mask


def filter_mask(
    mask: np.ndarray,
    area: tuple[float, float] | None = None,
    circularity: tuple[float, float] | None = None,
    solidity: tuple[float, float] | None = None,
    aspect_ratio: tuple[float, float] | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """Filter a binary mask based on area, circularity, solidity, and/or aspect_ratio criteria.

    Args:
        mask (np.ndarray): Binary mask stack to filter. Can be 2D (H, W) or 3D (T, H, W).
        area (tuple[float, float] | None, optional): Tuple of (min_area, max_area) to filter by.
            Objects outside this range will be removed. Defaults to None (no filtering).
        circularity (tuple[float, float] | None, optional): Tuple of (min_circularity, max_circularity)
            to filter by. Objects outside this range will be removed. Defaults to None (no filtering).
        solidity (tuple[float, float] | None, optional): Tuple of (min_solidity, max_solidity)
            to filter by. Objects outside this range will be removed. Defaults to None (no filtering).
        aspect_ratio (tuple[float, float] | None, optional): Tuple of (min_aspect_ratio, max_aspect_ratio)
            to filter by. Aspect ratio is calculated as minor_axis_length / major_axis_length of an
            ellipsoid fitted to the region. Objects outside this range will be removed.
            Defaults to None (no filtering).
        inplace (bool): If True, filter the mask in place. Defaults to False.
    Returns:
        np.ndarray: Filtered binary mask with same shape as input.
    """
    logger.info(
        "Filtering mask with area: %s, circularity: %s, solidity: %s, and aspect_ratio: %s",
        area,
        circularity,
        solidity,
        aspect_ratio,
    )
    from skimage.measure import label as sk_label, regionprops

    if mask.ndim < 2:
        raise ValueError("mask must be at least 2D with last two axes as (Y, X)")

    # Extract area, circularity, solidity, and aspect_ratio bounds
    area_min, area_max = area if area is not None else (None, None)
    circularity_min, circularity_max = (
        circularity if circularity is not None else (None, None)
    )
    solidity_min, solidity_max = solidity if solidity is not None else (None, None)
    aspect_ratio_min, aspect_ratio_max = (
        aspect_ratio if aspect_ratio is not None else (None, None)
    )

    # Prepare output
    if inplace:
        result = mask
    else:
        result = mask.copy()

    # Handle 2D case - wrap in list for uniform loop
    if mask.ndim == 2:
        planes_to_process = [result]
    else:
        # Handle 3D case - create list of plane views
        planes_to_process = [result[i] for i in range(result.shape[0])]

    # Process each plane
    for i, plane in enumerate(planes_to_process):
        plane_bool = plane.astype(bool, copy=False)
        if np.any(plane_bool):
            lbl = sk_label(plane_bool, connectivity=1)
            props = regionprops(lbl)

            # Identify regions to remove
            for prop in props:
                # Check area filter
                area_ok = True
                if area_min is not None and area_max is not None:
                    area_ok = area_min <= prop.area <= area_max

                # Check circularity filter
                circularity_ok = True
                if circularity_min is not None and circularity_max is not None:
                    if prop.perimeter > 0:
                        circ = (
                            4.0
                            * np.pi
                            * float(prop.area)
                            / (float(prop.perimeter) ** 2)
                        )
                        circularity_ok = circularity_min <= circ <= circularity_max
                    else:
                        circularity_ok = False

                # Check solidity filter
                solidity_ok = True
                if solidity_min is not None and solidity_max is not None:
                    if hasattr(prop, "solidity") and not np.isnan(prop.solidity):
                        sol = float(prop.solidity)
                        solidity_ok = solidity_min <= sol <= solidity_max
                    else:
                        solidity_ok = False

                # Check aspect_ratio filter
                aspect_ratio_ok = True
                if aspect_ratio_min is not None and aspect_ratio_max is not None:
                    if (
                        hasattr(prop, "major_axis_length")
                        and hasattr(prop, "minor_axis_length")
                        and prop.major_axis_length > 0
                    ):
                        ar = float(prop.minor_axis_length / prop.major_axis_length)
                        aspect_ratio_ok = aspect_ratio_min <= ar <= aspect_ratio_max
                    else:
                        aspect_ratio_ok = False

                # Remove region if it doesn't pass filters
                if not (area_ok and circularity_ok and solidity_ok and aspect_ratio_ok):
                    plane[lbl == prop.label] = 0

    return result


def circularity(regionmask):
    """Compute circularity: 4π * area / perimeter² (perfect circle = 1.0)."""
    # We need to compute perimeter from the mask
    # For now, we'll compute it from the regionprops, but we need to pass it differently
    # Actually, we can't access other properties in extra_properties
    # So we'll compute it manually or use a different approach
    from skimage.measure import perimeter

    perim = perimeter(regionmask)
    area = np.sum(regionmask)
    if perim > 0:
        return float(4.0 * np.pi * area / (perim**2))
    return float("nan")


def _bbox_width(regionmask, intensity_image):
    """Compute bounding box width from region mask.

    Uses numpy operations to find column bounds more efficiently.
    """
    if not np.any(regionmask):
        return float("nan")
    # Find columns that contain any True values
    cols = np.any(regionmask, axis=0)
    if not np.any(cols):
        return float("nan")
    # bbox format: (min_row, min_col, max_row, max_col)
    # Width = max_col - min_col + 1
    col_indices = np.where(cols)[0]
    return float(col_indices[-1] - col_indices[0] + 1)


def _bbox_height(regionmask, intensity_image):
    """Compute bounding box height from region mask.

    Uses numpy operations to find row bounds more efficiently.
    """
    if not np.any(regionmask):
        return float("nan")
    # Find rows that contain any True values
    rows = np.any(regionmask, axis=1)
    if not np.any(rows):
        return float("nan")
    # Height = max_row - min_row + 1
    row_indices = np.where(rows)[0]
    return float(row_indices[-1] - row_indices[0] + 1)


def aspect_ratio(regionmask):
    """Compute aspect ratio: minor_axis_length / major_axis_length."""
    # We need to compute major/minor axis from the mask
    # For simplicity, we'll use the covariance matrix approach
    coords = np.where(regionmask)
    if len(coords[0]) == 0:
        return float("nan")

    coords_array = np.array([coords[0], coords[1]], dtype=np.float64)
    centroid = np.mean(coords_array, axis=1)
    coords_centered = coords_array - centroid[:, np.newaxis]

    if coords_centered.shape[1] < 2:
        return float("nan")

    cov = np.cov(coords_centered)
    eigenvalues = np.linalg.eigvals(cov)
    if len(eigenvalues) < 2 or eigenvalues[0] <= 0:
        return float("nan")

    # Sort eigenvalues (largest first)
    eigenvalues = np.sort(eigenvalues)[::-1]
    if eigenvalues[1] <= 0:
        return float("nan")

    return float(np.sqrt(eigenvalues[1] / eigenvalues[0]))


def object_stats_plane(plane: np.ndarray, connectivity: int = 1) -> list:
    """Compute object statistics for a single plane.

    Uses regionprops with custom properties to compute circularity, bounding box
    dimensions, and aspect ratio for each region.

    Args:
        plane (np.ndarray): Binary mask.
        connectivity (int): Connectivity for labeling (1=4-connected, 2=8-connected).

    Returns:
        list: List of regionprops objects with additional custom properties:
            - circularity: 4π * area / perimeter²
            - bbox_width: Width of bounding box
            - bbox_height: Height of bounding box
            - aspect_ratio: minor_axis_length / major_axis_length

            Note: bbox_area is a standard regionprops attribute and is available directly.
    """
    logger.info("Computing object statistics for a single plane")

    lbl = sk_label(plane, connectivity=connectivity)
    props = regionprops(
        lbl,
        extra_properties=(
            circularity,
            _bbox_width,
            _bbox_height,
            aspect_ratio,
        ),
    )

    # Return list of regionprops
    # Note: nn_min_dist is computed in _compute_stats_from_props when needed
    return props


def standard_stats(arr, prefix: list[str] | None = None, agg: list[str] | None = None):
    """Compute statistical aggregations for an array.

    Args:
        arr: Array of values to compute statistics for.
        prefix: Optional list of prefixes to build dictionary keys.
            If provided, returns dict with keys like "{prefix}_{agg}", etc.
            If None, returns tuple of values in the order specified by agg.
        agg: Optional list of aggregation types to compute.
            Allowed values: "min", "max", "mean", "median", "std".
            Default is ["mean", "std", "median", "min", "max"].

    Returns:
        If prefix is None: tuple of values in the order specified by agg
        If prefix is provided: dict with keys built from prefix and agg suffixes
    """
    if agg is None:
        agg = ["mean", "std", "median", "min", "max"]

    # Validate agg values
    allowed_agg = {"min", "max", "mean", "median", "std"}
    invalid = set(agg) - allowed_agg
    if invalid:
        raise ValueError(
            f"Invalid aggregation types: {invalid}. Allowed: {allowed_agg}"
        )

    # Compute values
    values = {}
    if arr.size == 0:
        arr_valid = np.array([], dtype=arr.dtype)
    else:
        arr_valid = arr[~np.isnan(arr)]

    if arr_valid.size == 0:
        for a in agg:
            values[a] = float("nan")
    else:
        if "min" in agg:
            values["min"] = float(np.min(arr_valid))
        if "max" in agg:
            values["max"] = float(np.max(arr_valid))
        if "mean" in agg:
            values["mean"] = float(np.mean(arr_valid))
        if "median" in agg:
            values["median"] = float(np.median(arr_valid))
        if "std" in agg:
            values["std"] = float(np.std(arr_valid, ddof=0))

    if prefix is None:
        # Return tuple in the order specified by agg
        return tuple(values[a] for a in agg)

    # Build dictionary with prefix keys
    result = {}
    for p in prefix:
        for a in agg:
            result[f"{p}_{a}"] = values[a]
    return result


def _compute_stats_from_props(props: list) -> dict:
    """Compute summary statistics from a list of regionprops objects.

    Args:
        props: List of regionprops objects.

    Returns:
        dict: Dictionary with summary statistics (min, max, mean, median, std)
            for all metrics, plus count and centroids.
    """
    if len(props) == 0:
        return {
            "count": 0,
            "centroids": np.array([], dtype=np.float64).reshape(0, 2),
        }

    # Extract all metrics into arrays
    areas = np.array([p.area for p in props], dtype=np.float64)
    perimeters = np.array([p.perimeter for p in props], dtype=np.float64)
    circularities = np.array(
        [p.circularity if hasattr(p, "circularity") else np.nan for p in props],
        dtype=np.float64,
    )
    bbox_widths = np.array(
        [p.bbox_width if hasattr(p, "bbox_width") else np.nan for p in props],
        dtype=np.float64,
    )
    bbox_heights = np.array(
        [p.bbox_height if hasattr(p, "bbox_height") else np.nan for p in props],
        dtype=np.float64,
    )
    bbox_areas = np.array(
        [p.bbox_area if hasattr(p, "bbox_area") else np.nan for p in props],
        dtype=np.float64,
    )
    solidities = np.array(
        [p.solidity if hasattr(p, "solidity") else np.nan for p in props],
        dtype=np.float64,
    )
    aspect_ratios = np.array(
        [p.aspect_ratio if hasattr(p, "aspect_ratio") else np.nan for p in props],
        dtype=np.float64,
    )
    eccentricities = np.array(
        [p.eccentricity if hasattr(p, "eccentricity") else np.nan for p in props],
        dtype=np.float64,
    )
    centroids = np.array([p.centroid for p in props], dtype=np.float64)

    # Compute nearest neighbor distances
    if len(props) >= 2:
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)
        nn_dists = dists[:, 1]
    else:
        nn_dists = np.array([], dtype=np.float64)

    result = {"count": int(len(props)), "centroids": centroids}
    result.update(standard_stats(areas, prefix=["area"]))
    result.update(standard_stats(perimeters, prefix=["perimeter"]))
    result.update(standard_stats(circularities, prefix=["circularity"]))
    result.update(standard_stats(bbox_widths, prefix=["bbox_width"]))
    result.update(standard_stats(bbox_heights, prefix=["bbox_height"]))
    result.update(standard_stats(bbox_areas, prefix=["bbox_area"]))
    result.update(standard_stats(solidities, prefix=["solidity"]))
    result.update(standard_stats(aspect_ratios, prefix=["aspect_ratio"]))
    result.update(standard_stats(eccentricities, prefix=["eccentricity"]))
    # Only compute mean, median, std for nn_min_dist (not min/max)
    result.update(
        standard_stats(nn_dists, prefix=["nn_min_dist"], agg=["mean", "median", "std"])
    )

    return result


def summarize(stats_list: list) -> dict:
    """Aggregate statistics across a list of regionprops lists.

    Collects all raw values across all planes and computes aggregated statistics
    using standard_stats function.

    Args:
        stats_list: List of regionprops lists (from object_stats_plane).

    Returns:
        dict: Aggregated statistics dictionary with aggregated values across all planes.
    """
    if not stats_list:
        raise ValueError("stats_list cannot be empty")

    # Collect all raw values across all planes
    all_areas = []
    all_perimeters = []
    all_circularities = []
    all_bbox_widths = []
    all_bbox_heights = []
    all_bbox_areas = []
    all_solidities = []
    all_aspect_ratios = []
    all_eccentricities = []
    all_nn_dists = []

    for props in stats_list:
        for prop in props:
            all_areas.append(prop.area)
            all_perimeters.append(prop.perimeter)
            if hasattr(prop, "circularity"):
                all_circularities.append(prop.circularity)
            if hasattr(prop, "bbox_width"):
                all_bbox_widths.append(prop.bbox_width)
            if hasattr(prop, "bbox_height"):
                all_bbox_heights.append(prop.bbox_height)
            if hasattr(prop, "bbox_area"):
                all_bbox_areas.append(prop.bbox_area)
            if hasattr(prop, "solidity"):
                all_solidities.append(prop.solidity)
            if hasattr(prop, "aspect_ratio"):
                all_aspect_ratios.append(prop.aspect_ratio)
            if hasattr(prop, "eccentricity"):
                all_eccentricities.append(prop.eccentricity)

        # Compute nearest neighbor distances for this plane separately
        if len(props) >= 2:
            centroids = np.array([p.centroid for p in props], dtype=np.float64)
            tree = cKDTree(centroids)
            dists, _ = tree.query(centroids, k=2)
            all_nn_dists.extend(dists[:, 1].tolist())

    # Convert to arrays and compute stats
    result = {"count": int(len(all_areas))}

    # Compute stats for each metric using prefix
    result.update(
        standard_stats(np.array(all_areas, dtype=np.float64), prefix=["area"])
    )
    result.update(
        standard_stats(np.array(all_perimeters, dtype=np.float64), prefix=["perimeter"])
    )
    result.update(
        standard_stats(
            np.array(all_circularities, dtype=np.float64), prefix=["circularity"]
        )
    )
    result.update(
        standard_stats(
            np.array(all_bbox_widths, dtype=np.float64), prefix=["bbox_width"]
        )
    )
    result.update(
        standard_stats(
            np.array(all_bbox_heights, dtype=np.float64), prefix=["bbox_height"]
        )
    )
    result.update(
        standard_stats(np.array(all_bbox_areas, dtype=np.float64), prefix=["bbox_area"])
    )
    result.update(
        standard_stats(np.array(all_solidities, dtype=np.float64), prefix=["solidity"])
    )
    result.update(
        standard_stats(
            np.array(all_aspect_ratios, dtype=np.float64), prefix=["aspect_ratio"]
        )
    )
    result.update(
        standard_stats(
            np.array(all_eccentricities, dtype=np.float64), prefix=["eccentricity"]
        )
    )

    # Only compute mean, median, std for nn_min_dist (not min/max)
    nn_stats = standard_stats(
        np.array(all_nn_dists, dtype=np.float64),
        prefix=["nn_min_dist"],
        agg=["mean", "median", "std"],
    )
    result.update(nn_stats)

    return result


def object_stats(
    stack: np.ndarray,
    connectivity: int = 1,
    summary_only: bool = False,
    workers: int = -1,
    verbose: int = 0,
) -> dict:
    """Compute per-slice object statistics for a multi-dimensional binary stack.

    The last two axes are interpreted as (Y, X). All preceding axes form the
    indexing over slices. For each 2D slice, connected components are measured
    to compute area statistics, perimeter, circularity, bounding box, solidity,
    aspect ratio, and nearest-neighbor distances between object centroids.

    Args:
        stack (np.ndarray): Binary-like array. Non-zero pixels are treated as foreground.
        connectivity (int): Connectivity for labeling (1=4-connected, 2=8-connected).
        summary_only (bool): If True, compute regionprops for all planes but only return
            combined metrics. If False (default), return both per-slice and combined results.
        workers (int): Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose (int): Verbosity level for joblib.Parallel. Defaults to 0 (no output).
    Returns:
        dict: A dictionary containing:
            - "combined": Combined statistics across all slices
            - "spatial_shape": Spatial dimensions (Y, X)
            - "stack_shape": Full stack shape
            - "planes": (only if summary_only=False) Dictionary mapping plane_id
                (tuple[int, ...]) to regionprops lists. Each plane_id is a tuple
                of indices for the non-spatial axes. Each list contains regionprops
                objects with custom properties (circularity, bbox_width, bbox_height,
                aspect_ratio). Standard properties like bbox_area are also available.
    """
    logger.info("Computing object statistics")

    spatial_shape = stack.shape[-2:]
    if stack.ndim < 2:
        raise ValueError("stack must be at least 2D with last two axes as (Y, X)")

    # Handle 2D case by wrapping in list
    if stack.ndim == 2:
        non_spatial_shape = ()
        planes_list = [stack]
    else:
        non_spatial_shape = stack.shape[:-2]
        # Flatten non-spatial dimensions for iteration
        planes_list = []
        for idx in np.ndindex(non_spatial_shape):
            planes_list.append(stack[idx])

    # Ensure boolean mask
    mask = [plane.astype(bool, copy=False) for plane in planes_list]

    results = Parallel(n_jobs=workers, verbose=verbose)(
        delayed(object_stats_plane)(plane, connectivity) for plane in mask
    )
    print(
        f"first result: {results[0][0].circularity}, type: {type(results[0][0].circularity)}, last result: {results[-1][-1].circularity}, type: {type(results[-1][-1].circularity)}"
    )

    combined = summarize(results)  # type: ignore

    return_dict = {
        "combined": combined,
        "spatial_shape": spatial_shape,
        "stack_shape": stack.shape,
    }

    # Only include per-slice results if summary_only is False
    if not summary_only:
        # Create dictionary mapping plane_id tuples to results
        planes_dict = {}
        if stack.ndim == 2:
            # Single plane with empty tuple as key
            planes_dict[()] = results[0]
        else:
            # Generate plane_id tuples from non_spatial_shape
            for i, idx in enumerate(np.ndindex(non_spatial_shape)):
                planes_dict[idx] = results[i]
        return_dict["planes"] = planes_dict

    return return_dict


def mark_large_objects(
    stack: np.ndarray,
    area_mean: float,
    area_std: float,
    nn_min_dist_mean: float,
    nn_min_dist_std: float,
    connectivity: int = 1,
) -> np.ndarray:
    """Mark objects larger than a global threshold in a binary stack.

    The last two axes are interpreted as (Y, X). All preceding axes form the
    indexing over slices. Foreground (non-zero) pixels are copied as value 1 in
    the output; objects with area > (area_mean + 2 * area_std) are set to value 2.

    Args:
        stack (np.ndarray): Binary-like array. Non-zero treated as foreground.
        area_mean (float): Combined mean area across slices.
        area_std (float): Combined std of area across slices.
        nn_min_dist_mean (float): Combined mean of nearest-neighbor distances (unused).
        nn_min_dist_std (float): Combined std of nearest-neighbor distances (unused).
        connectivity (int): Connectivity for labeling (1=4-connected, 2=8-connected).

    Returns:
        np.ndarray: Integer array of same shape. Background=0, normal objects=1,
            large objects=2.
    """
    logger.info("Marking large objects with area: %s and std: %s", area_mean, area_std)
    from skimage.measure import label as sk_label, regionprops

    if stack.ndim < 2:
        raise ValueError("stack must be at least 2D with last two axes as (Y, X)")

    threshold = float(area_mean) + 2.0 * float(area_std)

    out = np.zeros_like(stack, dtype=np.int32)
    # Initialize foreground to 1 where input is non-zero
    out[stack.astype(bool, copy=False)] = 1

    non_spatial_shape = stack.shape[:-2]
    if non_spatial_shape == ():
        indices_iter = [()]
    else:
        indices_iter = np.ndindex(non_spatial_shape)

    for idx in indices_iter:
        plane_bin = (stack[idx] if idx != () else stack).astype(bool, copy=False)
        if not plane_bin.any():
            continue
        lbl = sk_label(plane_bin, connectivity=connectivity)
        if lbl.max() == 0:
            continue
        props = regionprops(lbl)
        for p in props:
            if float(p.area) > threshold:
                if idx == ():
                    out[lbl == p.label] = 2
                else:
                    # Build full boolean mask for this slice
                    mask2d = lbl == p.label
                    out[idx][mask2d] = 2

    return out


def voronois(
    stack: np.ndarray,
    connectivity: int = 1,
) -> tuple:
    """Create napari layers for Voronoi tessellation per YX plane.

    Args:
        stack (np.ndarray): Binary array. Last two axes are (Y, X). Supports 2D (H,W)
            and 3D (T,H,W); for 3D, tessellation is computed per time-plane and
            layers contain coordinates with leading time dimension.
        connectivity (int): Connectivity for labeling objects.

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
    else:
        planes = list(range(stack.shape[0]))

    def _process_plane(plane_idx):
        if plane_idx == ():
            binary = mask
        else:
            binary = mask[plane_idx]
        lbl = sk_label(binary, connectivity=connectivity)
        props = regionprops(lbl)
        if not props or len(props) < 2:
            return
        # Collect centroids (x,y) for Voronoi
        all_centroids_xy = np.array(
            [[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float64
        )
        # Build Voronoi diagram
        vor = Voronoi(all_centroids_xy)
        for pi, prop in enumerate(props):
            reg_index = vor.point_region[pi]
            vert_index = vor.regions[reg_index]
            if -1 in vert_index or len(vert_index) == 0:
                continue
            else:
                if stack.ndim == 2:
                    vertices = np.array(
                        [
                            [vor.vertices[v_index][1], vor.vertices[v_index][0]]
                            for v_index in vert_index
                        ],
                        dtype=np.float32,
                    )
                    point = np.array(
                        [prop.centroid[0], prop.centroid[1]], dtype=np.float32
                    )
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
                    and np.all(vertices[:, -2] <= stack.shape[-2])
                    and np.all(vertices[:, -1] >= 0.0)
                    and np.all(vertices[:, -1] <= stack.shape[-1])
                ):
                    # Calculate distances from centroid to each vertex
                    # Extract spatial coordinates (y, x) from centroid
                    if stack.ndim == 2:
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

                    polygons.append(vertices)
                    points_list.append(point)
                    feat_area.append(float(prop.area))
                    feat_nverts.append(int(len(vert_index)))
                    feat_plane.append(int(plane_idx if plane_idx != () else 0))
                    feat_y.append(float(prop.centroid[0]))
                    feat_x.append(float(prop.centroid[1]))
                    feat_vertex_dist_mean.append(dist_mean)
                    feat_vertex_dist_std.append(dist_std)

    for p in planes:
        _process_plane(p)

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

    return points_layer, polygons
