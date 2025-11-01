from ftplib import all_errors
import logging
from pathlib import Path

import numpy as np
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_tracking,     
)

from skimage.measure import label as sk_label, regionprops
from scipy.spatial import cKDTree, Voronoi

from .utils import auto_value_range, load_mp4, check_device, to_mp4
from .preprocess import noise2stack_denoise, dog_gpu


logger = logging.getLogger(__name__)


def analyze(
    input: str | np.ndarray,
    sigma_low: float | tuple[float, ...] = 1.0,
    sigma_high: float | tuple[float, ...] = 5.0,
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
    mask = segment_spots(dog)
    stats = object_stats(mask, summary_only=True)
    global_stats = stats["combined"]
    print (global_stats)
    f_mask = filter_mask(mask, area=auto_value_range(global_stats["area_mean"], tolerance=0.5), circularity=auto_value_range(global_stats["circularity_mean"], tolerance=0.25))
    holes = identify_lattice_holes(f_mask)
    final_stats = object_stats(mask)
    points_layer, polygons = voronois(mask)   

    # Ensure destination directory exists
    #output.parent.mkdir(parents=True, exist_ok=True)
    #to_mp4(str(output), dog)
    #logger.info("Labels stacked and written to: %s", output)
    return denoised, dog, mask, f_mask, holes, final_stats, points_layer, polygons


def segment(
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
        raise ValueError("stack must be at least (T, H, W) or (T, C, H, W)/(T, H, W, C)")

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
        gray_stack = (255.0 * (f32 / mx)).astype(np.uint8) if mx > 0 else f32.astype(np.uint8)

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

    print (type(segmentation), type(lineage ))
    print (segmentation.shape, len(lineage))
    for i,item in enumerate(lineage):
        print(i, item)
    return segmentation.astype(np.int32, copy=False)



def segment_spots(
    stack: np.ndarray,
    block_size: int = 51,
    offset: float = 0.0,
    sigma: float = None,
    min_area: int = 20,
    opening_radius: int = 1,
    method: str = "local",
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

    Returns:
        np.ndarray | list[np.ndarray]: Binary mask stack (T, H, W) or list of masks (T, H, W) for each slice.
    """
    logger.info("Segmenting spots with method: %s", method)
    from scipy.ndimage import gaussian_filter
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
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.feature import peak_local_max
    import scipy.ndimage as ndi
    from skimage.segmentation import watershed
    
    if stack.ndim < 2:
        raise ValueError("stack must be at least 2D with shape (H, W)")
    
    # Normalize stack to float and prepare for processing
    img_f = stack.astype(np.float32, copy=False)
    if img_f.max() > 0:
        img_f = img_f / float(img_f.max())
    
    # Handle 2D case - wrap in list for uniform loop
    if stack.ndim == 2:
        img_f = [img_f]
    # Handle 3D case - loop over first axis
    masks = []
    for plane in img_f:
        im = plane
        if sigma and sigma > 0:
            from scipy.ndimage import gaussian_filter
            im = gaussian_filter(im, sigma=sigma)
        bw = _apply_threshold(im, method, block_size, offset)
        bw = _postprocess_mask(bw, opening_radius, min_area)
        masks.append(bw)
    
    # Return result in same shape as input
    if stack.ndim == 2:
        return masks[0]
    else:
        return np.stack(masks, axis=0).astype(bool, copy=False)


def iterative_threshold(
    img: np.ndarray,
    area: tuple[float, float],
    mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    circularity: tuple[float, float] | None = None,
    iterations: int = 10,
) -> list[np.ndarray]:
    """Generate binary masks by iteratively thresholding an image at different intensity levels.

    Iterates over a range of threshold values from maximum to minimum intensity,
    creating a binary mask at each step where pixels exceed the threshold.

    Args:
        img (np.ndarray): Input image to threshold.
        area (tuple[float, float]): Area range (currently unused, reserved for future filtering).
        mask (np.ndarray | None, optional): Optional mask to restrict intensity computation.
            If provided, maximum intensity is computed only within the masked region.
            Defaults to None.
        exclude_mask (np.ndarray | None, optional): Optional mask to exclude regions from
            intensity computation and final binary masks. Pixels where exclude_mask is True
            will be excluded. Defaults to None.
        iterations (int, optional): Number of threshold iterations. Defaults to 10.

    Returns:
        list[np.ndarray]: List of binary masks, one for each threshold value.
            Thresholds range from max (first mask) to min (last mask).
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
    
    #img_min = 0.0
    step_size = img_max / iterations
    
    # Generate threshold values from max to min (reverse the ascending array)
    thresholds = np.arange(img_max, img_min, -step_size)
    
    # Create binary mask for each threshold
    binary_masks = []
    for th in thresholds:
        binary_mask = (img > th).astype(bool)
        # Optionally apply mask to restrict output
        if mask is not None:
            binary_mask = binary_mask & mask.astype(bool)
        # Optionally exclude exclude_mask regions
        if exclude_mask is not None:
            binary_mask = binary_mask & ~exclude_mask.astype(bool)
        filtered_binary_mask = filter_mask(binary_mask, area=area, circularity=circularity)
        binary_masks.append(filtered_binary_mask)
    
    return binary_masks


def _apply_threshold(im: np.ndarray, method_name: str, block_size: int, offset: float) -> np.ndarray:
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
        area_max = float('inf')
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
        raise ValueError(f"circularity must be 'auto', None, or a tuple of (min, max), got: {circularity}")
    
    # Determine area thresholds and perform segmentation
    if spot_size == "auto":
        # Get initial segmentation to compute median area
        if method == "all":
            temp_mask = _segment_with_method_stack(img_f, methods_all[0], block_size, offset, sigma, opening_radius, min_area)
        else:
            res = _segment_with_method_stack(img_f, method, block_size, offset, sigma, opening_radius, min_area)
            temp_mask = res
        
        # Calculate area thresholds from median
        area_min, area_max = _calculate_area_thresholds(temp_mask)
        
        # Apply filtering
        if method == "all":
            results = []
            for m in methods_all:
                mask = _segment_with_method_stack(img_f, m, block_size, offset, sigma, opening_radius, min_area)
                # Determine circularity thresholds if needed
                circ_min_final = None
                circ_max_final = None
                if circularity == "auto":
                    circ_min_final, circ_max_final = _calculate_circularity_thresholds_from_props(mask, area_min, area_max)
                elif circularity is not None:
                    circ_min_final = circularity_min
                    circ_max_final = circularity_max
                filtered = _apply_filters(mask, area_min, area_max, circ_min_final, circ_max_final)
                results.append(filtered)
            return results
        else:
            # Determine circularity thresholds if needed
            circ_min_final = None
            circ_max_final = None
            if circularity == "auto":
                circ_min_final, circ_max_final = _calculate_circularity_thresholds_from_props(res, area_min, area_max)
            elif circularity is not None:
                circ_min_final = circularity_min
                circ_max_final = circularity_max
            filtered_res = _apply_filters(res, area_min, area_max, circ_min_final, circ_max_final)
            return filtered_res if stack.ndim >= 3 else filtered_res[0]
    elif isinstance(spot_size, tuple):
        area_min = float(spot_size[0])
        area_max = float(spot_size[1])
        
        # Apply segmentation and filtering
        if method == "all":
            results = []
            for m in methods_all:
                mask = _segment_with_method_stack(img_f, m, block_size, offset, sigma, opening_radius, min_area)
                # Determine circularity thresholds if needed
                circ_min_final = None
                circ_max_final = None
                if circularity == "auto":
                    circ_min_final, circ_max_final = _calculate_circularity_thresholds_from_props(mask, area_min, area_max)
                elif circularity is not None:
                    circ_min_final = circularity_min
                    circ_max_final = circularity_max
                filtered = _apply_filters(mask, area_min, area_max, circ_min_final, circ_max_final)
                results.append(filtered)
            return results
        
        res = _segment_with_method_stack(img_f, method, block_size, offset, sigma, opening_radius, min_area)
        # Determine circularity thresholds if needed
        circ_min_final = None
        circ_max_final = None
        if circularity == "auto":
            circ_min_final, circ_max_final = _calculate_circularity_thresholds_from_props(res, area_min, area_max)
        elif circularity is not None:
            circ_min_final = circularity_min
            circ_max_final = circularity_max
        filtered_res = _apply_filters(res, area_min, area_max, circ_min_final, circ_max_final)
        return filtered_res if stack.ndim >= 3 else filtered_res[0]
    else:
        raise ValueError(f"spot_size must be 'auto' or a tuple of (min, max), got: {spot_size}")


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
        filtered = _apply_filters_plane(bw, lbl, props, area_min, area_max, circularity_min, circularity_max)
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


def identify_lattice_holes(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Identify holes in a lattice of regularly spaced white spots.

    A hole is defined as an empty area (black region in the mask) that is larger
    than 1.5*π*r², where r is the median distance between spots in the image.

    Args:
        mask (np.ndarray): Binary mask where white (True) represents spots and
            black (False) represents background. Supports 2D (H, W) or 3D (T, H, W)
            arrays. For 3D arrays, processing is performed independently for each
            plane defined by the last two axes.
        connectivity (int): Connectivity for labeling spots. Defaults to 1 (4-connected).

    Returns:
        np.ndarray: Binary mask of the same shape as input, where True indicates
            holes in the lattice (empty regions larger than threshold).

    Raises:
        ValueError: If mask is not 2D or 3D.
    """
    logger.info("Identifying lattice holes")
    if mask.ndim == 2:
        # Process 2D mask directly
        return _identify_lattice_holes_2d(mask, connectivity)
    elif mask.ndim == 3:
        # Process each 2D plane independently
        results = []
        for t in range(mask.shape[0]):
            plane = mask[t]
            holes_plane = _identify_lattice_holes_2d(plane, connectivity)
            results.append(holes_plane)
        return np.stack(results, axis=0).astype(bool, copy=False)
    else:
        raise ValueError("mask must be 2D (H, W) or 3D (T, H, W)")


def _identify_lattice_holes_2d(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Internal helper function to identify holes in a 2D mask.
    
    Args:
        mask (np.ndarray): 2D binary mask with shape (H, W).
        connectivity (int): Connectivity for labeling spots.
        
    Returns:
        np.ndarray: 2D binary mask with holes marked as True.
    """
    from skimage.measure import label as sk_label, regionprops
    from scipy.spatial import cKDTree
    
    # Find all spots (white regions) in the mask
    binary = mask.astype(bool, copy=False)
    if not np.any(binary):
        # No spots found, return empty mask
        return np.zeros_like(mask, dtype=bool)
    
    lbl = sk_label(binary, connectivity=connectivity)
    props = regionprops(lbl)
    
    if len(props) < 2:
        # Need at least 2 spots to calculate distances
        return np.zeros_like(mask, dtype=bool)
    
    # Get centroids of all spots
    centroids = np.array([p.centroid for p in props], dtype=np.float64)
    
    # Calculate distances to nearest neighbors for each spot
    tree = cKDTree(centroids)
    # Query for k=2 (nearest neighbor is the point itself)
    distances, _ = tree.query(centroids, k=2)
    # Extract distances to actual nearest neighbor (second column)
    nn_distances = distances[:, 1]
    
    # Calculate median distance r
    r = float(np.median(nn_distances))
    
    # Calculate threshold area: 1.5 * π * r²
    threshold_area = 1.5 * np.pi * r * r
    
    # Invert mask to find empty regions
    inverted_mask = ~binary
    
    # Find connected components in inverted mask (holes)
    hole_labels = sk_label(inverted_mask, connectivity=connectivity)
    hole_props = regionprops(hole_labels)
    
    # Create output mask with holes larger than threshold
    holes_mask = np.zeros_like(mask, dtype=bool)
    for prop in hole_props:
        if prop.area > threshold_area:
            holes_mask[hole_labels == prop.label] = True
    
    return holes_mask


def filter_mask(
    mask: np.ndarray,
    area: tuple[float, float] | None = None,
    circularity: tuple[float, float] | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """Filter a binary mask based on area and/or circularity criteria.

    Args:
        mask (np.ndarray): Binary mask stack to filter. Can be 2D (H, W) or 3D (T, H, W).
        area (tuple[float, float] | None, optional): Tuple of (min_area, max_area) to filter by.
            Objects outside this range will be removed. Defaults to None (no filtering).
        circularity (tuple[float, float] | None, optional): Tuple of (min_circularity, max_circularity)
            to filter by. Objects outside this range will be removed. Defaults to None (no filtering).
        inplace (bool): If True, filter the mask in place. Defaults to False.
    Returns:
        np.ndarray: Filtered binary mask with same shape as input.
    """
    logger.info("Filtering mask with area: %s and circularity: %s", area, circularity)
    from skimage.measure import label as sk_label, regionprops
    
    if mask.ndim < 2:
        raise ValueError("mask must be at least 2D with last two axes as (Y, X)")
    
    # Extract area and circularity bounds
    area_min, area_max = area if area is not None else (None, None)
    circularity_min, circularity_max = circularity if circularity is not None else (None, None)
    
    # Prepare output
    if inplace:
        result = mask
    else:
        result = mask.copy()
    
    # Handle 2D case
    if mask.ndim == 2:
        result = [result]
    # Handle 3D case - loop over first axis
    for plane in result:
        plane = plane.astype(bool, copy=False)
        if np.any(plane):
            lbl = sk_label(plane, connectivity=1)
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
                        circ = 4.0 * np.pi * float(prop.area) / (float(prop.perimeter) ** 2)
                        circularity_ok = circularity_min <= circ <= circularity_max
                    else:
                        circularity_ok = False
                
                # Remove region if it doesn't pass filters
                if not (area_ok and circularity_ok):
                    plane[lbl == prop.label] = 0
    
    return result


def object_stats(
    stack: np.ndarray,
    connectivity: int = 1,
    summary_only: bool = False,
) -> dict:
    """Compute per-slice object statistics for a multi-dimensional binary stack.

    The last two axes are interpreted as (Y, X). All preceding axes form the
    indexing over slices. For each 2D slice, connected components are measured
    to compute area statistics, perimeter, circularity, and nearest-neighbor distances between
    object centroids.

    Args:
        stack (np.ndarray): Binary-like array. Non-zero pixels are treated as foreground.
        connectivity (int): Connectivity for labeling (1=4-connected, 2=8-connected).
        summary_only (bool): If True, compute regionprops for all planes but only return
            combined metrics. If False (default), return both per-slice and combined results.

    Returns:
        dict: A dictionary containing:
            - "combined": Combined statistics across all slices
            - "spatial_shape": Spatial dimensions (Y, X)
            - "stack_shape": Full stack shape
            - "slices": (only if summary_only=False) List of per-slice dicts.
                Each per-slice dict contains:
                  - index: tuple of indices for non-spatial axes
                  - count: number of objects
                  - area_min, area_max, area_mean, area_median, area_std
                  - perimeter_min, perimeter_max, perimeter_mean, perimeter_median, perimeter_std
                  - circularity_min, circularity_max, circularity_mean, circularity_median, circularity_std
                  - nn_min_dist_mean, nn_min_dist_median, nn_min_dist_std
    """
    logger.info("Computing object statistics")
    from skimage.measure import label as sk_label, regionprops
    from scipy.spatial import cKDTree
    
    if stack.ndim < 2:
        raise ValueError("stack must be at least 2D with last two axes as (Y, X)")

    spatial_shape = stack.shape[-2:]
    non_spatial_shape = stack.shape[:-2]

    # Ensure boolean mask
    mask = stack.astype(bool, copy=False)

    results: list[dict] | None = [] if not summary_only else None
    all_areas: list[float] = []
    all_perimeters: list[float] = []
    all_circularities: list[float] = []
    all_nn: list[float] = []
    if non_spatial_shape == ():
        indices_iter = [()]
    else:
        indices_iter = np.ndindex(non_spatial_shape)

    for idx in indices_iter:
        plane = mask[idx] if idx != () else mask

        lbl = sk_label(plane, connectivity=connectivity)
        props = regionprops(lbl)

        areas = np.array([p.area for p in props], dtype=np.float64)
        if areas.size > 0:
            area_min = float(np.min(areas))
            area_max = float(np.max(areas))
            area_mean = float(np.mean(areas))
            area_median = float(np.median(areas))
            area_std = float(np.std(areas, ddof=0))
            all_areas.extend(areas.tolist())
        else:
            area_min = area_max = area_mean = area_median = area_std = float("nan")

        perimeters = np.array([p.perimeter for p in props], dtype=np.float64)
        if perimeters.size > 0:
            perimeter_min = float(np.min(perimeters))
            perimeter_max = float(np.max(perimeters))
            perimeter_mean = float(np.mean(perimeters))
            perimeter_median = float(np.median(perimeters))
            perimeter_std = float(np.std(perimeters, ddof=0))
            all_perimeters.extend(perimeters.tolist())
        else:
            perimeter_min = perimeter_max = perimeter_mean = perimeter_median = perimeter_std = float("nan")

        # Circularity: 4π * area / perimeter² (perfect circle = 1.0)
        circularities = []
        for p in props:
            if p.perimeter > 0:
                circ = 4.0 * np.pi * p.area / (p.perimeter ** 2)
                circularities.append(circ)
        
        if len(circularities) > 0:
            circ_arr = np.array(circularities, dtype=np.float64)
            circularity_min = float(np.min(circ_arr))
            circularity_max = float(np.max(circ_arr))
            circularity_mean = float(np.mean(circ_arr))
            circularity_median = float(np.median(circ_arr))
            circularity_std = float(np.std(circ_arr, ddof=0))
            all_circularities.extend(circularities)
        else:
            circularity_min = circularity_max = circularity_mean = circularity_median = circularity_std = float("nan")

        # Nearest neighbor distances between centroids
        if len(props) >= 2:
            centroids = np.array([p.centroid for p in props], dtype=np.float64)
            tree = cKDTree(centroids)
            dists, _ = tree.query(centroids, k=2)
            # dists[:,0] is distance to self (0); dists[:,1] is nearest neighbor
            nn = dists[:, 1]
            nn_min_dist_mean = float(np.mean(nn))
            nn_min_dist_median = float(np.median(nn))
            nn_min_dist_std = float(np.std(nn, ddof=0))
            all_nn.extend(nn.tolist())
        else:
            nn_min_dist_mean = nn_min_dist_median = nn_min_dist_std = float("nan")

        # Only build per-slice results if summary_only is False
        if not summary_only:
            results.append(
                {
                    "index": idx,
                    "count": int(len(props)),
                    "area_min": area_min,
                    "area_max": area_max,
                    "area_mean": area_mean,
                    "area_median": area_median,
                    "area_std": area_std,
                    "perimeter_min": perimeter_min,
                    "perimeter_max": perimeter_max,
                    "perimeter_mean": perimeter_mean,
                    "perimeter_median": perimeter_median,
                    "perimeter_std": perimeter_std,
                    "circularity_min": circularity_min,
                    "circularity_max": circularity_max,
                    "circularity_mean": circularity_mean,
                    "circularity_median": circularity_median,
                    "circularity_std": circularity_std,
                    "nn_min_dist_mean": nn_min_dist_mean,
                    "nn_min_dist_median": nn_min_dist_median,
                    "nn_min_dist_std": nn_min_dist_std,
                }
            )

    # Combined statistics across all slices
    all_areas_arr = np.asarray(all_areas, dtype=np.float64)
    all_perimeters_arr = np.asarray(all_perimeters, dtype=np.float64)
    all_circularities_arr = np.asarray(all_circularities, dtype=np.float64)
    all_nn_arr = np.asarray(all_nn, dtype=np.float64)

    if all_areas_arr.size > 0:
        comb_area_min = float(np.min(all_areas_arr))
        comb_area_max = float(np.max(all_areas_arr))
        comb_area_mean = float(np.mean(all_areas_arr))
        comb_area_median = float(np.median(all_areas_arr))
        comb_area_std = float(np.std(all_areas_arr, ddof=0))
    else:
        comb_area_min = comb_area_max = comb_area_mean = comb_area_median = comb_area_std = float("nan")

    if all_perimeters_arr.size > 0:
        comb_perimeter_min = float(np.min(all_perimeters_arr))
        comb_perimeter_max = float(np.max(all_perimeters_arr))
        comb_perimeter_mean = float(np.mean(all_perimeters_arr))
        comb_perimeter_median = float(np.median(all_perimeters_arr))
        comb_perimeter_std = float(np.std(all_perimeters_arr, ddof=0))
    else:
        comb_perimeter_min = comb_perimeter_max = comb_perimeter_mean = comb_perimeter_median = comb_perimeter_std = float("nan")

    if all_circularities_arr.size > 0:
        comb_circularity_min = float(np.min(all_circularities_arr))
        comb_circularity_max = float(np.max(all_circularities_arr))
        comb_circularity_mean = float(np.mean(all_circularities_arr))
        comb_circularity_median = float(np.median(all_circularities_arr))
        comb_circularity_std = float(np.std(all_circularities_arr, ddof=0))
    else:
        comb_circularity_min = comb_circularity_max = comb_circularity_mean = comb_circularity_median = comb_circularity_std = float("nan")

    if all_nn_arr.size > 0:
        comb_nn_mean = float(np.mean(all_nn_arr))
        comb_nn_median = float(np.median(all_nn_arr))
        comb_nn_std = float(np.std(all_nn_arr, ddof=0))
    else:
        comb_nn_mean = comb_nn_median = comb_nn_std = float("nan")

    combined = {
        "count": int(np.sum([r["count"] for r in results]) if results else len(all_areas)),
        "area_min": comb_area_min,
        "area_max": comb_area_max,
        "area_mean": comb_area_mean,
        "area_median": comb_area_median,
        "area_std": comb_area_std,
        "perimeter_min": comb_perimeter_min,
        "perimeter_max": comb_perimeter_max,
        "perimeter_mean": comb_perimeter_mean,
        "perimeter_median": comb_perimeter_median,
        "perimeter_std": comb_perimeter_std,
        "circularity_min": comb_circularity_min,
        "circularity_max": comb_circularity_max,
        "circularity_mean": comb_circularity_mean,
        "circularity_median": comb_circularity_median,
        "circularity_std": comb_circularity_std,
        "nn_min_dist_mean": comb_nn_mean,
        "nn_min_dist_median": comb_nn_median,
        "nn_min_dist_std": comb_nn_std,
    }

    return_dict = {
        "combined": combined,
        "spatial_shape": spatial_shape,
        "stack_shape": stack.shape,
    }
    
    # Only include per-slice results if summary_only is False
    if not summary_only:
        return_dict["slices"] = results
    
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
                    mask2d = (lbl == p.label)
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
    from skimage.measure import label as sk_label, regionprops
    from scipy.spatial import Voronoi
    
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
        all_centroids_xy = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float64)
        # Build Voronoi diagram
        vor = Voronoi(all_centroids_xy)
        for pi, prop in enumerate(props):
            reg_index = vor.point_region[pi]
            vert_index = vor.regions[reg_index]
            if -1 in vert_index or len(vert_index) == 0:
                continue
            else:
                if stack.ndim == 2:
                    vertices=np.array([[vor.vertices[v_index][1], vor.vertices[v_index][0]] for v_index in vert_index], dtype=np.float32)
                    point=np.array([prop.centroid[0], prop.centroid[1]], dtype=np.float32)
                else:
                    vertices=np.array([[float(plane_idx), vor.vertices[v_index][1], vor.vertices[v_index][0]] for v_index in vert_index], dtype=np.float32)
                    point=np.array([float(plane_idx), prop.centroid[0], prop.centroid[1]], dtype=np.float32)
                # only add points if all cell vertices are within the stack boundaries
                if np.all(vertices[:, -2] >=0.0) and np.all(vertices[:, -2] <= stack.shape[-2]) and np.all(vertices[:, -1] >=0.0) and np.all(vertices[:, -1] <= stack.shape[-1]):
                    # Calculate distances from centroid to each vertex
                    # Extract spatial coordinates (y, x) from centroid
                    if stack.ndim == 2:
                        centroid_yx = np.array([prop.centroid[0], prop.centroid[1]], dtype=np.float64)
                        vertices_yx = vertices
                    else:
                        centroid_yx = np.array([prop.centroid[0], prop.centroid[1]], dtype=np.float64)
                        vertices_yx = vertices[:, -2:]  # Extract (y, x) from (N, 3) -> (N, 2)
                    
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
    color_cycle = 20*["red", "green", "blue", "yellow", "purple", "orange", "brown", "pink", "gray", "black"]
    color_cycle=color_cycle[:np.max(feat_nverts)]
    print (f"color_cycle: {color_cycle}")
    features = {
        "plane": feat_plane,
        "centroid_y": feat_y,
        "centroid_x": feat_x,
        "area": feat_area,
        "n_vertices": feat_nverts,
        "vertex_dist_mean": feat_vertex_dist_mean,
        "vertex_dist_std": feat_vertex_dist_std,
    }
    points_layer = Points(data=pts, name="Atoms", face_color="n_vertices", face_color_cycle=color_cycle, features=features)

    return points_layer, polygons
        
