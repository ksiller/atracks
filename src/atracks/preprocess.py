from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.filters import gaussian
from joblib import Parallel, delayed


def preprocess(
    input: str | np.ndarray,
    sigma_low: float | tuple[float, ...] = 1.0,
    sigma_high: float | tuple[float, ...] = 5.0,
    workers: int = -1,
    verbose: int = 10,
) -> None:
    """Preprocess the input image stack by denoising and applying a Difference of Gaussians filter.

    Args:
        input (str | np.ndarray): Input image stack.
        sigma_low: Sigma for the first (lower) Gaussian blur.
        sigma_high: Sigma for the second (higher) Gaussian blur.
        workers: Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose: Verbosity level for joblib.Parallel. Defaults to 10 (medium output).
    """
    denoised = noise2stack_denoise(input)
    dog = dog_3d(
        input,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        workers=workers,
        verbose=verbose,
    )
    return denoised, dog


def noise2stack_denoise(
    stack: np.ndarray, window: int = 5, exclude_center: bool = True
) -> np.ndarray:
    """Denoise an image stack by averaging temporal neighbors (Noise2Stack-inspired).

    This implements a simple, non-learning variant inspired by the Noise2Stack idea:
    predict each frame from its neighboring frames in time. Here we compute a
    temporal moving average over the first axis (time), optionally excluding the
    center frame from the average.

    Args:
        stack (np.ndarray): Input stack with time as first axis. Shapes supported:
            (T, H, W) or (T, H, W, C).
        window (int): Temporal window size (odd recommended). Defaults to 5.
        exclude_center (bool): If True, remove the current frame from the
            average (classic Noise2Stack idea). Defaults to True.

    Returns:
        np.ndarray: Denoised stack with the same shape and dtype as the input.

    Notes:
        - This is a deterministic, filter-based approximation inspired by
          Noise2Stack; it does not train a neural network.
        - For integer inputs, values are clipped to the valid dtype range.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if exclude_center and window < 2:
        raise ValueError("window must be >= 2 when exclude_center=True")

    input_dtype = stack.dtype
    work = stack.astype(np.float32, copy=False)

    # Apply temporal moving average along T axis
    avg = uniform_filter1d(work, size=window, axis=0, mode="nearest")

    if exclude_center:
        # With mode='nearest', uniform_filter1d uses an effective window of exactly `window`.
        # Exclude the center by subtracting the original frame and renormalize.
        denoised = (avg * float(window) - work) / float(window - 1)
    else:
        denoised = avg

    # Cast back to input dtype with clipping for integer types
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        denoised = np.clip(denoised, info.min, info.max).astype(input_dtype, copy=False)
    else:
        denoised = denoised.astype(input_dtype, copy=False)

    return denoised


def dog_plane(
    plane: np.ndarray,
    sigma_low: float | tuple[float, ...],
    sigma_high: float | tuple[float, ...],
    mode: str = "reflect",
    normalize: bool = True,
) -> np.ndarray:
    """Compute Difference of Gaussians (DoG) for a single plane.

    Args:
        plane: Input plane.
        sigma_low: Sigma for the first (lower) Gaussian blur.
        sigma_high: Sigma for the second (higher) Gaussian blur.
        mode: Border handling mode passed to skimage.filters.gaussian (default: "reflect").
        normalize: If True, normalize the output intensity will be rescaled to min:max range of the particular dtype. Defaults to True.
    Returns:
        np.ndarray: Difference of Gaussians (DoG) for the single plane.
    """
    g_low = gaussian(plane, sigma=sigma_low, mode=mode, preserve_range=True)
    g_high = gaussian(plane, sigma=sigma_high, mode=mode, preserve_range=True)
    return g_low - g_high


def dog_3d(
    array: np.ndarray,
    sigma_low: float | tuple[float, ...],
    sigma_high: float | tuple[float, ...],
    mode: str = "reflect",
    normalize: bool = True,
    workers: int = -1,
    verbose: int = 0,
) -> np.ndarray:
    """Compute Difference of Gaussians (DoG) using NumPy + scikit-image.

    Args:
        array (np.ndarray): Input n-dimensional array. Filtering is applied across all axes.
        sigma_low: Sigma for the first (lower) Gaussian blur.
        sigma_high: Sigma for the second (higher) Gaussian blur.
        mode: Border handling mode passed to skimage.filters.gaussian (default: "reflect").
        normalize: If True, normalize the output intensity will be rescaled to min:max range of the particular dtype. Defaults to True.
        workers: Number of workers for parallel processing. Defaults to -1 (all available cores).
        verbose: Verbosity level for joblib.Parallel. Defaults to 0 (no output).
    Returns:
        np.ndarray: g(sigma_low) - g(sigma_high) with same shape and dtype as input.
            If normalize=True, output will be normalized to min:max range of the particular dtype.
    """
    input_dtype = array.dtype
    # If 3D, treat as sequence of 2D planes along axis 0 (last two axes are Y, X)
    if array.ndim == 3:
        dog = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(dog_plane)(array[i], sigma_low, sigma_high, mode, normalize)
            for i in range(array.shape[0])
        )
        dog = np.stack(dog, axis=0)
    else:
        # Generic nD filtering across all axes
        g_low = gaussian(array, sigma=sigma_low, mode=mode, preserve_range=True)
        g_high = gaussian(array, sigma=sigma_high, mode=mode, preserve_range=True)
        dog = g_low - g_high

    # Normalize to [0, 1] if requested
    if normalize:
        dog_min = np.min(dog)
        dog_max = np.max(dog)
        if dog_max > dog_min:
            dog = (dog - dog_min) / (dog_max - dog_min)
        else:
            # All values are the same; set to 0
            dog = np.zeros_like(dog, dtype=np.float32)
        return dog.astype(np.float32, copy=False)

    # Cast back to original dtype
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        dog = np.clip(dog, info.min, info.max).astype(input_dtype, copy=False)
    else:
        dog = dog.astype(input_dtype, copy=False)
    return dog
