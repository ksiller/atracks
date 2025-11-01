from __future__ import annotations

import os
from typing import Union
import logging

import cv2
import numpy as np
import torch
import tifffile
from skimage.measure import label as sk_label, regionprops
from scipy.spatial import Voronoi, cKDTree

logger = logging.getLogger(__name__)


def load_mp4(
    path: Union[str, os.PathLike],
    to_grayscale: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> np.ndarray:
    """Load an MP4 file into a NumPy array.

    Args:
        path (Union[str, os.PathLike]): Path to the MP4 video file.
        to_grayscale (bool, optional): If True, convert frames to single-channel
            grayscale before stacking. If False, keep color and convert BGR→RGB.
            Defaults to False.
        start_frame (int | None): Optional starting frame index (0-based). If None, starts at 0.
        end_frame (int | None): Optional inclusive ending frame index. If None, reads until EOF.

    Returns:
        np.ndarray: Video data stacked along time axis.
            - If `to_grayscale` is False: shape (T, H, W, C) in RGB order.
            - If `to_grayscale` is True: shape (T, H, W) with uint8 frames.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the video cannot be opened.
    """
    file_path = os.fspath(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    frames_rgb: list[np.ndarray] = []
    try:
        # Seek to start index when specified
        current_index = 0
        # Normalize and clamp indices
        s_idx = 0 if start_frame is None else max(0, int(start_frame))
        e_idx = None if end_frame is None else int(end_frame)
        if e_idx is not None and e_idx < s_idx:
            raise ValueError("end_frame must be >= start_frame")
        if s_idx > 0:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if e_idx is not None and total > 0 and e_idx >= total:
                e_idx = total - 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(s_idx))
            current_index = s_idx

        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if e_idx is not None and current_index > e_idx:
                break
            if to_grayscale:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frames_rgb.append(frame_gray)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            current_index += 1
    finally:
        capture.release()

    if not frames_rgb:
        logger.info("No frames read from: %s", file_path)
        return np.empty((0,), dtype=np.uint8)

    video_array = np.stack(frames_rgb, axis=0)
    logger.info(
        "Loaded MP4: %s -> shape=%s dtype=%s grayscale=%s",
        file_path,
        video_array.shape,
        video_array.dtype,
        to_grayscale,
    )
    return video_array


def check_device() -> torch.device:
    """Determine the best available PyTorch device (cuda, mps, or cpu).

    Returns:
        torch.device: The selected device in priority order cuda > mps > cpu.
    """
    # Prefer CUDA when available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        logger.info("Using CUDA device: %s", name)
        return device

    # Fallback to Apple Metal Performance Shaders on macOS
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Using MPS device")
        return torch.device("mps")

    # Default to CPU
    logger.info("Using CPU device")
    return torch.device("cpu")


def to_mp4(path: Union[str, os.PathLike], stack: np.ndarray, fps: int = 15) -> None:
    """Write an n-dimensional numpy stack to an MP4 file.

    Args:
        path (Union[str, os.PathLike]): Output file path (should end with .mp4).
        stack (np.ndarray): Video stack with time as first axis (T, ..., H, W).
            Supported frame layouts:
              - (T, H, W): single-channel; written as grayscale.
              - (T, H, W, C): RGB(A) with C in {3, 4}; written as color.
        fps (int, optional): Frames per second. Defaults to 15.

    Returns:
        None

    Raises:
        ValueError: If stack dimensionality or shape is not supported.
        RuntimeError: If the video writer cannot be opened.
    """
    file_path = os.fspath(path)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    if stack.ndim < 3:
        raise ValueError("stack must be at least 3D: (T, H, W) or (T, H, W, C)")

    t_dim = stack.shape[0]
    if t_dim == 0:
        logger.info("Empty stack; nothing to write: %s", file_path)
        return

    # Determine frame size and color mode
    color = False
    if stack.ndim == 3:
        height, width = int(stack.shape[-2]), int(stack.shape[-1])
    elif stack.ndim == 4 and stack.shape[-1] in (3, 4):
        height, width = int(stack.shape[-3]), int(stack.shape[-2])
        color = True
    else:
        raise ValueError("Unsupported stack shape; expected (T,H,W) or (T,H,W,C [3|4])")

    # FourCC for mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height), color)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter for: {file_path}")

    try:
        for t in range(t_dim):
            frame = stack[t]
            # Normalize dtype to uint8
            if frame.dtype != np.uint8:
                f32 = frame.astype(np.float32)
                maxv = float(f32.max()) if f32.size > 0 else 0.0
                if maxv > 0:
                    frame_u8 = np.clip(255.0 * (f32 / maxv), 0, 255).astype(np.uint8)
                else:
                    frame_u8 = f32.astype(np.uint8)
            else:
                frame_u8 = frame

            if color:
                # frame_u8 expected as RGB(A); convert to BGR for OpenCV
                if frame_u8.shape[-1] == 4:
                    frame_u8 = frame_u8[..., :3]
                rgb = frame_u8
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            else:
                # Grayscale expected shape (H, W)
                if frame_u8.ndim == 3 and frame_u8.shape[-1] == 1:
                    frame_u8 = frame_u8[..., 0]
                if frame_u8.ndim != 2:
                    raise ValueError("Grayscale frames must be 2D (H, W) per time step")
                writer.write(frame_u8)
    finally:
        writer.release()

    logger.info("Wrote MP4: %s (frames=%d, fps=%d, size=%dx%d, color=%s)", file_path, t_dim, fps, width, height, color)



def to_tif(path: Union[str, os.PathLike], stack: np.ndarray) -> None:
    """Write an n-dimensional numpy array as a multi-page TIFF.

    Args:
        path (Union[str, os.PathLike]): Output file path (should end with .tif/.tiff).
        stack (np.ndarray): N-dimensional array. The first axis is interpreted as
            the page/time dimension when writing multi-page TIFF. Higher
            dimensions are supported by TIFF and will be stored accordingly.

    Returns:
        None

    Raises:
        RuntimeError: If the optional dependency `tifffile` is not available.
    """
    file_path = os.fspath(path)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    # Write as-is; tifffile supports arbitrary numpy dtypes and shapes.
    # Use a common lossless compression to reduce size.
    tifffile.imwrite(file_path, stack, compression="deflate")
    logger.info("Wrote TIFF: %s shape=%s dtype=%s", file_path, tuple(stack.shape), stack.dtype)


def auto_value_range(
    target_value: float,
    tolerance: float = 0.25,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, float]:
    """Compute a value range around a target value with optional clipping.

    Calculates a symmetric range around the target value using the tolerance factor,
    then optionally clips the result to specified minimum and maximum bounds.

    Args:
        target_value (float): The central target value around which to compute the range.
        tolerance (float, optional): Fractional tolerance applied symmetrically.
            Defaults to 0.25 (25%).
        min_value (float | None, optional): Optional minimum bound. If set, the lower
            bound of the returned range will be clipped to this value. Defaults to None.
        max_value (float | None, optional): Optional maximum bound. If set, the upper
            bound of the returned range will be clipped to this value. Defaults to None.

    Returns:
        tuple[float, float]: A tuple (lower_bound, upper_bound) where:
            - lower_bound = (1 - tolerance) * target_value, clipped to min_value if provided
            - upper_bound = (1 + tolerance) * target_value, clipped to max_value if provided

    Examples:
        >>> auto_value_range(100.0, tolerance=0.25)
        (75.0, 125.0)
        >>> auto_value_range(100.0, tolerance=0.25, min_value=80.0, max_value=110.0)
        (80.0, 110.0)
    """
    lower = (1.0 - tolerance) * target_value
    upper = (1.0 + tolerance) * target_value

    if min_value is not None:
        lower = max(lower, min_value)
    if max_value is not None:
        upper = min(upper, max_value)

    return (lower, upper)

