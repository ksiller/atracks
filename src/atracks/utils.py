from __future__ import annotations

import os
from typing import Union
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_mp4(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Load an MP4 video into a NumPy array with shape (T, H, W, C) in RGB order.
    """
    file_path = os.fspath(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    frames_rgb: list[np.ndarray] = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
    finally:
        capture.release()

    if not frames_rgb:
        logger.info("No frames read from: %s", file_path)
        return np.empty((0,), dtype=np.uint8)

    video_array = np.stack(frames_rgb, axis=0)
    logger.info(
        "Loaded MP4: %s -> shape=%s dtype=%s", file_path, video_array.shape, video_array.dtype
    )
    return video_array


