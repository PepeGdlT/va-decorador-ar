from pathlib import Path

import cv2 as cv
import numpy as np

from .geometry import Camera
from .image_io import read_image


def calibrate(folder, columns, rows, square_m):
    if min(columns, rows) < 3 or not 0 < square_m < np.inf:
        raise ValueError("Dimensiones del tablero no válidas.")
    pattern = (columns, rows)
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_m
    object_points, image_points = [], []
    size = None
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        gray = read_image(path, cv.IMREAD_GRAYSCALE)
        found, corners = cv.findChessboardCornersSB(gray, pattern)
        if found:
            current_size = gray.shape[::-1]
            if size is not None and current_size != size:
                raise ValueError("Las imágenes aceptadas deben tener la misma resolución.")
            size = current_size
            object_points.append(points)
            image_points.append(corners)
    if len(image_points) < 10:
        raise ValueError(f"Solo hay {len(image_points)} vistas válidas. Se necesitan al menos 10.")
    rms, matrix, distortion, _, _ = cv.calibrateCamera(
        object_points, image_points, size, None, None
    )
    return Camera(matrix, distortion, size), rms, len(image_points)
