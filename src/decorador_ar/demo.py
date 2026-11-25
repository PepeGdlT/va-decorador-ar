from pathlib import Path

import cv2 as cv
import numpy as np

from .geometry import Camera, Pose, marker_points
from .marker import marker_pattern
from .render import furniture
from .video import augment_video, open_writer


def make_demo(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    size = (960, 540)
    camera = Camera(np.array([[790., 0, 480], [0, 790, 250], [0, 0, 1]]),
                    np.zeros(5), size)
    camera.save(folder / "camera.npz")
    side = 0.24
    pattern = marker_pattern(0, 320)
    edge = pattern.shape[0] - 1
    source = np.float32([[0, 0], [edge, 0], [edge, edge], [0, edge]])
    pattern_mask = np.full_like(pattern, 255)
    points = marker_points(side)
    target = folder / "entrada_sintetica.mp4"
    writer = open_writer(target, 24, size)
    rng = np.random.default_rng(42)
    floor = np.full((size[1], size[0], 3), (161, 178, 192), np.uint8)
    noise = rng.integers(-4, 5, floor.shape[:2], dtype=np.int16)
    floor = np.clip(floor.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)
    for y in range(0, size[1], 46):
        cv.line(floor, (0, y), (size[0], y), (132, 148, 160))
    try:
        for index in range(120):
            time = index / 24
            motion = np.sin(max(0, time - 1) * 1.6)
            pose = Pose(np.array([[2.35], [0.24 * motion], [0.10 * motion]]),
                        np.array([[0.18 * motion], [0.13], [1.15]]))
            corners = pose.project(points, camera).astype(np.float32)
            transform = cv.getPerspectiveTransform(source, corners)
            frame = floor.copy()
            if not 80 <= index < 90:
                warp = cv.warpPerspective(pattern, transform, size, flags=cv.INTER_NEAREST)
                mask = cv.warpPerspective(pattern_mask, transform, size, flags=cv.INTER_NEAREST) > 0
                frame[mask] = warp[mask, None]
            cv.putText(frame, "SECUENCIA SINTETICA", (18, 510),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 2, cv.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return target, camera, side


def run_demo(folder):
    """Process the same synthetic clip with and without pose smoothing."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Usa una carpeta nueva o vacía para la demo.")
    source, camera, side = make_demo(folder)
    pieces = tuple(furniture(0.32, 0.24, 0.24, -0.34, 0.01, 10))
    return {
        name: augment_video(source, folder / f"{name}.mp4", folder / f"{name}.csv",
                            camera, 0, side, pieces, alpha, 3)
        for name, alpha in (("sin_filtro", 1.0), ("suavizado", 0.55))
    }
