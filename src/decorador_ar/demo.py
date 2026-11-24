from pathlib import Path

import cv2 as cv
import numpy as np

from .geometry import Camera, Pose, marker_points
from .marker import dictionary
from .video import open_writer


def make_demo(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    size = (960, 540)
    camera = Camera(np.array([[790., 0, 480], [0, 790, 250], [0, 0, 1]]),
                    np.zeros(5), size)
    camera.save(folder / "camera.npz")
    side = 0.24
    pattern = cv.aruco.generateImageMarker(dictionary(), 0, 320)
    source = np.float32([[0, 0], [319, 0], [319, 319], [0, 319]])
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
            corners = pose.project(marker_points(side), camera).astype(np.float32)
            transform = cv.getPerspectiveTransform(source, corners)
            frame = floor.copy()
            if not 80 <= index < 90:
                warp = cv.warpPerspective(pattern, transform, size, flags=cv.INTER_NEAREST)
                mask = cv.warpPerspective(np.full_like(pattern, 255), transform, size,
                                          flags=cv.INTER_NEAREST) > 0
                frame[mask] = warp[mask, None]
            cv.putText(frame, "SECUENCIA SINTETICA", (18, 510),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 2, cv.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return target, camera, side
