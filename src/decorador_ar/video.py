import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2 as cv

from .geometry import Pose, estimate_pose, marker_points, reprojection_error
from .marker import MarkerDetector
from .render import draw_furniture


@dataclass(frozen=True)
class TrackingResult:
    state: str
    pose: Pose | None = None
    raw_error: float | None = None
    rendered_error: float | None = None
    anchor: tuple[float | None, float | None] = (None, None)


class PoseTracker:
    """Convert marker observations into tracked poses without reading or drawing video."""

    def __init__(self, camera, side, alpha, max_error):
        if not (0 < side < math.inf and 0 < alpha <= 1 and 0 < max_error < math.inf):
            raise ValueError("Lado del marcador, filtro o umbral no válidos.")
        self.camera = camera
        self.points = marker_points(side)
        self.origin = self.points.mean(axis=0)[None]
        self.alpha = alpha
        self.max_error = max_error
        self.previous = None

    def observe(self, pixels):
        if pixels is None:
            self.previous = None
            return TrackingResult("missing")

        pose, error = estimate_pose(self.points, pixels, self.camera)
        if pose is None or error is None or not math.isfinite(error) or error > self.max_error:
            self.previous = None
            return TrackingResult("rejected", raw_error=error)

        tracked = pose if self.previous is None else pose.blend(self.previous, self.alpha)
        self.previous = tracked
        return TrackingResult(
            "anchored", tracked, error,
            reprojection_error(self.points, pixels, tracked, self.camera),
            tuple(tracked.project(self.origin, self.camera)[0]),
        )


def open_writer(path, fps, size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"No se pudo abrir la salida MP4: {path}")
    return writer


def augment_video(source, output, metrics, camera, marker_id, side, pieces,
                  alpha, max_error):
    paths = [Path(path).resolve() for path in (source, output, metrics)]
    if len(set(paths)) != 3:
        raise ValueError("La entrada, el vídeo de salida y las métricas deben tener rutas distintas.")
    for path in paths[1:]:
        if path.exists():
            raise FileExistsError(f"La salida ya existe: {path}")
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    detector = MarkerDetector(marker_id)
    tracker = PoseTracker(camera, side, alpha, max_error)
    pieces = tuple(pieces)
    capture = cv.VideoCapture(str(source))
    writer = None
    count = accepted = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"No se pudo abrir {source}")
        fps = capture.get(cv.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("El vídeo no informa de una frecuencia válida.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("El vídeo no contiene fotogramas legibles.")
        if frame.shape[1::-1] != camera.size:
            raise ValueError(f"El vídeo debe tener la resolución calibrada {camera.size}.")
        with open(metrics, "x", encoding="utf-8", newline="") as stream:
            writer = open_writer(output, fps, camera.size)
            log = csv.writer(stream)
            log.writerow(["frame", "time_s", "state", "raw_error_px", "rendered_error_px",
                          "anchor_x_px", "anchor_y_px", "processing_ms"])
            while ok:
                if frame.shape[1::-1] != camera.size:
                    raise ValueError(f"Resolución inesperada en el fotograma {count}.")
                start = perf_counter()
                result = tracker.observe(detector.detect(frame))
                if result.pose is not None:
                    draw_furniture(frame, pieces, result.pose, camera)
                    accepted += 1
                cv.putText(frame, result.state.upper(), (18, 32), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2, cv.LINE_AA)
                log.writerow([count, count / fps, result.state, result.raw_error,
                              result.rendered_error, *result.anchor, (perf_counter() - start) * 1000])
                writer.write(frame)
                count += 1
                ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return count, accepted
