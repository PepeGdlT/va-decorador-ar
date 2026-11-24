import csv
import math
from pathlib import Path
from time import perf_counter

import cv2 as cv

from .geometry import estimate_pose, marker_points, reprojection_error
from .marker import MarkerDetector
from .render import draw_furniture


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
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(source))
    writer = None
    previous = None
    points = marker_points(side)
    detector = MarkerDetector(marker_id)
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
        writer = open_writer(output, fps, camera.size)
        with open(metrics, "w", encoding="utf-8", newline="") as stream:
            log = csv.writer(stream)
            log.writerow(["frame", "time_s", "state", "raw_error_px", "rendered_error_px",
                          "anchor_x_px", "anchor_y_px", "processing_ms"])
            while ok:
                start = perf_counter()
                pixels = detector.detect(frame)
                state = "missing"
                rendered_error = anchor_x = anchor_y = error = None
                pose = None
                if pixels is not None:
                    pose, error = estimate_pose(points, pixels, camera)
                    state = "rejected"
                if pose is not None and error <= max_error:
                    previous = pose if previous is None else pose.blend(previous, alpha)
                    rendered_error = reprojection_error(points, pixels, previous, camera)
                    anchor_x, anchor_y = previous.project(points.mean(axis=0)[None], camera)[0]
                    draw_furniture(frame, pieces, previous, camera)
                    accepted += 1
                    state = "anchored"
                else:
                    previous = None
                cv.putText(frame, state.upper(), (18, 32), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2, cv.LINE_AA)
                log.writerow([count, count / fps, state, error, rendered_error, anchor_x,
                              anchor_y, (perf_counter() - start) * 1000])
                writer.write(frame)
                count += 1
                ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return count, accepted
