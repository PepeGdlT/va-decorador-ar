"""Project a planar decoration into captured RGB-D frames using reference poses."""
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .catalog import texture
from .placement import composite
from .tum import CAMERA, FR1_ROOM_URL, read_images, read_index
from .video import open_writer


@dataclass(frozen=True)
class Clip:
    name: str
    start: float
    duration: float
    quad: tuple
    surface: str
    asset: str


ROOM_CLIPS = (
    Clip("suelo-ventilador", 17, 1.75, ((206, 398), (382, 352), (465, 407), (272, 462)),
         "floor", "geometrica"),
    Clip("suelo-radiador", 27.5, 1.5, ((310, 323), (490, 351), (470, 430), (259, 394)),
         "floor", "clasica"),
)


def _plane(points):
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[-1]
    return normal, normal @ center


def _anchor(frame, depth, corners):
    region = cv.fillConvexPoly(np.zeros(depth.shape, np.uint8), np.int32(corners), 1)
    region = cv.erode(region, np.ones((7, 7), np.uint8))
    y, x = np.nonzero(region & (depth > 0))
    if len(x) < 100:
        raise ValueError("La referencia tiene muy poca profundidad medida.")
    y, x = y[::4], x[::4]
    rays = _rays(np.column_stack((x, y)))
    points = rays * (depth[y, x, None] / 5000)
    for _ in range(3):
        normal, distance = _plane(points)
        deviation = abs(points @ normal - distance)
        points = points[deviation <= max(.02, np.quantile(deviation, .75))]
    normal, distance = _plane(points)
    errors = points @ normal - distance
    corner_rays = _rays(corners)
    z = distance / (corner_rays @ normal)
    if np.any(~np.isfinite(z)) or np.any(z <= 0) or np.max(z) > 8:
        raise ValueError("La profundidad del plano de anclaje no es válida.")
    rotation, _ = cv.Rodrigues(frame.pose.rotation)
    world = (corner_rays * z[:, None] - frame.pose.translation.ravel()) @ rotation
    return world, float(np.sqrt(np.mean(errors**2)))


def _rays(pixels):
    pixels = np.asarray(pixels, dtype=float)
    return np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(CAMERA.matrix).T


def _render(frame, depth, pose, world, asset, surface):
    if pose is None:
        return frame, "sin_pose"
    points = pose.camera_points(world)
    if np.any(points[:, 2] <= .05):
        return frame, "fuera_de_vista"
    quad = pose.project(world, CAMERA).astype(np.float32)
    if (not np.isfinite(quad).all() or np.max(abs(quad)) > 10000
            or not cv.isContourConvex(quad) or cv.contourArea(quad, oriented=True) < 16):
        return frame, "fuera_de_vista"
    region = cv.fillConvexPoly(np.zeros(frame.shape[:2], np.uint8), np.rint(quad).astype(np.int32), 1)
    if not region.any():
        return frame, "fuera_de_vista"
    if depth is not None:
        y, x = np.nonzero(region)
        normal, distance = _plane(points)
        expected = distance / (_rays(np.column_stack((x, y))) @ normal)
        measured = depth[y, x] / 5000
        # Keep unmeasured pixels; hide the piece behind nearer measured surfaces.
        region[y, x] = (measured == 0) | (measured >= expected - .08)
    if not region.any():
        return frame, "oculto"
    output = frame.copy()
    footprint = composite(output, asset, quad, region.astype(bool), surface)
    return output, "visible" if footprint.any() else "fuera_de_vista"


def render_recordings(archive, folder, clips):
    """Save original and augmented clips with per-frame timestamps and status."""
    folder = Path(folder)
    frames = read_index(archive)
    first = frames[0].timestamp
    selections = []
    pending = set()
    for clip in clips:
        chosen = [f for f in frames if clip.start <= f.timestamp - first < clip.start + clip.duration][::2]
        if len(chosen) < 2 or chosen[0].pose is None or chosen[0].depth is None:
            raise ValueError(f"No hay una referencia RGB-D con pose al inicio de {clip.name}.")
        selections.append((clip, chosen))
        pending.update(path for frame in chosen for path in (frame.image, frame.depth) if path)
    images = read_images(archive, pending)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "fuente.txt").write_text(
        f"TUM RGB-D fr1/room — Sturm et al. (IROS 2012), CC BY 3.0\n{FR1_ROOM_URL}\n"
        "Cambios: recorte temporal, muestreo de un fotograma de cada dos y composición.\n",
        encoding="utf-8",
    )
    results = {}
    for clip, chosen in selections:
        reference = chosen[0]
        world, rms = _anchor(reference, images[reference.depth], clip.quad)
        item = texture(clip.surface, clip.asset)
        target = folder / clip.name
        target.mkdir()
        fps = (len(chosen) - 1) / (chosen[-1].timestamp - reference.timestamp)
        original = open_writer(target / "original.mp4", fps, CAMERA.size)
        augmented = None
        previews = {}
        last_visible = None
        counts = {}
        try:
            augmented = open_writer(target / "decorado.mp4", fps, CAMERA.size)
            with (target / "fotogramas.csv").open("x", newline="", encoding="utf-8") as stream:
                log = csv.writer(stream)
                log.writerow(("fotograma", "timestamp_rgb_s", "estado", "pose_disponible", "profundidad_disponible"))
                for index, frame in enumerate(chosen):
                    image = images[frame.image]
                    result, state = _render(image, images.get(frame.depth), frame.pose,
                                            world, item, clip.surface)
                    original.write(image)
                    augmented.write(result)
                    log.writerow((index, f"{frame.timestamp:.6f}", state, frame.pose is not None,
                                  frame.depth is not None))
                    counts[state] = counts.get(state, 0) + 1
                    preview = (image, result, state)
                    if state == "visible":
                        last_visible = index, preview
                    if index in (0, len(chosen)//2, len(chosen)-1):
                        previews[index] = preview
        finally:
            original.release()
            if augmented is not None:
                augmented.release()
        if previews[len(chosen)//2][2] != "visible" and last_visible is not None:
            index, preview = last_visible
            if 0 < index < len(chosen) - 1:
                del previews[len(chosen)//2]
                previews[index] = preview
        results[clip.name] = ([previews[index] for index in sorted(previews)], counts, rms)
    return results
