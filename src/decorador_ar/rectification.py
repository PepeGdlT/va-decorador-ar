"""Rectify a referenced surface before projecting decoration into the photo."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .placement import Placement, composite, default_asset, polygon_coverage
from .image_io import read_image


def project(matrix, points):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    if np.any(abs(q[:, 2]) < 1e-9):
        raise ValueError("La transformación cruza el horizonte del plano.")
    return q[:, :2] / q[:, 2:3]


@dataclass(frozen=True)
class Plane:
    to_image: np.ndarray
    width: float
    height: float

    @classmethod
    def from_reference(cls, corners, width, height):
        """Calibrate from a rectangle and its dimensions, as in the practice."""
        points = np.asarray(corners, np.float32)
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or not np.isfinite([width, height]).all() or min(width, height) <= 0
                or not cv.isContourConvex(points) or cv.contourArea(points, oriented=True) <= 4):
            raise ValueError("Marca cuatro esquinas en orden SI, SD, ID, II y da su ancho y alto.")
        reference = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        return cls(cv.getPerspectiveTransform(reference, points, solveMethod=cv.DECOMP_SVD),
                   width, height)

    def coordinates(self, pixels):
        return project(np.linalg.inv(self.to_image), pixels)

    def rectify(self, image, bounds=None, max_side=480):
        """Sample the photograph with a single scale for both planar axes."""
        x0, y0, x1, y1 = (0, 0, self.width, self.height) if bounds is None else bounds
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0 or max_side < 2:
            raise ValueError("El recorte rectificado necesita límites crecientes y un tamaño válido.")
        scale = max_side / max(x1 - x0, y1 - y0)
        output_to_plane = np.array([[1 / scale, 0, x0], [0, 1 / scale, y0], [0, 0, 1]])
        size = (max(2, round((x1 - x0) * scale)), max(2, round((y1 - y0) * scale)))
        return cv.warpPerspective(image, self.to_image @ output_to_plane, size,
                                  flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)


def room_planes(ground, height, camera, camera_right):
    """Build wall homographies and floor axes from camera-relative room edges."""
    down = np.array([0, -1, 0])
    walls = []
    for a, b in zip(ground, np.roll(ground, -1, axis=0)):
        length = np.linalg.norm(b - a)
        if length < .05:
            continue
        if (b - a) @ camera_right < 0:
            a, b = b, a
        right = (b - a) / length
        inward = np.cross(right, down)
        if inward @ (-a) < 0:
            inward = -inward
        wall = Plane(camera @ np.column_stack((right, down, a - height * down)), length, height)
        floor = Plane(camera @ np.column_stack((right, inward, a)), length, length)
        walls.append((wall, floor))
    return walls


def select_wall(pairs, anchor):
    visible = []
    for wall, floor in pairs:
        try:
            x, y = wall.coordinates([anchor])[0]
        except ValueError:
            continue
        depth = wall.to_image[2] @ [x, y, 1]
        if depth > 0 and 0 <= x <= wall.width and 0 <= y <= wall.height:
            visible.append((depth, wall, floor))
    if not visible:
        raise ValueError("No hay una pared anotada en ese punto.")
    _, wall, floor = min(visible, key=lambda pair: pair[0])
    return wall, floor


def place_rectified(image, mask, plane, surface, anchor, length, *,
                    angle=0, vertical=False, min_coverage=.97, texture=None):
    """Keep the anchor fixed and measure/rotate in plane coordinates."""
    if surface not in ("wall", "floor"):
        raise ValueError("La superficie debe ser wall o floor.")
    if not np.isfinite([*anchor, length, angle]).all() or length <= 0:
        raise ValueError("Indica una esquina válida, un tamaño positivo y un ángulo finito.")
    if texture is None:
        texture = default_asset(surface, read_image)
    if vertical and surface == "floor":
        texture = cv.rotate(texture, cv.ROTATE_90_CLOCKWISE)
    h, w = texture.shape[:2]
    width, height = length * np.array([w, h]) / max(w, h)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = plane.coordinates([anchor])[0]
    flat = xy + np.array([[-width, 0], [0, 0], [0, height], [-width, height]]) @ rotation.T
    depth = np.column_stack((flat, np.ones(4))) @ plane.to_image[2]
    if np.any(abs(depth) < 1e-9) or np.any(np.sign(depth) != np.sign(depth[1])):
        raise ValueError("La pieza atraviesa el horizonte del plano.")
    quad = project(plane.to_image, flat).astype(np.float32)
    if (np.any(quad < 0) or np.any(quad >= np.array(image.shape[1::-1]) - 1)
            or not cv.isContourConvex(quad) or cv.contourArea(quad, oriented=True) < 16):
        raise ValueError("La pieza queda fuera del encuadre o es demasiado pequeña.")
    if surface == "wall" and (np.any(flat < 0)
                              or np.any(flat > [plane.width, plane.height])):
        raise ValueError("La pieza cruza los límites de la pared elegida.")
    coverage = polygon_coverage(quad, mask)
    if coverage < min_coverage:
        raise ValueError(f"Encaje de {coverage:.0%}. Cambia la esquina o reduce el tamaño.")
    output = image.copy()
    composite(output, texture, quad, mask, surface)
    return Placement(output, quad, coverage, surface), flat
