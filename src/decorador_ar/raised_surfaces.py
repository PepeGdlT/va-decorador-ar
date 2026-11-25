"""Suggest visible horizontal supports using depth and the calibrated ground plane."""
import cv2 as cv
import numpy as np

from .depth import estimate_depth
from .photo_furniture import Support, ground_pose
from .supported_objects import support_plane, support_points


def _geometry(scene, depth):
    wall, floor = scene.reference_planes
    ground = ground_pose(floor, wall, scene.intrinsics, scene.floor_anchor)
    ground_normal = cv.Rodrigues(ground.rotation)[0][:, 2]
    camera_height = -ground.translation.ravel() @ ground_normal
    if camera_height <= 0:
        raise ValueError("La referencia del suelo no deja la cámara por encima de él.")

    y, x = np.indices(depth.shape, dtype=np.float32)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(scene.intrinsics).T
    downward = -(rays @ ground_normal)
    valid = np.isfinite(depth) & (depth > .05)
    floor_mask = cv.erode(scene.masks["floor"][0].astype(np.uint8), np.ones((9, 9), np.uint8))
    samples = (floor_mask != 0) & valid & (downward > .18)
    if np.count_nonzero(samples) < 400:
        raise ValueError("Se ve demasiado poco suelo para estimar alturas en esta foto.")
    factors = camera_height / (depth * downward)[samples]
    scale = np.median(factors)
    if np.quantile(abs(factors / scale - 1), .75) > .28:
        raise ValueError("La profundidad estimada no encaja suficientemente bien con el suelo.")
    points = rays * np.where(valid, depth * scale / camera_height, 0)[..., None]
    height = 1 + points @ ground_normal
    return points, height, valid


def _horizontal(points, height):
    stride = max(2, round(height.shape[1] / 192))
    dx = points[:, 2 * stride:] - points[:, :-2 * stride]
    dy = points[2 * stride:, :] - points[:-2 * stride, :]
    dz_x = height[:, 2 * stride:] - height[:, :-2 * stride]
    dz_y = height[2 * stride:, :] - height[:-2 * stride, :]
    dx_len = np.linalg.norm(dx, axis=-1)
    dy_len = np.linalg.norm(dy, axis=-1)
    x_flat = (abs(dz_x) <= .004 + .25 * dx_len) & (dx_len < .20) & (dx_len > .002)
    y_flat = (abs(dz_y) <= .004 + .25 * dy_len) & (dy_len < .20) & (dy_len > .002)
    return np.pad(x_flat[stride:-stride] & y_flat[:, stride:-stride], stride)


def _largest_rectangle(mask, min_side):
    """Return the largest fully occupied grid rectangle as exclusive bounds."""
    heights = np.zeros(mask.shape[1] + 1, dtype=int)
    best = None
    best_area = 0
    for y, row in enumerate(mask):
        heights[:-1] = (heights[:-1] + 1) * row
        stack = []
        for x, height in enumerate(heights):
            start = x
            while stack and stack[-1][1] > height:
                start, top = stack.pop()
                width = x - start
                if width >= min_side and top >= min_side and width * top > best_area:
                    best_area = width * top
                    best = start, y + 1 - top, x, y + 1
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best


def _fit_support(scene, region, height, name):
    pose, camera = support_plane(scene, height)
    y, x = np.nonzero(region)
    pixels = np.column_stack((x, y))
    plane = support_points(pose, camera, pixels)
    lower, upper = plane.min(axis=0), plane.max(axis=0)
    density = 150
    grid_size = np.ceil((upper - lower) * density).astype(int) + 1
    if np.any(grid_size < 12) or np.any(grid_size > 500):
        return None
    rotation = cv.Rodrigues(pose.rotation)[0]
    projection = camera.matrix @ np.column_stack((rotation[:, :2], pose.translation))
    grid_transform = np.array([[density, 0, -density * lower[0]],
                               [0, density, -density * lower[1]], [0, 0, 1]])
    grid = cv.warpPerspective(region, grid_transform @ np.linalg.inv(projection), tuple(grid_size),
                              flags=cv.INTER_NEAREST)
    grid = cv.erode(grid, np.ones((5, 5), np.uint8), borderType=cv.BORDER_CONSTANT, borderValue=0)
    box = _largest_rectangle(grid, min_side=12)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if (x1 - x0) * (y1 - y0) < .012 * density**2:
        return None
    corners = np.float32([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])
    return Support(name, pose, camera, (corners / density + lower).astype(np.float32))


def detect_supports(scene, depth=None):
    furniture = scene.masks.get("support")
    if furniture is None or not furniture.any():
        return ()
    depth = estimate_depth(scene.image) if depth is None else np.asarray(depth)
    if depth.shape != scene.image.shape[:2]:
        raise ValueError("La profundidad debe estar alineada con la foto.")
    points, height, valid = _geometry(scene, depth)
    eligible = valid & furniture & (height >= .18) & (height <= .94)
    mask = (_horizontal(points, height) & eligible).astype(np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask &= eligible
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    minimum = max(450, int(mask.size * .0012))
    supports = []
    for index in np.argsort(-stats[1:, cv.CC_STAT_AREA]) + 1:
        if stats[index, cv.CC_STAT_AREA] < minimum or len(supports) == 5:
            break
        region = (labels == index).astype(np.uint8)
        level = float(np.median(height[region != 0]))
        region[abs(height - level) > .045] = 0
        if np.count_nonzero(region) < minimum:
            continue
        try:
            support = _fit_support(scene, region, level, f"Apoyo detectado {len(supports) + 1}")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if support is not None:
            supports.append(support)
    return tuple(supports)
