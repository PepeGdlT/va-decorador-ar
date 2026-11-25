"""Conservative placement and photographic compositing."""
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


ASSETS = Path(__file__).resolve().parents[2] / "assets"


@dataclass
class Placement:
    image: np.ndarray
    quad: np.ndarray
    coverage: float
    surface: str


def polygon_coverage(polygon, region):
    footprint = cv.fillConvexPoly(
        np.zeros(region.shape, np.uint8), np.rint(polygon).astype(np.int32), 1
    ).astype(bool)
    return float(np.mean(region[footprint])) if footprint.any() else 0.0


def default_asset(surface, read_image):
    if surface == "wall":
        art = read_image(ASSETS / "landscape-met.jpg")[18:-18, 16:-17]
        art = cv.resize(art, (480, 370), interpolation=cv.INTER_AREA)
        art = cv.copyMakeBorder(art, 23, 23, 23, 23, cv.BORDER_CONSTANT,
                                value=(213, 222, 225))
        return cv.copyMakeBorder(art, 12, 12, 12, 12, cv.BORDER_CONSTANT,
                                 value=(40, 48, 61))
    rug = read_image(ASSETS / "carpet-met.jpg")
    return cv.rotate(rug[29:-23, 23:-19], cv.ROTATE_90_CLOCKWISE)


def floor_direction(gray, mask):
    """Estimate a horizontal vanishing coordinate from converging floor lines."""
    h, w = gray.shape
    lines = cv.HoughLinesP(cv.Canny(gray, 45, 120), 1, np.pi / 180, 23,
                           minLineLength=max(20, int(.11 * min(w, h))), maxLineGap=9)
    if lines is None:
        return w / 2, .40 * h
    equations = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = int(x2 - x1), int(y2 - y1)
        if abs(dy) < 9 or abs(dx) < .15 * abs(dy):
            continue
        xs = np.rint(np.linspace(x1, x2, 12)).astype(int)
        ys = np.rint(np.linspace(y1, y2, 12)).astype(int)
        if np.mean(mask[ys, xs]) < .85:
            continue
        a = np.array([dy, -dx], dtype=float) / np.hypot(dx, dy)
        equations.append((*a, float(a @ [x1, y1])))
        if len(equations) == 60:
            break
    if len(equations) < 4:
        return w / 2, .40 * h
    lines = np.asarray(equations)
    best = None
    for i, first in enumerate(lines):
        for second in lines[i + 1:]:
            matrix = np.array([first[:2], second[:2]])
            if abs(np.linalg.det(matrix)) < .24:
                continue
            x, y = xy = np.linalg.solve(matrix, [first[2], second[2]])
            if not (-.2 * w < x < 1.2 * w and .05 * h < y < .62 * h):
                continue
            errors = abs(lines[:, :2] @ xy - lines[:, 2])
            inliers = errors < max(3, .018 * h)
            support = int(inliers.sum())
            if support >= 4 and (best is None or support > best[0]):
                best = support, xy
    return best[1] if best is not None else (w / 2, .40 * h)


def choose_quad(image, mask, confidence, surface, aspect):
    h, w = mask.shape
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    wall = surface == "wall"
    reliable = mask & (confidence >= (.57 if wall else .60))
    if not wall:
        border = max(2, round(.035 * min(w, h)))
        reliable[:border] = reliable[-border:] = False
        reliable[:, :border] = reliable[:, -border:] = False
    clearance = max(2, round(.012 * min(w, h)))
    safe = cv.erode(reliable.astype(np.uint8), np.ones((2 * clearance + 1,) * 2, np.uint8),
                    borderType=cv.BORDER_CONSTANT, borderValue=0)
    if not safe.any():
        return None
    edges = (cv.Canny(gray, 45, 115) > 0).astype(np.float32)
    target_y = .35 if wall else .78
    vx, vy = (w / 2, h / 2) if wall else floor_direction(gray, reliable)
    candidates = []
    sizes = (.23, .19, .15, .12) if wall else (.35, .29, .23, .17)
    for fraction in sizes:
        width = max(6, round(min(w * fraction, h * fraction * 1.65)))
        height = max(4, round(width / aspect * (1 if wall else .51)))
        if width >= w or height >= h:
            continue
        kernel = (width, height)
        local_edges = cv.boxFilter(edges, -1, kernel)
        local_confidence = cv.boxFilter(confidence, cv.CV_32F, kernel)
        valid = cv.erode(safe, np.ones((height, width), np.uint8),
                         borderType=cv.BORDER_CONSTANT, borderValue=0).astype(bool)
        if wall:
            valid &= (local_edges <= .065) & (local_confidence >= .70)
        else:
            valid &= (local_edges <= .13) & (local_confidence >= .72)
        ys, xs = np.nonzero(valid)
        if not xs.size:
            continue
        distance = ((xs / w - .5) ** 2 + 1.4 * (ys / h - target_y) ** 2)
        score = local_confidence[ys, xs] - 2.1 * local_edges[ys, xs] - .75 * distance
        index = int(np.argmax(score))
        x, y = int(xs[index]), int(ys[index])
        left, top = x - width // 2, y - height // 2
        right, bottom = left + width - 1, top + height - 1
        if wall:
            quad = [[left, top], [right, top], [right, bottom], [left, bottom]]
        else:
            taper = float(np.clip(height / max(bottom - vy, height), .12, .42))
            inset = width * taper / 2
            shift = float(np.clip((vx - x) * taper, -inset, inset))
            quad = [[left + inset + shift, top], [right - inset + shift, top],
                    [right, bottom], [left, bottom]]
        candidates.append((float(score[index]) + fraction * (.55 if wall else 1.05),
                           np.asarray(quad, np.float32)))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def placement_alpha(polygon, region):
    outline = cv.fillConvexPoly(np.zeros(region.shape, np.uint8),
                               np.rint(polygon * 16).astype(np.int32),
                               255, lineType=cv.LINE_AA, shift=4)
    return (outline.astype(np.float32) / 255) * region


def composite(output, asset, polygon, region, surface):
    h, w = asset.shape[:2]
    far_width = np.linalg.norm(polygon[1] - polygon[0])
    near_width = np.linalg.norm(polygon[2] - polygon[3])
    side_height = min(np.linalg.norm(polygon[3] - polygon[0]),
                      np.linalg.norm(polygon[2] - polygon[1]))
    # Area resampling before the warp prevents weave and painting detail aliasing.
    reduced = (max(2, min(w, round((far_width + near_width) * .65))),
               max(2, min(h, round(side_height * 1.3))))
    if reduced != (w, h):
        asset = cv.resize(asset, reduced, interpolation=cv.INTER_AREA)
        h, w = asset.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv.getPerspectiveTransform(corners, polygon)
    size = output.shape[1::-1]
    projected = cv.warpPerspective(asset, matrix, size, borderMode=cv.BORDER_REPLICATE)
    alpha = placement_alpha(polygon, region)
    blur = max(.65, min(size) * .0028)
    shadow = cv.GaussianBlur(alpha, (0, 0), blur)
    offset = max(1, round(blur))
    shadow = cv.warpAffine(shadow, np.float32([[1, 0, offset], [0, 1, offset]]), size)
    original = output.astype(np.float32)
    background = original * (1 - (.29 if surface == "wall" else .22)
                             * (shadow * (1 - alpha) * region)[:, :, None])
    light = cv.GaussianBlur(cv.cvtColor(output, cv.COLOR_BGR2GRAY).astype(np.float32),
                            (0, 0), max(3, min(size) * .025))
    active = alpha >= .5
    if not active.any():
        return active
    mean_light = float(np.median(light[active]))
    exposure = np.clip(.70 + mean_light / 640, .73, 1.06)
    shading = np.clip((light + 35) / (mean_light + 35), .75, 1.24)
    foreground = np.clip(projected * (exposure * shading)[:, :, None], 0, 255)
    affected = region & ((alpha > 0) | (shadow > 0))
    a = alpha[affected, None]
    output[affected] = np.rint(background[affected] * (1 - a) + foreground[affected] * a).astype(np.uint8)
    return active
