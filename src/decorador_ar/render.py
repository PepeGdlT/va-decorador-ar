import cv2 as cv
import numpy as np


FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
LIGHT = (0.65, 1.0, 0.82, 0.70, 0.85, 0.75)


def box(x, y, z, width, depth, height, color):
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    return corners * [width, depth, height] + [x, y, z], color


def furniture(width, depth, height, x, y, yaw):
    leg = min(width, depth, height) * 0.13
    top = height * 0.12
    pieces = [box(-width / 2, -depth / 2, height - top, width, depth, top, (65, 156, 220))]
    for lx in (-width / 2, width / 2 - leg):
        for ly in (-depth / 2, depth / 2 - leg):
            pieces.append(box(lx, ly, 0, leg, leg, height - top, (90, 108, 126)))
    angle = np.deg2rad(yaw)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for points, color in pieces:
        points[:, :2] = points[:, :2] @ rotation.T + [x, y]
        yield points, color


def draw_furniture(frame, pieces, pose, camera):
    faces = []
    for points, color in pieces:
        camera_points = pose.camera_points(points)
        if np.any(camera_points[:, 2] <= 1e-3):
            continue
        pixels = pose.project(points, camera)
        if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e6:
            continue
        for indices, shade in zip(FACES, LIGHT):
            indices = list(indices)
            polygon = camera_points[indices]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            if np.dot(normal, polygon[0]) >= 0:
                continue
            faces.append((polygon[:, 2].mean(), np.rint(pixels[indices]).astype(np.int32),
                          tuple(int(channel * shade) for channel in color)))
    for _, polygon, color in sorted(faces, key=lambda face: face[0], reverse=True):
        cv.fillConvexPoly(frame, polygon, color, lineType=cv.LINE_AA)
        cv.polylines(frame, [polygon], True, (55, 65, 75), 1, cv.LINE_AA)
