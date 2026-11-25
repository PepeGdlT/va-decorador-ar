import cv2 as cv
import numpy as np


FACES = np.array(((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)))
LIGHT = (0.65, 1.0, 0.82, 0.70, 0.85, 0.75)


def box(x, y, z, width, depth, height, color):
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    return corners * [width, depth, height] + [x, y, z], color


def cylinder(z, radius, height, color, *, top_radius=None, sides=24):
    theta = np.arange(sides) * (2 * np.pi / sides)
    ring = np.column_stack((np.cos(theta), np.sin(theta)))
    lower = np.column_stack((radius * ring, np.full(sides, z)))
    upper = np.column_stack(((radius if top_radius is None else top_radius) * ring,
                             np.full(sides, z + height)))
    faces = (tuple(range(sides - 1, -1, -1)), tuple(range(sides, 2 * sides)),
             *((i, (i + 1) % sides, (i + 1) % sides + sides, i + sides)
               for i in range(sides)))
    return np.vstack((lower, upper)), color, faces


def furniture(width, depth, height, x, y, yaw, *,
              top_color=(65, 156, 220), leg_color=(90, 108, 126)):
    leg = min(width, depth, height) * 0.13
    top = height * 0.12
    pieces = [box(-width / 2, -depth / 2, height - top, width, depth, top, top_color)]
    for lx in (-width / 2, width / 2 - leg):
        for ly in (-depth / 2, depth / 2 - leg):
            pieces.append(box(lx, ly, 0, leg, leg, height - top, leg_color))
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
            polygon = camera_points[indices]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            if np.dot(normal, polygon[0]) >= 0:
                continue
            faces.append((polygon[:, 2].mean(), np.rint(pixels[indices]).astype(np.int32),
                          tuple(int(channel * shade) for channel in color)))
    for _, polygon, color in sorted(faces, key=lambda face: face[0], reverse=True):
        cv.fillConvexPoly(frame, polygon, color, lineType=cv.LINE_AA)
        cv.polylines(frame, [polygon], True, (55, 65, 75), 1, cv.LINE_AA)


def load_obj(path, color, *, scale=1.0, center_xy=True, ground_z=0.0):
    """Load a Wavefront .obj mesh into (vertices, color, faces) for draw_furniture_depth."""
    vertices, faces = [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face = [int(token.split("/")[0]) - 1 for token in parts[1:]]
                faces.append(tuple(face))
    if not vertices or not faces:
        raise ValueError(f"El archivo OBJ no contiene vértices o caras válidas: {path}")
    pts = np.asarray(vertices, dtype=float) * scale
    if center_xy:
        pts[:, 0] -= (pts[:, 0].min() + pts[:, 0].max()) / 2
        pts[:, 1] -= (pts[:, 1].min() + pts[:, 1].max()) / 2
    pts[:, 2] += ground_z - pts[:, 2].min()
    return pts, color, tuple(faces)


def draw_furniture_depth(frame, pieces, pose, camera, *, inverse_depth=None, light=1.):
    """Rasterize visible faces with directional Lambert shading and a shared depth buffer."""
    height, width = frame.shape[:2]
    if inverse_depth is None:
        inverse_depth = np.zeros((height, width), dtype=np.float32)
    # Key light from upper-front-left in camera coordinates.
    light_dir = np.array([-0.35, -0.65, -0.67], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    for points, color, *mesh_faces in pieces:
        local = pose.camera_points(points)
        pixels = pose.project(points, camera)
        faces = mesh_faces[0] if mesh_faces else FACES
        for face in faces:
            face = np.asarray(face)
            polygon = local[face]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9 or np.dot(normal, polygon[0]) >= 0:
                continue
            unit_normal = normal / norm
            lambert = max(0.0, float(np.dot(unit_normal, light_dir)))
            shade = 0.56 + 0.44 * lambert
            shaded = np.clip(np.rint(np.asarray(color) * shade * light), 0, 255).astype(np.uint8)
            for i in range(1, len(face) - 1):
                indices = face[[0, i, i + 1]]
                xy = pixels[indices]
                x0, y0 = np.maximum(0, np.floor(xy.min(axis=0)).astype(int))
                x1, y1 = np.minimum([width, height], np.ceil(xy.max(axis=0)).astype(int) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                matrix = np.column_stack((xy, np.ones(3)))
                if abs(np.linalg.det(matrix)) < 1e-8:
                    continue
                inv = np.linalg.inv(matrix)
                y, x = np.ogrid[y0:y1, x0:x1]
                weights = x[..., None] * inv[0] + y[..., None] * inv[1] + inv[2]
                depth = weights @ (1 / local[indices, 2])
                tile = inverse_depth[y0:y1, x0:x1]
                visible = np.all(weights >= -1e-8, axis=2) & (depth > tile)
                tile[visible] = depth[visible]
                frame[y0:y1, x0:x1][visible] = shaded
