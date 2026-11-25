"""Objects placed on virtual or manually referenced horizontal surfaces."""
import cv2 as cv
import numpy as np

from .geometry import Camera, Pose
from .photo_furniture import FurnitureLayer, Support, ground_pose, rectangle, validate_projection
from .render import box, cylinder

ACCESSORIES = ("plato", "libros", "maceta", "jarron", "taza", "cuenco", "bandeja")


def support_points(pose, camera, pixels):
    xy = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays = np.linalg.solve(camera.matrix, np.column_stack((xy, np.ones(len(xy)))).T).T
    axes = cv.Rodrigues(pose.rotation)[0]
    normal = axes[:, 2]
    denom = rays @ normal
    if np.any(abs(denom) < 1e-7):
        raise ValueError("La superficie está demasiado cerca del horizonte.")
    depth = (pose.translation.ravel() @ normal) / denom
    if np.any(depth <= .05) or not np.isfinite(depth).all():
        raise ValueError("El punto no está delante de la cámara a esa altura.")
    return ((rays * depth[:, None] - pose.translation.ravel()) @ axes)[:, :2]


def support_plane(scene, height):
    wall, floor = scene.reference_planes
    ground = ground_pose(floor, wall, scene.intrinsics, scene.floor_anchor)
    up = cv.Rodrigues(ground.rotation)[0][:, 2]
    camera_height = -ground.translation.ravel() @ up
    camera = Camera(scene.intrinsics, np.zeros(5), scene.image.shape[1::-1])
    origin = ground.camera_points(np.array([[0., 0., height * camera_height]])).reshape(3, 1)
    return Pose(ground.rotation, origin), camera


def mark_support(scene, corners, height):
    corners = np.asarray(corners, np.float32)
    if (corners.shape != (4, 2) or not np.isfinite(corners).all()
            or np.any(corners < 0) or np.any(corners >= scene.image.shape[1::-1])
            or not cv.isContourConvex(corners) or abs(cv.contourArea(corners)) < 40):
        raise ValueError("Marca las cuatro esquinas visibles en orden alrededor del tablero.")
    if not np.isfinite(height) or not .03 <= height <= 1.4:
        raise ValueError("La altura debe estar entre 0.03 y 1.40 alturas de cámara.")
    pose, camera = support_plane(scene, height)
    boundary = support_points(pose, camera, corners).astype(np.float32)
    if not cv.isContourConvex(boundary) or abs(cv.contourArea(boundary)) < .002:
        raise ValueError("El tablero marcado es demasiado pequeño para usarlo.")
    return Support("Superficie marcada", pose, camera, boundary)


def _mesh(name):
    if name == "taza":
        return .128, .096, (
            cylinder(0, .039, .071, (166, 132, 76), top_radius=.045),
            cylinder(.070, .036, .002, (60, 70, 69)),
            box(.035, -.012, .052, .027, .024, .011, (166, 132, 76)),
            box(.051, -.012, .025, .011, .024, .031, (166, 132, 76)),
            box(.035, -.012, .021, .027, .024, .010, (166, 132, 76)),
        )
    if name == "cuenco":
        return .156, .156, (
            cylinder(0, .049, .037, (96, 180, 212), top_radius=.078),
            cylinder(.036, .067, .002, (122, 203, 230)),
        )
    if name == "bandeja":
        return .22, .13, (
            box(-.11, -.065, 0, .22, .13, .012, (86, 124, 160)),
            box(-.11, -.065, .012, .22, .009, .010, (70, 108, 143)),
            box(-.11, .056, .012, .22, .009, .010, (70, 108, 143)),
        )
    if name == "plato":
        return .204, .204, (
            cylinder(0, .068, .008, (222, 223, 214), top_radius=.101),
            cylinder(.0075, .071, .003, (248, 247, 238), top_radius=.08),
        )
    if name == "libros":
        return .175, .116, (
            box(-.0875, -.058, 0, .175, .116, .022, (112, 93, 173)),
            box(-.074, -.05, .022, .15, .10, .023, (137, 166, 108)),
            box(-.068, -.047, .045, .135, .094, .019, (85, 131, 181)),
        )
    if name == "maceta":
        return .15, .15, (
            cylinder(0, .045, .091, (94, 131, 185), top_radius=.068),
            cylinder(.09, .058, .004, (47, 58, 65)),
            box(-.008, -.008, .093, .016, .016, .095, (68, 116, 53)),
            cylinder(.16, .075, .07, (84, 145, 74), top_radius=.039),
            cylinder(.229, .047, .046, (104, 159, 77), top_radius=.012),
        )
    if name == "jarron":
        return .12, .12, (
            cylinder(0, .039, .06, (173, 132, 83), top_radius=.057),
            cylinder(.06, .057, .069, (187, 147, 99), top_radius=.032),
            cylinder(.129, .032, .036, (208, 169, 115), top_radius=.035),
        )
    raise ValueError(f"Objeto de altura desconocido: {name}.")


def prepare_accessory(name, anchor, support, *, scale=1., angle=0.):
    if not np.isfinite([*anchor, scale, angle]).all() or scale <= 0:
        raise ValueError("La posición, el giro y la escala deben ser válidos.")
    width, depth, parts = _mesh(name)
    center = support_points(support.pose, support.camera, [anchor])[0]
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    base = rectangle(width * scale, depth * scale) @ rotation.T + center
    boundary = support.boundary.astype(np.float32)
    if any(cv.pointPolygonTest(boundary, tuple(point), False) < 0 for point in base):
        raise ValueError("El objeto se sale del apoyo. Muévelo hacia dentro o reduce la escala.")
    transformed = []
    for points, *material in parts:
        mesh = points * scale
        mesh[:, :2] = mesh[:, :2] @ rotation.T + center
        transformed.append((mesh, *material))
    if max(mesh[0][:, 2].max() for mesh in transformed) >= support.clearance:
        raise ValueError("El objeto es demasiado alto para esa balda.")
    validate_projection(transformed, support.pose, support.camera)
    footprint = support.pose.project(np.column_stack((base, np.zeros(4))),
                                     support.camera).astype(np.float32)
    return FurnitureLayer(tuple(transformed), support.pose, support.camera, footprint, 1.,
                          attached_to=support)
