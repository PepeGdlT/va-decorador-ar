"""Place a parametric mesh on the annotated floor of a perspective photograph."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .geometry import Camera, Pose
from .placement import polygon_coverage, visible_polygon
from .render import box, cylinder, draw_furniture_depth, furniture


@dataclass(frozen=True)
class Model:
    dimensions: tuple[float, float, float]
    top_color: tuple[int, int, int]
    leg_color: tuple[int, int, int]
    kind: str = "table"


# Dimensions are relative to the nominal camera height of the ZInD layout.
MODELS = {
    "mesa_baja": Model((.64, .38, .29), (110, 151, 182), (62, 73, 84)),
    "banco": Model((.63, .25, .32), (144, 161, 125), (75, 93, 86)),
    "taburete": Model((.29, .29, .38), (122, 161, 201), (68, 81, 97)),
    "consola": Model((.75, .24, .57), (103, 137, 171), (55, 63, 73)),
    "mostrador": Model((.77, .38, .72), (155, 169, 184), (100, 111, 123), "counter"),
    "estanteria": Model((.64, .27, .83), (100, 143, 179), (79, 111, 139), "shelves"),
    "mesa_redonda": Model((.53, .53, .42), (114, 155, 192), (65, 69, 74), "round"),
    "aparador": Model((.70, .30, .43), (123, 157, 188), (77, 116, 146), "counter"),
    "puf": Model((.35, .35, .24), (117, 144, 96), (74, 91, 66), "ottoman"),
    "lampara_de_pie": Model((.31, .31, .82), (176, 206, 232), (78, 85, 89), "lamp"),
    "sofa": Model((.92, .44, .42), (172, 188, 204), (58, 66, 78), "sofa"),
    "silla": Model((.34, .34, .50), (154, 174, 196), (62, 74, 92), "chair"),
    "planta_de_pie": Model((.34, .34, .74), (68, 132, 62), (178, 192, 206), "plant"),
}


@dataclass(frozen=True)
class FurniturePreview:
    image: np.ndarray
    footprint: np.ndarray
    coverage: float


@dataclass(frozen=True)
class Support:
    name: str
    pose: Pose
    camera: Camera
    boundary: np.ndarray
    clearance: float = np.inf

    @property
    def outline(self):
        return self.pose.project(np.column_stack((self.boundary, np.zeros(len(self.boundary)))),
                                 self.camera)

    @property
    def center(self):
        point = np.array([[*self.boundary.mean(axis=0), 0.]])
        return self.pose.project(point, self.camera)[0]


@dataclass(frozen=True)
class FurnitureLayer:
    parts: tuple
    pose: Pose
    camera: Camera
    footprint: np.ndarray
    coverage: float
    supports: tuple[Support, ...] = ()
    attached_to: Support | None = None


def ground_pose(floor, wall, intrinsics, anchor):
    x, y = floor.coordinates([anchor])[0]
    origin = np.linalg.solve(intrinsics, floor.to_image @ [x, y, 1.])
    right = np.linalg.solve(intrinsics, floor.to_image[:, 0])
    up = np.linalg.solve(intrinsics, -wall.to_image[:, 1])
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    rotation = np.column_stack((right, np.cross(up, right), up))
    return Pose(cv.Rodrigues(rotation)[0], origin[:, None])


def rectangle(width, depth):
    return np.float32([[-width/2, -depth/2], [width/2, -depth/2],
                       [width/2, depth/2], [-width/2, depth/2]])


def validate_projection(parts, pose, camera):
    points = np.concatenate([part[0] for part in parts])
    if np.any(pose.camera_points(points)[:, 2] <= .05):
        raise ValueError("El mueble queda demasiado cerca o detrás de la cámara.")
    projected = pose.project(points, camera)
    if not np.isfinite(projected).all() or np.max(abs(projected)) > 1e6:
        raise ValueError("No se puede proyectar el objeto en la foto.")
    outline = cv.convexHull(projected.astype(np.float32)).reshape(-1, 2)
    area, _ = visible_polygon(outline, camera.size[::-1])
    if area < 16:
        raise ValueError("Queda muy poco objeto visible en la foto.")


def _floor_mesh(spec, width, depth, height):
    if spec.kind == "table":
        base = list(furniture(width, depth, height, 0, 0, 0,
                              top_color=spec.top_color, leg_color=spec.leg_color))
        leg = min(width, depth, height) * 0.13
        shelf_z = height * 0.28
        base.append(box(-width / 2 + leg, -depth / 2 + leg, shelf_z,
                        width - 2 * leg, depth - 2 * leg, height * 0.06, spec.leg_color))
        return tuple(base), [("Tablero", height, np.inf)]
    slab = .06 * height
    radius = width / 2
    if spec.kind == "sofa":
        leg_h = .16 * height
        plinth_h = .10 * height
        seat_h = .20 * height
        arm_w = .11 * width
        back_d = .18 * depth
        leg = .055 * min(width, depth)
        parts = [
            box(lx, ly, 0, leg, leg, leg_h, spec.leg_color)
            for lx in (-width / 2 + .06 * width, width / 2 - .06 * width - leg)
            for ly in (-depth / 2 + .06 * depth, depth / 2 - .06 * depth - leg)
        ]
        # Recessed wooden/dark base plinth.
        parts.append(box(-width / 2 + .02 * width, -depth / 2 + .02 * depth, leg_h,
                         .96 * width, .96 * depth, plinth_h, spec.leg_color))
        z_seat = leg_h + plinth_h
        # Two seat cushions + two backrest cushions + outer backrest + slim armrests.
        inner_w = width - 2 * arm_w
        gap = .02 * width
        cw = (inner_w - gap) / 2
        cushion_d = depth - back_d * .75
        parts.append(box(-width / 2 + arm_w, -depth / 2, z_seat, cw, cushion_d, seat_h, spec.top_color))
        parts.append(box(-width / 2 + arm_w + cw + gap, -depth / 2, z_seat, cw, cushion_d, seat_h, spec.top_color))
        # Backrest frame + two back cushions.
        parts.append(box(-width / 2 + arm_w, depth / 2 - back_d, z_seat,
                         inner_w, back_d, height - z_seat, spec.top_color))
        back_c_h = (height - z_seat - seat_h) * .92
        parts.append(box(-width / 2 + arm_w + .01 * width, depth / 2 - back_d - .10 * depth,
                         z_seat + seat_h, cw - .01 * width, .11 * depth, back_c_h, spec.top_color))
        parts.append(box(-width / 2 + arm_w + cw + gap, depth / 2 - back_d - .10 * depth,
                         z_seat + seat_h, cw - .01 * width, .11 * depth, back_c_h, spec.top_color))
        for ax in (-width / 2, width / 2 - arm_w):
            parts.append(box(ax, -depth / 2 + .02 * depth, z_seat, arm_w, depth - .02 * depth,
                             .44 * height, spec.top_color))
        return tuple(parts), [("Asiento", z_seat + seat_h, height - z_seat - seat_h)]
    if spec.kind == "chair":
        seat_z = .46 * height
        leg = .08 * min(width, depth)
        parts = [
            box(lx, ly, 0, leg, leg, seat_z, spec.leg_color)
            for lx in (-width / 2 + .04 * width, width / 2 - .04 * width - leg)
            for ly in (-depth / 2 + .04 * depth, depth / 2 - .04 * depth - leg)
        ]
        parts.append(box(-width / 2, -depth / 2, seat_z, width, depth, .08 * height, spec.top_color))
        # Two vertical back posts + padded backrest panel.
        for lx in (-width / 2 + .04 * width, width / 2 - .04 * width - leg):
            parts.append(box(lx, depth / 2 - .10 * depth, seat_z + .08 * height,
                             leg, .10 * depth, height - seat_z - .08 * height, spec.leg_color))
        parts.append(box(-width / 2 + .04 * width, depth / 2 - .12 * depth, seat_z + .24 * height,
                         width - .08 * width, .11 * depth, (height - seat_z - .24 * height) * .92, spec.top_color))
        return tuple(parts), [("Asiento", seat_z + .08 * height, height - seat_z - .08 * height)]
    if spec.kind == "plant":
        # Ceramic planter + soil + central stem + organic multi-height broad leaf clusters.
        pot = cylinder(0, .44 * radius, .28 * height, spec.leg_color, top_radius=.54 * radius)
        rim = cylinder(.25 * height, .55 * radius, .04 * height, (158, 174, 190))
        soil = cylinder(.27 * height, .48 * radius, .02 * height, (48, 58, 68))
        stem = cylinder(.28 * height, .055 * radius, .48 * height, (62, 86, 52), top_radius=.035 * radius)
        leaf_specs = (
            (.12 * radius, .08 * radius, .46 * height, .26 * radius, .04 * height, (62, 122, 56)),
            (-.14 * radius, -.06 * radius, .52 * height, .28 * radius, .04 * height, (72, 138, 64)),
            (.04 * radius, -.15 * radius, .58 * height, .25 * radius, .04 * height, (84, 152, 74)),
            (-.08 * radius, .14 * radius, .64 * height, .27 * radius, .04 * height, (74, 142, 66)),
            (.11 * radius, -.04 * radius, .72 * height, .23 * radius, .04 * height, (92, 162, 80)),
            (-.03 * radius, .02 * radius, .80 * height, .19 * radius, .04 * height, (104, 174, 88)),
        )
        leaves = []
        for ox, oy, lz, lr, lh, lcol in leaf_specs:
            pts, col, faces = cylinder(lz, lr, lh, lcol, top_radius=.82 * lr, sides=16)
            pts[:, 0] += ox
            pts[:, 1] += oy
            leaves.append((pts, col, faces))
        return (pot, rim, soil, stem, *leaves), []
    if spec.kind == "round":
        return (
            cylinder(0, .63 * radius, slab, spec.leg_color),
            cylinder(slab, .15 * radius, height - 2 * slab, spec.leg_color),
            cylinder(height - slab, radius, slab, spec.top_color),
        ), [("Tablero", height, np.inf)]
    if spec.kind == "ottoman":
        return (
            cylinder(0, .83 * radius, .22 * height, spec.leg_color),
            cylinder(.19 * height, .94 * radius, .64 * height, spec.top_color, top_radius=radius),
            cylinder(.83 * height, radius, .17 * height, spec.top_color, top_radius=.89 * radius),
        ), []
    if spec.kind == "lamp":
        return (
            cylinder(0, .76 * radius, slab, spec.leg_color),
            cylinder(slab, .10 * radius, .77 * height, spec.leg_color),
            cylinder(.77 * height, radius, .23 * height, spec.top_color, top_radius=.60 * radius),
        ), []
    if spec.kind == "counter":
        body = box(-.46*width, -.44*depth, 0, .92*width, .88*depth, height-slab, spec.leg_color)
        top = box(-width/2, -depth/2, height-slab, width, depth, slab, spec.top_color)
        return (body, top), [("Encimera", height, np.inf)]
    side = .06 * width
    shelves = [
        box(x, -depth/2, 0, side, depth, height, spec.leg_color)
        for x in (-width/2, width/2-side)
    ]
    heights = [height * ratio for ratio in (.16, .43, .70, 1.)]
    shelves.extend(box(-width/2+side, -depth/2, level-slab, width-2*side, depth, slab, spec.top_color)
                   for level in heights)
    levels = [(f"Balda {i + 1}", level, heights[i + 1] - level - slab)
              for i, level in enumerate(heights[:-1])]
    return tuple(shelves), [*levels, ("Parte superior", height, np.inf)]


def prepare_furniture(mask, floor, wall, intrinsics, anchor, *,
                      model="mesa_baja", scale=1., yaw=0.):
    if model not in MODELS:
        raise ValueError(f"Mueble desconocido: {model}.")
    if not np.isfinite([*anchor, scale, yaw]).all() or scale <= 0:
        raise ValueError("La posición, el giro y la escala deben ser válidos.")
    pose = ground_pose(floor, wall, intrinsics, anchor)
    rotation = cv.Rodrigues(pose.rotation)[0]
    angle = np.deg2rad(yaw)
    rotation = rotation @ np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pose = Pose(cv.Rodrigues(rotation)[0], pose.translation)
    camera = Camera(intrinsics, np.zeros(5), mask.shape[::-1])
    spec = MODELS[model]
    width, depth, height = np.asarray(spec.dimensions) * scale
    parts, levels = _floor_mesh(spec, width, depth, height)
    validate_projection(parts, pose, camera)
    boundary = rectangle(width, depth)
    quad = pose.project(np.column_stack((boundary, np.zeros(4))), camera).astype(np.float32)
    coverage = polygon_coverage(quad, mask)
    if coverage < .96:
        raise ValueError(f"Solo el {coverage:.0%} de la base visible coincide con suelo libre detectado.")
    usable = .64 if spec.kind == "round" else .87 if spec.kind == "shelves" else .95
    supports = tuple(
        Support(name, Pose(pose.rotation, pose.camera_points([[0, 0, z]]).reshape(3, 1)),
                camera, rectangle(width * usable, depth * (.64 if spec.kind == "round" else .95)),
                clearance)
        for name, z, clearance in levels
    )
    return FurnitureLayer(parts, pose, camera, quad, coverage, supports)


def _contact_shadow(image, polygons, receiver, strength, blur):
    shadow = np.zeros(image.shape[:2], np.uint8)
    for polygon in polygons:
        cv.fillConvexPoly(shadow, np.rint(polygon).astype(np.int32), 255)
    alpha = (strength / 255) * cv.GaussianBlur(shadow, (0, 0), blur) * receiver
    image[:] = np.rint(image * (1 - alpha[..., None])).astype(np.uint8)


def _support_shadow(image, layers, support, depth):
    polygon = np.rint(support.outline).astype(np.int32)
    receiver = cv.fillConvexPoly(np.zeros(depth.shape, np.uint8), polygon, 1)
    if depth.any():
        y, x = np.nonzero(receiver)
        rays = np.linalg.solve(support.camera.matrix, np.stack((x, y, np.ones_like(x)))).T
        normal = cv.Rodrigues(support.pose.rotation)[0][:, 2]
        expected = (rays @ normal) / (support.pose.translation.ravel() @ normal)
        receiver[y, x] = depth[y, x] <= expected * 1.015
    contacts = []
    for layer in layers:
        vertices = np.concatenate([part[0] for part in layer.parts])
        base = vertices[np.isclose(vertices[:, 2], vertices[:, 2].min())]
        pixels = layer.pose.project(base, layer.camera)
        contacts.append(cv.convexHull(pixels.astype(np.float32)))
    _contact_shadow(image, contacts, receiver, .36, 4)


def _ambient_gain(image, footprint):
    x, y, w, h = cv.boundingRect(footprint)
    patch = image[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12]
    if not patch.size:
        return 1.
    channel = np.median(patch[::3, ::3], axis=(0, 1)) / 255
    neutral = channel.mean()
    return np.clip(.80 + .40 * neutral + .22 * (channel - neutral), .78, 1.15)


def render_furniture_layers(image, mask, layers, *, lighting=True, depth_map=None):
    if not layers:
        return image.copy()
    output = image.copy()
    standing = [layer for layer in layers if layer.attached_to is None]
    elevated = {}
    for layer in layers:
        if layer.attached_to is not None:
            elevated.setdefault(id(layer.attached_to), []).append(layer)
    if lighting and standing:
        _contact_shadow(output, [layer.footprint for layer in standing], mask, .34, 10)
    buffer = np.zeros(mask.shape, np.float32)
    if depth_map is not None:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.shape != mask.shape:
            raise ValueError("El mapa de profundidad debe coincidir con el tamaño de la foto.")
        occluders = np.isfinite(depth) & (depth > .05) & (~mask)
        buffer[occluders] = 1.0 / depth[occluders]

    def draw(layer, reference):
        gain = _ambient_gain(reference, layer.footprint) if lighting else 1.
        draw_furniture_depth(output, layer.parts, layer.pose, layer.camera,
                             inverse_depth=buffer, light=gain)

    for layer in standing:
        draw(layer, image)
    if lighting:
        for group in elevated.values():
            _support_shadow(output, group, group[0].attached_to, buffer)
    if elevated:
        reference = output.copy() if lighting else image
        for group in elevated.values():
            for layer in group:
                draw(layer, reference)
    return output


def place_furniture(image, mask, floor, wall, intrinsics, anchor, *,
                    model="mesa_baja", scale=1., yaw=0.):
    layer = prepare_furniture(mask, floor, wall, intrinsics, anchor,
                              model=model, scale=scale, yaw=yaw)
    return FurniturePreview(render_furniture_layers(image, mask, [layer]),
                            layer.footprint, layer.coverage)
