"""Accumulate planar decorations and floor furniture in one photograph."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .catalog import ART, RUGS, texture
from .photo_furniture import MODELS, FurnitureLayer, Support, prepare_furniture, render_furniture_layers
from .placement import composite
from .supported_objects import ACCESSORIES, mark_support, prepare_accessory

CHOICES = {
    "wall": tuple((name.capitalize(), name) for name in ART),
    "floor": (*((f"Alfombra · {name}", name) for name in RUGS),
              *((name.replace("_", " ").capitalize() + " · 3D", name) for name in MODELS)),
    "raised": tuple((name.capitalize(), name) for name in ACCESSORIES),
}


@dataclass(frozen=True)
class FlatLayer:
    surface: str
    asset: np.ndarray
    quad: np.ndarray


class Composition:
    def __init__(self, scene):
        self.scene = scene
        self.items: list[tuple[str, FlatLayer | FurnitureLayer | Support]] = []
        self.detected_supports: tuple[Support, ...] = ()

    def supports(self):
        for support in self.detected_supports:
            yield f"Foto · {support.name}", support
        for index, (label, layer) in enumerate(self.items):
            if isinstance(layer, Support):
                yield f"{index + 1}. {layer.name}", layer
            elif isinstance(layer, FurnitureLayer):
                for support in layer.supports:
                    yield f"{index + 1}. {label} · {support.name}", support

    def add_marked_support(self, corners, height):
        self.items.append(("Superficie marcada", mark_support(self.scene, corners, height)))

    def prepare(self, surface, name, anchor, *, scale=1., angle=0., vertical=False, support=None):
        if surface not in CHOICES or name not in dict(CHOICES[surface]).values():
            raise ValueError("Esa pieza no corresponde a la superficie elegida.")
        if surface == "raised":
            if not any(candidate is support for _, candidate in self.supports()):
                raise ValueError("Selecciona un apoyo de esta vista.")
            layer = prepare_accessory(name, anchor, support, scale=scale, angle=angle)
        elif name in MODELS:
            wall, floor = self.scene.reference_planes
            layer = prepare_furniture(
                self.scene.masks["floor"][0], floor, wall, self.scene.intrinsics, anchor,
                model=name, scale=scale, yaw=angle,
            )
        else:
            asset = texture(surface, name)
            if surface == "floor" and vertical:
                asset = cv.rotate(asset, cv.ROTATE_90_CLOCKWISE)
            placement, _ = self.scene.place(surface, anchor, .45 * scale, angle=angle, asset=asset)
            layer = FlatLayer(surface, asset, placement.quad)
        if isinstance(layer, FurnitureLayer):
            for _, existing in self.items:
                if isinstance(existing, FurnitureLayer) and existing.attached_to is layer.attached_to:
                    area, _ = cv.intersectConvexConvex(layer.footprint, existing.footprint)
                    if area > 1:
                        raise ValueError("La base coincide con otra pieza del mismo apoyo.")
        label = {"raised": "Objeto", "wall": "Cuadro"}.get(
            surface, "Mueble" if name in MODELS else "Alfombra"
        )
        return f"{label} · {name.replace('_', ' ')}", layer

    def add(self, surface, name, anchor, **options):
        self.items.append(self.prepare(surface, name, anchor, **options))

    def remove(self, index):
        _, removed = self.items.pop(index)
        supports = (removed,) if isinstance(removed, Support) else getattr(removed, "supports", ())
        self.items[:] = [(label, layer) for label, layer in self.items
                         if not any(getattr(layer, "attached_to", None) is support
                                    for support in supports)]

    def render(self, pending=None, *, lighting=True):
        output = self.scene.image.copy()
        furniture = []
        items = self.items if pending is None else [*self.items, pending]
        for _, layer in items:
            if isinstance(layer, FurnitureLayer):
                furniture.append(layer)
            elif isinstance(layer, FlatLayer):
                composite(output, layer.asset, layer.quad,
                          self.scene.masks[layer.surface][0], layer.surface)
        if furniture:
            output = render_furniture_layers(output, self.scene.masks["floor"][0], furniture,
                                             lighting=lighting)
        return output
