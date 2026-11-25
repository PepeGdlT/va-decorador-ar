"""Presentation and orchestration for the main project notebook."""
from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import mkdtemp

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .catalog import DESIGN_ANCHORS, DESIGNS, texture
from .guided import InteriorDecorator
from .image_io import read_image
from .panorama import CONTROL_CASES, FURNITURE_VIEWS, SUPPORT_VIEWS, ZindLayout, prepare_zind_sample
from .photo_furniture import place_furniture
from .photo_study import study
from .rectification import place_rectified, project, select_wall
from .recorded_room import render_recordings


@dataclass
class Scene:
    image: np.ndarray
    masks: dict
    planes: list
    wall_anchor: tuple[int, int]
    floor_anchor: tuple[int, int]
    rug_size: float
    intrinsics: np.ndarray

    @property
    def reference_planes(self):
        return select_wall(self.planes, self.wall_anchor)

    def place(self, surface, anchor, length, *, angle=0, vertical=False, image=None, asset=None):
        if surface == "wall":
            plane, _ = select_wall(self.planes, anchor)
        else:
            _, plane = self.reference_planes
        return place_rectified(
            self.image if image is None else image, self.masks[surface][0],
            plane, surface, anchor, length, angle=angle, vertical=vertical, texture=asset,
        )


def _load_views(root, segmenter, photos, cases):
    layout = ZindLayout(root)
    scenes = {}
    for path in photos:
        image = read_image(path)
        scenes[path.stem] = Scene(
            image, segmenter.predict(image, supports=True), layout.planes(path), *cases[path.stem],
            layout.intrinsics(path),
        )
    return scenes


def load_scenes(root, segmenter):
    return _load_views(root, segmenter, prepare_zind_sample(root), CONTROL_CASES)


def load_furniture_stock(root, segmenter):
    # Reference wall point, test support point and an unused 2D rug size.
    cases = {
        "bonus_03": ((430, 270), (420, 538), .4),
        "salon_03": ((432, 300), (515, 581), .4),
        "pasillo_01": ((680, 310), (380, 460), .4),
        "dormitorio_03": ((670, 320), (510, 568), .4),
        "dormitorio_04": ((515, 370), (475, 600), .4),
        "pasillo_02": ((210, 290), (525, 525), .4),
    }
    photos = prepare_zind_sample(root, views=FURNITURE_VIEWS)
    return _load_views(root, segmenter, photos, cases)


def load_support_scenes(root, segmenter):
    cases = {
        "cocina_01": ((620, 260), (674, 597), .4),
        "cocina_02": ((460, 220), (410, 560), .4),
    }
    return _load_views(root, segmenter, prepare_zind_sample(root, views=SUPPORT_VIEWS), cases)


def _show(ax, image, title):
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def show_original(scene, name):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    _show(ax, scene.image, f"Foto original · {name}")
    return fig


def show_gallery(scenes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, scene) in zip(axes.flat, scenes.items()):
        art, rug = DESIGNS[name][:2]
        _show(ax, decorate_2d(name, scene), f"{name} · {art} + {rug}")
    return fig


def decorate_2d(name, scene):
    art, rug, art_size, rug_size, vertical = DESIGNS[name]
    art_anchor, rug_anchor = DESIGN_ANCHORS[name]
    picture, _ = scene.place("wall", art_anchor, art_size, asset=texture("wall", art))
    design, _ = scene.place("floor", rug_anchor, rug_size, vertical=vertical,
                            image=picture.image, asset=texture("floor", rug))
    return design.image


def furniture_preview(scene, anchor, *, model, scale=1., yaw=0., image=None):
    wall, floor = scene.reference_planes
    return place_furniture(
        scene.image if image is None else image, scene.masks["floor"][0],
        floor, wall, scene.intrinsics, anchor, model=model, scale=scale, yaw=yaw,
    )


def show_furniture_stock(scenes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    results = {}
    for ax, (name, scene) in zip(axes.flat, scenes.items()):
        model = "banco" if name.startswith("dormitorio") else "mesa_baja"
        try:
            result = furniture_preview(scene, scene.floor_anchor, model=model)
        except ValueError as error:
            _show(ax, scene.image, f"{name} · colocación rechazada")
            results[name] = str(error)
        else:
            title = "bloquea el paso · revisión visual" if name == "pasillo_02" else model
            _show(ax, result.image, f"{name} · {title}")
            results[name] = f"{result.coverage:.0%} de la base sobre suelo detectado"
    plt.show()
    for name, result in results.items():
        print(f"{name}: {result}")


def show_furniture_detail(scene, result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    _show(axes[0], scene.image, "Foto original")
    _show(axes[1], scene.image, f"Huella del mueble · {result.coverage:.0%} de suelo")
    axes[1].plot(*result.footprint[[0, 1, 2, 3, 0]].T, "o-", color="cyan")
    _show(axes[2], result.image, "Composición 2D + mueble 3D")
    return fig


def show_rectification(scene):
    wall, floor = scene.reference_planes
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    cases = (
        ("wall", "Pared", wall, scene.wall_anchor, .4, [40, 185, 250]),
        ("floor", "Suelo", floor, scene.floor_anchor, scene.rug_size, [160, 215, 60]),
    )
    for (photo_ax, plane_ax, result_ax), (surface, label, plane, anchor, size, color) in zip(axes, cases):
        placement, flat = scene.place(surface, anchor, size)
        mask = scene.masks[surface][0]
        overlay = scene.image.copy()
        overlay[mask] = (.65 * overlay[mask] + .35 * np.asarray(color)).astype(np.uint8)
        _show(photo_ax, overlay, f"{label} · máscara y referencias")
        _show(result_ax, placement.image, f"{label} · encaje {placement.coverage:.0%}")

        x0, y0 = flat.min(axis=0) - .12
        x1, y1 = flat.max(axis=0) + .12
        seam = wall.height if surface == "wall" else 0
        y0, y1 = min(y0, seam - .04), max(y1, seam + .04)
        rectified = plane.rectify(scene.image, (x0, y0, x1, y1))
        plane_ax.imshow(cv.cvtColor(rectified, cv.COLOR_BGR2RGB), extent=(x0, x1, y1, y0))
        plane_ax.set_title("Pared de frente" if surface == "wall" else "Suelo desde arriba")
        plane_ax.grid(alpha=.45)
        plane_ax.axhline(seam, color="gold", linewidth=2)

        edge = np.array([[.18 * wall.width, wall.height], [.82 * wall.width, wall.height]])
        if np.all(np.column_stack((edge, np.ones(2))) @ wall.to_image[2] > 0):
            photo_ax.plot(*project(wall.to_image, edge).T, color="gold", linewidth=2)
        outline = [0, 1, 2, 3, 0]
        photo_ax.plot(*placement.quad[outline].T, color="cyan")
        plane_ax.plot(*flat[outline].T, color="cyan")
        h, w = scene.image.shape[:2]
        photo_ax.set(xlim=(0, w), ylim=(h, 0))
    return fig


def start_editor(scene, art="paisaje", rug="clasica"):
    assets = {"wall": texture("wall", art), "floor": texture("floor", rug)}
    def place(_image, _masks, surface, anchor, *, size, angle_deg, orientation):
        result, _ = scene.place(surface, anchor, size * 1.5,
                                angle=angle_deg, vertical=orientation == "vertical",
                                asset=assets[surface])
        return result

    return InteriorDecorator(scene.image, scene.masks, place=place).show()


def new_run(root):
    configured = os.environ.get("DECORADOR_AR_NOTEBOOK_OUTPUT")
    if configured:
        return Path(configured)
    result_dir = Path(root) / "results"
    result_dir.mkdir(exist_ok=True)
    return Path(mkdtemp(prefix="notebook-", dir=result_dir))


def evaluate_masks(archive, output, segmenter, count=9):
    rows = study(archive, output, count, metrics_only=True, segmenter=segmenter)
    means = {}
    for surface in ("wall", "floor"):
        values = [row[f"{surface}_iou"] for row in rows if row[f"{surface}_iou"] is not None]
        means[surface] = float(np.mean(values)) if values else None
    return means


def show_room_clips(archive, folder, clips):
    results = render_recordings(archive, folder, clips)
    for name, (previews, counts, rms) in results.items():
        fig, axes = plt.subplots(2, len(previews), figsize=(13, 7), layout="constrained")
        fig.suptitle(name)
        for index, (original, decorated, state) in enumerate(previews):
            _show(axes[0, index], original, "Captura real")
            _show(axes[1, index], decorated, state.replace("_", " "))
        plt.show()
        print(f"{name}: {counts}. RMS del ajuste inicial: {rms * 1000:.1f} mm.")
        print(folder / name / "decorado.mp4")
    return results
