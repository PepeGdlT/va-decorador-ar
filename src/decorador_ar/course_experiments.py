"""Reproducible comparisons used by the course notebook."""
from pathlib import Path
from zipfile import ZipFile

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .ade import ADEValidation
from .composition import Composition
from .photo import LABELS, Segmenter, iou
from .panorama import ZIND_VIEWS, view_camera
from .rectification import project
from .raised_surfaces import detect_supports


def _binary_step(mask, operation):
    padded = np.pad(mask, 1, constant_values=operation is np.logical_and)
    h, w = mask.shape
    return operation.reduce([padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def clean_mask(mask):
    """Opening then closing with a 3 × 3 square, implemented with NumPy."""
    for operation in (np.logical_and, np.logical_or, np.logical_or, np.logical_and):
        mask = _binary_step(mask, operation)
    return mask


def mask_comparison(archive, segmenter, *, compare_cnn=False):
    """Compare fixed variants on annotated photographs, with no threshold tuning."""
    variants = {"argmax": [], "morfología 3×3": [], "probabilidad >= 0.55": []}
    cnn = None
    if compare_cnn:
        try:
            cnn = Segmenter("876ffc56b819a829448f7e81e9f8606deef6fb65",
                            model_id="openmmlab/upernet-convnext-tiny")
        except OSError as error:
            print(f"Comparación CNN no disponible: {error}")
        else:
            variants["CNN ConvNeXt-Tiny + UperNet"] = []
            print(f"CNN: {cnn.model_id} @ {cnn.revision}")
    examples = []
    with ZipFile(Path(archive)) as source:
        data = ADEValidation(source)
        for index in (117, 506, 1517):
            image, truth = data.read_pair(f"ADE_val_{index:08d}")
            prediction = segmenter.predict(image)
            cnn_prediction = cnn.predict(image) if cnn is not None else None
            for label, class_id in LABELS.items():
                raw, probability = prediction[label]
                clean = clean_mask(raw)
                reference = cv.morphologyEx(raw.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
                reference = cv.morphologyEx(reference, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                np.testing.assert_array_equal(clean, reference.astype(bool))
                masks = (raw, clean, probability >= .55)
                if cnn_prediction is not None:
                    masks += (cnn_prediction[label][0],)
                for name, mask in zip(variants, masks):
                    score = iou(mask, truth, class_id)
                    if score is not None:
                        variants[name].append((label, score))
                if index == 506 and label == "floor":
                    examples = (image[:, :, ::-1], raw, probability, clean)
                    if cnn_prediction is not None:
                        examples += (cnn_prediction[label][0],)
                    examples += (truth == class_id,)
    titles = ["Foto", "Suelo por argmax", "Probabilidad", "Morfología"]
    if cnn is not None:
        titles.append("CNN")
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 3))
    for ax, example, title in zip(axes, examples, [*titles, "Anotación"]):
        ax.imshow(example, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    for name, scores in variants.items():
        formatted = ", ".join(f"{label} {np.mean(values):.3f}" for label in LABELS
                              if (values := [s for c, s in scores if c == label]))
        print(f"{name}: {formatted}")
    return fig


def show_support_comparison(scene):
    composition = Composition(scene)
    composition.detected_supports = detect_supports(scene)
    if not composition.detected_supports:
        raise ValueError("La escena de demostración no presenta un apoyo utilizable.")
    patch = composition.detected_supports[0]
    composition.add("raised", "plato", patch.center, support=patch)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    photos = (scene.image, composition.render(lighting=False), composition.render(lighting=True))
    for ax, image, title in zip(axes, photos, ("Zona estimada", "Material base", "Luz local y sombra")):
        ax.imshow(image[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    axes[0].plot(*patch.outline[[0, 1, 2, 3, 0]].T, color="cyan")
    print(f"Zonas de apoyo propuestas en la foto: {len(composition.detected_supports)}")
    return fig


def _rotation_reference(first, second):
    directions = {name[:-4]: (file, yaw, pitch) for file, yaw, pitch, name in ZIND_VIEWS}
    file_a, yaw_a, pitch_a = directions[first]
    file_b, yaw_b, pitch_b = directions[second]
    if file_a != file_b:
        raise ValueError("El control requiere dos vistas del mismo panorama.")
    rotation_a = view_camera(yaw_a, pitch_a)[3]
    rotation_b = view_camera(yaw_b, pitch_b)[3]
    flip = np.diag([1., -1., 1.])
    return flip @ rotation_b.T @ rotation_a @ flip


def show_registration(scenes, first="bonus_01", second="bonus_02"):
    """Register two views, measure error against known rotation, transfer a footprint."""
    a, b = scenes[first], scenes[second]
    reference = b.intrinsics @ _rotation_reference(first, second) @ np.linalg.inv(a.intrinsics)
    orb = cv.ORB_create(nfeatures=2200)
    keys_a, desc_a = orb.detectAndCompute(cv.cvtColor(a.image, cv.COLOR_BGR2GRAY), None)
    keys_b, desc_b = orb.detectAndCompute(cv.cvtColor(b.image, cv.COLOR_BGR2GRAY), None)
    if desc_a is None or desc_b is None:
        raise ValueError("No hay suficientes puntos ORB en una de las vistas.")
    matches = [pair[0] for pair in cv.BFMatcher(cv.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
               if len(pair) == 2 and pair[0].distance < .8 * pair[1].distance]
    if len(matches) < 8:
        raise ValueError("No hay suficientes correspondencias para comparar los ajustes.")
    src = np.float32([keys_a[m.queryIdx].pt for m in matches])
    dst = np.float32([keys_b[m.trainIdx].pt for m in matches])
    width, height = a.image.shape[1::-1]
    check = np.array([(x, y) for x in np.linspace(40, width - 40, 12)
                      for y in np.linspace(40, height - 40, 9)])
    expected = project(reference, check)
    visible = np.all((expected >= 0) & (expected < [width, height]), axis=1)
    cv.setRNGSeed(42)
    fit, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.)
    baseline, _ = cv.findHomography(src, dst, 0)
    if fit is None or baseline is None:
        raise ValueError("Los puntos no permiten ajustar una homografía.")
    for label, matrix in (("RANSAC", fit), ("Sin rechazo robusto", baseline)):
        errors = np.linalg.norm(project(matrix, check[visible]) - expected[visible], axis=1)
        print(f"{label}: mediana {np.median(errors):.2f} px, RMSE {np.sqrt(np.mean(errors**2)):.2f} px")
    print(f"ORB: {len(matches)} coincidencias, {int(inliers.sum())} dentro del consenso de 3 px.")
    matched = cv.drawMatches(a.image, keys_a, b.image, keys_b, matches,
                             None, matchesMask=inliers.ravel().tolist(),
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    placement, _ = a.place("wall", a.wall_anchor, .3)
    moved = project(fit, placement.quad)
    target = project(reference, placement.quad)
    print(f"Error medio de las cuatro esquinas del cuadro: {np.linalg.norm(moved-target, axis=1).mean():.2f} px")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(matched[:, :, ::-1])
    axes[0].set_title("Correspondencias ORB aceptadas por RANSAC")
    axes[1].imshow(b.image[:, :, ::-1])
    for points, label, style in ((target, "Giro conocido", "c-"), (moved, "ORB + RANSAC", "y--")):
        axes[1].plot(*points[[0, 1, 2, 3, 0]].T, style, label=label)
    axes[1].set_title("Traslado de la huella del cuadro")
    axes[1].legend()
    for ax in axes:
        ax.axis("off")
    return fig
