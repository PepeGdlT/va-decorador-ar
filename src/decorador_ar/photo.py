"""Surface segmentation and planar decoration of photographs."""
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np

from .image_io import decode_image, read_image, write_image
from .placement import choose_quad, composite, default_asset, placement_alpha, polygon_coverage

MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
LABELS = {"wall": 1, "floor": 4}
SUPPORT_CLASSES = {
    "table", "coffee table", "desk", "shelf", "bookcase", "counter", "countertop",
    "kitchen island", "cabinet", "chest of drawers", "bench", "buffet", "bar",
}


class Segmenter:
    model_id = MODEL

    def __init__(self, revision=None, *, model_id=MODEL):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        revision = revision or (MODEL_REVISION if model_id == MODEL else None)
        self.model_id = model_id
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id, revision=revision)
        self.model.eval()
        self.revision = self.model.config._commit_hash or revision
        self.indices = {
            name: next(i for i, label in self.model.config.id2label.items()
                       if label.split(",")[0].strip().lower() == name)
            for name in LABELS
        }
        self.support_indices = [i for i, label in self.model.config.id2label.items()
                                if label.strip().lower() in SUPPORT_CLASSES]

    def predict(self, bgr, *, supports=False):
        from PIL import Image

        image = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = resized_class_map(logits, bgr.shape[:2]).cpu().numpy()
            probabilities = logits.softmax(dim=1)[0, list(self.indices.values())].cpu().numpy()
        masks = {
            name: (prediction == index, cv.resize(probability, bgr.shape[1::-1],
                                                   interpolation=cv.INTER_LINEAR))
            for (name, index), probability in zip(self.indices.items(), probabilities)
        }
        if supports:
            masks["support"] = np.isin(prediction, self.support_indices)
        return masks


def resized_class_map(logits, shape, channels_per_batch=4):
    """Resize class logits before argmax without allocating all full-size channels."""
    import torch
    from torch.nn.functional import interpolate

    highest = indices = None
    for start in range(0, logits.shape[1], channels_per_batch):
        resized = interpolate(logits[:, start:start + channels_per_batch], size=shape,
                              mode="bilinear", align_corners=False)
        values, local_ids = resized[0].max(dim=0)
        if highest is None:
            highest, indices = values, local_ids
        else:
            better = values > highest
            indices = torch.where(better, local_ids + start, indices)
            highest = torch.maximum(highest, values)
    return indices


def iou(prediction, truth, label):
    expected = truth == label
    prediction = prediction & (truth != 0)
    union = np.count_nonzero(prediction | expected)
    return float(np.count_nonzero(prediction & expected) / union) if union else None


def read_planes(path, size):
    with Path(path).open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict) or not set(spec) <= LABELS.keys():
        raise ValueError("El archivo de planos admite las claves wall y floor.")
    planes = {}
    for name, corners in spec.items():
        points = np.asarray(corners, dtype=np.float32)
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or np.any(points < 0) or np.any(points >= size)
                or not cv.isContourConvex(points)
                or cv.contourArea(points, oriented=True) < 25):
            raise ValueError(f"Cuatro esquinas válidas necesarias para {name}, en sentido horario.")
        planes[name] = points
    return planes


@dataclass
class PhotoResult:
    measurements: dict
    images: dict[str, np.ndarray]


def analyze_photo(bgr, name, segmenter, *, truth=None, planes=None,
                  textures=None, experimental=False, include_images=True):
    """Analyze image arrays independently of paths, CLI and the ADE20K archive."""
    if truth is not None and truth.shape != bgr.shape[:2]:
        raise ValueError("La anotación debe ser un PNG indexado ADE del tamaño de la foto.")
    planes = planes or {}
    textures = {} if textures is None else textures
    masks = segmenter.predict(bgr)
    decorated = bgr.copy() if include_images else None
    measurements = {"image": name, "model": segmenter.model_id,
                    "revision": segmenter.revision, "surfaces": {}}
    for surface in LABELS:
        region, confidence = masks[surface]
        polygon = planes.get(surface)
        reason = "sin_plano"
        if experimental or polygon is not None:
            if surface not in textures:
                textures[surface] = default_asset(surface, read_image)
            texture = textures[surface]
        if polygon is not None:
            if polygon_coverage(polygon, region) < .97:
                polygon = None
                reason = "mascara_no_respalda_el_plano"
        elif experimental:
            polygon = choose_quad(bgr, region, confidence, surface, texture.shape[1] / texture.shape[0])
            reason = "sin_region_fiable"
        footprint = None
        if polygon is not None:
            if include_images:
                footprint = composite(decorated, texture, polygon, region, surface)
            else:
                footprint = placement_alpha(polygon, region) >= .5
            reason = "plano_indicado" if surface in planes else "propuesta_experimental"
        correct_surface = None
        if footprint is not None and truth is not None:
            labeled = footprint & (truth != 0)
            if labeled.any():
                correct_surface = float(np.mean(truth[labeled] == LABELS[surface]))
        measurements["surfaces"][surface] = {
            "fraction": float(region.mean()),
            "quad_px": polygon.tolist() if polygon is not None else None,
            "decision": reason,
            "iou": iou(region, truth, LABELS[surface]) if truth is not None else None,
            "placement_on_labeled_surface": correct_surface,
        }
    images = {}
    if include_images:
        overlay = bgr.copy()
        for surface, color in (("wall", (44, 177, 246)), ("floor", (169, 211, 61))):
            region = masks[surface][0]
            overlay[region] = np.rint(.55 * bgr[region] + .45 * np.asarray(color)).astype(np.uint8)
        images = {"original": bgr, "superficies": overlay, "decorada": decorated,
                  "pared": masks["wall"][0].astype(np.uint8) * 255,
                  "suelo": masks["floor"][0].astype(np.uint8) * 255}
    return PhotoResult(measurements, images)


def save_photo_result(output, result):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"La salida ya existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".foto-", dir=output.parent) as temporary:
        staging = Path(temporary) / "salida"
        staging.mkdir()
        for name, picture in result.images.items():
            write_image(staging / f"{name}.png", picture)
        (staging / "resultado.json").write_text(
            json.dumps(result.measurements, indent=2, allow_nan=False), encoding="utf-8"
        )
        staging.rename(output)


def process_photo(image, output, segmenter, annotation=None, art=None, rug=None,
                  plane_file=None, experimental=False):
    if Path(output).exists():
        raise FileExistsError(f"La salida ya existe: {output}")
    bgr = read_image(image)
    planes = read_planes(plane_file, bgr.shape[1::-1]) if plane_file is not None else {}
    result = analyze_photo(
        bgr, Path(image).name, segmenter,
        truth=decode_labels(Path(annotation).read_bytes()) if annotation is not None else None,
        planes=planes,
        textures={surface: read_image(path) for surface, path in (("wall", art), ("floor", rug))
                  if path is not None and (experimental or surface in planes)},
        experimental=experimental,
    )
    save_photo_result(output, result)
    return result.measurements


def decode_labels(data):
    from PIL import Image

    with Image.open(BytesIO(data)) as annotation:
        return np.asarray(annotation)
