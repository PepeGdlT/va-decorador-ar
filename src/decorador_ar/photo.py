"""Surface segmentation and planar decoration of photographs."""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from .placement import choose_quad, composite, default_asset

MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
LABELS = {"wall": 1, "floor": 4}


class Segmenter:
    def __init__(self, revision=None):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        revision = revision or MODEL_REVISION
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(MODEL, revision=revision, use_fast=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL, revision=revision)
        self.model.eval()
        self.revision = self.model.config._commit_hash or revision
        self.indices = {
            name: next(i for i, label in self.model.config.id2label.items()
                       if label.split(",")[0].strip().lower() == name)
            for name in LABELS
        }

    def predict(self, bgr):
        from PIL import Image

        image = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[bgr.shape[:2]]
            )[0].cpu().numpy()
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        return {
            name: (prediction == index, cv.resize(probabilities[index], bgr.shape[1::-1],
                                                   interpolation=cv.INTER_LINEAR))
            for name, index in self.indices.items()
        }


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


def process_photo(image, output, segmenter, annotation=None, art=None, rug=None,
                  plane_file=None, experimental=False):
    from PIL import Image

    bgr = read_image(image)
    planes = read_planes(plane_file, bgr.shape[1::-1]) if plane_file is not None else {}
    truth = np.asarray(Image.open(annotation)) if annotation is not None else None
    if truth is not None and truth.shape != bgr.shape[:2]:
        raise ValueError("La anotación debe ser un PNG indexado ADE del tamaño de la foto.")
    masks = segmenter.predict(bgr)
    decorated, overlay = bgr.copy(), bgr.copy()
    measurements = {"image": Path(image).name, "model": MODEL,
                    "revision": segmenter.revision, "surfaces": {}}
    for surface, asset, color in (("wall", art, (44, 177, 246)),
                                  ("floor", rug, (169, 211, 61))):
        region, confidence = masks[surface]
        overlay[region] = np.rint(.55 * bgr[region] + .45 * np.asarray(color)).astype(np.uint8)
        polygon = planes.get(surface)
        reason = "sin_plano"
        if experimental or polygon is not None:
            texture = read_image(asset) if asset else default_asset(surface, read_image)
        if polygon is not None:
            area = cv.fillConvexPoly(np.zeros(region.shape, np.uint8),
                                     np.rint(polygon).astype(np.int32), 1).astype(bool)
            if np.mean(region[area]) < .97:
                polygon = None
                reason = "mascara_no_respalda_el_plano"
        elif experimental:
            polygon = choose_quad(bgr, region, confidence, surface, texture.shape[1] / texture.shape[0])
            reason = "sin_region_fiable"
        footprint = None
        if polygon is not None:
            footprint = composite(decorated, texture, polygon, region, surface)
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, picture in (("original", bgr), ("superficies", overlay), ("decorada", decorated),
                          ("pared", masks["wall"][0].astype(np.uint8) * 255),
                          ("suelo", masks["floor"][0].astype(np.uint8) * 255)):
        ok, encoded = cv.imencode(".png", picture)
        if not ok:
            raise ValueError(f"No se pudo guardar {name}.")
        encoded.tofile(output / f"{name}.png")
    (output / "resultado.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    return measurements


def read_image(path):
    image = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer {path}")
    return image
