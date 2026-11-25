import json
from functools import lru_cache
from pathlib import Path

import numpy as np


DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"


@lru_cache(maxsize=1)
def _estimator():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, revision=DEPTH_REVISION,
                                                   use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, revision=DEPTH_REVISION)
    return processor, model.eval()


def estimate_depth(bgr):
    """Estimate depth aligned with the input photograph."""
    import torch
    from PIL import Image

    processor, model = _estimator()
    inputs = processor(images=Image.fromarray(bgr[:, :, ::-1]), return_tensors="pt")
    with torch.inference_mode():
        result = model(**inputs)
        depth = processor.post_process_depth_estimation(result, [bgr.shape[:2]])[0]["predicted_depth"]
    return depth.cpu().numpy()


def score_depth(reference, prediction, minimum, maximum):
    if not 0 < minimum < maximum < np.inf:
        raise ValueError("Intervalo de profundidad no válido.")
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.ndim != 2 or reference.shape != prediction.shape:
        raise ValueError("Se necesitan dos mapas 2D registrados con la misma forma.")
    valid = np.isfinite(reference) & (reference >= minimum) & (reference <= maximum)
    estimated = np.isfinite(prediction) & (prediction > 0)
    common = valid & estimated
    if not np.any(common):
        raise ValueError("No hay píxeles válidos comunes.")
    truth = reference[common].astype(np.float64)
    error = prediction[common].astype(np.float64) - truth
    return {
        "reference_pixels": int(valid.sum()),
        "evaluated_pixels": int(common.sum()),
        "coverage": float(common.sum() / valid.sum()),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "abs_rel": float(np.mean(np.abs(error) / truth)),
    }


def evaluate_files(reference, prediction, output, minimum, maximum):
    scores = score_depth(np.load(reference, allow_pickle=False),
                         np.load(prediction, allow_pickle=False), minimum, maximum)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(scores, stream, indent=2, allow_nan=False)
    return scores
