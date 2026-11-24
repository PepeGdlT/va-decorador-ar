import json
from pathlib import Path

import numpy as np


def score_depth(reference, prediction, minimum, maximum):
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
