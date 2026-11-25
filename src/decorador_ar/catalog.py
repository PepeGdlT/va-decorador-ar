"""Open Access textures and hand-curated combinations for the ZInD rooms."""
from functools import lru_cache

import cv2 as cv
import numpy as np

from .image_io import read_image
from .placement import ASSETS, default_asset

ART_FILES = {
    "ola": "wave-met.jpg", "bodegon": "still-life-met.jpg",
    "lirios": "irises-met.jpg", "cipreses": "cypresses-met.jpg",
}
ART = ("paisaje", *ART_FILES)
RUGS = ("clasica", "geometrica", "floral", "medallon", "roja")

# artwork, rug, artwork size, rug size, rug orientation
DESIGNS = {
    "bonus_01": ("paisaje", "clasica", .63, .65, False),
    "bonus_02": ("ola", "geometrica", .60, .56, False),
    "salon_01": ("bodegon", "floral", .58, .66, False),
    "salon_02": ("ola", "geometrica", .52, .40, False),
    "dormitorio_01": ("bodegon", "clasica", .38, .49, True),
    "dormitorio_02": ("ola", "floral", .35, .50, False),
}

# Aesthetic anchors are separate from the fixed rectification checks.
DESIGN_ANCHORS = {
    "bonus_01": ((520, 225), (560, 540)),
    "bonus_02": ((355, 210), (475, 520)),
    "salon_01": ((530, 225), (600, 550)),
    "salon_02": ((345, 222), (460, 560)),
    "dormitorio_01": ((375, 270), (550, 530)),
    "dormitorio_02": ((245, 260), (640, 585)),
}


@lru_cache(maxsize=16)
def texture(surface, name):
    if (surface, name) in (("wall", "paisaje"), ("floor", "clasica")):
        return default_asset(surface, read_image)
    if surface == "wall" and name in ART_FILES:
        image = read_image(ASSETS / ART_FILES[name])
        matted = cv.copyMakeBorder(image, 25, 25, 25, 25, cv.BORDER_CONSTANT,
                                   value=(211, 220, 225))
        return cv.copyMakeBorder(matted, 12, 12, 12, 12, cv.BORDER_CONSTANT,
                                 value=(39, 43, 47))
    if surface == "floor" and name == "floral":
        image = read_image(ASSETS / "rug-floral-met.jpg")[16:-9, 9:-10]
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    if surface == "floor" and name in ("geometrica", "medallon", "roja"):
        file, corners = {
            "geometrica": ("rug-geometric-met.jpg", [[34, 23], [570, 39], [567, 392], [30, 408]]),
            "medallon": ("rug-medallion-met.jpg", [[111, 15], [473, 13], [576, 310], [20, 311]]),
            "roja": ("rug-runner-met.jpg", [[119, 44], [445, 71], [542, 365], [110, 350]]),
        }[name]
        image = read_image(ASSETS / file)
        rectangle = np.float32([[0, 0], [539, 0], [539, 359], [0, 359]])
        rug = cv.warpPerspective(image, cv.getPerspectiveTransform(np.float32(corners), rectangle),
                                 (540, 360))
        if name == "roja":
            rug = cv.copyMakeBorder(rug, 12, 12, 12, 12, cv.BORDER_CONSTANT, value=(39, 48, 83))
        return rug
    raise ValueError(f"Pieza de catálogo desconocida: {name} ({surface}).")
