"""Read and encode images without passing Unicode paths to OpenCV."""
from pathlib import Path

import cv2 as cv
import numpy as np


def decode_image(data, name, flags=cv.IMREAD_COLOR):
    image = cv.imdecode(np.frombuffer(data, dtype=np.uint8), flags) if data else None
    if image is None:
        raise ValueError(f"No se pudo leer {name}")
    return image


def read_image(path, flags=cv.IMREAD_COLOR):
    return decode_image(Path(path).read_bytes(), path, flags)


def encode_image(image, suffix, params=()):
    ok, encoded = cv.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {suffix}.")
    return encoded.tobytes()


def write_image(path, image, *, exclusive=False):
    path = Path(path)
    data = encode_image(image, path.suffix.lower())
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(data)
