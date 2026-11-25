"""Access ADE20K validation pairs without extracting a dataset to disk."""
from itertools import islice, zip_longest
from pathlib import PurePosixPath

import numpy as np

from .image_io import decode_image
from .photo import LABELS, decode_labels

ROOMS = ("bedroom", "living_room", "dining_room")


class ADEValidation:
    def __init__(self, archive):
        self.archive = archive
        files = archive.namelist()
        metadata = next((name for name in files if name.endswith("/sceneCategories.txt")), None)
        if metadata is None:
            raise ValueError("El ZIP debe incluir sceneCategories.txt.")
        self.root = PurePosixPath(metadata).parent
        self.categories = dict(
            line.split(maxsplit=1)
            for line in archive.read(metadata).decode("utf-8").splitlines() if line.strip()
        )

    def _read(self, stem, folder, suffix):
        name = self.root / folder / "validation" / (stem + suffix)
        try:
            return self.archive.read(name.as_posix())
        except KeyError as error:
            raise ValueError(f"No se encontró {name} en el ZIP.") from error

    def encoded_pair(self, stem):
        return self.encoded_image(stem), self._read(stem, "annotations", ".png")

    def encoded_image(self, stem):
        return self._read(stem, "images", ".jpg")

    def read_pair(self, stem):
        image, annotation = self.encoded_pair(stem)
        return decode_image(image, stem), decode_labels(annotation)

    def sample(self, count, offset=0):
        """Yield IDs and categories in deterministic round-robin order."""
        def eligible(room):
            for stem, category in sorted(self.categories.items()):
                if not stem.startswith("ADE_val_") or category != room:
                    continue
                encoded = self._read(stem, "annotations", ".png")
                mask = decode_labels(encoded)
                if all(np.mean(mask == label) >= .08 for label in LABELS.values()):
                    yield stem, room

        rounds = zip_longest(*(eligible(room) for room in ROOMS))
        pairs = (pair for batch in rounds for pair in batch if pair is not None)
        yield from islice(pairs, offset, offset + count)
