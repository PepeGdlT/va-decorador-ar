"""Reproducible development sample from the official ADE20K ZIP."""
import csv
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .photo import LABELS, Segmenter, process_photo

ROOMS = ("bedroom", "living_room", "dining_room")


def study(archive, output, count, offset=0):
    from PIL import Image

    if count <= 0 or offset < 0:
        raise ValueError("Usa un número de fotos positivo y un desplazamiento no negativo.")
    output = Path(output)
    with ZipFile(archive) as source:
        category_file = next((name for name in source.namelist()
                              if name.endswith("/sceneCategories.txt")), None)
        if category_file is None:
            raise ValueError("El ZIP debe incluir sceneCategories.txt.")
        categories = dict(line.split(maxsplit=1)
                          for line in source.read(category_file).decode().splitlines()
                          if line.strip())
        names = sorted(name for name in source.namelist()
                       if "/annotations/validation/ADE_val_" in name and name.endswith(".png"))
        by_room = {room: [] for room in ROOMS}
        for name in names:
            room = categories.get(Path(name).stem)
            if room in by_room:
                by_room[room].append(name)

        def eligible(paths):
            for name in paths:
                encoded = source.read(name)
                with Image.open(BytesIO(encoded)) as image:
                    mask = np.asarray(image)
                if all(np.mean(mask == label) >= 0.08 for label in LABELS.values()):
                    yield name, encoded

        rounds = zip_longest(*(eligible(by_room[room]) for room in ROOMS))
        selected = []
        for batch in rounds:
            selected.extend(pair for pair in batch if pair is not None)
            if len(selected) >= offset + count:
                break
        selected = selected[offset:offset + count]
        if len(selected) < count:
            raise ValueError(f"Solo se encontraron {len(selected)} pares elegibles.")
        segmenter = Segmenter()
        inputs = output / "entradas"
        inputs.mkdir(parents=True, exist_ok=False)
        rows = []
        for name, encoded in selected:
            stem = Path(name).stem
            annotation, image = inputs / f"{stem}.png", inputs / f"{stem}.jpg"
            photograph = name.replace("/annotations/", "/images/").removesuffix(".png") + ".jpg"
            annotation.write_bytes(encoded)
            image.write_bytes(source.read(photograph))
            result = process_photo(image, output / stem, segmenter, annotation, experimental=True)
            rows.append({
                "image": stem,
                "scene": categories[stem],
                "wall_iou": result["surfaces"]["wall"]["iou"],
                "floor_iou": result["surfaces"]["floor"]["iou"],
                "picture_placed": result["surfaces"]["wall"]["quad_px"] is not None,
                "rug_placed": result["surfaces"]["floor"]["quad_px"] is not None,
                "picture_on_wall": result["surfaces"]["wall"]["placement_on_labeled_surface"],
                "rug_on_floor": result["surfaces"]["floor"]["placement_on_labeled_surface"],
            })
    with (output / "resumen.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
