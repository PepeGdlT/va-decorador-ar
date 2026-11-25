"""Run reproducible experiments over ADE20K validation images."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from .ade import ADEValidation
from .image_io import decode_image
from .photo import Segmenter, analyze_photo, decode_labels, save_photo_result

FIELDS = ("image", "scene", "wall_iou", "floor_iou", "picture_placed",
          "rug_placed", "picture_on_wall", "rug_on_floor")


def study(archive, output, count, offset=0, *, metrics_only=False, segmenter=None):
    if count <= 0 or offset < 0:
        raise ValueError("Usa un número de fotos positivo y un desplazamiento no negativo.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"La salida ya existe: {output}")

    rows = []
    textures = {}
    with ZipFile(archive) as source:
        dataset = ADEValidation(source)
        selected = list(dataset.sample(count, offset))
        if len(selected) < count:
            raise ValueError(f"Solo se encontraron {len(selected)} pares elegibles.")
        if segmenter is None:
            segmenter = Segmenter()
        output.mkdir(parents=True, exist_ok=False)
        (output / "estudio.json").write_text(json.dumps({
            "archive": Path(archive).name, "split": "validation",
            "requested_count": count, "offset": offset,
            "model": segmenter.model_id, "revision": segmenter.revision,
            "placement": "experimental", "metrics_only": metrics_only,
        }, indent=2), encoding="utf-8")
        inputs = output / "entradas"
        if not metrics_only:
            inputs.mkdir()
        with (output / "resumen.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for stem, room in selected:
                name = stem + ".jpg"
                encoded, annotation = dataset.encoded_pair(stem)
                result = analyze_photo(
                    decode_image(encoded, name), name, segmenter, truth=decode_labels(annotation),
                    textures=textures, experimental=True, include_images=not metrics_only,
                )
                if not metrics_only:
                    save_photo_result(output / stem, result)
                    (inputs / name).write_bytes(encoded)
                    (inputs / (stem + ".png")).write_bytes(annotation)
                surfaces = result.measurements["surfaces"]
                wall, floor = surfaces["wall"], surfaces["floor"]
                row = dict(zip(FIELDS, (
                    stem, room, wall["iou"], floor["iou"], wall["quad_px"] is not None,
                    floor["quad_px"] is not None, wall["placement_on_labeled_surface"],
                    floor["placement_on_labeled_surface"],
                )))
                writer.writerow(row)
                stream.flush()
                rows.append(row)
    return rows
