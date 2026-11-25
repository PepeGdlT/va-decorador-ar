import argparse
from pathlib import Path
from zipfile import BadZipFile

import cv2 as cv
import numpy as np

from .calibration import calibrate
from .demo import run_demo
from .depth import evaluate_files
from .geometry import Camera
from .image_io import write_image
from .marker import DICTIONARY_SIZE, marker_image
from .render import furniture
from .video import augment_video


def parser():
    root = argparse.ArgumentParser(description="Decorador AR")
    commands = root.add_subparsers(dest="command", required=True)
    marker = commands.add_parser("marker", help="Genera un marcador PNG con margen blanco")
    marker.add_argument("output", type=Path)
    marker.add_argument("--id", type=int, choices=range(DICTIONARY_SIZE), default=0)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("images", type=Path)
    calibration.add_argument("output", type=Path)
    calibration.add_argument("--columns", type=int, default=9)
    calibration.add_argument("--rows", type=int, default=6)
    calibration.add_argument("--square-m", type=float, required=True)
    video = commands.add_parser("video", help="Procesa un vídeo grabado")
    video.add_argument("source", type=Path)
    video.add_argument("camera", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--metrics", type=Path)
    video.add_argument("--id", type=int, choices=range(DICTIONARY_SIZE), default=0)
    video.add_argument("--side", type=float, required=True, help="Lado negro en metros")
    video.add_argument("--size", nargs=3, type=float, default=(0.65, 0.42, 0.44),
                       metavar=("WIDTH", "DEPTH", "HEIGHT"))
    video.add_argument("--xy", nargs=2, type=float, default=(0.34, 0.12))
    video.add_argument("--yaw", type=float, default=10)
    video.add_argument("--alpha", type=float, default=1, help="1 desactiva el suavizado")
    video.add_argument("--max-error", type=float, default=3)
    demo = commands.add_parser("demo", help="Genera y procesa una secuencia sintética")
    demo.add_argument("folder", type=Path)
    depth = commands.add_parser("depth-score", help="Compara dos mapas .npy en metros")
    depth.add_argument("reference", type=Path)
    depth.add_argument("prediction", type=Path)
    depth.add_argument("output", type=Path)
    depth.add_argument("--min-m", type=float, default=0.1)
    depth.add_argument("--max-m", type=float, default=10)
    photo = commands.add_parser("photo", help="Distingue pared y suelo y decora una fotografía")
    photo.add_argument("image", type=Path)
    photo.add_argument("output", type=Path)
    photo.add_argument("--annotation", type=Path)
    photo.add_argument("--art", type=Path)
    photo.add_argument("--rug", type=Path)
    photo.add_argument("--revision")
    placement = photo.add_mutually_exclusive_group()
    placement.add_argument("--planes", type=Path, help="JSON con cuatro esquinas por superficie")
    placement.add_argument("--experimental", action="store_true", help="Ensaya ubicación automática aproximada")
    study = commands.add_parser("photo-study", help="Prueba fotos reales del ZIP oficial ADE20K")
    study.add_argument("archive", type=Path)
    study.add_argument("output", type=Path)
    study.add_argument("--count", type=int, default=8)
    study.add_argument("--offset", type=int, default=0)
    study.add_argument("--metrics-only", action="store_true", help="Guarda métricas y metadatos, sin imágenes")
    return root


def unused(path, suffix):
    if path.suffix.lower() != suffix or path.exists():
        raise ValueError(f"Se necesita un archivo nuevo {suffix}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _marker(args):
    unused(args.output, ".png")
    write_image(args.output, marker_image(args.id), exclusive=True)


def _calibrate(args):
    unused(args.output, ".npz")
    camera, rms, views = calibrate(args.images, args.columns, args.rows, args.square_m)
    with args.output.open("xb") as stream:
        camera.save(stream, rms)
    print(f"{views} vistas | RMS de calibración: {rms:.3f} px")


def _depth(args):
    print(evaluate_files(args.reference, args.prediction, args.output, args.min_m, args.max_m))


def _photo(args):
    from .photo import Segmenter, process_photo

    if not args.image.is_file():
        raise FileNotFoundError(f"No se encontró {args.image}")
    if args.output.exists():
        raise FileExistsError(f"La salida ya existe: {args.output}")
    print(process_photo(args.image, args.output, Segmenter(args.revision),
                        args.annotation, args.art, args.rug,
                        plane_file=args.planes, experimental=args.experimental))


def _study(args):
    from .photo_study import study

    for result in study(args.archive, args.output, args.count, args.offset,
                        metrics_only=args.metrics_only):
        print(result)


def _demo(args):
    for name, (count, accepted) in run_demo(args.folder).items():
        print(f"{name}: {accepted}/{count} fotogramas anclados")


def _video(args):
    if not np.isfinite([*args.size, *args.xy, args.yaw]).all() or min(args.size) <= 0:
        raise ValueError("Dimensiones, posición o giro del mueble no válidos.")
    metrics = args.metrics or args.output.with_suffix(".csv")
    unused(args.output, ".mp4")
    unused(metrics, ".csv")
    count, accepted = augment_video(
        args.source, args.output, metrics, Camera.load(args.camera), args.id, args.side,
        furniture(*args.size, *args.xy, args.yaw), args.alpha, args.max_error,
    )
    print(f"{accepted}/{count} fotogramas anclados | {metrics}")


def run(args):
    handlers = {
        "calibrate": _calibrate, "demo": _demo, "depth-score": _depth,
        "marker": _marker, "photo": _photo, "photo-study": _study, "video": _video,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, OSError, KeyError, BadZipFile, cv.error) as error:
        cli.exit(2, f"Error: {error}\n")
