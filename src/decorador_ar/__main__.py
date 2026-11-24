import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import calibrate
from .demo import make_demo
from .depth import evaluate_files
from .geometry import Camera
from .marker import marker_image
from .render import furniture
from .video import augment_video


def parser():
    root = argparse.ArgumentParser(description="Decorador AR")
    commands = root.add_subparsers(dest="command", required=True)
    marker = commands.add_parser("marker", help="Genera un marcador PNG con margen blanco")
    marker.add_argument("output", type=Path)
    marker.add_argument("--id", type=int, choices=range(50), default=0)
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
    video.add_argument("--id", type=int, choices=range(50), default=0)
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
    return root


def unused(path, suffix):
    if path.suffix.lower() != suffix or path.exists():
        raise ValueError(f"Se necesita un archivo nuevo {suffix}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def run(args):
    if args.command == "marker":
        unused(args.output, ".png")
        ok, encoded = cv.imencode(".png", marker_image(args.id))
        if not ok:
            raise ValueError("No se pudo codificar el marcador.")
        encoded.tofile(args.output)
    elif args.command == "calibrate":
        if min(args.columns, args.rows) < 3 or not 0 < args.square_m < np.inf:
            raise ValueError("Dimensiones del tablero no válidas.")
        unused(args.output, ".npz")
        camera, rms, views = calibrate(args.images, args.columns, args.rows, args.square_m)
        camera.save(args.output, rms)
        print(f"{views} vistas | RMS de calibración: {rms:.3f} px")
    elif args.command == "depth-score":
        if not 0 < args.min_m < args.max_m < np.inf:
            raise ValueError("Intervalo de profundidad no válido.")
        print(evaluate_files(args.reference, args.prediction, args.output, args.min_m, args.max_m))
    elif args.command == "photo":
        from .photo import Segmenter, process_photo

        print(process_photo(args.image, args.output, Segmenter(args.revision),
                            args.annotation, args.art, args.rug,
                            plane_file=args.planes, experimental=args.experimental))
    elif args.command == "photo-study":
        from .photo_study import study

        for result in study(args.archive, args.output, args.count, args.offset):
            print(result)
    elif args.command == "demo":
        if args.folder.exists() and any(args.folder.iterdir()):
            raise ValueError("Usa una carpeta nueva o vacía para la demo.")
        source, camera, side = make_demo(args.folder)
        pieces = list(furniture(0.32, 0.24, 0.24, -0.34, 0.01, 10))
        for name, alpha in (("sin_filtro", 1.0), ("suavizado", 0.55)):
            count, accepted = augment_video(source, args.folder / f"{name}.mp4",
                                            args.folder / f"{name}.csv", camera, 0, side,
                                            pieces, alpha, 3)
            print(f"{name}: {accepted}/{count} fotogramas anclados")
    else:
        if not (np.isfinite([*args.size, *args.xy, args.side, args.yaw,
                             args.alpha, args.max_error]).all()
                and 0 < args.alpha <= 1 and min(*args.size, args.side, args.max_error) > 0):
            raise ValueError("Dimensiones, filtro o umbral no válidos.")
        metrics = args.metrics or args.output.with_suffix(".csv")
        unused(args.output, ".mp4")
        unused(metrics, ".csv")
        count, accepted = augment_video(
            args.source, args.output, metrics, Camera.load(args.camera), args.id, args.side,
            list(furniture(*args.size, *args.xy, args.yaw)), args.alpha, args.max_error,
        )
        print(f"{accepted}/{count} fotogramas anclados | {metrics}")


if __name__ == "__main__":
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, OSError, KeyError, cv.error) as error:
        cli.exit(2, f"Error: {error}\n")
