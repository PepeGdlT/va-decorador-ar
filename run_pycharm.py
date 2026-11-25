"""Repeatable entry points for the PyCharm run configurations."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "data" / "raw" / "ADEChallengeData2016.zip"
NOTEBOOK = ROOT / "notebooks" / "Decorador_AR.ipynb"


def output_folder(task):
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    return Path(mkdtemp(prefix=f"pycharm-{task}-", dir=results))


def require_dataset():
    if not DATASET.is_file():
        raise SystemExit(f"Falta {DATASET}. La descarga oficial está en data/README.md.")


def notebook(output):
    import nbformat
    from nbclient import NotebookClient

    document = nbformat.read(NOTEBOOK, as_version=4)
    run = NotebookClient(
        document, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    # Use the same Python that launched the configuration, even if a user
    # kernel named python3 points to a different environment.
    run.create_kernel_manager().kernel_spec.argv[0] = sys.executable
    run.execute(env={**os.environ, "DECORADOR_AR_NOTEBOOK_OUTPUT": str(output)})
    nbformat.write(document, output / "decorador-ejecutado.ipynb")


def photo(output):
    from decorador_ar.ade import ADEValidation
    from decorador_ar.photo import Segmenter, process_photo

    stem = "ADE_val_00000506"
    image = output / f"{stem}.jpg"
    annotation = output / f"{stem}.png"
    with ZipFile(DATASET) as dataset:
        jpg, png = ADEValidation(dataset).encoded_pair(stem)
        image.write_bytes(jpg)
        annotation.write_bytes(png)
    print(process_photo(image, output / "montaje", Segmenter(), annotation,
                        plane_file=ROOT / "assets" / "planes" / f"{stem}.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("notebook", "photo", "study", "demo", "marker"))
    task = parser.parse_args().task
    if task in ("photo", "study"):
        require_dataset()
    output = output_folder(task)
    print(f"Salida: {output}", flush=True)
    if task == "notebook":
        notebook(output)
    elif task == "photo":
        photo(output)
    else:
        from decorador_ar.__main__ import parser as cli_parser, run

        commands = {
            "study": ["photo-study", str(DATASET), str(output / "estudio"), "--count", "9"],
            "demo": ["demo", str(output / "video")],
            "marker": ["marker", str(output / "aruco.png")],
        }
        run(cli_parser().parse_args(commands[task]))


if __name__ == "__main__":
    main()
