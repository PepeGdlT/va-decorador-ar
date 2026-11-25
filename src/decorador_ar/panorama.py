"""Perspective views and a small reproducible subset of the ZInD sample tour."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import cv2 as cv
import numpy as np

from .image_io import decode_image, encode_image
from .rectification import room_planes


ZIND_RAW = "https://raw.githubusercontent.com/zillow/zind/main/sample_tour/000/panos"

# Public one-home sample shipped by the official ZInD repository.  These are
# not a substitute for the train/validation partitions of the full dataset.
ZIND_VIEWS = (
    ("floor_01_partial_room_01_pano_14.jpg", 55, 13, "bonus_01.jpg"),
    ("floor_01_partial_room_01_pano_14.jpg", 80, 15, "bonus_02.jpg"),
    ("floor_01_partial_room_09_pano_2.jpg", 100, 12, "salon_01.jpg"),
    ("floor_01_partial_room_12_pano_3.jpg", 125, 14, "salon_02.jpg"),
    ("floor_01_partial_room_07_pano_18.jpg", 0, 15, "dormitorio_01.jpg"),
    ("floor_01_partial_room_11_pano_25.jpg", 0, 15, "dormitorio_02.jpg"),
)

# Additional viewpoints for testing furniture, including a third bedroom.
FURNITURE_VIEWS = (
    ("floor_01_partial_room_01_pano_15.jpg", 145, 25, "bonus_03.jpg"),
    ("floor_01_partial_room_09_pano_5.jpg", 110, 25, "salon_03.jpg"),
    ("floor_01_partial_room_09_pano_6.jpg", 150, 25, "pasillo_01.jpg"),
    ("floor_01_partial_room_07_pano_19.jpg", 35, 25, "dormitorio_03.jpg"),
    ("floor_01_partial_room_19_pano_28.jpg", 0, 25, "dormitorio_04.jpg"),
    ("floor_01_partial_room_19_pano_27.jpg", 115, 25, "pasillo_02.jpg"),
)

SUPPORT_VIEWS = (
    ("floor_01_partial_room_06_pano_11.jpg", 60, 24, "cocina_01.jpg"),
    ("floor_01_partial_room_06_pano_12.jpg", 0, 24, "cocina_02.jpg"),
)

# Control clicks in photo pixels and long side of the rug in camera-height units.
CONTROL_CASES = {
    "bonus_01": ((600, 270), (560, 540), .48),
    "bonus_02": ((380, 255), (475, 520), .48),
    "salon_01": ((600, 265), (600, 540), .48),
    "salon_02": ((365, 255), (460, 560), .36),
    "dormitorio_01": ((375, 270), (620, 565), .48),
    "dormitorio_02": ((260, 275), (650, 585), .36),
}


class ZindLayout:
    def __init__(self, root):
        path = Path(root) / "data/raw/zind-sample/zind_data.json"
        layout = json.loads(path.read_text(encoding="utf-8"))
        self.panoramas = {
            pano["image_path"]: pano
            for room in layout["merger"]["floor_01"].values()
            for partial in room.values()
            for pano in partial.values()
        }
        self.views = {name: (source, yaw, pitch)
                      for source, yaw, pitch, name in (*ZIND_VIEWS, *FURNITURE_VIEWS, *SUPPORT_VIEWS)}

    def _view(self, photo):
        name = Path(photo).name
        if name not in self.views:
            raise ValueError(f"No hay parámetros de cámara para {name}.")
        return self.views[name]

    def intrinsics(self, photo):
        _, yaw, pitch = self._view(photo)
        focal, cx, cy, _ = view_camera(yaw, pitch)
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    def planes(self, photo):
        source, yaw, pitch = self._view(photo)
        pano = self.panoramas[f"panos/{source}"]
        vertices = np.asarray(pano["layout_visible"]["vertices"], dtype=float)
        # ZInD's azimuth uses -x; our perspective camera uses +x to the right.
        ground = np.column_stack((
            -vertices[:, 0], np.full(len(vertices), -pano["camera_height"]), vertices[:, 1]
        ))
        focal, cx, cy, rotation = view_camera(yaw, pitch)
        camera = np.array([[focal, 0, cx], [0, -focal, cy], [0, 0, 1]]) @ rotation.T
        return room_planes(ground, pano["ceiling_height"], camera, rotation[:, 0])


def view_camera(yaw_deg, pitch_deg, hfov_deg=82, size=(960, 720)):
    """Return focal length, principal point and camera-to-panorama rotation."""
    width, height = size
    if not 1 < hfov_deg < 179 or width < 2 or height < 2:
        raise ValueError("Campo de visión o tamaño de salida no válido.")
    focal = .5 * width / np.tan(.5 * np.deg2rad(hfov_deg))
    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
    yaw_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                           [0, 1, 0],
                           [-np.sin(yaw), 0, np.cos(yaw)]], np.float32)
    pitch_matrix = np.array([[1, 0, 0],
                             [0, np.cos(pitch), -np.sin(pitch)],
                             [0, np.sin(pitch), np.cos(pitch)]], np.float32)
    return focal, (width - 1) / 2, (height - 1) / 2, yaw_matrix @ pitch_matrix


def perspective_view(panorama, yaw_deg, pitch_deg=0, hfov_deg=82, size=(960, 720)):
    """Project an equirectangular panorama onto a rectilinear camera view."""
    width, height = size
    focal, cx, cy, rotation = view_camera(yaw_deg, pitch_deg, hfov_deg, size)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    rays = np.stack(((x - cx) / focal, -(y - cy) / focal, np.ones_like(x)), axis=-1)
    rays = rays @ rotation.T
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arctan2(rays[..., 1],
                          np.hypot(rays[..., 0], rays[..., 2]))
    source_h, source_w = panorama.shape[:2]
    map_x = ((longitude / (2 * np.pi) + .5) * (source_w - 1)).astype(np.float32)
    map_y = ((.5 - latitude / np.pi) * (source_h - 1)).astype(np.float32)
    return cv.remap(panorama, map_x, map_y, cv.INTER_LINEAR,
                    borderMode=cv.BORDER_WRAP)


def _download(url, path):
    request = Request(url, headers={"User-Agent": "va-decorador-ar academic project"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) < 10_000:
        raise ValueError(f"Descarga incompleta: {url}")
    _replace_bytes(path, data)


def _replace_bytes(path, data):
    # A download or render that stops midway must not leave a valid-looking cache entry.
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_zind_sample(root, *, force=False, views=ZIND_VIEWS):
    """Download and cache the selected views from the official ZInD sample."""
    root = Path(root)
    raw = root / "data" / "raw" / "zind-sample"
    output = root / "data" / "processed" / "zind-rectified"
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    layout = raw / "zind_data.json"
    if force or not layout.is_file():
        _download(ZIND_RAW.rsplit("/", 1)[0] + "/zind_data.json", layout)
    panoramas = {}
    for filename, yaw, pitch, output_name in views:
        target = output / output_name
        if target.is_file() and not force:
            continue
        source = raw / filename
        if filename not in panoramas:
            if force or not source.is_file():
                _download(f"{ZIND_RAW}/{filename}", source)
            panoramas[filename] = decode_image(source.read_bytes(), source)
        view = perspective_view(panoramas[filename], yaw, pitch)
        _replace_bytes(target, encode_image(view, ".jpg", [cv.IMWRITE_JPEG_QUALITY, 93]))
    return [output / name for *_, name in views]
