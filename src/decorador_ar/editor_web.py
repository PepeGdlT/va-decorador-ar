"""Serve the notebook's pointer-controlled editor on the local machine."""
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock, Thread
from urllib.parse import parse_qs, urlsplit

import cv2 as cv
import numpy as np

from .composition import CHOICES, Composition
from .image_io import encode_image, write_image
from .photo_furniture import MODELS
from .raised_surfaces import detect_supports


ASSETS = Path(__file__).with_name("editor_assets")


class EditorSession:
    def __init__(self, scenes, output):
        self.rooms = {name: Composition(scene) for name, scene in scenes.items()}
        self.output = Path(output)
        self.scanned = set()
        self.frames = {}

    def dispatch(self, action, data):
        room = data["room"]
        composition = self.rooms[room]
        surface = data.get("surface", "wall")
        name = data.get("piece", "paisaje")
        lighting = data.get("lighting", True)
        if not isinstance(lighting, bool) or surface not in CHOICES:
            raise ValueError("Selección inválida.")
        frame_key = room, lighting
        scene = composition.scene
        supports = tuple(composition.supports())
        i = int(data.get("support", -1))
        support = supports[i][1] if 0 <= i < len(supports) and surface == "raised" else None
        if action == "detect" and room not in self.scanned:
            composition.detected_supports = detect_supports(scene)
            self.scanned.add(room)
        if action == "mark":
            composition.add_marked_support(data["corners"], float(data["height"]))
        if action == "remove":
            index = int(data["index"])
            if not 0 <= index < len(composition.items):
                raise ValueError("Selecciona una pieza de la composición.")
            composition.remove(index)
        if action == "save":
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = self.output / f"composicion-{room}-{stamp}.png"
            write_image(path, composition.render(lighting=lighting), exclusive=True)
            return {"path": str(path)}
        if action in {"preview", "add"}:
            anchor = tuple(map(float, data["point"]))
            if len(anchor) != 2 or not all(0 <= n < limit for n, limit in zip(anchor, scene.image.shape[1::-1])):
                raise ValueError("Elige un punto dentro de la foto.")
            scale, angle = float(data["scale"]), float(data["angle"])
            if not .25 <= scale <= 1.6 or not -90 <= angle <= 90:
                raise ValueError("Escala o giro fuera de rango.")
            pending = composition.prepare(surface, name, anchor, scale=scale, angle=angle,
                                          vertical=data.get("vertical", False), support=support)
            if action == "preview":
                base = self.frame(composition, frame_key)
                draft = composition.render(pending, lighting=lighting)
                return encode_image(cv.addWeighted(base, .45, draft, .55, 0), ".jpg")
            composition.items.append(pending)
        if action in {"add", "remove"}:
            for key in ((room, False), (room, True)):
                self.frames.pop(key, None)
        if action == "photo":
            return encode_image(self.frame(composition, frame_key), ".jpg")
        return self.state(composition, room, surface)

    def frame(self, composition, key):
        if key not in self.frames:
            self.frames[key] = composition.render(lighting=key[1])
        return self.frames[key]

    def state(self, composition, room, surface):
        scene = composition.scene
        return {
            "size": scene.image.shape[1::-1],
            "anchor": scene.wall_anchor if surface == "wall" else scene.floor_anchor,
            "items": [label for label, _ in composition.items],
            "supports": [{"label": label, "outline": support.outline.tolist(),
                          "center": support.center.tolist()} for label, support in composition.supports()],
            "scanned": room in self.scanned,
        }


def start_editor(scenes, initial, output):
    """Start a session bound to loopback. Return a handle that can be closed."""
    session = EditorSession(scenes, output)
    token = token_urlsafe(24)
    mount = f"/{token}/"
    lock = Lock()
    pages = {
        "": ("index.html", "text/html; charset=utf-8"),
        "editor.css": ("editor.css", "text/css; charset=utf-8"),
        "editor.js": ("editor.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, data, status=200):
            self.reply(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

        def do_GET(self):
            path = urlsplit(self.path)
            route = path.path.removeprefix(mount) if path.path.startswith(mount) else None
            if route in pages:
                file, content_type = pages[route]
                self.reply((ASSETS / file).read_bytes(), content_type)
            elif route == "config":
                self.json({"rooms": list(scenes), "initial": initial, "choices": CHOICES,
                           "models": list(MODELS)})
            elif route == "photo":
                query = parse_qs(path.query)
                with lock:
                    try:
                        photo = session.dispatch("photo", {
                            "room": query["room"][0], "lighting": query.get("lit") != ["0"],
                        })
                    except (ValueError, KeyError):
                        return self.json({"error": "Vista no disponible."}, 400)
                self.reply(photo, "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            action = self.path.removeprefix(mount) if self.path.startswith(mount) else None
            if action not in {"state", "preview", "add", "remove", "detect", "mark", "save"}:
                return self.send_error(404)
            if (self.headers.get_content_type() != "application/json"
                    or self.headers.get("Origin") not in (None, origin)):
                return self.send_error(403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4096:
                    return self.send_error(400)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    return self.send_error(400)
                with lock:
                    result = session.dispatch(action, data)
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.json({"error": str(error)}, 422)
            if isinstance(result, bytes):
                self.reply(result, "image/jpeg")
            else:
                self.json(result)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = f"http://127.0.0.1:{server.server_port}"
    server.url = origin + mount
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def display_editor(server):
    from IPython.display import HTML, display

    url = escape(server.url, quote=True)
    display(HTML(f'<p><a href="{url}" target="_blank" rel="noopener">'
                 'Abrir editor en el navegador</a></p>'
                 f'<iframe title="Editor de habitaciones" src="{url}" '
                 'style="width:100%;height:690px;border:0"></iframe>'))
