"""Notebook controls for room selection, placement and PNG export."""
from datetime import datetime
from html import escape
from pathlib import Path

import cv2 as cv
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from .composition import CHOICES, Composition
from .image_io import encode_image, write_image
from .photo_furniture import MODELS
from .raised_surfaces import detect_supports


class CompositionEditor:
    def __init__(self, scenes, initial, output):
        self.rooms = {name: Composition(scene) for name, scene in scenes.items()}
        self.output = Path(output)
        self.room = widgets.Dropdown(options=list(scenes), value=initial, description="Vista")
        self.surface = widgets.ToggleButtons(options=[("Pared", "wall"), ("Suelo", "floor"),
                                                       ("En altura", "raised")],
                                             description="Superficie")
        self.piece = widgets.Dropdown(options=CHOICES["wall"], description="Pieza")
        self.size = widgets.FloatSlider(value=1., min=.25, max=1.6, step=.05, description="Escala",
                                        continuous_update=False)
        self.angle = widgets.IntSlider(value=0, min=-90, max=90, description="Giro °",
                                       continuous_update=False)
        self.vertical = widgets.Checkbox(description="Alfombra vertical", indent=False)
        self.support = widgets.Dropdown(description="Apoyo")
        self.support_height = widgets.FloatSlider(value=.55, min=.10, max=1.40, step=.05,
                                                   description="Altura relativa",
                                                   style={"description_width": "initial"})
        self.mark_button = widgets.Button(description="Marcar tablero")
        self.detect_button = widgets.Button(description="Detectar apoyos")
        self.support_controls = widgets.HBox([self.support, self.detect_button,
                                               self.support_height, self.mark_button],
                                              layout=widgets.Layout(flex_flow="row wrap"))
        self.x = widgets.BoundedIntText(description="X", min=0)
        self.y = widgets.BoundedIntText(description="Y", min=0)
        self.items = widgets.Select(description="Piezas", rows=4, layout=widgets.Layout(width="98%"))
        self.message = widgets.HTML()
        self.preview_status = widgets.HTML()
        self.add_button = widgets.Button(description="Añadir en X, Y")
        self.remove_button = widgets.Button(description="Quitar seleccionada")
        self.save_button = widgets.Button(description="Guardar PNG")
        self.preview = widgets.Image(format="png", layout=widgets.Layout(
            width="560px", max_width="100%", object_fit="contain"
        ))
        self.grid = widgets.Checkbox(value=True, description="Guía de coordenadas", indent=False)
        self.live_preview = widgets.Checkbox(value=True, description="Previsualizar pieza", indent=False)
        self.lighting = widgets.Checkbox(value=True, description="Luz y sombra", indent=False)
        self.fig = None
        self._ready = False
        self._frame = None
        self._marking = False
        self._corners = []
        self._supports = {}
        self._scanned = set()
        self._confirmed = None
        self.room.observe(self._change_room, names="value")
        self.surface.observe(self._change_surface, names="value")
        self.piece.observe(self._change_piece, names="value")
        self.support.observe(self._change_support, names="value")
        self.grid.observe(self._draw_preview, names="value")
        for control in (self.x, self.y, self.size, self.angle, self.vertical, self.live_preview):
            control.observe(self._draw_preview, names="value")
        self.lighting.observe(self._refresh, names="value")
        self.detect_button.on_click(self._detect_supports)
        self.mark_button.on_click(self._toggle_marking)
        self.add_button.on_click(self._add_from_coordinates)
        self.remove_button.on_click(self._remove)
        self.save_button.on_click(self._save)
        self._change_room()
        self._ready = True

    @property
    def current(self):
        return self.rooms[self.room.value]

    def _change_room(self, _=None):
        self._frame = None
        self._confirmed = None
        self._stop_marking()
        height, width = self.current.scene.image.shape[:2]
        self.x.max, self.y.max = width - 1, height - 1
        self._change_piece()
        self._refresh()

    def _change_surface(self, _=None):
        self._stop_marking()
        self.piece.options = CHOICES[self.surface.value]
        self._change_piece()

    def _change_piece(self, _=None):
        if self._marking:
            return
        floor = self.surface.value == "floor"
        raised = self.surface.value == "raised"
        furniture = self.piece.value in MODELS or raised
        self.support_controls.layout.display = "" if raised else "none"
        self.vertical.layout.display = "" if floor and not furniture else "none"
        scene = self.current.scene
        self.x.value, self.y.value = scene.floor_anchor if floor or raised else scene.wall_anchor
        point = ("un punto sobre el apoyo" if raised else
                 "el centro de apoyo del mueble" if furniture else
                 "la esquina superior derecha de la pieza")
        self.message.value = f"Haz clic en {point}." if self.fig else f"Indica X e Y para {point}."
        if raised:
            self._change_support()
        else:
            self._draw_preview()

    def _change_support(self, _=None):
        if self.surface.value != "raised" or self._marking:
            return
        support = self._supports.get(self.support.value)
        self.add_button.disabled = support is None
        if support is not None:
            point = support.center
            if np.isfinite(point).all():
                self.x.value, self.y.value = (round(value) for value in point)
        else:
            self.message.value = "Detecta apoyos en la foto, añade un mueble o marca un tablero."
        self._draw_preview()

    def _detect_supports(self, _):
        room = self.room.value
        if self._marking or room in self._scanned:
            return
        self.detect_button.disabled = True
        self.message.value = "Buscando superficies horizontales en la foto…"
        try:
            self.current.detected_supports = detect_supports(self.current.scene)
        except (OSError, ValueError) as error:
            self.message.value = escape(f"No se pudo detectar un apoyo. {error}")
            self.detect_button.disabled = False
            return
        self._scanned.add(room)
        self._refresh()
        count = len(self.current.detected_supports)
        if count:
            self.support.value = id(self.current.detected_supports[0])
            self.message.value = f"Apoyos detectados en la foto: {count}. Revisa la zona marcada en cian."
        else:
            self.message.value = "La foto no ofrece un apoyo claro. Puedes marcar un tablero si lo ves."

    def _support_pixels(self):
        support = self._supports.get(self.support.value)
        if self.surface.value != "raised" or self._marking or support is None:
            return np.empty((0, 2))
        boundary = support.outline
        return np.vstack((boundary, boundary[0]))

    def _draw_preview(self, _=None):
        if not self._ready or self._frame is None:
            return
        frame = self._frame
        self.preview_status.value = ""
        if (self.live_preview.value and not self._marking and not self.add_button.disabled
                and self._placement_key() != self._confirmed):
            try:
                pending = self.current.prepare(*self._placement_args(), **self._placement_options())
            except ValueError as error:
                self.preview_status.value = escape(f"Vista previa · {error}")
            else:
                frame = self.current.render(pending, lighting=self.lighting.value)
                self.preview_status.value = "<b>Vista previa</b> · Pulsa «Añadir en X, Y» para fijar la pieza."
        support = self._support_pixels()
        if self.fig is not None:
            self.artist.set_data(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            self.support_outline.set_data(*support.T)
            self.fig.canvas.draw_idle()
            return
        image = frame.copy()
        height, width = image.shape[:2]
        cyan = (255, 214, 44)
        if self.grid.value:
            guides = image.copy()
            for x in range(100, width, 100):
                cv.line(guides, (x, 0), (x, height - 1), (225, 225, 225), 1)
                cv.putText(guides, str(x), (x + 3, 22), cv.FONT_HERSHEY_SIMPLEX,
                           .6, (255, 255, 255), 2, cv.LINE_AA)
            for y in range(100, height, 100):
                cv.line(guides, (0, y), (width - 1, y), (225, 225, 225), 1)
                cv.putText(guides, str(y), (6, y - 4), cv.FONT_HERSHEY_SIMPLEX,
                           .6, (255, 255, 255), 2, cv.LINE_AA)
            image = cv.addWeighted(image, .60, guides, .40, 0)
            cv.drawMarker(image, (self.x.value, self.y.value), cyan, cv.MARKER_CROSS, 20, 2)
        if len(support):
            cv.polylines(image, [np.rint(support).astype(np.int32)], True, cyan, 2, cv.LINE_AA)
        if self._corners:
            corners = np.asarray(self._corners, np.int32)
            cv.polylines(image, [corners], False, cyan, 2, cv.LINE_AA)
            for point in self._corners:
                cv.circle(image, point, 4, cyan, -1, cv.LINE_AA)
        self.preview.value = encode_image(image, ".png")

    def _stop_marking(self):
        self._marking = False
        self._corners = []
        self.mark_button.description = "Marcar tablero"
        self.detect_button.disabled = self.room.value in self._scanned
        self.add_button.description = "Añadir en X, Y"
        self.add_button.disabled = self.surface.value == "raised" and self.support.value is None
        if self.fig is not None:
            self.markers.set_data([], [])

    def _toggle_marking(self, _):
        if self._marking:
            self._stop_marking()
            self._change_piece()
            self._refresh()
            return
        self._marking = True
        self.detect_button.disabled = True
        self.mark_button.description = "Cancelar marcado"
        self.add_button.description = "Marcar X, Y"
        self.add_button.disabled = False
        self.message.value = "Marca las cuatro esquinas visibles del tablero en orden alrededor."
        self._draw_preview()

    def _mark_point(self):
        self._corners.append((self.x.value, self.y.value))
        if self.fig is not None:
            self.markers.set_data(*zip(*self._corners))
        self._draw_preview()
        if len(self._corners) < 4:
            self.message.value = f"Esquinas marcadas: {len(self._corners)} de 4."
            return
        try:
            self.current.add_marked_support(self._corners, self.support_height.value)
        except ValueError as error:
            self._stop_marking()
            self._refresh()
            self.message.value = escape(str(error))
            return
        self._stop_marking()
        self._refresh()
        self.support.value = id(self.current.items[-1][1])
        self.message.value = "Apoyo marcado. Coloca el objeto dentro del tablero."

    def _refresh(self, _=None):
        self._frame = None
        available = tuple(self.current.supports())
        supports = [(label, id(support)) for label, support in available]
        self._supports = {id(support): support for _, support in available}
        if list(self.support.options) != supports:
            previous = self.support.value
            self.support.options = supports
            ids = set(self._supports)
            if previous in ids:
                self.support.value = previous
        if not self._marking:
            self.add_button.disabled = self.surface.value == "raised" and self.support.value is None
            self.detect_button.disabled = self.room.value in self._scanned
        self.items.options = [(f"{i + 1}. {label}", i)
                              for i, (label, _) in enumerate(self.current.items)]
        empty = not self.current.items
        self.remove_button.disabled = self.save_button.disabled = empty
        self._frame = self.current.render(lighting=self.lighting.value)
        self._draw_preview()

    def _placement_args(self):
        return self.surface.value, self.piece.value, (self.x.value, self.y.value)

    def _placement_options(self):
        return dict(scale=self.size.value, angle=self.angle.value, vertical=self.vertical.value,
                    support=self._supports.get(self.support.value))

    def _placement_key(self):
        return (self.room.value, *self._placement_args(), self.size.value, self.angle.value,
                self.vertical.value, self.support.value)

    def _add_from_coordinates(self, _):
        if self._marking:
            self._mark_point()
            return
        if self.live_preview.value and self._placement_key() == self._confirmed:
            self.message.value = "Cambia la posición o la pieza para añadir otra."
            return
        try:
            self.current.add(*self._placement_args(), **self._placement_options())
        except ValueError as error:
            self.message.value = escape(str(error))
            return
        self._confirmed = self._placement_key()
        self._refresh()
        self.items.index = len(self.current.items) - 1
        self.message.value = f"Pieza añadida. Total en esta vista: {len(self.current.items)}."

    def _click(self, event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if (event.inaxes is self.ax and event.button == 1 and not getattr(toolbar, "mode", "")
                and event.xdata is not None and event.ydata is not None):
            self.x.value, self.y.value = round(event.xdata), round(event.ydata)
            if self._marking or not self.live_preview.value:
                self._add_from_coordinates(None)

    def _remove(self, _):
        if self.items.value is not None:
            before = len(self.current.items)
            self.current.remove(self.items.value)
            self._confirmed = None
            self._refresh()
            removed = before - len(self.current.items)
            self.message.value = ("Pieza quitada." if removed == 1 else
                                  f"Quitadas {removed} piezas, incluido el apoyo.")

    def _save(self, _):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.output / f"composicion-{self.room.value}-{stamp}.png"
        try:
            write_image(path, self.current.render(lighting=self.lighting.value), exclusive=True)
        except OSError as error:
            self.message.value = escape(f"No se pudo guardar el PNG. {error}")
            return
        self.message.value = f"Guardado en <code>{escape(str(path))}</code>."

    def show(self, *, clickable=False):
        if clickable:
            with plt.ioff():
                self.fig, self.ax = plt.subplots(figsize=(8, 6), layout="constrained")
            if not isinstance(self.fig.canvas, widgets.Widget):
                plt.close(self.fig)
                self.fig = None
                raise ValueError("Para usar el clic activa primero %matplotlib widget.")
            self.artist = self.ax.imshow(cv.cvtColor(self.current.scene.image, cv.COLOR_BGR2RGB))
            self.markers, = self.ax.plot([], [], "o-", color="#31d8ff")
            self.support_outline, = self.ax.plot([], [], "--", color="#31d8ff", linewidth=1.3)
            self.ax.axis("off")
            self.fig.canvas.mpl_connect("button_press_event", self._click)
        viewer = self.fig.canvas if clickable else self.preview
        options = widgets.HBox([self.live_preview, self.lighting]
                               + ([] if clickable else [self.grid]),
                               layout=widgets.Layout(flex_flow="row wrap"))
        controls = widgets.VBox([
            widgets.HBox([self.room, self.surface, self.piece], layout=widgets.Layout(flex_flow="row wrap")),
            widgets.HBox([self.size, self.angle, self.vertical], layout=widgets.Layout(flex_flow="row wrap")),
            self.support_controls,
            widgets.HBox([self.x, self.y, self.add_button]),
            self.items, widgets.HBox([self.remove_button, self.save_button]), self.message,
        ])
        self._refresh()
        display(widgets.VBox([viewer, self.preview_status, options, controls]))
        return self
