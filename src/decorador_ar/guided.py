"""One-click guided placement for the teaching notebook.

The click selects an anchor.  Object dimensions and angle are defined in the
surface plane, not by dragging four unrelated image corners.
"""
import cv2 as cv
import numpy as np

from .placement import Placement, composite, default_asset, polygon_coverage
from .image_io import read_image


def wall_quad(top_right, width_px, aspect, angle_deg=0):
    """Create a straight wall rectangle anchored at its top-right corner."""
    if width_px <= 0 or aspect <= 0:
        raise ValueError("El ancho y la proporción deben ser positivos.")
    point = np.asarray(top_right, dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    right = np.array([np.cos(angle), np.sin(angle)], np.float32)
    down = np.array([-np.sin(angle), np.cos(angle)], np.float32)
    top_left = point - width_px * right
    bottom_left = top_left + width_px / aspect * down
    bottom_right = point + width_px / aspect * down
    return np.float32([top_left, point, bottom_right, bottom_left])


def _row_limits(mask, y, radius):
    ys = slice(max(0, y - radius), min(mask.shape[0], y + radius + 1))
    _, xs = np.nonzero(mask[ys])
    if not xs.size:
        raise ValueError("No se encontró suelo suficiente en esa franja.")
    return float(np.quantile(xs, .03)), float(np.quantile(xs, .97))


def floor_plane(mask):
    """Approximate the visible floor by a robust far/near trapezoid."""
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(binary, 8)
    if count < 2:
        raise ValueError("No se ha detectado una región de suelo.")
    component = labels == (1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA])))
    ys, _ = np.nonzero(component)
    if ys.size < 500:
        raise ValueError("La región de suelo es demasiado pequeña.")
    far_y = int(np.quantile(ys, .10))
    near_y = int(np.quantile(ys, .97))
    if near_y - far_y < .12 * mask.shape[0]:
        raise ValueError("No hay profundidad visible suficiente para una alfombra.")
    radius = max(3, round(.012 * mask.shape[0]))
    far_left, far_right = _row_limits(component, far_y, radius)
    near_left, near_right = _row_limits(component, near_y, radius)
    plane = np.float32([[far_left, far_y], [far_right, far_y],
                        [near_right, near_y], [near_left, near_y]])
    if not cv.isContourConvex(np.rint(plane).astype(np.int32)):
        raise ValueError("El contorno de suelo detectado no forma un plano estable.")
    return plane


def floor_transforms(mask, extent=1000):
    """Return image↔bird's-eye homographies for the detected floor."""
    source = floor_plane(mask)
    target = np.float32([[0, 0], [extent, 0], [extent, extent], [0, extent]])
    return (cv.getPerspectiveTransform(source, target),
            cv.getPerspectiveTransform(target, source), source)


def transform_points(points, matrix):
    points = np.asarray(points, np.float32).reshape(1, -1, 2)
    return cv.perspectiveTransform(points, matrix)[0]


def floor_quad(top_right, mask, long_side=360, orientation="horizontal",
               angle_deg=0, aspect=1.6, extent=1000):
    """Build a rug rectangle in bird's-eye space and project it to the photo."""
    if long_side <= 0 or aspect <= 0:
        raise ValueError("El tamaño y la proporción deben ser positivos.")
    image_to_plane, plane_to_image, _ = floor_transforms(mask, extent)
    anchor = transform_points([top_right], image_to_plane)[0]
    short_side = long_side / aspect
    width, depth = ((long_side, short_side) if orientation == "horizontal"
                    else (short_side, long_side))
    angle = np.deg2rad(angle_deg)
    right = np.array([np.cos(angle), np.sin(angle)], np.float32)
    down = np.array([-np.sin(angle), np.cos(angle)], np.float32)
    top_left = anchor - width * right
    plane_quad = np.float32([top_left, anchor, anchor + depth * down,
                             top_left + depth * down])
    return transform_points(plane_quad, plane_to_image)


def place_from_corner(image, masks, surface, top_right, *, size=.28,
                      angle_deg=0, orientation="horizontal", min_coverage=.88,
                      texture=None):
    """Place an object from one click, rejecting a contradictory surface."""
    if surface not in ("wall", "floor"):
        raise ValueError("La superficie debe ser wall o floor.")
    region = masks[surface][0] if isinstance(masks[surface], tuple) else masks[surface]
    texture = default_asset(surface, read_image) if texture is None else texture
    aspect = texture.shape[1] / texture.shape[0]
    if surface == "wall":
        quad = wall_quad(top_right, image.shape[1] * size, aspect, angle_deg)
    else:
        quad = floor_quad(top_right, region, long_side=1000 * size,
                          orientation=orientation, angle_deg=angle_deg, aspect=aspect)
    coverage = polygon_coverage(quad, region)
    if coverage < min_coverage:
        raise ValueError(
            f"La colocación solo coincide un {coverage:.0%} con el {surface}; "
            "elige otra esquina o reduce el tamaño."
        )
    output = image.copy()
    composite(output, texture, quad, region, surface)
    return Placement(output, quad, coverage, surface)


class InteriorDecorator:
    """Notebook controls with an injectable placement engine."""

    def __init__(self, image, masks, *, place=None):
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt

        self.image = image
        self.masks = masks
        self.place = place or place_from_corner
        self.last = None
        self._plt = plt
        self._display = display
        self.surface = widgets.ToggleButtons(
            options=[("Cuadro · pared", "wall"), ("Alfombra · suelo", "floor")],
            description="Objeto:")
        self.orientation = widgets.ToggleButtons(
            options=[("Horizontal", "horizontal"), ("Vertical", "vertical")],
            description="Alfombra:")
        self.size = widgets.FloatSlider(value=.28, min=.10, max=.55, step=.01,
                                        description="Tamaño:", readout_format=".0%")
        self.angle = widgets.FloatSlider(value=0, min=-45, max=45, step=1,
                                         description="Grados:")
        self.message = widgets.HTML("Haz clic en la <b>esquina superior derecha</b>.")
        self.output = widgets.Output()
        self.controls = widgets.VBox([
            self.surface, self.orientation, self.size, self.angle, self.message
        ])
        self.surface.observe(self._surface_changed, names="value")

    def _surface_changed(self, change):
        self.orientation.layout.display = "" if change["new"] == "floor" else "none"
        self.message.value = "Haz clic en la <b>esquina superior derecha</b>."

    def _clicked(self, event):
        if (event.inaxes is not self.ax or event.button != 1
                or event.xdata is None or event.ydata is None):
            return
        try:
            self.last = self.place(
                self.image, self.masks, self.surface.value,
                (event.xdata, event.ydata), size=self.size.value,
                angle_deg=self.angle.value, orientation=self.orientation.value,
            )
            self.ax.clear()
            self.ax.imshow(cv.cvtColor(self.last.image, cv.COLOR_BGR2RGB))
            self.ax.plot(*self.last.quad[[0, 1, 2, 3, 0]].T,
                         "o-", color="#36d1ff", linewidth=1.5)
            self.message.value = f"✅ Encaje con la superficie: <b>{self.last.coverage:.1%}</b>."
        except ValueError as error:
            self.message.value = f"❌ {error}"
        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def show(self):
        self.orientation.layout.display = "none"
        with self.output:
            self.fig, self.ax = self._plt.subplots(figsize=(11, 7))
            self.ax.imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
            self.ax.set_title("Clic = esquina superior derecha")
            self.ax.set_axis_off()
            self.fig.canvas.mpl_connect("button_press_event", self._clicked)
            self._plt.show()
        self._display(self.controls, self.output)
        return self
