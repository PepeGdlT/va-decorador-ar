import unittest
from types import SimpleNamespace

import cv2 as cv
import numpy as np

from decorador_ar.composition import Composition
from decorador_ar.notebook_workflow import Scene
from decorador_ar.raised_surfaces import detect_supports
from decorador_ar.rectification import room_planes
from decorador_ar.supported_objects import support_plane


class CompositionTest(unittest.TestCase):
    def test_draft_renders_without_entering_the_composition(self):
        floor = np.zeros((240, 320), bool)
        image = np.full((240, 320, 3), 120, np.uint8)
        quad = np.float32([[100, 70], [170, 70], [170, 120], [100, 120]])
        scene = SimpleNamespace(
            image=image, masks={"wall": (~floor,), "floor": (floor,)},
            place=lambda *_args, **_options: (SimpleNamespace(quad=quad), None),
        )
        composition = Composition(scene)
        pending = composition.prepare("wall", "paisaje", (170, 70))
        draft = composition.render(pending)
        self.assertFalse(composition.items)
        self.assertFalse(np.array_equal(draft, image))
        np.testing.assert_array_equal(composition.render(), image)
        composition.add("wall", "paisaje", (170, 70))
        np.testing.assert_array_equal(composition.render(), draft)

    def test_detected_support_excludes_gap_and_accepts_an_accessory(self):
        intrinsics = np.array([[200., 0, 160], [0, 200, 120], [0, 0, 1]])
        ground = np.array([[-2., -1, 1], [2, -1, 1], [2, -1, 6], [-2, -1, 6]])
        planes = room_planes(ground, 2.5, intrinsics @ np.diag([1., -1., 1.]), [1, 0, 0])
        shape = (240, 320)
        y, _ = np.indices(shape)
        depth = np.divide(200., y - 120, out=np.zeros(shape), where=y > 120)
        tabletop = np.zeros(shape, np.uint8)
        scene = Scene(np.full((*shape, 3), 125, np.uint8), {}, planes,
                      (160, 90), (160, 185), .4, intrinsics)
        pose, camera = support_plane(scene, .55)
        corners = pose.project(np.array([[-.75, -1.45, 0], [.75, -1.45, 0],
                                         [.75, -.45, 0], [-.75, -.45, 0]]), camera)
        cv.fillConvexPoly(tabletop, np.rint(corners).astype(np.int32), 1)
        gap = np.zeros(shape, bool)
        gap[:, 155:185] = True
        scene.masks = {"wall": (y < 120,), "floor": ((y > 120) & (tabletop == 0),),
                       "support": (tabletop != 0) & ~gap}
        depth[tabletop != 0] *= .45
        composition = Composition(scene)
        composition.detected_supports = detect_supports(scene, depth)
        self.assertTrue(composition.detected_supports)
        for support in composition.detected_supports:
            projected = support.outline
            self.assertTrue((projected[:, 0] < 155).all() or (projected[:, 0] >= 185).all())
        support = composition.detected_supports[0]
        composition.add("raised", "plato", support.center, support=support)
        self.assertFalse(np.array_equal(composition.render(), composition.render(lighting=False)))
