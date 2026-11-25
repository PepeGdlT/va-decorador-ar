import unittest

import numpy as np
import torch
from torch.nn.functional import interpolate

from decorador_ar.photo import analyze_photo, resized_class_map


class FixedSurfaces:
    model_id = "test"
    revision = "local"

    def predict(self, image):
        wall = np.ones(image.shape[:2], dtype=bool)
        wall[35:, :] = False
        return {"wall": (wall, wall.astype(np.float32)),
                "floor": (~wall, (~wall).astype(np.float32))}


class PhotoPipelineTest(unittest.TestCase):
    def test_chunked_logits_match_full_interpolation_with_ties(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 13, 11, 17)
        logits[:, 9] = logits[:, 2]
        shape = (93, 151)
        reference = interpolate(logits, size=shape, mode="bilinear",
                                align_corners=False)[0].argmax(dim=0)
        torch.testing.assert_close(resized_class_map(logits, shape), reference)

    def test_metrics_match_render_and_obstacles_are_rejected(self):
        image = np.full((70, 100, 3), 160, np.uint8)
        truth = np.full(image.shape[:2], 4, np.uint8)
        truth[:35] = 1
        planes = {"wall": np.float32([[10, 8], [40, 8], [40, 25], [10, 25]]),
                  "floor": np.float32([[55, 8], [75, 8], [75, 25], [55, 25]])}
        texture = np.full((20, 30, 3), 60, np.uint8)
        options = dict(truth=truth, planes=planes, textures={"wall": texture, "floor": texture})
        rendered = analyze_photo(image, "photo", FixedSurfaces(), **options)
        metrics = analyze_photo(image, "photo", FixedSurfaces(), include_images=False, **options)
        self.assertEqual(rendered.measurements, metrics.measurements)
        self.assertFalse(metrics.images)
        self.assertEqual(metrics.measurements["surfaces"]["wall"]["placement_on_labeled_surface"], 1)
        self.assertEqual(metrics.measurements["surfaces"]["floor"]["decision"],
                         "mascara_no_respalda_el_plano")
        np.testing.assert_array_equal(rendered.images["decorada"][35:], image[35:])
