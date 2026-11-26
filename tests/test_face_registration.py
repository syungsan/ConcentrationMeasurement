import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.face_registration import padded_detection_image, registration_crop, rotate_frame, detect_capture_faces


class FaceRegistrationTests(unittest.TestCase):
    def test_padding_keeps_portrait_aspect_ratio(self):
        image = np.full((300, 100, 3), 200, np.uint8)
        square = padded_detection_image(image, pad=80)
        y, x = np.where(square[:, :, 0] > 100)
        ratio = (y.max()-y.min()+1) / (x.max()-x.min()+1)
        self.assertAlmostEqual(ratio, 3, delta=.03)
        self.assertEqual(square.shape, (640, 640, 3))

    def test_crop_keeps_context_and_original_pixels_at_edge(self):
        image = np.full((1920, 1080, 3), 200, np.uint8)
        crop = registration_crop(image, [0, 0, 200, 300])
        self.assertEqual(crop.shape, (450, 450, 3))
        self.assertTrue(np.any(crop == 200))
        self.assertTrue(np.any(crop == 24))
        self.assertIsNone(registration_crop(image, [10, 0, 0, 1]))

    def test_portrait_tiles_restore_source_coordinates(self):
        face = SimpleNamespace(bbox=np.array([10., 20., 110., 120.]))
        class Detector:
            def __init__(self):
                self.calls = 0
            def get(self, image):
                self.calls += 1
                return [face] if self.calls == 3 else []
        detector = Detector()
        found = detect_capture_faces(detector, np.zeros((1920, 1080, 3), np.uint8))
        np.testing.assert_array_equal(found[0].bbox, [10, 560, 110, 660])
        np.testing.assert_array_equal(face.bbox, [10, 20, 110, 120])

    def test_rotation_fallback_uses_consistent_transformed_image(self):
        spec = importlib.util.spec_from_file_location("build_db_test", ROOT / "scripts/tools/build_face_db.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = np.zeros((150, 70, 3), np.uint8)
        source[:30, :10] = 255
        face = SimpleNamespace(normed_embedding=np.array([1., 0.]))
        class Detector:
            def __init__(self):
                self.images = []
            def get(self, image):
                self.images.append(image.copy())
                return [face] if len(self.images) == 3 else []
        detector = Detector()
        self.assertEqual(module.detect_face_with_fallback(detector, source), [face])
        np.testing.assert_array_equal(detector.images[2], padded_detection_image(rotate_frame(source, 90)))


if __name__ == "__main__":
    unittest.main()
