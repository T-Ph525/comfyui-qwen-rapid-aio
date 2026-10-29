"""Run the pinned upstream crop/association code with mocked ML inference.

Set MIVOLO_TEST_SOURCE to the pinned extension's py/age_gender_node.py.
No torch, GPU, weights or inference packages are needed for these checks.
"""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from tests.test_age_gate import module
from rapidgen_age_gate.policy import PolicyRejected, ScannerUnavailable


class Tensor:
    def __init__(self, pixels): self.pixels = np.asarray(pixels)
    @property
    def shape(self): return self.pixels.shape
    @property
    def ndim(self): return self.pixels.ndim
    def __getitem__(self, key): return Tensor(self.pixels[key])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.pixels
    def item(self): return self.pixels.item()
    def to(self, **kwargs): return self


def box(kind, coords):
    return SimpleNamespace(cls=np.asarray(kind), xyxy=Tensor([coords]))


def batch(count=1):
    return Tensor(np.zeros((count, 128, 128, 3), dtype=np.float32))


def engine(predictor_type, detections, ages):
    instance = module.AgeEngine.__new__(module.AgeEngine)
    instance.device = 'cpu'
    instance.minimum_age = 18.
    instance.predictions = []
    instance.torch = SimpleNamespace(inference_mode=nullcontext)
    instance.detector = lambda pixels, **kwargs: [SimpleNamespace(boxes=detections)]
    instance.processor = lambda **kwargs: {'pixel_values': Tensor([0.])}
    age_iter = iter(ages)
    instance.model = SimpleNamespace(dtype='float32')
    class Model:
        dtype = 'float32'
        def __call__(self, **kwargs):
            try: age = next(age_iter)
            except StopIteration: return SimpleNamespace(age_output=None)
            return SimpleNamespace(age_output=Tensor([[age]]))
    instance.model = Model()
    instance.predictor = predictor_type()
    instance.predictor._predict_single = instance.predict_raw
    return instance


@unittest.skipUnless(os.environ.get('MIVOLO_TEST_SOURCE'), 'Set MIVOLO_TEST_SOURCE for pinned upstream integration')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['MIVOLO_TEST_SOURCE']).read_text()
        parsed = ast.parse(source)
        names = {'crop_box', 'tensor_to_np_bgr', 'MiVOLOAgeGenderPredictorWithDetector'}
        selected = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        assert {node.name for node in selected} == names
        namespace = {'np': np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned MiVOLO backend>', 'exec'), namespace)
        cls.predictor_type = namespace['MiVOLOAgeGenderPredictorWithDetector']

    def make_engine(self, detections, ages):
        return engine(self.predictor_type, detections, ages)

    def test_all_faces_must_pass(self):
        e = self.make_engine([box(1, [0, 0, 40, 40]), box(1, [60, 60, 100, 100])], [35., 40.])
        e.check(batch())
        self.assertEqual(e.predictions, [35., 40.])

    def test_later_young_face_blocks_whole_image(self):
        e = self.make_engine([box(1, [0, 0, 40, 40]), box(1, [60, 60, 100, 100])], [35., 17.6])
        with self.assertRaises(PolicyRejected): e.check(batch())

    def test_every_batch_image_is_checked(self):
        e = self.make_engine([box(1, [0, 0, 40, 40])], [35., 17.6])
        with self.assertRaises(PolicyRejected): e.check(batch(2))

    def test_person_and_face_association_uses_both_crops(self):
        e = self.make_engine([box(0, [0, 0, 120, 120]), box(1, [10, 10, 60, 60])], [35.])
        crops = []
        def processor(images):
            crops.append(images[0]); return {'pixel_values': Tensor([0.])}
        e.processor = processor
        e.check(batch())
        self.assertEqual([crop.shape for crop in crops], [(50, 50, 3), (120, 120, 3)])
        self.assertEqual(e.predictions, [35.])

    def test_unassociated_face_is_also_checked(self):
        e = self.make_engine([box(0, [0, 0, 64, 128]), box(1, [0, 0, 40, 40]), box(1, [70, 0, 110, 40])], [35., 12.])
        with self.assertRaises(PolicyRejected): e.check(batch())

    def test_person_without_face_is_policy_rejection(self):
        e = self.make_engine([box(0, [0, 0, 100, 128])], [35.])
        with self.assertRaises(PolicyRejected): e.check(batch())

    def test_no_detection_is_policy_rejection(self):
        with self.assertRaises(PolicyRejected): self.make_engine([], []).check(batch())

    def test_small_face_is_policy_rejection(self):
        with self.assertRaises(PolicyRejected): self.make_engine([box(1, [0, 0, 10, 10])], []).check(batch())

    def test_invalid_bounds_cannot_be_silently_skipped(self):
        with self.assertRaises(PolicyRejected): self.make_engine([box(1, [127, 127, 160, 160])], []).check(batch())

    def test_subpixel_person_cannot_be_silently_skipped(self):
        e = self.make_engine([box(0, [0.1, 0.1, 0.9, 0.9]), box(1, [60, 60, 100, 100])], [35.])
        with self.assertRaises(PolicyRejected): e.check(batch())

    def test_invalid_class_is_scanner_fault(self):
        with self.assertRaises(ScannerUnavailable): self.make_engine([box(2, [0, 0, 40, 40])], []).check(batch())

    def test_missing_prediction_is_scanner_fault(self):
        with self.assertRaises(ScannerUnavailable): self.make_engine([box(1, [0, 0, 40, 40])], []).check(batch())

    def test_nonfinite_age_is_scanner_fault(self):
        with self.assertRaises(ScannerUnavailable): self.make_engine([box(1, [0, 0, 40, 40])], [float('nan')]).check(batch())

    def test_nonfinite_image_is_scanner_fault(self):
        pixels = batch(); pixels.pixels[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ScannerUnavailable): self.make_engine([], []).check(pixels)

    def test_more_than_four_batch_images_is_policy_rejection(self):
        with self.assertRaises(PolicyRejected): self.make_engine([], []).check(batch(5))
