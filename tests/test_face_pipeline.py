from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import numpy as np

from tests.test_age_gate import module
from rapidgen_age_gate.policy import PolicyRejected, ScannerUnavailable


class Sample:
    def __init__(self, pixels): self.pixels=pixels
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.pixels


class Batch:
    shape=(1,128,128,3)
    def __iter__(self): return iter([Sample(np.zeros(self.shape[1:]))])


def engine(faces,scores):
    instance=module.AgeEngine.__new__(module.AgeEngine)
    instance.device='cpu'
    instance.adult_confidence=.90
    instance.max_young_score=.05
    instance.cv2=SimpleNamespace(COLOR_RGB2BGR=1,cvtColor=lambda pixels,mode:pixels)
    instance.detector=SimpleNamespace(setInputSize=lambda size:None,detect=lambda pixels:(None,faces))
    instance.torch=SimpleNamespace(inference_mode=nullcontext)
    instance.processor=lambda **kwargs:SimpleNamespace(to=lambda device:{})
    instance.model=lambda **kwargs:SimpleNamespace(logits=SimpleNamespace(softmax=lambda dim:SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:scores))))
    return instance


class PipelineTests(unittest.TestCase):
    def test_no_detected_face_is_policy_rejection(self):
        with self.assertRaises(PolicyRejected): engine(None,[]).check(Batch())

    def test_small_face_is_policy_rejection(self):
        with self.assertRaises(PolicyRejected):engine(np.array([[0,0,10,10]]),[]).check(Batch())

    def test_second_young_face_rejects_whole_image(self):
        adult=[0,0,1.,0,0,0,0,0]
        young=[0,1.,0,0,0,0,0,0]
        with self.assertRaises(PolicyRejected):
            engine(np.array([[0,0,40,40],[60,60,40,40]]),[adult,young]).check(Batch())

    def test_all_faces_must_pass(self):
        adult=[0,0,1.,0,0,0,0,0]
        engine(np.array([[0,0,40,40],[60,60,40,40]]),[adult,adult]).check(Batch())

    def test_missing_face_prediction_is_server_failure(self):
        adult=[0,0,1.,0,0,0,0,0]
        with self.assertRaises(ScannerUnavailable):
            engine(np.array([[0,0,40,40],[60,60,40,40]]),[adult]).check(Batch())
