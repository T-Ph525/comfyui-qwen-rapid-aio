"""Mandatory MiVOLO V2 gate, retaining the existing Qwen workflow sockets."""
import importlib.util
import os
from pathlib import Path
import threading

from .policy import PolicyRejected, ScannerUnavailable, check_age

_ENGINE = None
_LOCK = threading.RLock()


def model_paths():
    import folder_paths
    root = Path(folder_paths.models_dir)
    model = Path(os.environ.get('RAPIDGEN_AGE_MODEL_DIR', root / 'mivolo/iitolstykh/mivolo_v2'))
    detector = Path(os.environ.get('RAPIDGEN_FACE_DETECTOR', root / 'yolo/yolov8x_person_face.pt'))
    return model, detector


def load_predictor_module():
    # Load the installed, pinned extension's detector/crop/association pipeline.
    # Do not rely on the arbitrary module name assigned by ComfyUI's loader.
    import folder_paths
    candidates = [Path(root) / 'ComfyUI-MiVolo-V2/py/age_gender_node.py'
                  for root in folder_paths.get_folder_paths('custom_nodes')]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: ComfyUI-MiVolo-V2 is not installed')
    spec = importlib.util.spec_from_file_location('rapidgen_mivolo_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgeEngine:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from ultralytics import YOLO

        model_dir, detector_path = model_paths()
        required = ('config.json', 'preprocessor_config.json', 'model.safetensors',
                    'configuration_mivolo.py', 'modeling_mivolo.py', 'mivolo_image_processor.py')
        if any(not (model_dir / name).is_file() for name in required) or not detector_path.is_file():
            raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: MiVOLO model or detector is not installed')
        self.torch = torch
        self.device = os.environ.get('RAPIDGEN_AGE_DEVICE', 'cpu')
        self.minimum_age = float(os.environ.get('RAPIDGEN_MIN_AGE', '18'))
        check_age(120., self.minimum_age)  # Validate configuration before inference.
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True,
            use_safetensors=True, torch_dtype=torch.float32,
        ).to(self.device).eval()
        self.detector = YOLO(str(detector_path))
        labels = {int(key): str(value).lower() for key, value in self.detector.names.items()}
        if labels != {0: 'person', 1: 'face'}:
            raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: unexpected detector classes')
        upstream = load_predictor_module()
        self.predictor = upstream.MiVOLOAgeGenderPredictorWithDetector()
        # Upstream rounds ages to integers. Replace only its single-subject
        # inference hook; keep its All-mode face/body association and crops.
        self.predictor._predict_single = self.predict_raw
        self.predictions = []

    def predict_raw(self, model_pack, face_crop, body_crop):
        # Body-only estimates are not sufficient for this service's policy.
        if face_crop is None or min(face_crop.shape[:2]) < 32:
            raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: no assessable face for a subject')
        inputs = {}
        for key, crop in (('faces_input', face_crop), ('body_input', body_crop)):
            values = self.processor(images=[crop])['pixel_values']
            inputs[key] = None if values is None else values.to(device=self.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            output = self.model(**inputs).age_output
        if output is None or tuple(output.shape) != (1, 1):
            raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: missing age prediction')
        age = check_age(output[0].item(), self.minimum_age)
        self.predictions.append(age)
        # Do not emit gender or round age; only the numeric result is relevant.
        return ('Age checked', age, '')

    def detect(self, bgr, **kwargs):
        """Validate detections before upstream can silently skip invalid crops."""
        import numpy as np
        results = self.detector(bgr, device=self.device, **kwargs)
        if not results or len(results) != 1 or results[0].boxes is None:
            raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: no assessable subjects')
        boxes = results[0].boxes
        if len(boxes) == 0:
            raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: no assessable subjects')
        if len(boxes) > 64:
            raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: too many subjects to assess')
        height, width = bgr.shape[:2]
        for box in boxes:
            class_id = int(box.cls.item())
            coords = box.xyxy[0].cpu().numpy()
            if class_id not in (0, 1) or not np.isfinite(coords).all():
                raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: invalid detector output')
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or int(x2) <= int(x1) or int(y2) <= int(y1):
                raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: unassessable subject bounds')
            if class_id == 1 and min(x2 - x1, y2 - y1) < 32:
                raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: face too small to assess')
        return results

    def check(self, tensor):
        import numpy as np
        if len(tensor.shape) != 4 or tensor.shape[0] < 1 or tensor.shape[-1] != 3:
            raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: expected a ComfyUI RGB IMAGE batch')
        if tensor.shape[0] > 4:
            raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: too many images')
        # Upstream only processes batch[0]; invoke it once per image instead.
        for index in range(tensor.shape[0]):
            sample = tensor[index:index + 1]
            pixels = sample.detach().cpu().numpy()
            if not np.isfinite(pixels).all():
                raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: non-finite image')
            if tensor.shape[1] * tensor.shape[2] > 16000000:
                raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: image too large for age check')
            self.predictions = []
            self.predictor.predict_with_detector(
                None, self.detect, sample, 'Use persons and faces', 'All', 0.4, 0.7)
            if not self.predictions:
                raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: no assessable face')


def get_engine():
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = AgeEngine()
        return _ENGINE


class RapidGenAgeGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {'image1': ('IMAGE',), 'prompt': ('STRING', {'forceInput': True})},
            'optional': {'image2': ('IMAGE',), 'image3': ('IMAGE',), 'image4': ('IMAGE',)},
        }

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('checked_image1', 'checked_image2', 'checked_image3', 'checked_image4', 'prompt')
    FUNCTION = 'check'
    CATEGORY = 'RapidGen/Safety'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def check(self, image1, prompt, image2=None, image3=None, image4=None):
        images = (image1, image2, image3, image4)
        try:
            with _LOCK:
                engine = get_engine()
                for image in images:
                    if image is not None:
                        engine.check(image)
        except (PolicyRejected, ScannerUnavailable):
            raise
        except Exception as error:
            raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: MiVOLO age check could not run') from error
        return (*images, prompt)


NODE_CLASS_MAPPINGS = {'RapidGenAgeGate': RapidGenAgeGate}
NODE_DISPLAY_NAME_MAPPINGS = {'RapidGenAgeGate': 'RapidGen MiVOLO Age Safety Gate'}
