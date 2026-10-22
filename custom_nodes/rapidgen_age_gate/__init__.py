"""ComfyUI passthrough gate. Only checked image outputs may reach SaveImage."""
import os
import threading
from pathlib import Path

from .policy import EXPECTED_LABELS, MODEL_ID, PolicyRejected, ScannerUnavailable, check_scores

_ENGINE = None
_LOCK = threading.RLock()


def model_paths():
    import folder_paths
    root = Path(folder_paths.models_dir) / "rapidgen"
    model = Path(os.environ.get("RAPIDGEN_AGE_MODEL_DIR", root / "facial-age-detection"))
    detector = Path(os.environ.get("RAPIDGEN_FACE_DETECTOR", root / "face_detection_yunet_2023mar.onnx"))
    return model, detector


class AgeEngine:
    def __init__(self):
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_dir, detector_path = model_paths()
        if not model_dir.is_dir() or not detector_path.is_file():
            raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: age model or face detector is not installed")
        self.torch = torch
        self.cv2 = cv2
        self.device = os.environ.get("RAPIDGEN_AGE_DEVICE", "cpu")
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).to(self.device).eval()
        labels = tuple(self.model.config.id2label.get(i) for i in range(8))
        if labels != EXPECTED_LABELS or len(self.model.config.id2label) != 8:
            raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: installed model has unexpected age labels")
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.7, 0.3, 5000)
        self.adult_confidence = float(os.environ.get("RAPIDGEN_ADULT_CONFIDENCE", "0.90"))
        self.max_young_score = float(os.environ.get("RAPIDGEN_MAX_YOUNG_SCORE", "0.05"))
        if not (0 < self.adult_confidence <= 1 and 0 <= self.max_young_score < 1):
            raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: invalid age thresholds")

    def check(self, tensor):
        import numpy as np
        from PIL import Image
        if len(tensor.shape) != 4 or tensor.shape[0] < 1 or tensor.shape[-1] != 3:
            raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: expected a ComfyUI RGB IMAGE batch")
        if tensor.shape[0] > 4:
            raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: too many images")
        for sample in tensor:
            rgb = sample.detach().cpu().numpy()
            if not np.isfinite(rgb).all():
                raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: non-finite image")
            rgb = np.uint8(np.clip(rgb, 0, 1) * 255)
            height, width = rgb.shape[:2]
            if width * height > 16000000:
                raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: image too large for age check")
            scale = min(1.0, 1600 / max(width, height))
            bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
            if scale < 1:
                bgr = self.cv2.resize(bgr, (round(width * scale), round(height * scale)))
            # Pad to the detector's stride without stretching facial geometry.
            pad_h, pad_w = (-bgr.shape[0]) % 32, (-bgr.shape[1]) % 32
            if pad_h or pad_w:
                bgr = self.cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, self.cv2.BORDER_CONSTANT, value=(0, 0, 0))
            self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = self.detector.detect(bgr)
            if faces is None or len(faces) == 0:
                raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: no assessable face")
            if len(faces) > 16:
                raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: too many faces to assess")
            crops = []
            for face in faces:
                x, y, w, h = [float(value) / scale for value in face[:4]]
                if min(w, h) < 32:
                    raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: face too small to assess")
                # Small margin retains facial context without classifying the
                # whole scene or choosing only the largest/oldest person.
                left, top = max(0, int(x - w * .1)), max(0, int(y - h * .1))
                right, bottom = min(width, int(x + w * 1.1)), min(height, int(y + h * 1.1))
                if right <= left or bottom <= top:
                    raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: invalid face bounds")
                crops.append(Image.fromarray(rgb[top:bottom, left:right]))
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                scores = self.model(**inputs).logits.softmax(dim=-1).cpu().tolist()
            if len(scores) != len(crops):
                raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: missing age predictions")
            for probabilities in scores:
                check_scores(probabilities, self.adult_confidence, self.max_young_score)


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
            "required": {"image1": ("IMAGE",), "prompt": ("STRING", {"forceInput": True})},
            "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("checked_image1", "checked_image2", "checked_image3", "checked_image4", "prompt")
    FUNCTION = "check"
    CATEGORY = "RapidGen/Safety"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Policy/configuration may have changed since the previous run.
        return float("nan")

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
            # Never turn dependency/OOM/decoder/model errors into an approval.
            raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: age check could not run") from error
        return (*images, prompt)


NODE_CLASS_MAPPINGS = {"RapidGenAgeGate": RapidGenAgeGate}
NODE_DISPLAY_NAME_MAPPINGS = {"RapidGenAgeGate": "RapidGen Face Age Gate"}
