"""Docker build-time offline CPU smoke check; no user images are involved."""
import sys
import numpy as np

# ComfyUI disables argument parsing for imported modules, so adding --cpu to
# sys.argv does not select CPU here. Set the parsed option before any upstream
# node can import comfy.model_management and initialize CUDA. Build process only.
sys.path.insert(0, '/comfyui')
from comfy.cli_args import args
args.cpu = True
from custom_nodes.rapidgen_age_gate import get_engine, model_paths
from custom_nodes.rapidgen_age_gate.policy import PolicyRejected

model_dir, detector = model_paths()
assert detector.stat().st_size > 1000000, 'Detector weights are missing/truncated'
assert (model_dir / 'model.safetensors').stat().st_size > 1000000, 'Age weights are missing/truncated'
engine = get_engine()
engine.torch.set_num_threads(2)
face = np.zeros((64, 64, 3), dtype=np.uint8)
# Exercise the actual processor/model, including missing body handling.
for body in (None, np.zeros((128, 64, 3), dtype=np.uint8)):
    engine.predictions = []
    try:
        engine.predict_raw(None, face, body)
    except PolicyRejected:
        pass  # A synthetic blank face is not expected to pass the age policy.
# Exercise the real detector; blank images should be a policy block, not a fault.
try:
    engine.detect(np.zeros((320, 320, 3), dtype=np.uint8), conf=0.4, iou=0.7, verbose=False)
except PolicyRejected:
    pass
try:
    engine.check(engine.torch.zeros((1, 320, 320, 3)))
except PolicyRejected:
    pass
else:
    raise AssertionError('An empty image must not pass the full gate')
print('RapidGen offline MiVOLO CPU inference and YOLO smoke checks passed')
