"""Docker build-time offline CPU smoke check; no user images are involved."""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/comfyui')
from custom_nodes.rapidgen_age_gate import get_engine

engine = get_engine()
engine.torch.set_num_threads(2)
inputs = engine.processor(images=[Image.new('RGB', (224, 224))], return_tensors='pt').to(engine.device)
with engine.torch.inference_mode():
    logits = engine.model(**inputs).logits
assert tuple(logits.shape) == (1, 8), 'Age model has an unexpected output shape'
assert engine.torch.isfinite(logits).all().item(), 'Age model produced invalid values'
engine.detector.setInputSize((320, 320))
engine.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
print('RapidGen offline CPU model inference and face detector smoke checks passed')
