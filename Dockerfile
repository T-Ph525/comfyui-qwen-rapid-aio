# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.5-base

RUN hf download "Phr00t/Qwen-Image-Edit-Rapid-AIO" --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir "/comfyui/models/checkpoints"
RUN comfy node install "https://github.com/yolain/ComfyUI-Easy-Use.git" "https://github.com/lrzjason/Comfyui-QwenEditUtils.git"
RUN comfy model download --url "https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora/resolve/main/qwen-image-edit-plus-nsfw-lora.safetensors" --relative-path "models/loras" --filename "qwen-image-edit-plus-nsfw-lora.safetensors"

# MiVOLO V2: pin both executable model code and the ComfyUI wrapper.
ARG MIVOLO_NODE_REVISION=54354ff4cdd388974d7b3ac938a59ac88a58d27f
ARG MIVOLO_LIBRARY_REVISION=37475e3f8818b5f22448003feec3e64b01bfb188
ARG AGE_MODEL_REVISION=53393526c220e34cdd7b722b36d22b6f9e5f4241
ARG FACE_DETECTOR_REVISION=cc9d63ff895e8d980d98bed2b141e83a4e7f5959
RUN git clone https://github.com/vdeng-ai/ComfyUI-MiVolo-V2.git /comfyui/custom_nodes/ComfyUI-MiVolo-V2 \
    && git -C /comfyui/custom_nodes/ComfyUI-MiVolo-V2 checkout "${MIVOLO_NODE_REVISION}"

# Keep the base worker's torch/CUDA wheels. The upstream MiVOLO package declares
# old detector dependencies; install its code separately, with explicit deps here.
COPY requirements-age-gate.txt /requirements-age-gate.txt
RUN python -m pip freeze | sed -n '/^torch==/p; /^torchvision==/p; /^torchaudio==/p' > /torch-constraints.txt \
    && python -m pip install --no-cache-dir -c /torch-constraints.txt -r /requirements-age-gate.txt \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git@${MIVOLO_LIBRARY_REVISION}"
RUN hf download "iitolstykh/mivolo_v2" \
    --revision "${AGE_MODEL_REVISION}" \
    --include "*.json" "*.py" "model.safetensors" "README.md" \
    --local-dir "/comfyui/models/mivolo/iitolstykh/mivolo_v2"
RUN hf download "iitolstykh/demo_yolov8_detector" \
    --revision "${FACE_DETECTOR_REVISION}" \
    --include "yolov8x_person_face.pt" "README.md" \
    --local-dir "/comfyui/models/yolo"

ENV RAPIDGEN_AGE_MODEL_DIR=/comfyui/models/mivolo/iitolstykh/mivolo_v2 \
    RAPIDGEN_FACE_DETECTOR=/comfyui/models/yolo/yolov8x_person_face.pt \
    RAPIDGEN_AGE_DEVICE=cpu \
    RAPIDGEN_MIN_AGE=18
COPY custom_nodes/rapidgen_age_gate /comfyui/custom_nodes/rapidgen_age_gate

# Keep the pinned base worker's WebSocket, S3 and startup behavior. The wrapper
# adds mandatory gates and rejects failed history BEFORE its S3 upload loop.
RUN python -c "from pathlib import Path; import hashlib; data=Path('/handler.py').read_bytes(); assert hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest() == '817cad8a9a94a9e66c0d0013a3d7a35baee0316f', 'Base worker handler changed; review the pre-upload guard before building'" \
    && mv /handler.py /upstream_handler.py
COPY rapidgen_handler.py /handler.py
COPY workflow_guard.py safety_outcome.py /rapidgen/
ENV PYTHONPATH=/rapidgen

# Verify offline CPU inference and detector execution while building; no GPU needed.
COPY scripts/validate_models.py /rapidgen/validate_models.py
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAPIDGEN_AGE_DEVICE=cpu python /rapidgen/validate_models.py
