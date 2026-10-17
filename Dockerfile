# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.5-base

RUN hf download "Phr00t/Qwen-Image-Edit-Rapid-AIO" --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir "/comfyui/models/checkpoints"
RUN comfy node install "https://github.com/yolain/ComfyUI-Easy-Use.git" "https://github.com/lrzjason/Comfyui-QwenEditUtils.git"
RUN comfy model download --url "https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora/resolve/main/qwen-image-edit-plus-nsfw-lora.safetensors" --relative-path "models/loras" --filename "qwen-image-edit-plus-nsfw-lora.safetensors"

# Local face-age gate assets. Pin snapshots so rebuilds use the same models.
# Download only inference files: the repository also contains training checkpoints.
ARG AGE_MODEL_REVISION=ceb570360eb639cf8496deabc3eeb4da4cb66bcc
ARG FACE_DETECTOR_REVISION=3cc26e7f1014a5ee5d74a42acee58bafc9d0a310
RUN hf download "prithivMLmods/facial-age-detection" \
    --revision "${AGE_MODEL_REVISION}" \
    --include "config.json" "preprocessor_config.json" "model.safetensors" \
    --local-dir "/comfyui/models/rapidgen/facial-age-detection"
RUN hf download "opencv/face_detection_yunet" \
    --revision "${FACE_DETECTOR_REVISION}" \
    --include "face_detection_yunet_2023mar.onnx" "LICENSE" \
    --local-dir "/comfyui/models/rapidgen"

# Match the RapidGenAgeGate node's paths. CPU preserves generation GPU memory.
ENV RAPIDGEN_AGE_MODEL_DIR=/comfyui/models/rapidgen/facial-age-detection \
    RAPIDGEN_FACE_DETECTOR=/comfyui/models/rapidgen/face_detection_yunet_2023mar.onnx \
    RAPIDGEN_AGE_DEVICE=cpu

# Fail the build if a download did not produce the required inference assets.
RUN test -s "${RAPIDGEN_AGE_MODEL_DIR}/config.json" \
    && test -s "${RAPIDGEN_AGE_MODEL_DIR}/preprocessor_config.json" \
    && test -s "${RAPIDGEN_AGE_MODEL_DIR}/model.safetensors" \
    && test -s "${RAPIDGEN_FACE_DETECTOR}"
