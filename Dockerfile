# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.5-base

RUN hf download "Phr00t/Qwen-Image-Edit-Rapid-AIO" --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir "/comfyui/models/checkpoints"
RUN comfy node install "https://github.com/yolain/ComfyUI-Easy-Use.git" "https://github.com/lrzjason/Comfyui-QwenEditUtils.git"
RUN comfy model download --url "https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora/resolve/main/qwen-image-edit-plus-nsfw-lora.safetensors" --relative-path "models/loras" --filename "qwen-image-edit-plus-nsfw-lora.safetensors"
