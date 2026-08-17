# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.5-base

RUN hf download "Phr00t/Qwen-Image-Edit-Rapid-AIO" --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir "/comfyui/models/checkpoints"
RUN comfy node install "https://github.com/yolain/ComfyUI-Easy-Use.git" "https://github.com/lrzjason/Comfyui-QwenEditUtils.git"
RUN comfy model download --url "https://huggingface.co/lrzjason/Anything2Real/resolve/main/anything2real_2601_A_final_patched.safetensors" --relative-path "models/loras"
RUN comfy model download --url "https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors" --relative-path "models/loras"
RUN comfy model download --url "https://huggingface.co/vafipas663/Qwen-Edit-2509-Upscale-LoRA/resolve/main/qwen-edit-enhance_64-v3_000001000.safetensors" --relative-path "models/loras"
