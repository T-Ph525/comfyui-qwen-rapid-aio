# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.5-base

# build-time tokens for gated downloads — never baked into final image.
# pass via: docker build --build-arg HF_TOKEN=$HF_TOKEN ...
ARG HF_TOKEN=""

RUN hf download "Phr00t/Qwen-Image-Edit-Rapid-AIO" "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir "/comfyui/models/checkpoints"
RUN comfy node install "https://github.com/yolain/ComfyUI-Easy-Use.git"
RUN wget "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/fixed-textencode-node/nodes_qwen.v2.py" -O "/comfyui/custom_nodes"
