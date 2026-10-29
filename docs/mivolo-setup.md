# Install and connect the MiVOLO safety gate

## Docker / RunPod

You should rebuild the repository's Dockerfile and redeploy that image. It
installs the upstream extension and local gate, bakes the pinned model assets
into the image and validates offline CPU inference during the build. You do not
need to install a second node pack on the running container.

## Manual installation

You should activate the Python environment that runs ComfyUI, then run these
commands from the ComfyUI directory:

```bash
git clone https://github.com/vdeng-ai/ComfyUI-MiVolo-V2.git custom_nodes/ComfyUI-MiVolo-V2
git -C custom_nodes/ComfyUI-MiVolo-V2 checkout 54354ff4cdd388974d7b3ac938a59ac88a58d27f
python -m pip freeze | sed -n '/^torch==/p; /^torchvision==/p; /^torchaudio==/p' > /tmp/rapidgen-torch-constraints.txt
python -m pip install -c /tmp/rapidgen-torch-constraints.txt -r /path/to/comfyui-qwen-rapid-aio/requirements-age-gate.txt
python -m pip install --no-deps --no-build-isolation 'git+https://github.com/WildChlamydia/MiVOLO.git@37475e3f8818b5f22448003feec3e64b01bfb188'
cp -R /path/to/comfyui-qwen-rapid-aio/custom_nodes/rapidgen_age_gate custom_nodes/
```

If you use Manager, you can search for `ComfyUI-MiVolo-V2` in Custom Nodes
Manager and install it there. Manager installs the upstream nodes only; you
still need this repository's local `rapidgen_age_gate` for mandatory blocking.
Use the Docker build for a reproducible production dependency installation.

## Model placement

You can download the pinned complete model and detector with ComfyUI's Python:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(
    'iitolstykh/mivolo_v2',
    revision='53393526c220e34cdd7b722b36d22b6f9e5f4241',
    allow_patterns=['*.json', '*.py', 'model.safetensors', 'README.md'],
    local_dir='models/mivolo/iitolstykh/mivolo_v2',
)
hf_hub_download(
    'iitolstykh/demo_yolov8_detector',
    'yolov8x_person_face.pt',
    revision='cc9d63ff895e8d980d98bed2b141e83a4e7f5959',
    local_dir='models/yolo',
)
PY
```

If you must transfer files manually, you should put the complete age snapshot
from the pinned Hugging Face revision under
`ComfyUI/models/mivolo/iitolstykh/mivolo_v2/`. You need `config.json`,
`preprocessor_config.json`, `configuration_mivolo.py`, `modeling_mivolo.py`,
`mivolo_image_processor.py` and `model.safetensors`. Put the detector at
`ComfyUI/models/yolo/yolov8x_person_face.pt`. Restart ComfyUI after installation.

## Connect the supplied workflows

You can open `workflow.json` in ComfyUI or submit `api-workflow.json` to the
RunPod worker. The examples already contain both mandatory gates.

| Connection | Purpose |
| --- | --- |
| LoadImage IMAGE → input RapidGenAgeGate `image1` | Scan the original reference |
| Edit prompt → input gate `prompt` | Make conditioning wait for all references |
| Additional references → input gate `image2`–`image4` | Scan every supplied reference |
| Input gate checked images → Qwen image inputs | Prevent image-conditioning bypass |
| Input gate `prompt` → Qwen positive `prompt` | Prevent sampling before the precheck |
| VAEDecode IMAGE → output RapidGenAgeGate `image1` | Scan the edited result |
| Input gate `prompt` → output gate `prompt` | Preserve the precheck dependency |
| Output gate `checked_image1` → SaveImage `images` | Save only approved results |

You should search for **RapidGen MiVOLO Age Safety Gate**. Its API class remains
`RapidGenAgeGate`, preserving existing workflow references. It loads the models
internally, calls the upstream **MiVOLO Predictor (with Detector)** pipeline in
`Use persons and faces` / `All` mode, and checks raw numeric ages. It raises
`PolicyRejected` when any age is below `RAPIDGEN_MIN_AGE` (default 18). ComfyUI
then aborts that workflow; the worker returns `POLICY_REJECTED` without images.
It raises `ScannerUnavailable` for confirmed scanner faults.

The upstream **Load MiVOLO Model**, **Load MiVOLO Detector (YOLO)**,
**MiVOLO Predictor (with Detector)** and **MiVOLO Predictor (from Crops)** nodes
are also installed for inspection. Their rounded age STRING outputs are not
used to authorize a result. You do not need an external string-to-number or
comparison node. You must not substitute an ordinary image switch: an unused
branch can still execute unless it implements lazy evaluation correctly.

You should connect every save, preview and upload to the checked output and
avoid exposing an unguarded ComfyUI API to end users. The RunPod wrapper rejects
unknown nodes and rebuilds gates server-side; direct ComfyUI UI/API edits are an
operator boundary, not protected against a malicious operator.

## Validate before release

You should verify a later person and a later batch image can both block a job,
that no-face/body-only subjects remain blocked, that an input block stops
sampling, and that an output block produces no upload. You should also verify
that a scanner fault returns `REFUND_SERVER_FAILURE`, while an age policy block
returns `CHARGE`. Run the regression suite and rebuild the Docker image before
performing a live end-to-end S3 check on the target worker.

MiVOLO estimates age and does not prove adulthood. The detector may miss a person
entirely. An estimate near 18 is uncertain; a higher operator threshold can add
a buffer but cannot guarantee minor detection. No contextual scan is included.
