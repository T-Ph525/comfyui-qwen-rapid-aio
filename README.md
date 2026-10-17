# Qwen-Rapid-AIO with local face-age checks

ComfyUI Qwen image-edit worker based on `runpod/worker-comfyui:5.8.5-base`.
The image contains the local `RapidGenAgeGate` node, dependencies, pinned model
assets, guarded API and UI workflows, and a RunPod wrapper. No contextual or
vision-LLM scan is performed.

## Build and deploy

```bash
docker build -t my-comfy-workflow .
docker run --rm --gpus all -p 8188:8188 -e SERVE_API_LOCALLY=true my-comfy-workflow
```

Deploy your rebuilt image on RunPod, or build this repository through RunPod's
GitHub deployment. The older wizard image
`runpod/comfyui-wizard:kd735q8y3ek2k59tz7v7w3tah58c4c3q` does **not** include these
changes. No live endpoint is updated merely by changing this repository.

The Docker build checks required files and runs offline age-model inference and the face detector on CPU.
It also verifies the pinned upstream handler before installing the wrapper:
if the base handler changes, the build refuses to proceed until its pre-upload
integration has been reviewed.

## Model paths inside the backend

| Asset | Location |
| --- | --- |
| `prithivMLmods/facial-age-detection` snapshot | `/comfyui/models/rapidgen/facial-age-detection/` |
| YuNet face detector | `/comfyui/models/rapidgen/face_detection_yunet_2023mar.onnx` |
| ComfyUI node | `/comfyui/custom_nodes/rapidgen_age_gate/` |

The age snapshot contains `config.json`, `preprocessor_config.json` and
`model.safetensors`; training checkpoints are excluded. Model revisions are
pinned by the `AGE_MODEL_REVISION` and `FACE_DETECTOR_REVISION` build arguments.
These files are baked into the image; no startup download is needed.

Environment overrides:

```dotenv
RAPIDGEN_AGE_MODEL_DIR=/comfyui/models/rapidgen/facial-age-detection
RAPIDGEN_FACE_DETECTOR=/comfyui/models/rapidgen/face_detection_yunet_2023mar.onnx
RAPIDGEN_AGE_DEVICE=cpu
```

CPU is the default to preserve generation VRAM. Use `cuda` only after validating
memory/latency. A mounted volume over `/comfyui/models` hides these baked files;
populate the volume or override the paths. Do not put age weights in checkpoints.

## Mandatory gates and supported requests

Send `api-workflow.json` as `input.workflow`, and include each reference in
`input.images` using the usual worker format:

```json
{
  "input": {
    "workflow": "<the API workflow object, not a string>",
    "images": [{"name": "images.png", "image": "<base64 image>"}]
  }
}
```

The wrapper inserts input and output gates, including for older ungated requests.
It checks all 1–4 uploaded references before generation, and checks every image
output before SaveImage/PreviewImage and the upstream worker's S3 uploader.
Caller-supplied gates are replaced. Unsupported node types, missing uploads,
unsafe filenames and cyclic graphs are refused before queueing. This is a guard
for the supported Qwen graph, not a general-purpose ComfyUI workflow sandbox.
The RunPod API key must remain server-side.

Only a fully successful execution history reaches the S3 upload loop. A policy
block or scanner fault cannot return partial images. The inherited worker keeps
its usual S3 configuration (`BUCKET_ENDPOINT_URL` and the other RunPod bucket
settings) and falls back to base64 when S3 is not configured. S3 retention and
bucket access policies remain the operator's responsibility.

The checked example output is `SaveImage`, so it produces a result the upstream
worker can return; that worker skips temporary `PreviewImage` outputs. Both
examples use the actual `v23/` checkpoint location.

## Outcomes and credit handling

Errors are returned as normal structured worker output, without `images`:

| `error_code` | `billing_outcome` | Meaning |
| --- | --- | --- |
| `POLICY_REJECTED` | `CHARGE` | Young, uncertain, missing or unassessable face |
| `INPUT_REJECTED` | `CHARGE` | Unsupported/malformed submitted workflow or image input |
| `SCANNER_FAILURE` | `REFUND_SERVER_FAILURE` | Missing model/dependency, model failure or invalid scores |
| `WORKFLOW_FAILURE` | `REVIEW` | Other worker error; investigate the cause |

Successful output retains `images` and adds `scanner: "face-age-v1"` and
`billing_outcome: "CHARGE"`. The application backend must inspect the structured
outcome even when RunPod's transport status is `COMPLETED`; it must apply a
server-failure refund at most once per charged job. It must not blindly refund
all generic RunPod `FAILED` statuses. User cancellation and safety rejection
remain charged. This repository reports billing outcomes; it cannot mutate the
application wallet itself.

## Policy limits

This model estimates age groups; it does not verify actual age or consent.
Both 01–10 and **11–20** are blocked, including adults aged 18–20. No face, tiny
faces or uncertain ages are policy rejections, so this face-only service also
blocks images without assessable faces. It does not understand whole-image
context, prompts, consent, or reliably handle every drawing/occlusion.

Default thresholds are provisional: top older-group confidence at least 0.90,
combined 01–20 score at most 0.05. These are not calibrated under-18 probabilities.
Set `RAPIDGEN_ADULT_CONFIDENCE` / `RAPIDGEN_MAX_YOUNG_SCORE` only after representative
validation. Missing weights and inference failures never bypass scanning.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q custom_nodes rapidgen_handler.py workflow_guard.py safety_outcome.py
```

Unit tests use mocked inference and worker history to verify all-reference
checks, workflow gate injection, pre-upload blocking, no partial-image leakage,
and billing outcomes. They do not establish model accuracy or live GPU/runtime
compatibility. The Docker image includes an offline CPU inference/detector build check;
validate inference and end-to-end S3 delivery on your target worker before launch.
