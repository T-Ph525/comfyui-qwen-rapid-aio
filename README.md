# Qwen-Rapid-AIO with MiVOLO V2 age gates

This ComfyUI Qwen image-edit worker uses `runpod/worker-comfyui:5.8.5-base`.
It installs the pinned [ComfyUI-MiVolo-V2 extension](https://github.com/vdeng-ai/ComfyUI-MiVolo-V2)
and uses its all-subject detector/crop pipeline inside the mandatory local
`RapidGenAgeGate`. You keep the existing API sockets, input/output gates and
pre-upload RunPod history guard. There is no contextual or prompt scan.

The adapter reads MiVOLO's **raw, unrounded age estimate** and rejects an image
if any detected subject is estimated below 18. It scans every image in batches
of up to four, not just the first image. Model estimates do not verify actual
age, consent or safety. Detection misses and age errors remain possible.

## Build and deployment

You can build and run the worker with:

```bash
docker build -t my-comfy-workflow .
docker run --rm --gpus all -p 8188:8188 -e SERVE_API_LOCALLY=true my-comfy-workflow
```

You must deploy the rebuilt image to RunPod before the endpoint changes. The old
wizard image does not contain these changes. The build pins the extension,
MiVOLO library, age model and detector revisions, preserves the base torch/CUDA
wheels, and performs an offline CPU processor/model/detector smoke check. A model
or dependency loading error fails the build. The upstream handler checksum guard
also refuses an unreviewed change to the pre-upload integration.

You can find the installation commands, manual weight placement and node wiring
in [docs/mivolo-setup.md](docs/mivolo-setup.md).

## Model paths inside the backend

| Asset | Location |
| --- | --- |
| Complete `iitolstykh/mivolo_v2` snapshot | `/comfyui/models/mivolo/iitolstykh/mivolo_v2/` |
| Person/face detector | `/comfyui/models/yolo/yolov8x_person_face.pt` |
| Upstream extension | `/comfyui/custom_nodes/ComfyUI-MiVolo-V2/` |
| Mandatory service gate | `/comfyui/custom_nodes/rapidgen_age_gate/` |

The age snapshot needs its JSON configuration, processor, Python model code and
`model.safetensors`, not just one weight file. These are downloaded at build time;
the service gate loads them locally and does not download them during a job.
MiVOLO uses custom Python model code (`trust_remote_code=True`); the Dockerfile
pins that code to a specific Hugging Face snapshot. The YOLO checkpoint is a
trusted, pinned upstream PyTorch artifact.

You can override these server settings:

```dotenv
RAPIDGEN_AGE_MODEL_DIR=/comfyui/models/mivolo/iitolstykh/mivolo_v2
RAPIDGEN_FACE_DETECTOR=/comfyui/models/yolo/yolov8x_person_face.pt
RAPIDGEN_AGE_DEVICE=cpu
RAPIDGEN_MIN_AGE=18
```

CPU/FP32 is the default to preserve generation VRAM. You can select `cuda` only
after validating memory and latency. You can raise the minimum age, but values
below 18 or invalid configuration are scanner faults, never bypasses. Retire the
old age-group confidence settings: `RAPIDGEN_ADULT_CONFIDENCE` and
`RAPIDGEN_MAX_YOUNG_SCORE` are no longer used. If you mounted the old model path,
update it; a volume over `/comfyui/models` hides baked-in model files.

## Mandatory gates and supported requests

You send `api-workflow.json` as the `input.workflow` object and include each
reference in `input.images`, using the inherited worker upload format:

```json
{
  "input": {
    "workflow": "<replace with the API workflow object, not a string>",
    "images": [{"name": "images.png", "image": "<base64 image>"}]
  }
}
```

The wrapper replaces caller-supplied gates and inserts trusted input and output
gates, including for older requests. It checks all 1–4 references before sampling
and every exported output before SaveImage/PreviewImage and the S3 upload loop.
Unknown node types, missing uploads, unsafe filenames and graph cycles are
rejected before queueing. This supports the repository's Qwen graph; it is not a
general-purpose ComfyUI sandbox. Keep the RunPod API key server-side.

Only a fully successful history reaches the inherited S3 uploader. Neither
policy blocks nor scanner faults return partial images. S3 bucket settings and
retention stay under the operator's control. The inherited worker can return
base64 when S3 is not configured. Both example workflows use SaveImage (the
worker skips temporary PreviewImage results) and the actual `v23/` checkpoint.

## Outcomes and credit handling

Errors remain structured worker output without `images`:

| `error_code` | `billing_outcome` | Meaning |
| --- | --- | --- |
| `POLICY_REJECTED` | `CHARGE` | Estimated age below minimum, missing/tiny face, or unassessable subject |
| `INPUT_REJECTED` | `CHARGE` | Unsupported/malformed workflow or submitted image input |
| `SCANNER_FAILURE` | `REFUND_SERVER_FAILURE` | Missing dependency/model, inference failure or invalid age output |
| `WORKFLOW_FAILURE` | `REVIEW` | Other failure requiring investigation |

Successful output retains the inherited `images` format and adds
`scanner: "face-age-v1"`, `scanner_model: "mivolo-v2-age-v1"` and
`billing_outcome: "CHARGE"`. The scanner protocol tag stays compatible with the
existing application backend; the model tag identifies MiVOLO. Your application
must inspect this result even when RunPod says `COMPLETED`, and apply a confirmed
server-fault refund at most once. Cancellation, security/policy rejection and
unsuitable results remain charged. A generic worker failure is not automatically
refundable. This repository does not modify application wallets.

## Policy limits

The service requires an assessable face for every detected person, even though
MiVOLO supports body-only estimation. It rejects body-only detections, images
with no detections, faces smaller than 32 pixels, invalid bounds, more than 64
detections, more than four batch images, and images larger than 16 megapixels.
The detector uses `conf=0.4` and `iou=0.7`, with upstream `Use persons and faces`
and `All` modes fixed by server code. No client-controlled age, comparison,
selection mode or threshold is accepted.

Unlike the old age-bin model, MiVOLO can return a continuous estimate such as
17.6. That value blocks without rounding to 18. An estimate of exactly 18 passes
the literal default comparison. This does not establish actual adulthood.
There is no calibrated under-18 probability or confidence threshold here; you
must evaluate boundary cases and detection coverage on representative data
before public deployment. Drawings, occlusion, groups and small faces can fail.

## Validation

You can run the light regression suite without installing ML dependencies:

```bash
python -m pip install numpy Pillow
git clone https://github.com/vdeng-ai/ComfyUI-MiVolo-V2.git /tmp/rapidgen-mivolo-tests
git -C /tmp/rapidgen-mivolo-tests checkout 54354ff4cdd388974d7b3ac938a59ac88a58d27f
MIVOLO_TEST_SOURCE=/tmp/rapidgen-mivolo-tests/py/age_gender_node.py python -m unittest discover -s tests -v
python -m compileall -q custom_nodes scripts rapidgen_handler.py workflow_guard.py safety_outcome.py
```

The pipeline tests run the actual pinned upstream crop/association code with
mocked model/detector results. If you omit `MIVOLO_TEST_SOURCE`, those integration
tests are explicitly skipped; CI supplies it. Tests cover raw age boundaries,
later faces/batch images, all-reference gating, pre-upload blocking, faults and
billing outcomes. They do not establish model accuracy, GPU compatibility or
live S3 delivery. The Docker build runs a separate real, offline CPU ML smoke
check; you must still validate the built image on your target worker.

## Upstream licenses

The extension declares CC-BY-SA-4.0; its library has an attribution/share-alike
license. The MiVOLO V2 Hugging Face model card lists Apache-2.0, while the
person/face detector lists AGPL-3.0 and Ultralytics has its own licensing options.
Review the relevant upstream terms for your commercial deployment and any
redistribution. Pinned source and model revisions are recorded in the Dockerfile.
