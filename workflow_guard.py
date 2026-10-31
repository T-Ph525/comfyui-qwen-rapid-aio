"""Install mandatory age gates around the supported Qwen image-edit graph.

This is a graph transformer for this service, not an arbitrary ComfyUI workflow
sandbox. Unknown node classes are refused, including alternate file/S3 outputs.
"""
from copy import deepcopy
import re

ALLOWED_NODES = {
    'CheckpointLoaderSimple', 'LoraLoader', 'KSampler', 'VAEDecode',
    'TextEncodeQwenImageEditPlus', 'TextEncodeQwenImageEdit', 'LoadImage', 'EmptyLatentImage',
    'SaveImage', 'PreviewImage', 'RapidGenAgeGate',
}
GATE_TYPE = 'RapidGenAgeGate'


class InputRejected(ValueError):
    pass


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int


def protect_workflow(workflow, images, mode='edit'):
    if mode not in ('edit', 'text-to-image'):
        raise InputRejected('Unsupported generation mode.')
    text_mode = mode == 'text-to-image'
    if not isinstance(workflow, dict) or not workflow or len(workflow) > 40:
        raise InputRejected('Unsupported workflow.')
    graph = deepcopy(workflow)
    if any(not isinstance(key, str) or not isinstance(node, dict) for key, node in graph.items()):
        raise InputRejected('Invalid workflow nodes.')
    if any(node.get('class_type') not in ALLOWED_NODES or not isinstance(node.get('inputs'), dict) for node in graph.values()):
        raise InputRejected('Unsupported workflow node.')
    # Don't trust caller-supplied gates, modes, scores, or a bypass flag. Strip
    # previous gates and restore their passthrough links, then build our own.
    caller_gates = {key for key, node in graph.items() if node['class_type'] == GATE_TYPE}

    def unwrap(value, seen=None):
        if not is_link(value) or value[0] not in caller_gates:
            return value
        seen = set() if seen is None else seen
        if value[0] in seen:
            raise InputRejected('Cyclic workflow gate.')
        seen.add(value[0])
        slot = value[1]
        name = 'prompt' if slot == 4 else 'image' + str(slot + 1)
        if slot not in range(5) or name not in graph[value[0]]['inputs']:
            raise InputRejected('Invalid gate connection.')
        return unwrap(graph[value[0]]['inputs'][name], seen)

    for key, node in graph.items():
        if key not in caller_gates:
            node['inputs'] = {name: unwrap(value) for name, value in node['inputs'].items()}
    for key in caller_gates:
        del graph[key]
    loads = [key for key, node in graph.items() if node['class_type'] == 'LoadImage']
    samplers = [key for key, node in graph.items() if node['class_type'] == 'KSampler']
    outputs = [key for key, node in graph.items() if node['class_type'] in {'SaveImage', 'PreviewImage'}]
    if not ((len(loads) == 0 if text_mode else 1 <= len(loads) <= 4) and len(samplers) == 1 and 1 <= len(outputs) <= 4):
        raise InputRejected('Unexpected image inputs, sampler count or outputs for the generation mode.')
    if not isinstance(images, list) or len(images) != len(loads):
        raise InputRejected('Upload every reference image with the request.')
    names = []
    for image in images:
        name = image.get('name') if isinstance(image, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:png|jpg|jpeg|webp)', name, flags=re.I):
            raise InputRejected('Invalid input image filename.')
        if not isinstance(image.get('image'), str) or not image['image']:
            raise InputRejected('Missing input image data.')
        names.append(name)
    referenced = [graph[key]['inputs'].get('image') for key in loads]
    if len(set(names)) != len(names) or set(referenced) != set(names) or len(set(referenced)) != len(loads):
        raise InputRejected('Reference image uploads must match the workflow.')

    positive = graph[samplers[0]]['inputs'].get('positive')
    if not is_link(positive) or positive[0] not in graph or graph[positive[0]]['class_type'] != ('TextEncodeQwenImageEdit' if text_mode else 'TextEncodeQwenImageEditPlus'):
        raise InputRejected('Unsupported positive conditioning.')
    encoder = graph[positive[0]]
    prompt = encoder['inputs'].get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise InputRejected('A prompt of 1–4000 characters is required.')

    # Ensure a distinct reserved node ID even when the original graph uses 20.
    def new_id():
        number = 1000
        while str(number) in graph:
            number += 1
        return str(number)

    prompt_source = prompt
    if not text_mode:
        pre = new_id()
        gate_inputs = {'prompt': prompt}
        for slot, load in enumerate(loads):
            gate_inputs['image' + str(slot + 1)] = [load, 0]
        replacements = {load: [pre, slot] for slot, load in enumerate(loads)}
        for node in graph.values():
            for name, value in list(node['inputs'].items()):
                if is_link(value) and value[0] in replacements and value[1] == 0:
                    node['inputs'][name] = replacements[value[0]]
        graph[pre] = {'class_type': GATE_TYPE, 'inputs': gate_inputs}
        encoder['inputs']['prompt'] = [pre, 4]
        prompt_source = [pre, 4]

    # Every exported image passes an independent output gate. Even a malicious
    # extra PreviewImage cannot export a reference or an unchecked decode.
    for output in outputs:
        source = graph[output]['inputs'].get('images')
        if not is_link(source) or source[0] not in graph:
            raise InputRejected('Invalid image output connection.')
        post = new_id()
        graph[post] = {'class_type': GATE_TYPE, 'inputs': {'image1': source, 'prompt': prompt_source}}
        graph[output]['inputs']['images'] = [post, 0]
    # Enforce graph validity/cycle checks before queueing any compute.
    visiting, done = set(), set()
    def visit(key):
        if key in visiting:
            raise InputRejected('Cyclic workflow.')
        if key in done:
            return
        visiting.add(key)
        for value in graph[key]['inputs'].values():
            if is_link(value):
                if value[0] not in graph:
                    raise InputRejected('Missing workflow dependency.')
                visit(value[0])
        visiting.remove(key)
        done.add(key)
    for key in graph:
        visit(key)
    return graph
