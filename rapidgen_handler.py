"""Guard the pinned worker-comfyui handler without forking its S3/upload logic."""
from copy import deepcopy
import importlib.util
import threading

from workflow_guard import InputRejected, protect_workflow
from safety_outcome import safety_outcome

_LOCK = threading.Lock()
_UPSTREAM = None


class StopBeforeUpload(BaseException):
    """Bypass upstream's broad Exception handlers, then handle in our wrapper."""
    def __init__(self, outcome):
        self.outcome = outcome


def inspect_history(history):
    outcome = safety_outcome(history)
    if outcome:
        outcome['billing_outcome'] = 'CHARGE' if outcome['error_code'] == 'POLICY_REJECTED' else 'REFUND_SERVER_FAILURE'
        return outcome
    status = history.get('status', {})
    errors = [item for item in status.get('messages', []) if isinstance(item, (list, tuple)) and item and item[0] in {'execution_error', 'execution_interrupted'}]
    if errors or status.get('status_str') != 'success' or status.get('completed') is not True:
        return {'error_code': 'WORKFLOW_FAILURE', 'billing_outcome': 'REVIEW', 'message': 'The workflow did not finish successfully.'}
    return None


def install_history_guard(upstream):
    original = upstream.get_history
    def guarded(prompt_id):
        history = original(prompt_id)
        if prompt_id not in history:
            raise StopBeforeUpload({'error_code': 'WORKFLOW_FAILURE', 'billing_outcome': 'REVIEW', 'message': 'The workflow history is unavailable.'})
        outcome = inspect_history(history[prompt_id])
        if outcome:
            raise StopBeforeUpload(outcome)
        return history
    upstream.get_history = guarded
    return upstream


def load_upstream():
    global _UPSTREAM
    if _UPSTREAM is None:
        spec = importlib.util.spec_from_file_location('rapidgen_upstream', '/upstream_handler.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        _UPSTREAM = install_history_guard(upstream)
    return _UPSTREAM


def handler(job, upstream=None):
    # One ComfyUI instance and shared temporary filenames per worker. Serialize
    # requests even if a future SDK/config enables concurrent handler calls.
    with _LOCK:
        try:
            protected = deepcopy(job)
            data = protected.get('input', {})
            if not isinstance(data, dict):
                raise InputRejected('Invalid job input.')
            data['workflow'] = protect_workflow(data.get('workflow'), data.get('images'))
        except (InputRejected, TypeError, AttributeError):
            return {'error_code': 'INPUT_REJECTED', 'billing_outcome': 'CHARGE', 'message': 'Unsupported or invalid image-edit request.'}
        try:
            result = (upstream or load_upstream()).handler(protected)
        except StopBeforeUpload as stopped:
            return stopped.outcome
        except Exception:
            # Unclassified worker errors need investigation; don't guess that
            # they qualify for refunds and don't leak prompts/stack traces.
            return {'error_code': 'WORKFLOW_FAILURE', 'billing_outcome': 'REVIEW', 'message': 'The worker could not complete the request.'}
        if not isinstance(result, dict) or not result.get('images') or result.get('error') or result.get('errors'):
            return {'error_code': 'WORKFLOW_FAILURE', 'billing_outcome': 'REVIEW', 'message': 'The worker did not deliver a complete result.'}
        return {**result, 'billing_outcome': 'CHARGE', 'scanner': 'face-age-v1', 'scanner_model': 'mivolo-v2-age-v1'}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
