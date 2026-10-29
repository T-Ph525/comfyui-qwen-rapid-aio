"""Translate ONLY errors from our known ComfyUI node into trusted outcomes.

Call from the existing RunPod handler after reading ComfyUI's execution history.
Return this object as the RunPod handler's normal result. Raising a generic worker
exception for a policy block loses its meaning and can cause incorrect refunds.
"""
GATE_NODE = 'RapidGenAgeGate'


def safety_outcome(history):
    messages = history.get('status', {}).get('messages', [])
    for message in messages:
        if not isinstance(message, (tuple, list)) or len(message) != 2:
            continue
        event, detail = message
        if event != 'execution_error' or not isinstance(detail, dict):
            continue
        if detail.get('node_type') != GATE_NODE:
            continue
        exception = str(detail.get('exception_type', '')).rsplit('.', 1)[-1]
        # Never classify by exception_message: it may contain untrusted text.
        if exception == 'PolicyRejected':
            return {'error_code': 'POLICY_REJECTED', 'message': 'The MiVOLO age check rejected this attempt.'}
        if exception == 'ScannerUnavailable':
            return {'error_code': 'SCANNER_FAILURE', 'message': 'The MiVOLO age scanner could not run.'}
    return None
