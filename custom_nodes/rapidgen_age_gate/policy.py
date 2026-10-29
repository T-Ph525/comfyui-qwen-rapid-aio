"""MiVOLO age decisions only; no contextual, prompt or gender policy."""
import math

MODEL_ID = 'iitolstykh/mivolo_v2'
SCANNER_ID = 'mivolo-v2-age-v1'


class PolicyRejected(RuntimeError):
    """A policy decision: the accepted attempt remains charged."""


class ScannerUnavailable(RuntimeError):
    """A scanner fault: the service can refund this confirmed failure."""


def check_age(age, minimum_age=18.0):
    """Compare raw regression output, never an age rounded for display.

    A passing prediction is not proof of actual age. This model does not expose
    a calibrated under-18 confidence score. Thresholds require validation.
    """
    if isinstance(minimum_age, bool) or not isinstance(minimum_age, (float, int)) or not math.isfinite(minimum_age) or not 18 <= minimum_age <= 100:
        raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: invalid minimum age')
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 120:
        raise ScannerUnavailable('RAPIDGEN_SCANNER_FAILURE: invalid age estimate')
    if age < minimum_age:
        raise PolicyRejected('RAPIDGEN_POLICY_REJECTED: estimated age below minimum')
    return age
