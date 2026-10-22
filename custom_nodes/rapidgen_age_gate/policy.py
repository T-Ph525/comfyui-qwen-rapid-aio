"""Face-age decisions only. No contextual, prompt, or vision-LLM scan."""
import math

MODEL_ID = "prithivMLmods/facial-age-detection"
EXPECTED_LABELS = (
    "age 01-10", "age 11-20", "age 21-30", "age 31-40",
    "age 41-55", "age 56-65", "age 66-80", "age 80 +",
)


class PolicyRejected(RuntimeError):
    """A policy decision: the accepted attempt remains charged."""


class ScannerUnavailable(RuntimeError):
    """A scanner fault: the service can refund this confirmed failure."""


def check_scores(scores, adult_confidence=0.90, max_young_score=0.05):
    """Scores are model probabilities, NOT calibrated under-18 probabilities.

    Both 01-10 and 11-20 are disallowed. This intentionally also rejects some
    adults aged 18-20. Deployment thresholds need representative validation.
    """
    if len(scores) != 8 or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1 for p in scores):
        raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: invalid age scores")
    if abs(sum(scores) - 1) > 0.01:
        raise ScannerUnavailable("RAPIDGEN_SCANNER_FAILURE: invalid probability total")
    winner = max(range(8), key=scores.__getitem__)
    if winner < 2:
        raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: young age group")
    if scores[winner] < adult_confidence or scores[0] + scores[1] > max_young_score:
        raise PolicyRejected("RAPIDGEN_POLICY_REJECTED: age uncertain")
    return EXPECTED_LABELS[winner]
