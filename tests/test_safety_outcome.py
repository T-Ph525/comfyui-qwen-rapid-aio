import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('safety_outcome', Path(__file__).parents[1] / 'safety_outcome.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def history(node_type, exception_type, text=''):
    return {'status': {'messages': [['execution_error', {'node_type': node_type, 'exception_type': exception_type, 'exception_message': text}]]}}


class OutcomeTests(unittest.TestCase):
    def test_policy_rejection_is_not_a_refundable_fault(self):
        result = module.safety_outcome(history('RapidGenAgeGate', 'rapidgen_age_gate.policy.PolicyRejected'))
        self.assertEqual(result['error_code'], 'POLICY_REJECTED')

    def test_scanner_fault_is_explicit(self):
        result = module.safety_outcome(history('RapidGenAgeGate', 'rapidgen_age_gate.policy.ScannerUnavailable'))
        self.assertEqual(result['error_code'], 'SCANNER_FAILURE')

    def test_untrusted_text_cannot_claim_a_refund(self):
        self.assertIsNone(module.safety_outcome(history('OtherNode', 'RuntimeError', 'SCANNER_FAILURE')))
        self.assertIsNone(module.safety_outcome(history('RapidGenAgeGate', 'RuntimeError', 'ScannerUnavailable')))

    def test_success_has_no_error_outcome(self):
        self.assertIsNone(module.safety_outcome({'status': {'messages': []}}))


if __name__ == '__main__':
    unittest.main()
