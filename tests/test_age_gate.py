import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1] / 'custom_nodes' / 'rapidgen_age_gate'
spec = importlib.util.spec_from_file_location('rapidgen_age_gate', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from rapidgen_age_gate.policy import check_age, PolicyRejected, ScannerUnavailable


class AgePolicyTests(unittest.TestCase):
    def test_raw_under_18_is_rejected_without_rounding(self):
        for age in [0., 10., 17., 17.6, 17.999]:
            with self.assertRaises(PolicyRejected):
                check_age(age)

    def test_18_and_older_pass_literal_threshold(self):
        for age in [18., 18.01, 20., 40.]:
            self.assertEqual(check_age(age), age)

    def test_operator_can_raise_but_not_lower_threshold(self):
        with self.assertRaises(PolicyRejected):
            check_age(20., 21.)
        for threshold in [17., math.nan, math.inf, 101., True]:
            with self.assertRaises(ScannerUnavailable):
                check_age(40., threshold)

    def test_bad_model_output_is_server_failure(self):
        for age in [None, math.nan, math.inf, -1., 200., '35', True]:
            with self.assertRaises(ScannerUnavailable):
                check_age(age)

    def test_every_reference_is_checked_and_passed_through(self):
        checked = []
        class Engine:
            def check(self, image):
                checked.append(image)
        images = [object() for _ in range(4)]
        with patch.object(module, 'get_engine', return_value=Engine()):
            result = module.RapidGenAgeGate().check(*[images[0], 'unchanged prompt', *images[1:]])
        self.assertEqual(checked, images)
        self.assertEqual(result, (*images, 'unchanged prompt'))

    def test_a_later_reference_cannot_bypass_gate(self):
        class Engine:
            def check(self, image):
                if image == 'young':
                    raise PolicyRejected('blocked')
        with patch.object(module, 'get_engine', return_value=Engine()):
            with self.assertRaises(PolicyRejected):
                module.RapidGenAgeGate().check('adult', 'prompt', 'young')

    def test_dependency_failure_never_passes(self):
        with patch.object(module, 'get_engine', side_effect=ImportError('missing package')):
            with self.assertRaises(ScannerUnavailable):
                module.RapidGenAgeGate().check(object(), 'prompt')

    def test_gate_is_not_cached(self):
        self.assertTrue(math.isnan(module.RapidGenAgeGate.IS_CHANGED()))


if __name__ == '__main__':
    unittest.main()
