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
from rapidgen_age_gate.policy import check_scores, PolicyRejected, ScannerUnavailable


class AgePolicyTests(unittest.TestCase):
    def test_young_and_11_to_20_are_rejections(self):
        for index in [0, 1]:
            scores = [0.] * 8
            scores[index] = 1.
            with self.assertRaises(PolicyRejected):
                check_scores(scores)

    def test_confident_older_group(self):
        self.assertEqual(check_scores([.001, .001, .99, .002, .002, .002, .001, .001]), 'age 21-30')

    def test_uncertainty_is_policy_not_server_failure(self):
        with self.assertRaises(PolicyRejected):
            check_scores([.05, .1, .7, .05, .04, .03, .02, .01])

    def test_bad_model_output_is_server_failure(self):
        for scores in [[], [math.nan] * 8, [1.] * 8, [0.] * 8]:
            with self.assertRaises(ScannerUnavailable):
                check_scores(scores)

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
