"""The build smoke script must select CPU before loading any scanner nodes."""
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class CpuSelected(Exception):
    pass


class BuildCpuTests(unittest.TestCase):
    def test_imported_comfy_args_are_set_before_engine_initialization(self):
        # ComfyUI's imported CLI configuration defaults to cpu=False even when
        # sys.argv contains --cpu. A real GPU is unavailable on image builders.
        comfy = ModuleType('comfy')
        comfy.__path__ = []
        cli = ModuleType('comfy.cli_args')
        cli.args = SimpleNamespace(cpu=False)
        comfy.cli_args = cli
        gate = ModuleType('custom_nodes.rapidgen_age_gate')
        policy = ModuleType('custom_nodes.rapidgen_age_gate.policy')
        policy.PolicyRejected = type('PolicyRejected', (RuntimeError,), {})
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            for name in ('model.safetensors', 'detector.pt'):
                with (model_dir / name).open('wb') as stream:
                    stream.truncate(1000001)
            gate.model_paths = lambda: (model_dir, model_dir / 'detector.pt')
            def initialize():
                self.assertTrue(cli.args.cpu, 'ComfyUI would initialize CUDA on the build runner')
                raise CpuSelected()
            gate.get_engine = initialize
            modules = {'comfy': comfy, 'comfy.cli_args': cli,
                       'custom_nodes.rapidgen_age_gate': gate,
                       'custom_nodes.rapidgen_age_gate.policy': policy}
            with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)):
                with self.assertRaises(CpuSelected):
                    runpy.run_path(str(Path(__file__).parents[1] / 'scripts/validate_models.py'), run_name='__main__')
