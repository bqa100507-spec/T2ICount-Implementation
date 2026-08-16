import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / 'notebooks' / 'train_colab.ipynb'
)


def _load_notebook():
    with NOTEBOOK_PATH.open('r', encoding='utf-8') as notebook_file:
        return json.load(notebook_file)


def _code_sources():
    return [
        ''.join(cell.get('source', []))
        for cell in _load_notebook()['cells']
        if cell.get('cell_type') == 'code'
    ]


def _find_code_cell(*markers):
    matches = [
        source for source in _code_sources()
        if all(marker in source for marker in markers)
    ]
    if len(matches) != 1:
        raise AssertionError(
            'Expected one notebook code cell containing {!r}, found {}'.format(
                markers, len(matches)
            )
        )
    return matches[0]


class NotebookWorkflowTests(unittest.TestCase):
    def test_smoke_cells_are_disabled_without_launching_subprocesses(self):
        globals_for_cells = {
            'RUN_INFRA_SMOKE_TESTS': False,
            'ASSET_ROOT': '/runtime-assets',
            'DRIVE_ASSET_ROOT': '/drive-assets',
            'CKPT': Path('/drive-assets/checkpoint.tar'),
        }

        with mock.patch('subprocess.run') as subprocess_run:
            for marker in (
                'NUMERICAL SMOKE PASSED',
                'TRAINING SMOKE PASSED',
                'RESUME SMOKE PASSED',
            ):
                source = _find_code_cell(marker)
                exec(compile(source, str(NOTEBOOK_PATH), 'exec'), globals_for_cells)

        subprocess_run.assert_not_called()

    def test_smoke_cells_retain_full_debug_output(self):
        for marker in (
            'NUMERICAL SMOKE PASSED',
            'TRAINING SMOKE PASSED',
            'RESUME SMOKE PASSED',
        ):
            source = _find_code_cell(marker)
            self.assertIn('if not RUN_INFRA_SMOKE_TESTS:', source)
            self.assertIn('subprocess.run(', source)
            self.assertIn('stdout=subprocess.PIPE', source)
            self.assertIn('stderr=subprocess.STDOUT', source)
            self.assertIn('print(result.stdout)', source)
            self.assertIn('RETURN CODE:', source)
            self.assertIn('FAILED', source)

    def test_limited_compute_pilot_configuration_and_live_streaming(self):
        source = _find_code_cell('baseline_500x10', 'subprocess.Popen(')

        for expected in (
            'PILOT_BATCH_SIZE = 1',
            'TRAIN_SAMPLES = 500',
            'TRAIN_SUBSET_SEED = 3407',
            'EPOCHS = 10',
            '"-u"',
            '"--train-samples", str(TRAIN_SAMPLES)',
            '"--train-subset-seed", str(TRAIN_SUBSET_SEED)',
            '"--epochs", str(EPOCHS)',
            '"--start-val", "9"',
            '"--val-epoch", "1"',
            '"--seed", "3407"',
            'subprocess.Popen(',
            'stdout=subprocess.PIPE',
            'stderr=subprocess.STDOUT',
            'text=True',
            'bufsize=1',
            'for line in process.stdout:',
            'print(line, end="", flush=True)',
            'return_code = process.wait()',
            'RETURN CODE:',
            'TOTAL TIME:',
            'PASSED',
            'FAILED',
        ):
            self.assertIn(expected, source)

        self.assertNotIn('--smoke-train-samples', source)
        self.assertNotIn('subprocess.run(', source)

    def test_notebook_declares_smoke_tests_disabled_by_default(self):
        self.assertIn(
            'RUN_INFRA_SMOKE_TESTS = False', '\n'.join(_code_sources())
        )


if __name__ == '__main__':
    unittest.main()
