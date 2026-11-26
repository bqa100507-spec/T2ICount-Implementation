"""Offline notebook/ZIP checks. No installation, Drive access or GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import zipfile

import pytest

from tools import train_rp011 as runner

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/RP011_Colab.ipynb'
ZIP = ROOT / 'outputs/rp011_colab_upload/RP011_source.zip'


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def cell(name):
    return ''.join(next(c['source'] for c in notebook()['cells'] if c['id'] == name))


def context(tmp_path):
    return {'Path': Path, 'json': json, 'sys': sys,
            'REPO': tmp_path / 'repo', 'ASSET_ROOT': tmp_path / 'assets',
            'RP010_RUN': tmp_path / 'RP-010B', 'OUTPUT_DIR': tmp_path / 'RP-011',
            'DATA_ROOT': tmp_path / 'assets/datasets/FSC147',
            'PROMPT_BANK': tmp_path / 'bank.json', 'NOTEBOOK_LOGS': tmp_path / 'logs'}


def test_notebook_is_clean_and_all_python_cells_compile():
    n = notebook()
    assert n['nbformat'] == 4
    for c in n['cells']:
        if c['cell_type'] == 'code':
            assert c['execution_count'] is None and c['outputs'] == []
            compile(''.join(c['source']), c['id'], 'exec')
    ids = [c['id'] for c in n['cells']]
    assert ids.index('validation') < ids.index('preflight') < ids.index('pilot')


def test_source_zip_contains_current_hnw_and_runner_without_assets():
    if not ZIP.exists():
        pytest.skip('Upload ZIP is a generated local artifact')
    with zipfile.ZipFile(ZIP) as archive:
        raw = archive.read('SOURCE_MANIFEST.json')
        manifest = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() in cell('source')
        assert set(archive.namelist()) == set(manifest['files']) | {'SOURCE_MANIFEST.json'}
        for name, expected in manifest['files'].items():
            assert not name.endswith(('.pth', '.ckpt', '.npy', '.npz', '.bin', '.safetensors', '.ipynb'))
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
            assert archive.read(name) == (ROOT / name).read_bytes(), name
        for name in ('train.py', 'utils/regression_trainer.py', 'tools/train_rp011.py',
                     'configs/rp011.json', 'configs/rp010b.json', 'requirements-colab.txt'):
            assert name in manifest['files']
    assert ZIP.stat().st_size < 2 * 1024**2


def test_source_cell_extracts_and_builds_usable_git_snapshot(tmp_path):
    if not ZIP.exists():
        pytest.skip('Upload ZIP is a generated local artifact')
    scope = context(tmp_path)
    scope.update(SOURCE_ZIP=ZIP, hashlib=hashlib, zipfile=zipfile)
    commands = []
    def git_command(command, log):
        commands.append(command)
        subprocess.run(command, cwd=scope['REPO'], check=True, capture_output=True)
    scope['stream'] = git_command
    exec(compile(cell('source'), 'source', 'exec'), scope)
    assert len(commands) == 3
    result = subprocess.check_output(['git', 'ls-files', '*.py', 'configs/*'], cwd=scope['REPO'], text=True)
    assert 'tools/train_rp011.py' in result and 'configs/rp011.json' in result
    assert 'utils/regression_trainer.py' in result
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=scope['REPO'], text=True).strip()


def test_dependency_install_protects_core_stack_and_reuses_existing_pins():
    source = cell('dependencies')
    assert "REPO / 'requirements-colab.txt'" in source
    assert "'-c', constraints" in source and "'--upgrade'," not in source
    assert "('torch', 'torchvision', 'torchaudio', 'numpy'" in source
    requirements = (ROOT / 'requirements-colab.txt').read_text()
    assert '3ba01b241669f5ade541ce990f7650a3b8f65318' in requirements
    assert 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6' in requirements


@pytest.mark.parametrize('name, variable', [('gpu', 'gpu_probe'), ('validation', 'validation')])
def test_subprocess_programs_compile_and_validation_arguments_are_real(tmp_path, name, variable):
    scope = context(tmp_path)
    commands = []
    scope['stream'] = lambda command, log: commands.append(command)
    exec(compile(cell(name), name, 'exec'), scope)
    compile(scope[variable], name + '-subprocess', 'exec')
    assert commands[0][:3] == [sys.executable, '-u', '-c']
    if name == 'validation':
        common = scope['COMMON']
        assert set(common[::2]) == {'--asset-root', '--rp010-run', '--output-dir', '--data-dir', '--prompt-bank'}


def test_gpu_probe_has_vram_and_host_memory_gates():
    source = cell('gpu')
    assert 'total >= 22 * 2**30' in source
    assert 'free >= 21 * 2**30' in source
    assert 'ram >= 8 * 2**30' in source
    assert 'torch.cuda.device_count() == 1' in source


def test_disabled_pilot_has_no_subprocess_or_drive_reads():
    scope = {'PREFLIGHT_PASSED': False}
    exec(compile(cell('pilot'), 'pilot', 'exec'), scope)
    assert scope['RUN_PILOT'] is False


def test_failed_preflight_never_opens_the_pilot_gate(tmp_path):
    scope = context(tmp_path)
    scope.update(RUNNER=tmp_path / 'runner.py', COMMON=[], PREFLIGHT_PASSED=True)
    def fail(*args):
        raise RuntimeError('preflight failed')
    scope['stream'] = fail
    with pytest.raises(RuntimeError, match='preflight failed'):
        exec(compile(cell('preflight'), 'preflight', 'exec'), scope)
    assert scope['PREFLIGHT_PASSED'] is False
    with pytest.raises(AssertionError, match='Both GPU smokes'):
        exec(compile(cell('pilot').replace('RUN_PILOT = False', 'RUN_PILOT = True'), 'pilot', 'exec'), scope)


def test_successful_preflight_and_enabled_pilot_use_only_supported_runner_commands(tmp_path, monkeypatch):
    scope = context(tmp_path)
    scope['stream'] = lambda *args: None
    exec(compile(cell('validation'), 'validation', 'exec'), scope)
    report_path = scope['OUTPUT_DIR'] / 'preflight/report.json'
    report_path.parent.mkdir(parents=True)
    arm = {'strict_model_load': True, 'starting_weights_sha256': 'same', 'loss': .1,
           'gradient_l2': 1., 'peak_allocated_bytes': 1, 'peak_reserved_bytes': 2}
    report_path.write_text(json.dumps({'status': 'passed', 'arms': {'control': arm, 'hnw': arm},
                                     'effective_config_differences': ['rrcl_hnw'], 'estimated_training_bytes': 3}))
    calls = []
    monkeypatch.setattr(runner, 'preflight', lambda *args: {'status': 'passed'})
    monkeypatch.setattr(runner, 'train_arm', lambda *args: calls.append(args[-1]))
    monkeypatch.setattr(runner, 'compare', lambda *args: [])
    commands = []
    def execute_command(command, log):
        assert str(command[2]) == str(scope['RUNNER'])
        commands.append(list(map(str, command[3:])))
        assert runner.main(commands[-1]) == 0
    scope['stream'] = execute_command
    exec(compile(cell('preflight'), 'preflight', 'exec'), scope)
    assert scope['PREFLIGHT_PASSED'] is True
    exec(compile(cell('pilot').replace('RUN_PILOT = False', 'RUN_PILOT = True'), 'pilot', 'exec'), scope)
    assert [command[0] for command in commands] == ['preflight', 'train', 'train', 'compare']
    assert calls == ['control', 'hnw']
    assert all('--allow-training' in command for command in commands[1:3])
