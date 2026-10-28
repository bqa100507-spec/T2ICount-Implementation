"""CPU-only coverage and notebook-order fixtures, no GitHub/GPU jobs."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import spatial_diagnostic_checks as checks

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/rich_semantic_spatial_diagnostic_230.ipynb'


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def configuration():
    namespace = {}
    exec(notebook()['cells'][1]['source'], namespace)
    return namespace


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def full_report(tmp_path):
    """Synthetic tables and filenames only; these are not model results."""
    config = configuration()
    images = config['EXPECTED_IMAGE_IDS']
    models, prompts = config['MODEL_SELECTION'], config['PROMPT_SELECTION']
    folder = tmp_path / 'full'
    (folder / 'maps').mkdir(parents=True)
    (folder / 'figures').mkdir()
    records = [{'image': name, 'patch_xywh': [[0, 0, 384, 384]]} for name in images]
    manifest = {'schema_version': 'fsc147s-spatial-v2', 'status': 'complete',
                'config': {'num_images': 230, 'models': models, 'prompts': prompts},
                'selected_images': records,
                'capture_equivalence': [{'model': m, 'prompt': p, 'image': '3319.jpg', 'bitwise_equal': True}
                                        for m in models for p in prompts],
                'visualization': {'count': 20, 'selected_images': config['VISUALIZATION_IMAGE_IDS'],
                                  'rule': 'Intersection with original immutable 20-image selection'},
                'storage': {'ephemeral_cache': {'status': 'removed', 'path': str(tmp_path / 'removed-cache')}}}
    rows = [{'model': m, 'image': image, 'prompt_type': p,
             'gt_count': 1, 'pred_count': 1, 'abs_error': 0}
            for m in models for image in images for p in prompts]
    keys = ('image', 'left_model', 'left_prompt', 'right_model', 'right_prompt', 'map_type')
    comparisons = [{**dict(zip(keys, key)), 'normalized_l1': 0, 'pearson': ''}
                   for key in sorted(checks.comparison_keys(images, models, prompts))]
    aggregate_keys = keys[1:]
    summary = {'status': 'complete',
               'counting': [{'model': m, 'prompt_type': p, 'n': 230, 'mae': 0, 'rmse': 0}
                            for m in models for p in prompts],
               'map_comparisons': [{**dict(zip(aggregate_keys, key)), 'n': 230,
                                    'mean_normalized_l1': 0, 'mean_pearson': None, 'pearson_valid_n': 0}
                                   for key in sorted({tuple(r[k] for k in aggregate_keys) for r in comparisons})]}
    for image in config['VISUALIZATION_IMAGE_IDS']:
        stem = Path(image).stem
        for m in models:
            for p in prompts:
                (folder / 'maps' / '{}__{}__{}.npz'.format(stem, m, p)).touch()
        for suffix in ('__full_density.png', '__patch00.png', '__attention_patch00.png'):
            (folder / 'figures' / (stem + suffix)).touch()
    (folder / 'figures/summary.png').touch()
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder / 'summary.json').write_text(json.dumps(summary))
    write_csv(folder / 'results.csv', rows)
    write_csv(folder / 'map_comparisons.csv', comparisons)
    return folder, config, manifest, summary, rows, comparisons


def validate(report):
    folder, config = report[:2]
    return checks.saved_run_state(folder, 230, config['MODEL_SELECTION'], config['PROMPT_SELECTION'],
                                  config['EXPECTED_IMAGE_IDS'], config['VISUALIZATION_IMAGE_IDS'])


def test_full1380_coverage_needs_only_six_capture_checks(full_report):
    assert len(full_report[4]) == 1380
    assert len(full_report[5]) == 14490
    assert len(full_report[2]['capture_equivalence']) == 6
    assert validate(full_report)[0]


@pytest.mark.parametrize('fault', ['counting', 'maps', 'capture', 'npz', 'cache', 'summary'])
def test_completion_rejects_incomplete_or_mismatched_artifacts(full_report, fault):
    folder, config, manifest, summary, rows, comparisons = full_report
    if fault == 'counting':
        rows[-1] = rows[0]  # Same row count, wrong identities.
        write_csv(folder / 'results.csv', rows)
    elif fault == 'maps':
        comparisons[-1] = comparisons[0]
        write_csv(folder / 'map_comparisons.csv', comparisons)
    elif fault == 'capture':
        manifest['capture_equivalence'][0]['image'] = config['EXPECTED_IMAGE_IDS'][1]
    elif fault == 'npz':
        (folder / 'maps/7__BASE-002__class.npz').touch()  # Unselected permanent map.
    elif fault == 'cache':
        Path(manifest['storage']['ephemeral_cache']['path']).mkdir()
    elif fault == 'summary':
        summary['map_comparisons'][0]['n'] = 229
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder / 'summary.json').write_text(json.dumps(summary))
    assert not validate(full_report)[0]


def test_legacy_preliminary_is_reusable_without_v2_storage_fields(full_report):
    folder, config, manifest, summary, rows, _ = full_report
    manifest['schema_version'] = 'fsc147s-spatial-v1'
    manifest['selected_images'] = manifest['selected_images'][:1]
    manifest['config']['num_images'] = 1
    del manifest['storage'], manifest['visualization']
    for r in summary['counting']:
        r['n'] = 1
    rows = [r for r in rows if r['image'] == '3319.jpg']
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder / 'summary.json').write_text(json.dumps(summary))
    write_csv(folder / 'results.csv', rows)
    assert checks.saved_run_state(folder, 1, config['MODEL_SELECTION'], config['PROMPT_SELECTION'],
                                   ['3319.jpg'], config['VISUALIZATION_IMAGE_IDS'])[0]


def test_notebook_embeds_exact_shared_checks_and_no_source_zip():
    cells = notebook()['cells']
    shared = (ROOT / 'tools/spatial_diagnostic_checks.py').read_text(encoding='utf-8')
    assert cells[4]['source'].startswith(shared)
    assert 'SOURCE_ZIP' not in ''.join(c['source'] for c in cells)
    assert 'git' in cells[6]['source'] and 'REPO_URL' in cells[6]['source']
    assert 'clone' in cells[6]['source'] and '--ff-only' in cells[6]['source']


@pytest.mark.parametrize('reuse_full', [False, True])
def test_notebook_all_cells_execute_in_order_without_new_inference(tmp_path, monkeypatch, full_report, reuse_full):
    cells = notebook()['cells']
    ns = {}
    exec(cells[1]['source'], ns)
    mount = tmp_path / 'drive'
    asset_root = mount / 'MyDrive/assets'
    asset_root.mkdir(parents=True)
    ns.update(ASSET_ROOT=asset_root, OUTPUT_DIR=full_report[0] if reuse_full else tmp_path / 'absent',
              PRELIMINARY_OUTPUT_DIR=tmp_path / 'no-preliminary', RUN_DIAGNOSTIC=reuse_full)
    # Remap only the notebook's Colab mount probe, without mounting anything.
    ns['Path'] = lambda value: mount if value == '/content/drive' else Path(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('No subprocess is allowed for saved-results/default mode')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    monkeypatch.setitem(sys.modules, 'IPython.display', SimpleNamespace(display=lambda *a: None, Image=lambda **k: None))
    for cell in cells[3:]:
        if cell['cell_type'] == 'code':
            exec(compile(cell['source'], cell['id'], 'exec'), ns)
    assert ns['NEED_EXECUTION'] is False
    assert ns['FULL_COMPLETE'] == reuse_full


def test_notebook_fresh_github_setup_and_preflight_cell_order(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    cells, ns = notebook()['cells'], configuration()
    repo = tmp_path / 'clone'
    ns.update(REPO=repo, NEED_EXECUTION=True)
    commands = []
    revision = 'a' * 40
    def stream(command, **kwargs):
        commands.append([str(v) for v in command])
        assert command[0:2] == ['git', 'clone']
        repo.mkdir()
        for relative in ('tools/spatial_diagnostic.py', 'tools/spatial_diagnostic_checks.py',
                         'tests/test_spatial_diagnostic.py', 'docs/SPATIAL_DIAGNOSTIC.md',
                         'docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json',
                         'notebooks/rich_semantic_spatial_diagnostic_230.ipynb',
                         'FSC-147-S.json', 'requirements-colab.txt', 'configs/v1-inference.yaml'):
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    def output(command, **kwargs):
        args = command[3:]
        if args[:1] == ['rev-parse']:
            return revision
        if args[:1] == ['branch']:
            return 'rich-prompt-alignment'
        if args[:1] == ['remote']:
            return ns['REPO_URL']
        return ''
    ns['stream'] = stream
    monkeypatch.setattr(subprocess, 'check_output', output)
    previous = Path.cwd()
    try:
        exec(cells[6]['source'], ns)
        assert commands[0][-2] == ns['REPO_URL']
        assert ns['SOURCE_REVISION'] == revision
        # Preflight function definitions must exist only after source/dependencies.
        # Mock the actual preflight, not the notebook's argument construction.
        from tools import spatial_diagnostic as diag
        expected = ns['EXPECTED_IMAGE_IDS']
        bank = tmp_path / 'bank'
        bank.write_text('bank fixture')
        ns['EXPECTED_BANK_SHA256'] = ns['sha256'](bank)
        ns.update(NEED_NEW_FULL=True, OUTPUT_DIR=tmp_path / 'new-output', PROMPT_BANK=bank)
        captured = []
        def preflight(args):
            captured.append(args)
            return None, {}, [], [{'image': i} for i in expected], bank
        monkeypatch.setattr(diag, 'preflight', preflight)
        # verify_source already exercised above with the original snapshot hash.
        ns['verify_source'] = lambda: None
        exec(cells[10]['source'], ns)
        assert captured[0].num_images == 230
        assert captured[0].visualize_first_n == 20
        assert captured[0].capture_image == '3319.jpg'
        preliminary_args = diag.parse_args(ns['command_args'](tmp_path / 'preliminary', 1))
        assert preliminary_args.num_images == 1
    finally:
        os.chdir(previous)


def test_git_ignore_runtime_payloads_but_not_source():
    for relative in ('outputs/run/results.csv', 'weights.tar', 'maps.npz', 'data.npy',
                     'datasets/FSC147/image.jpg', 'prompts/fsc147s_gemma4_26b_v3.json',
                     '.spatial_diagnostic_tmp/cache.npz'):
        assert subprocess.run(['git', 'check-ignore', '-q', '--no-index', relative], cwd=ROOT).returncode == 0
    for relative in ('tools/spatial_diagnostic.py', 'datasets/dataset.py',
                     'notebooks/rich_semantic_spatial_diagnostic_230.ipynb'):
        assert subprocess.run(['git', 'check-ignore', '-q', '--no-index', relative], cwd=ROOT).returncode == 1
