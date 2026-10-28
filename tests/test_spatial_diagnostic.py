"""CPU-only diagnostics tests; tiny substitutes, never load diffusion weights."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.reg_model import Count
from models.diff_unet import AttentionStore, UNetWrapper
from tools import spatial_diagnostic as diag
from tools.evaluate_rich_prompts import load_evaluation_samples, RichPromptEvaluationError
from utils.inference import predict_density, prepare_image_patches


class Tokenizer:
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 max_length=None, padding=None, **kwargs):
        ids = list(range(10, 10 + len(text.split())))
        if truncation:
            ids = ids[:max_length - 2 if add_special_tokens else max_length]
        if add_special_tokens:
            ids = [1] + ids + [2]
        if padding == 'max_length':
            ids += [2] * (max_length - len(ids))
        return {'input_ids': torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class TinyUNet(nn.Module):
    def forward(self, images):
        maps = []
        for size in (12, 24, 48):
            grid = torch.arange(size * size, dtype=images.dtype).reshape(1, 1, size, size)
            tokens = torch.arange(77, dtype=images.dtype).reshape(1, 77, 1, 1)
            maps.append((grid + tokens).repeat(len(images), 1, 1, 1))
        return [images], *maps


class TinyDecoder(nn.Module):
    def forward(self, features, text):
        density = F.avg_pool2d(features[0][:, :1], 8) + 3840
        return density, density / 4000, density / 5000


class TinyCount(Count):
    # Execute real Count.forward / real production inference, replacing only
    # expensive feature providers. No real-model equivalence is implied.
    def __init__(self):
        nn.Module.__init__(self)
        self.unet = TinyUNet()
        self.decoder = TinyDecoder()

    def extract_feat(self, images, prompts):
        return self.unet(images), torch.ones(len(images), 77, 768)


def test_checkpoint_paths_and_no_tar_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match='BASE-002'):
        diag.checkpoint_paths(tmp_path, ['BASE-002'])
    for name, relative in diag.CHECKPOINTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'weights placeholder')
    assert len(diag.checkpoint_paths(tmp_path, list(diag.CHECKPOINTS))) == 3
    assert all(p.suffix == '.pth' for p in diag.checkpoint_paths(tmp_path, ['RP-009']).values())
    with pytest.raises(ValueError):
        diag.checkpoint_paths(tmp_path, ['unknown'])
    with pytest.raises(ValueError):
        diag.checkpoint_paths(tmp_path, ['RP-003', 'RP-003'])


def test_bank_lookup_and_class_rejection(tmp_path):
    metadata = tmp_path / 'metadata.json'
    bank = tmp_path / 'bank.json'
    (tmp_path / '1.jpg').touch()
    metadata.write_text(json.dumps({'1.jpg': {'class': 'cups', 'count': 2}}))
    doc = {'metadata': {'protocol_version': 'rich-prompt-phase1-v3', 'generator': 'saved'},
           'prompts': {'1.jpg': {'class': 'cups', 'detailed': 'red cups',
                                 'generalized': 'red objects', 'status': 'ok'}}}
    bank.write_text(json.dumps(doc))
    samples, _ = load_evaluation_samples(metadata, bank, tmp_path)
    assert samples[0].detailed_prompt == 'red cups'
    doc['prompts']['1.jpg']['class'] = 'people'
    bank.write_text(json.dumps(doc))
    with pytest.raises(RichPromptEvaluationError, match='Class mismatch'):
        load_evaluation_samples(metadata, bank, tmp_path)


def test_bank_discovery_requires_explicit_choice_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, 'ROOT', tmp_path / 'repo')
    name = 'fsc147s_gemma4_26b_v3.json'
    for root, contents in ((tmp_path / 'drive', 'one'), (diag.ROOT, 'two')):
        (root / 'prompts').mkdir(parents=True)
        (root / 'prompts' / name).write_text(contents)
    with pytest.raises(ValueError, match='differ'):
        diag.discover_bank(tmp_path / 'drive')
    path = diag.ROOT / 'prompts' / name
    assert diag.discover_bank(tmp_path / 'drive', path) == path.resolve()


@pytest.mark.parametrize('words', [1, 2, 75, 90])
def test_clip_alignment_special_padding_and_truncation(words):
    mask, record = diag.prompt_tokens(Tokenizer(), ' '.join(['target'] * words), 'detailed')
    assert mask.sum() == min(words, 75)
    assert record['active_token_indices'] == list(range(1, min(words, 75) + 1))
    assert record['mask'][0] == record['mask'][76] == 0
    assert record['truncated'] == (words > 75)
    assert record['input_ids'][min(words, 75) + 1] == 2


def test_class_mask_must_match_standard_evaluator():
    diag.prompt_tokens(Tokenizer(), 'red cups', 'class')
    with pytest.raises(ValueError, match='Class mask differs'):
        diag.prompt_tokens(Tokenizer(), ' '.join(['x'] * 90), 'class')


def test_selection_is_reproducible_order_independent_and_balanced():
    samples = [SimpleNamespace(image_name='{:03d}.jpg'.format(i), class_name='class' + str(i),
                               gt_count=[1, 4, 10, 35][i % 4]) for i in range(32)]
    first = diag.select_samples(samples)
    second = diag.select_samples(list(reversed(samples)))
    assert [s.image_name for s in first] == [s.image_name for s in second]
    assert len(set(s.image_name for s in first)) == 20
    assert [sum(s.gt_count == count for s in first) for count in (1, 4, 10, 35)] == [5]*4
    assert diag.select_samples(samples, 1)[0].image_name == first[0].image_name


def test_multi_patch_coordinates_match_real_extractor():
    images = torch.arange(500 * 650).reshape(1, 1, 500, 650).float()
    prepared = prepare_image_patches(images)
    positions = diag.patch_positions(500, 650)
    assert positions == [[0, 0, 384, 384], [266, 0, 384, 384],
                         [0, 116, 384, 384], [266, 116, 384, 384]]
    for patch, (x, y, w, h) in zip(prepared[0], positions):
        assert torch.equal(patch, images[0, :, y:y+h, x:x+w])
    with pytest.raises(ValueError):
        diag.patch_positions(200, 384)


def test_existing_attention_store_resolution_averages():
    store = AttentionStore(base_size=48, max_size=48)
    for size in (12, 24, 48):
        store(torch.ones(1, size*size, 77), True, 'down')
        store(torch.ones(1, size*size, 77)*3, True, 'up')
    wrapper = UNetWrapper.__new__(UNetWrapper)
    nn.Module.__init__(wrapper)
    wrapper.size12, wrapper.size24, wrapper.size48 = 12, 24, 48
    wrapper.attn_selector = ['down_cross', 'up_cross']
    for size, attention in zip((12, 24, 48), wrapper.process_attn(store.get_average_attention())):
        assert attention.shape == (1, 77, size, size)
        assert torch.equal(attention, torch.ones_like(attention)*2)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_capture_keeps_count_forward_output_and_density_scaling(batch_size):
    model = TinyCount().eval()
    inputs = torch.zeros(1, 3, 384, 650)
    mask, _ = diag.prompt_tokens(Tokenizer(), 'red cups', 'class')
    expected = predict_density(model, inputs, 'red cups', mask, batch_size=batch_size)
    arrays, count = diag.capture_prediction(model, inputs, 'red cups', mask, batch_size)
    assert np.array_equal(expected.numpy(), arrays['density_full'])
    assert count == 384 * 650  # 3840 /64 /60 = 1 at every pixel, overlap averaged
    assert not model._forward_hooks and not model.unet._forward_hooks
    assert arrays['tokens_12'].shape == (2, 2, 12, 12)
    assert arrays['tokens_24'].shape == (2, 2, 24, 24)
    assert arrays['tokens_48'].shape == (2, 2, 48, 48)
    assert np.array_equal(arrays['attention_12'], arrays['tokens_12'].mean(1, keepdims=True))
    assert arrays['sim_x2'].shape == arrays['sim_x1'].shape == (2, 1, 48, 48)
    assert np.all(arrays['density_native'] == 3840)
    assert arrays['patch_xywh'].tolist() == diag.patch_positions(384, 650)


def test_hooks_removed_on_shape_failure():
    model = TinyCount()
    class InvalidUNet(TinyUNet):
        def forward(self, images):
            out = list(super().forward(images))
            out[1] = out[1][:, :, :6, :6]
            return tuple(out)
    model.unet = InvalidUNet()
    mask, _ = diag.prompt_tokens(Tokenizer(), 'cups', 'class')
    with pytest.raises(ValueError, match='cross-attention shape'):
        diag.capture_prediction(model, torch.zeros(1, 3, 384, 384), 'cups', mask)
    assert not model._forward_hooks and not model.unet._forward_hooks


def test_capture_does_not_change_any_count_forward_output():
    model = TinyCount().eval()
    images = torch.zeros(1, 3, 384, 384)
    mask, _ = diag.prompt_tokens(Tokenizer(), 'red cups', 'class')
    forward_mask = mask.reshape(1, 77, 1, 1)
    with torch.no_grad():
        normal = model(images, ['red cups'], forward_mask)
        with diag.SpatialCapture(model, [1, 2]):
            captured = model(images, ['red cups'], forward_mask)
    assert len(normal) == len(captured) == 4
    assert all(torch.equal(left, right) for left, right in zip(normal, captured))


def test_output_naming_and_overwrite_protection(tmp_path):
    output = diag.new_output(tmp_path / 'new')
    assert (output / 'maps').is_dir()
    with pytest.raises(FileExistsError):
        diag.new_output(output)
    assert diag.artifact_stem('31.jpg', 'RP-003', 'class') == '31__RP-003__class'
    with pytest.raises(ValueError):
        diag.artifact_stem('../31.jpg', 'RP-003', 'class')


def test_map_comparisons_are_descriptive_and_constant_safe():
    a = np.arange(24).reshape(2, 1, 3, 4).astype(float)
    assert diag.map_difference(a, a)['normalized_l1'] == 0
    assert diag.map_difference(a, a)['pearson'] == pytest.approx(1)
    assert diag.map_difference(a*0, a*0)['pearson'] is None
    with pytest.raises(ValueError):
        diag.map_difference(a, a[:1])


def test_summary_uses_absolute_and_squared_count_errors():
    rows = [{'model': 'BASE-002', 'prompt_type': 'class', 'abs_error': v} for v in (3, 4)]
    result = diag.summarize(rows)[0]
    assert result['mae'] == 3.5
    assert result['rmse'] == pytest.approx(np.sqrt(12.5))


def test_plotting_and_token_view_smoke(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = diag.new_output(tmp_path / 'run')
    image = tmp_path / '31.jpg'
    Image.new('RGB', (384, 384), 'white').save(image)
    mask, tokens = diag.prompt_tokens(Tokenizer(), 'red cups', 'class')
    arrays, count = diag.capture_prediction(TinyCount(), torch.zeros(1, 3, 384, 384), 'red cups', mask)
    np.savez_compressed(output / 'maps/31__BASE-002__class.npz', **arrays)
    record = {'image': '31.jpg', 'class': 'cups', 'image_file': {'path': str(image)},
              'patch_xywh': [[0, 0, 384, 384]], 'prompts': {'class': tokens}}
    rows = [{'image': '31.jpg', 'model': 'BASE-002', 'prompt_type': 'class',
             'gt_count': 2, 'pred_count': count, 'abs_error': count-2}]
    diag.plot_image(output, record, rows, ['BASE-002'], ['class'])
    assert len(list((output / 'figures').glob('*.png'))) == 3
    diag.write_json(output / 'manifest.json', {'selected_images': [record]})
    fig = diag.plot_token(output, '31.jpg', 'BASE-002', 'class', 1)
    plt.close(fig)
    with pytest.raises(ValueError):
        diag.plot_token(output, '31.jpg', 'BASE-002', 'class', 0)


def test_cli_defaults_do_not_run():
    args = diag.parse_args(['--asset-root', 'assets', '--output-dir', 'new'])
    assert not args.run
    assert args.num_images == 230 and args.batch_size == 1
    assert args.visualize_first_n == 20 and args.ephemeral_map_cache
    assert args.models == ['BASE-002', 'RP-003', 'RP-009']
    assert args.prompts == ['class', 'detailed']


@pytest.mark.parametrize('failure', [None, 'capture', 'comparison'])
def test_cpu_fixture_full_workflow_sequential_models_and_artifacts(tmp_path, monkeypatch, failure):
    import weakref
    import models.build
    image = tmp_path / 'fixture.jpg'
    Image.new('RGB', (384, 384), 'gray').save(image)
    # First image is NOT the designated capture image; 7.jpg is not retained.
    samples = [SimpleNamespace(image_name=name, class_name='cups', gt_count=2,
                                image_path=image, class_prompt='cups', detailed_prompt='red cups')
               for name in ('7.jpg', '31.jpg', '3319.jpg')]
    records = [{'image': sample.image_name, 'class': 'cups', 'gt_count': 2,
              'image_file': diag.fingerprint(image), 'image_hw': [384, 384],
              'patch_xywh': [[0, 0, 384, 384]],
              'prompts': {m: diag.prompt_tokens(Tokenizer(), p, m)[1]
                          for m, p in [('class', 'cups'), ('detailed', 'red cups')]}}
               for sample in samples]
    fixture_file = tmp_path / 'asset.txt'
    fixture_file.write_text('CPU fixture, not real model weights')
    clip_dir = tmp_path / 'clip'
    clip_dir.mkdir()
    (clip_dir / 'config.json').write_text('{}')
    bank = tmp_path / 'bank.json'
    bank.write_text('{"metadata": {"generator": "CPU fixture"}}')
    checkpoints = {m: fixture_file for m in diag.CHECKPOINTS}
    assets = SimpleNamespace(sd_checkpoint=fixture_file, clip_dir=clip_dir)
    monkeypatch.setattr(diag, 'preflight', lambda args: (
        assets, checkpoints, samples, records, bank, fixture_file, fixture_file, 'CPU fixture'))
    monkeypatch.setattr(diag, 'environment', lambda: {'validation': 'CPU substitutes'})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    references = []
    def build(*args, **kwargs):
        assert kwargs['device'] == 'cpu'
        assert all(reference() is None for reference in references), 'Previous model was not released'
        model = TinyCount()
        model.clip = SimpleNamespace(tokenizer=Tokenizer())
        references.append(weakref.ref(model))
        return model
    monkeypatch.setattr(models.build, 'build_t2icount', build)
    output = tmp_path / 'output'
    preliminary = tmp_path / 'preliminary'
    preliminary.mkdir()
    (preliminary / 'unchanged.txt').write_text('preserve this preliminary')
    args = diag.parse_args(['--asset-root', str(tmp_path), '--output-dir', str(output),
                            '--num-images', '3', '--device', 'cpu', '--temp-dir', str(tmp_path), '--run'])
    if failure == 'capture':
        original = diag.capture_prediction
        calls = []
        def capture(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('injected capture failure')
            return original(*args, **kwargs)
        monkeypatch.setattr(diag, 'capture_prediction', capture)
    elif failure == 'comparison':
        def compare(*args, **kwargs):
            raise RuntimeError('injected comparison failure')
        monkeypatch.setattr(diag, 'paired_maps', compare)
    if failure:
        with pytest.raises(RuntimeError, match='injected'):
            diag.run(args)
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['status'] == 'failed'
        assert manifest['storage']['ephemeral_cache']['status'] == 'removed'
        assert not Path(manifest['storage']['ephemeral_cache']['path']).exists()
        assert (preliminary / 'unchanged.txt').read_text() == 'preserve this preliminary'
        return
    diag.run(args)
    assert len(references) == 3 and all(reference() is None for reference in references)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert len(manifest['capture_equivalence']) == 6
    assert all(r['bitwise_equal'] for r in manifest['capture_equivalence'])
    assert {r['image'] for r in manifest['capture_equivalence']} == {'3319.jpg'}
    assert manifest['visualization']['selected_images'] == ['31.jpg', '3319.jpg']
    assert not Path(manifest['storage']['ephemeral_cache']['path']).exists()
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['counting']) == 6
    assert summary['dot_metrics']['status'] == 'unavailable'
    assert len(list((output / 'maps').glob('*.npz'))) == 12
    assert not list((output / 'maps').glob('7__*.npz'))
    assert not list((output / 'figures').glob('7__*.png'))
    with (output / 'results.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 18
    with (output / 'map_comparisons.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 3 * 9 * 7
    assert (output / 'results.csv').is_file() and (output / 'map_comparisons.csv').is_file()
    assert (output / 'run.log').is_file() and (output / 'figures/summary.png').is_file()


def test_notebook_code_compiles_and_is_safe_by_default():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/rich_semantic_spatial_diagnostic_230.ipynb').read_text(encoding='utf-8'))
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(cell['source'], 'notebook-cell-' + str(index), 'exec')
            assert cell['execution_count'] is None and cell['outputs'] == []
    config = notebook['cells'][1]['source']
    assert 'RUN_DIAGNOSTIC = False' in config and 'RUN_PRELIMINARY = False' in config
    assert '%%bash' not in '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')


def test_all230_selection_preserves_original20_and_preliminary():
    metadata = json.loads((diag.ROOT / 'FSC-147-S.json').read_text())
    samples = [SimpleNamespace(image_name=k, class_name=v['class'], gt_count=v['count'])
               for k, v in metadata.items()]
    original = json.loads((diag.ROOT / 'docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json').read_text())
    selected = diag.select_samples(samples, 230)
    assert len(selected) == len({s.image_name for s in selected}) == 230
    assert [s.image_name for s in selected[:20]] == [r['image'] for r in original['images']]
    assert selected[0].image_name == diag.CAPTURE_IMAGE


def test_cache_parent_rejects_drive_or_output_storage(tmp_path):
    args = diag.parse_args(['--asset-root', str(tmp_path), '--output-dir', str(tmp_path / 'output'),
                           '--temp-dir', str(tmp_path)])
    assert diag.cache_parent(args) == tmp_path.resolve()
    (tmp_path / 'output').mkdir()
    args.temp_dir = str(tmp_path / 'output')
    with pytest.raises(ValueError, match='local storage'):
        diag.cache_parent(args)


def test_selective_cache_map_metrics_equal_permanent_storage(tmp_path):
    output = diag.new_output(tmp_path / 'output')
    cache = tmp_path / 'scratch'
    cache.mkdir()
    grid = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    for model, value in [('BASE-002', grid), ('RP-003', grid[:, :, ::-1].copy())]:
        np.savez_compressed(output / 'maps' / ('31__' + model + '__class.npz'),
                            **{key: value for key in diag.MAPS})
    records = [{'image': '31.jpg'}]
    models = ['BASE-002', 'RP-003']
    expected = diag.paired_maps(output, records, models, ['class'])
    saved = output / 'maps/31__RP-003__class.npz'
    saved.rename(cache / saved.name)
    actual = diag.paired_maps(output, records, models, ['class'], cache)
    assert actual == expected
    assert not list(cache.iterdir())
