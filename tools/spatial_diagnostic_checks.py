"""Standard-library-only output checks shared with the notebook (no model imports)."""
import csv
import json
import math
from pathlib import Path

MAP_TYPES = ('fused_attention', 'sim_x2', 'sim_x1', 'density_native',
             'attention_12', 'attention_24', 'attention_48')


def comparison_keys(image_ids, models, prompts):
    conditions = [(model, prompt) for model in models for prompt in prompts]
    return {(image, *left, *right, kind)
            for image in image_ids
            for i, left in enumerate(conditions) for right in conditions[i+1:]
            if left[0] == right[0] or left[1] == right[1]
            for kind in MAP_TYPES}


def saved_run_state(folder, expected_images, models, prompts, expected_ids,
                    visualization_ids, capture_image='3319.jpg', require_complete=True):
    """Check coverage, not 1380 equivalence tests: only six on designated image.

    Legacy v1 preliminary outputs are accepted without the v2 storage fields.
    Full v2 output additionally requires all paired metrics, exact retained files,
    and a removed local cache. No inference/weights/dataset access is performed.
    """
    folder = Path(folder)
    if not folder.exists():
        return False, 'not found'
    try:
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        with (folder / 'results.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        valid_status = ('complete',) if require_complete else ('running', 'complete')
        if manifest.get('status') not in valid_status or summary.get('status') != 'complete':
            return False, 'manifest/summary is incomplete'
        config = manifest['config']
        if config['models'] != models or config['prompts'] != prompts or config['num_images'] != expected_images:
            return False, 'model/prompt/image-count configuration mismatch'
        selected = manifest['selected_images']
        image_ids = [record['image'] for record in selected]
        if (len(image_ids) != expected_images or len(set(image_ids)) != expected_images
                or image_ids != list(expected_ids)):
            return False, 'evaluated image identities/order mismatch'
        capture = manifest['capture_equivalence']
        capture_keys = {(r['model'], r['image'], r['prompt']) for r in capture}
        expected_capture = {(model, capture_image, prompt) for model in models for prompt in prompts}
        if (len(capture) != len(expected_capture) or capture_keys != expected_capture
                or not all(r['bitwise_equal'] is True for r in capture)):
            return False, 'designated preliminary image capture equivalence incomplete/failed'
        expected_rows = {(m, image, p) for m in models for image in image_ids for p in prompts}
        row_keys = {(r['model'], r['image'], r['prompt_type']) for r in rows}
        if len(rows) != len(expected_rows) or row_keys != expected_rows:
            return False, 'counting condition coverage mismatch'
        if any(not math.isfinite(float(r[k])) for r in rows for k in ('gt_count', 'pred_count', 'abs_error')):
            return False, 'nonfinite counting value'
        counts = summary['counting']
        if (len(counts) != len(models)*len(prompts)
                or {(r['model'], r['prompt_type']) for r in counts} != {(m, p) for m in models for p in prompts}
                or any(r['n'] != expected_images for r in counts)):
            return False, 'counting summary coverage mismatch'
        if any(not math.isfinite(float(r[k])) for r in counts for k in ('mae', 'rmse')):
            return False, 'nonfinite counting summary'
        # Existing v1 preliminary results predate selective storage and remain reusable.
        if manifest.get('schema_version') == 'fsc147s-spatial-v2' or expected_images == 230:
            retained = [image for image in image_ids if image in visualization_ids]
            if manifest['visualization'] != {'count': len(retained), 'selected_images': retained,
                                             'rule': 'Intersection with original immutable 20-image selection'}:
                return False, 'visualization selection mismatch'
            expected_npz = {'{}__{}__{}.npz'.format(Path(image).stem, m, p)
                            for image in retained for m in models for p in prompts}
            if {p.name for p in (folder / 'maps').glob('*.npz')} != expected_npz:
                return False, 'permanent NPZ identities mismatch'
            cache = manifest['storage']['ephemeral_cache']
            if cache['status'] != 'removed' or Path(cache['path']).exists():
                return False, 'temporary map cache not cleaned'
            with (folder / 'map_comparisons.csv').open(newline='', encoding='utf-8') as stream:
                maps = list(csv.DictReader(stream))
            keys = ('image', 'left_model', 'left_prompt', 'right_model', 'right_prompt', 'map_type')
            expected_maps = comparison_keys(image_ids, models, prompts)
            if len(maps) != len(expected_maps) or {tuple(r[k] for k in keys) for r in maps} != expected_maps:
                return False, 'paired map coverage mismatch'
            if any(not math.isfinite(float(r['normalized_l1'])) or
                   (r['pearson'] != '' and not math.isfinite(float(r['pearson']))) for r in maps):
                return False, 'nonfinite paired map value'
            aggregate_keys = ('left_model', 'left_prompt', 'right_model', 'right_prompt', 'map_type')
            aggregates = summary['map_comparisons']
            expected_aggregates = {key[1:] for key in expected_maps}
            if (len(aggregates) != len(expected_aggregates)
                    or {tuple(r[k] for k in aggregate_keys) for r in aggregates} != expected_aggregates
                    or any(r['n'] != expected_images for r in aggregates)):
                return False, 'paired map summary coverage mismatch'
            if any(not math.isfinite(float(r['mean_normalized_l1'])) or
                   (r['mean_pearson'] is not None and not math.isfinite(float(r['mean_pearson']))) or
                   not 0 <= r['pearson_valid_n'] <= expected_images for r in aggregates):
                return False, 'invalid paired map aggregate value'
            expected_figures = {'summary.png'}
            for record in selected:
                if record['image'] not in retained:
                    continue
                stem = Path(record['image']).stem
                expected_figures.add(stem + '__full_density.png')
                for index in range(len(record['patch_xywh'])):
                    expected_figures.update({stem + '__patch{:02d}.png'.format(index),
                                             stem + '__attention_patch{:02d}.png'.format(index)})
            if {p.name for p in (folder / 'figures').glob('*.png')} != expected_figures:
                return False, 'figure identities mismatch'
        return True, 'complete: {} images, {} conditions, {} designated-image capture checks'.format(
            expected_images, len(rows), len(capture))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, 'missing/unreadable/inconsistent output: ' + str(exc)
