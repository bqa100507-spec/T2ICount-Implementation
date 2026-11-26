#!/usr/bin/env python
"""DIAG-002: CPU-only post-hoc analysis of immutable DIAG-001 artifacts.

This module deliberately imports no T2ICount/model/checkpoint/torch utilities.
It also runs standalone when embedded in the analysis notebook.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata

MODELS = ('BASE-002', 'RP-003', 'RP-009')
PROMPTS = ('class', 'detailed')
MAP_TYPES = ('fused_attention', 'sim_x2', 'sim_x1', 'density_native',
             'attention_12', 'attention_24', 'attention_48')
ORIGINAL_IMAGES = ('3319.jpg', '7706.jpg', '918.jpg', '31.jpg', '605.jpg',
                   '290.jpg', '343.jpg', '336.jpg', '6305.jpg', '3169.jpg',
                   '6353.jpg', '3721.jpg', '3699.jpg', '3019.jpg', '5651.jpg',
                   '63.jpg', '3287.jpg', '75.jpg', '3703.jpg', '5356.jpg')
BINS = ('GT=1', 'GT=2-5', 'GT=6-20', 'GT>20')
EPSILON = 1e-12  # Minimum positive total density mass; never added to/clipped into D.
TIE_TOLERANCE = 1e-9
SCHEMA = 'diag002-cpu-v1'


class ValidationError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__('Input validation failed:\n- ' + '\n- '.join(self.issues))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    def clean(item):
        if isinstance(item,dict): return {str(k):clean(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)): return [clean(v) for v in item]
        if isinstance(item,np.generic): return clean(item.item())
        if isinstance(item,float) and not math.isfinite(item): return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict('records')
    if not rows:
        raise ValueError('Refusing an empty table without a declared schema: ' + str(path))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None or (isinstance(v, float) and not math.isfinite(v)) else v
                             for k, v in row.items()})


def bin_name(count):
    if count == 1:
        return BINS[0]
    if 2 <= count <= 5:
        return BINS[1]
    if 6 <= count <= 20:
        return BINS[2]
    if count > 20:
        return BINS[3]
    raise ValueError('GT count outside the approved bins: ' + str(count))


def condition_pairs():
    conditions = list(itertools.product(MODELS, PROMPTS))
    return [(left, right) for i, left in enumerate(conditions) for right in conditions[i+1:]
            if left[0] == right[0] or left[1] == right[1]]


def archive_name(image, model, prompt):
    if Path(image).name != image or model not in MODELS or prompt not in PROMPTS:
        raise ValueError('Unsafe artifact key')
    return '{}__{}__{}.npz'.format(Path(image).stem, model, prompt)


def expected_coordinates(height, width):
    if min(height, width) < 384:
        raise ValueError('Image smaller than original inference patch')
    return [[min(j*384, width-384), min(i*384, height-384), 384, 384]
            for i in range((height-1)//384+1) for j in range((width-1)//384+1)]


def normalize_density(density, epsilon=EPSILON):
    """No clipping or pseudo target: invalid probability assumptions stay explicit."""
    density = np.asarray(density, dtype=np.float64)
    if not np.isfinite(density).all():
        return None, 'nonfinite'
    if np.any(density < 0):
        return None, 'negative_values'
    mass = float(density.sum())
    if mass <= epsilon:
        return None, 'zero_or_near_zero_mass'
    return density / mass, 'valid'


def correlation(x, y, method='pearson', controls=None):
    """Pairwise complete cases; partial Spearman = OLS residuals of ranks.

    Controls are log1p(GT), log1p(reference absolute error). Ranks use average
    ties. No p-values/multiple-comparison selection or predictive fitting.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if controls is not None:
        controls = np.asarray(controls, float)
        valid &= np.isfinite(controls).all(1)
    x, y = x[valid], y[valid]
    n, rank = len(x), 0
    if n < 3:
        return {'r': None, 'n': n, 'status': 'insufficient_n', 'control_rank': rank}
    if method == 'spearman':
        x, y = rankdata(x), rankdata(y)
    elif method != 'pearson':
        raise ValueError('Unknown correlation method')
    if controls is not None:
        covariates = controls[valid]
        if method == 'spearman':
            covariates = np.column_stack([rankdata(covariates[:, j]) for j in range(covariates.shape[1])])
        design = np.column_stack([np.ones(n), covariates])
        rank = int(np.linalg.matrix_rank(design))
        if n <= rank + 2:
            return {'r': None, 'n': n, 'status': 'insufficient_residual_df', 'control_rank': rank}
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    # Relative tolerance handles exact linear predictions after residualization.
    if np.linalg.norm(x-x.mean()) <= 1e-10 * max(1., np.linalg.norm(x)) or \
            np.linalg.norm(y-y.mean()) <= 1e-10 * max(1., np.linalg.norm(y)):
        return {'r': None, 'n': n, 'status': 'constant_or_zero_residual', 'control_rank': rank}
    return {'r': float(np.corrcoef(x, y)[0, 1]), 'n': n, 'status': 'valid', 'control_rank': rank}


def bootstrap_mean(values, repetitions=2000, seed=3407):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or repetitions < 1:
        raise ValueError('Bootstrap requires observations and positive repetitions')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _read_table(path, columns, numeric, nullable=()):
    try:
        table = pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise ValidationError([str(path) + ': ' + str(exc)]) from exc
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValidationError([path.name + ': missing columns ' + repr(sorted(missing))])
    for key in numeric:
        empty = table[key].eq('')
        try:
            table[key] = pd.to_numeric(table[key].replace('', np.nan), errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValidationError([path.name + ': nonnumeric ' + key]) from exc
        allowed_missing = empty if key in nullable else np.zeros(len(table), dtype=bool)
        if ((~np.isfinite(table[key])) & ~allowed_missing).any():
            raise ValidationError([path.name + ': non-finite ' + key])
    return table


def validate_tables(root):
    """Require exact original keys/coverage; never fix rows, prompts or source summaries."""
    root = Path(root)
    try:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValidationError(['Missing/unreadable manifest or summary: ' + str(exc)]) from exc
    issues = []
    if manifest.get('status') != 'complete' or summary.get('status') != 'complete':
        issues.append('DIAG-001 is not complete')
    if manifest.get('schema_version') != 'fsc147s-spatial-v2':
        issues.append('Require actual full-230 fsc147s-spatial-v2 schema')
    records = manifest.get('selected_images', [])
    ids = [r.get('image') for r in records]
    if len(ids) != 230 or len(set(ids)) != 230 or any(not isinstance(i, str) or Path(i).name != i for i in ids):
        issues.append('Require 230 unique safe image identifiers')
    if manifest.get('visualization', {}).get('selected_images') != list(ORIGINAL_IMAGES) or \
            manifest.get('visualization', {}).get('count') != 20:
        issues.append('Original 20 visualization images changed/missing')
    if manifest.get('config', {}).get('models') != list(MODELS) or manifest.get('config', {}).get('prompts') != list(PROMPTS):
        issues.append('Exact three checkpoint/two prompt identifiers required')
    inference = manifest.get('inference', {})
    if inference.get('patch_size') != 384 or inference.get('stride') != 384:
        issues.append('Incompatible patch geometry')
    if issues:
        raise ValidationError(issues)
    record_by_id = {r['image']: r for r in records}
    for record in records:
        try:
            gt = float(record['gt_count'])
            if not math.isfinite(gt) or int(gt) != gt:
                raise ValueError('non-integer/non-finite GT')
            bin_name(gt)
            h, w = record['image_hw']
            if record['patch_xywh'] != expected_coordinates(h, w):
                raise ValueError('patch coordinates disagree with production extraction')
            for prompt in PROMPTS:
                token = record['prompts'][prompt]
                active = token['active_token_indices']
                if (len(token['input_ids']) != 77 or len(token['tokens']) != 77 or len(token['mask']) != 77
                        or active != list(range(1, len(active)+1)) or not 1 <= len(active) <= 75
                        or [i for i,v in enumerate(token['mask']) if v != 0] != active
                        or not all(token['mask'][i] == 1 for i in active)):
                    raise ValueError('incompatible saved token/mask alignment')
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(record['image'] + ': ' + str(exc))
    results = _read_table(root/'results.csv', ('image','class','model','prompt_type','prompt','gt_count','pred_count','abs_error','patch_count'),
                          ('gt_count','pred_count','abs_error','patch_count'))
    keys = ['image','model','prompt_type']
    actual = set(results[keys].itertuples(index=False, name=None))
    expected = set(itertools.product(ids, MODELS, PROMPTS))
    if len(results) != 1380 or results.duplicated(keys).any() or actual != expected:
        issues.append('Counting joins: expected 1380 exact unique image/model/prompt keys; missing={}, extra={}, duplicates={}'.format(
            len(expected-actual), len(actual-expected), int(results.duplicated(keys).sum())))
    if actual == expected:
        for row in results.itertuples():
            record = record_by_id[row.image]
            if row.gt_count != record['gt_count']:
                issues.append(row.image + ': GT mismatch')
            if row.prompt != record['prompts'][row.prompt_type]['prompt']:
                issues.append(row.image + ': prompt text mismatch')
            if row.patch_count != len(record['patch_xywh']):
                issues.append(row.image + ': patch count mismatch')
        if any(row['class'] != record_by_id[row['image']]['class'] for row in results.to_dict('records')):
            issues.append('Class mapping mismatch')
        if not np.allclose(results.abs_error, np.abs(results.pred_count-results.gt_count), rtol=1e-9, atol=1e-9):
            issues.append('Stored absolute errors disagree with predictions/GT')
    maps = _read_table(root/'map_comparisons.csv', ('image','left_model','left_prompt','right_model','right_prompt','map_type','normalized_l1','pearson'),
                       ('normalized_l1','pearson'), nullable=('pearson',))
    map_keys = ['image','left_model','left_prompt','right_model','right_prompt','map_type']
    expected_maps = {(image,*left,*right,kind) for image in ids for left,right in condition_pairs() for kind in MAP_TYPES}
    actual_maps = set(maps[map_keys].itertuples(index=False, name=None))
    if len(maps) != 14490 or maps.duplicated(map_keys).any() or actual_maps != expected_maps:
        issues.append('Map joins: expected 14490 exact unique keys; missing={}, extra={}, duplicates={}'.format(
            len(expected_maps-actual_maps),len(actual_maps-expected_maps),int(maps.duplicated(map_keys).sum())))
    if (maps.normalized_l1.lt(0) | maps.normalized_l1.gt(1+1e-7)).any() or \
            (maps.pearson.dropna().abs() > 1+1e-7).any():
        issues.append('Normalized map metric outside its documented range')
    summary_counts = {(r['model'],r['prompt_type']):r for r in summary.get('counting',[])}
    if len(summary.get('counting',[])) != 6 or set(summary_counts) != set(itertools.product(MODELS,PROMPTS)):
        issues.append('Counting summary condition coverage mismatch')
    else:
        for condition, data in results.groupby(['model','prompt_type']):
            item = summary_counts[condition]
            if item.get('n') != 230 or not np.allclose([item.get('mae'),item.get('rmse')],
                    [data.abs_error.mean(),np.sqrt(np.mean(data.abs_error**2))], rtol=1e-8,atol=1e-8):
                issues.append('Source summary count metrics disagree: ' + str(condition))
    aggregate_keys=['left_model','left_prompt','right_model','right_prompt','map_type']
    saved_aggregates=summary.get('map_comparisons',[])
    saved_lookup={tuple(row[k] for k in aggregate_keys):row for row in saved_aggregates}
    if len(saved_aggregates)!=63 or set(saved_lookup)!= {(*left,*right,kind) for left,right in condition_pairs() for kind in MAP_TYPES}:
        issues.append('Source map summary coverage mismatch')
    else:
        for key,data in maps.groupby(aggregate_keys,sort=True):
            item=saved_lookup[key];valid=data.pearson.dropna()
            expected_pearson=float(valid.mean()) if len(valid) else None
            if (item.get('n')!=230 or item.get('pearson_valid_n')!=len(valid)
                    or not np.isclose(item.get('mean_normalized_l1',np.nan),data.normalized_l1.mean(),rtol=1e-8,atol=1e-8)
                    or (expected_pearson is None and item.get('mean_pearson') is not None)
                    or (expected_pearson is not None and not np.isclose(item.get('mean_pearson',np.nan),expected_pearson,rtol=1e-8,atol=1e-8))):
                issues.append('Source map summary disagrees with keyed records: '+str(key))
    expected_npz = {archive_name(i,m,p) for i in ORIGINAL_IMAGES for m,p in itertools.product(MODELS,PROMPTS)}
    if {p.name for p in (root/'maps').glob('*.npz')} != expected_npz:
        issues.append('Require exactly 120 NPZ files for the unchanged 20 images')
    capture = manifest.get('capture_equivalence',[])
    if len(capture)!=6 or {(r['model'],r['image'],r['prompt']) for r in capture} != \
            {(m,'3319.jpg',p) for m,p in itertools.product(MODELS,PROMPTS)} or not all(r['bitwise_equal'] is True for r in capture):
        issues.append('DIAG-001 designated-image capture checks incomplete')
    if issues:
        raise ValidationError(issues)
    results['gt_bin'] = results.gt_count.map(bin_name)
    # Explicit stable joins; CSV row order never determines pairing or bootstrap.
    results = results.sort_values(keys).reset_index(drop=True)
    maps = maps.sort_values(map_keys).reset_index(drop=True)
    return manifest, summary, results, maps


def validate_npz(root, manifest, results, image_root=None):
    records = {r['image']:r for r in manifest['selected_images']}
    lookup = results.set_index(['image','model','prompt_type'])
    issues, statuses, images = [], [], {}
    for image in ORIGINAL_IMAGES:
        record = records[image]
        image_path = Path(image_root)/image if image_root else Path(record['image_file']['path'])
        if image_path.is_file():
            if sha256(image_path) != record['image_file']['sha256']:
                issues.append(image + ': RGB source fingerprint mismatch')
            else:
                with Image.open(image_path) as im:
                    if list(im.size) != record['image_hw'][::-1]:
                        issues.append(image + ': RGB dimension mismatch')
                images[image] = image_path.resolve()
        else:
            issues.append(image + ': original RGB unavailable; set --image-root to existing matching images')
        for model,prompt in itertools.product(MODELS,PROMPTS):
            path = Path(root)/'maps'/archive_name(image,model,prompt)
            try:
                with np.load(path,allow_pickle=False) as data:
                    expected = set(MAP_TYPES)|{'tokens_12','tokens_24','tokens_48','density_full','patch_xywh'}
                    if set(data.files) != expected:
                        raise ValueError('NPZ array keys mismatch: ' + str(set(data.files)^expected))
                    patches, tokens = len(record['patch_xywh']), len(record['prompts'][prompt]['active_token_indices'])
                    for kind in expected:
                        value = data[kind]
                        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                            raise ValueError('nonnumeric/nonfinite ' + kind)
                        if kind=='patch_xywh' and value.dtype.kind not in 'iu':
                            raise ValueError('patch coordinates must be integer arrays')
                        if kind.startswith('tokens_'):
                            size = int(kind.split('_')[-1]); shape=(patches,tokens,size,size)
                        elif kind.startswith('attention_'):
                            size = int(kind.split('_')[-1]); shape=(patches,1,size,size)
                        elif kind=='density_full': shape=(1,1,*record['image_hw'])
                        elif kind=='patch_xywh': shape=(patches,4)
                        else: shape=(patches,1,48,48)
                        if value.shape != shape:
                            raise ValueError('{} shape {} != {}'.format(kind,value.shape,shape))
                    if not np.array_equal(data['patch_xywh'],record['patch_xywh']):
                        raise ValueError('patch coordinate mismatch')
                    for size in (12,24,48):
                        if not np.allclose(data['attention_'+str(size)],data['tokens_'+str(size)].mean(1,keepdims=True),rtol=2e-6,atol=1e-7):
                            raise ValueError('content-token mean mismatch at '+str(size))
                    mass = float(data['density_full'].sum(dtype=np.float64))
                    if not np.isclose(mass,lookup.loc[(image,model,prompt),'pred_count'],rtol=2e-6,atol=1e-5):
                        raise ValueError('full density mass disagrees with source counting/scaling')
                    probability,status = normalize_density(data['density_full'])
                    statuses.append({'image':image,'model':model,'prompt_type':prompt,'mass':mass,
                                     'min_density':float(data['density_full'].min()),'normalization_status':status})
            except (OSError,ValueError,KeyError) as exc:
                issues.append(path.name+': '+str(exc))
    if issues:
        raise ValidationError(issues)
    return statuses, images


def counting_tables(results, repetitions=2000):
    summaries, paired, pair_summaries, prompt_rows, winner_rows = [],[],[],[],[]
    for (model,prompt), group in results.groupby(['model','prompt_type'],sort=True):
        for name, data in [('all',group)]+[(b,group[group.gt_bin==b]) for b in BINS]:
            signed = (data.pred_count-data.gt_count).to_numpy()
            n=len(data)
            summaries.append({'model':model,'prompt_type':prompt,'gt_bin':name,'n':n,
                'mae':float(data.abs_error.mean()) if n else None,
                'rmse':float(np.sqrt(np.mean(data.abs_error**2))) if n else None,
                'signed_bias':float(signed.mean()) if n else None,
                'median_absolute_error':float(data.abs_error.median()) if n else None,
                'over_n':int(np.sum(signed>TIE_TOLERANCE)),'under_n':int(np.sum(signed<-TIE_TOLERANCE)),
                'exact_n':int(np.sum(np.abs(signed)<=TIE_TOLERANCE)),
                'over_pct':float(100*np.mean(signed>TIE_TOLERANCE)) if n else None,
                'under_pct':float(100*np.mean(signed<-TIE_TOLERANCE)) if n else None,
                'over_error_mean':float(np.maximum(signed,0).mean()) if n else None,
                'under_error_mean':float(np.maximum(-signed,0).mean()) if n else None,
                'pred_mean':float(data.pred_count.mean()) if n else None,
                **{'pred_'+key:float(data.pred_count.quantile(q)) if n else None
                   for key,q in [('p05',.05),('p25',.25),('median',.5),('p75',.75),('p95',.95)]},
                'pred_max':float(data.pred_count.max()) if n else None})
    lookup=results.set_index(['image','model','prompt_type'])
    ids=sorted(results.image.unique())
    for prompt in PROMPTS:
        for reference,candidate in itertools.combinations(MODELS,2):
            these=[]
            for image in ids:
                left,right=lookup.loc[(image,reference,prompt)],lookup.loc[(image,candidate,prompt)]
                gain=float(left.abs_error-right.abs_error)
                row={'image':image,'prompt_type':prompt,'reference':reference,'candidate':candidate,
                     'gt_count':float(left.gt_count),'gt_bin':left.gt_bin,
                     'pred_reference':float(left.pred_count),'pred_candidate':float(right.pred_count),
                     'error_reference':float(left.abs_error),'error_candidate':float(right.abs_error),
                     'delta_pred':float(right.pred_count-left.pred_count),'error_gain':gain,
                     'outcome':'better' if gain>TIE_TOLERANCE else 'worse' if gain<-TIE_TOLERANCE else 'tie',
                     'over_error_gain':float(max(left.pred_count-left.gt_count,0)-max(right.pred_count-right.gt_count,0)),
                     'under_error_gain':float(max(left.gt_count-left.pred_count,0)-max(right.gt_count-right.pred_count,0))}
                paired.append(row);these.append(row)
            for name,data in [('all',these)]+[(b,[r for r in these if r['gt_bin']==b]) for b in BINS]:
                n=len(data)
                gains=np.array([r['error_gain'] for r in data])
                lo,hi=bootstrap_mean(gains,repetitions) if name=='all' else (None,None)
                pair_summaries.append({'reference':reference,'candidate':candidate,'prompt_type':prompt,'gt_bin':name,'n':n,
                    'candidate_wins':sum(r['outcome']=='better' for r in data),
                    'candidate_loses':sum(r['outcome']=='worse' for r in data),'ties':sum(r['outcome']=='tie' for r in data),
                    'mean_error_gain':float(gains.mean()) if n else None,'gain_ci_low':lo,'gain_ci_high':hi,
                    'mean_delta_pred':float(np.mean([r['delta_pred'] for r in data])) if n else None,
                    'prediction_lower_n':sum(r['delta_pred']<-TIE_TOLERANCE for r in data),
                    'mean_over_error_gain':float(np.mean([r['over_error_gain'] for r in data])) if n else None,
                    'mean_under_error_gain':float(np.mean([r['under_error_gain'] for r in data])) if n else None})
        for model in MODELS:
            sole,shared,not_best=0,0,0
            for image in ids:
                errors=np.array([lookup.loc[(image,m,prompt),'abs_error'] for m in MODELS])
                best=np.abs(errors-errors.min())<=TIE_TOLERANCE
                if best[MODELS.index(model)]:
                    if best.sum()==1: sole+=1
                    else: shared+=1
                else: not_best+=1
            winner_rows.append({'model':model,'prompt_type':prompt,'n':len(ids),
                                'sole_best':sole,'shared_best':shared,'not_best':not_best})
    for model in MODELS:
        for image in ids:
            left,right=lookup.loc[(image,model,'class')],lookup.loc[(image,model,'detailed')]
            prompt_rows.append({'image':image,'model':model,'gt_count':float(left.gt_count),'gt_bin':left.gt_bin,
                'class_pred':float(left.pred_count),'detailed_pred':float(right.pred_count),
                'class_error':float(left.abs_error),'detailed_error':float(right.abs_error),
                'delta_pred':float(right.pred_count-left.pred_count),'delta_abs_error':float(right.abs_error-left.abs_error)})
    return {'counting_summary':pd.DataFrame(summaries),'paired_counting':pd.DataFrame(paired),
            'paired_counting_summary':pd.DataFrame(pair_summaries),'prompt_sensitivity':pd.DataFrame(prompt_rows),
            'model_winners':pd.DataFrame(winner_rows)}


def spatial_tables(results,maps):
    lookup=results.set_index(['image','model','prompt_type'])
    joined=[]
    for item in maps.to_dict('records'):
        left=lookup.loc[(item['image'],item['left_model'],item['left_prompt'])]
        right=lookup.loc[(item['image'],item['right_model'],item['right_prompt'])]
        within=item['left_model']==item['right_model']
        joined.append({**item,'comparison':item['left_model']+'/'+item['left_prompt']+' -> '+item['right_model']+'/'+item['right_prompt'],
                       'kind':'prompt' if within else 'checkpoint',
                       'gt_count':float(left.gt_count),'gt_bin':left.gt_bin,
                       'reference_error':float(left.abs_error),'error_gain':float(left.abs_error-right.abs_error),
                       'delta_abs_error':float(right.abs_error-left.abs_error),
                       'delta_pred':float(right.pred_count-left.pred_count),
                       'one_minus_pearson':float(1-item['pearson']) if np.isfinite(item['pearson']) else np.nan})
    joined=pd.DataFrame(joined)
    associations, groups, extremes=[],[],[]
    for (comparison,kind,map_type),data in joined.groupby(['comparison','kind','map_type'],sort=True):
        outcomes=np.where(data.error_gain>TIE_TOLERANCE,'better',np.where(data.error_gain<-TIE_TOLERANCE,'worse','tie'))
        for label,selected in [('all',data)]+[(name,data[outcomes==name]) for name in ('better','worse','tie')]:
            valid=selected.pearson.dropna()
            groups.append({'comparison':comparison,'kind':kind,'map_type':map_type,'candidate_outcome':label,
                'n':len(selected),'pearson_valid_n':len(valid),
                'pearson_mean':float(valid.mean()) if len(valid) else None,
                'pearson_median':float(valid.median()) if len(valid) else None,
                **{'l1_'+key:float(selected.normalized_l1.quantile(q)) if len(selected) else None
                   for key,q in [('p25',.25),('median',.5),('p75',.75)]}})
        targets=['delta_pred','delta_abs_error'] if kind=='prompt' else ['error_gain']
        for target in targets:
            for feature in ('normalized_l1','one_minus_pearson'):
                strata=[('all',data)]+[(b,data[data.gt_bin==b]) for b in BINS]
                for stratum,selected in strata:
                    for method in ('pearson','spearman'):
                        for adjusted in ([False,True] if stratum=='all' else [False]):
                            controls=np.column_stack([np.log1p(selected.gt_count),np.log1p(selected.reference_error)]) if adjusted else None
                            result=correlation(selected[feature],selected[target],method,controls)
                            associations.append({'comparison':comparison,'kind':kind,'map_type':map_type,
                                'feature':feature,'target':target,'gt_bin':stratum,'method':method,
                                'adjustment':'log1p_gt_and_reference_error' if adjusted else 'none',**result})
        # Fixed top three per direction/metric, identities retained for inspection.
        for metric,ascending in [('normalized_l1',False),('pearson',True)]:
            for row in data.dropna(subset=[metric]).sort_values([metric,'image'],ascending=[ascending,True]).head(3).to_dict('records'):
                extremes.append({'comparison':comparison,'map_type':map_type,'selection_metric':metric,
                                 'image':row['image'],'normalized_l1':float(row['normalized_l1']),
                                 'pearson':float(row['pearson']) if np.isfinite(row['pearson']) else None,
                                 'error_gain':float(row['error_gain']),'gt_count':float(row['gt_count'])})
    return {'spatial_joined':joined,'spatial_associations':pd.DataFrame(associations),
            'spatial_groups':pd.DataFrame(groups),'spatial_extremes':pd.DataFrame(extremes)}


def density_shapes(root,statuses):
    rows=[]
    for image in ORIGINAL_IMAGES:
        densities={}
        for condition in itertools.product(MODELS,PROMPTS):
            with np.load(Path(root)/'maps'/archive_name(image,*condition),allow_pickle=False) as data:
                densities[condition]=data['density_full'][0,0].astype(np.float64)
        for left,right in condition_pairs():
            first,second=densities[left],densities[right]
            p,pstatus=normalize_density(first);q,qstatus=normalize_density(second)
            mass1,mass2=float(first.sum()),float(second.sum())
            valid=p is not None and q is not None
            rows.append({'image':image,'left_model':left[0],'left_prompt':left[1],
                'right_model':right[0],'right_prompt':right[1],
                'mass_reference':mass1,'mass_candidate':mass2,'delta_mass':mass2-mass1,
                'mass_ratio':mass2/mass1 if mass1>EPSILON else None,
                'reference_normalization':pstatus,'candidate_normalization':qstatus,
                'shape_valid':valid,'total_variation':float(.5*np.abs(p-q).sum()) if valid else None,
                'shape_pearson':correlation(p.ravel(),q.ravel())['r'] if valid else None})
    return pd.DataFrame(rows)


def save_figure(fig,path,qa):
    """Render every figure; detect clipped labels and blank/corrupt PNGs."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.canvas.get_width_height()
    outside=[]
    # Axis objects cache tick labels beyond the view limits; those are not
    # rendered by Matplotlib and must not be mistaken for clipped visible text.
    hidden_ticks=set()
    for axis_container in fig.axes:
        for axis in (axis_container.xaxis,axis_container.yaxis):
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not axis_container.axison or tick.get_loc()<low or tick.get_loc()>high:
                    hidden_ticks.update((id(tick.label1),id(tick.label2)))
    for text in fig.findobj(Text):
        if id(text) in hidden_ticks or not text.get_visible() or not text.get_text().strip():
            continue
        box=text.get_window_extent(renderer)
        if np.isfinite([box.x0,box.y0,box.x1,box.y1]).all() and \
                (box.x0 < -3 or box.y0 < -3 or box.x1 > width+3 or box.y1 > height+3):
            outside.append(text.get_text())
    fig.savefig(path,dpi=120,facecolor='white')
    plt.close(fig)
    with Image.open(path) as image:
        image.load()
        extrema=image.convert('RGB').getextrema()
        if all(low==high for low,high in extrema):
            raise ValueError('Blank figure: '+str(path))
        size=list(image.size)
    qa.append({'file':str(Path(path).name),'pixels':size,'clipped_text':outside,'sha256':sha256(path)})
    if outside:
        raise ValueError('Clipped figure text in {}: {}'.format(Path(path).name,outside[:4]))


def _matrix(associations,kind,target,adjustment):
    comparisons=list(dict.fromkeys(associations[associations.kind==kind].comparison))
    if kind=='checkpoint':
        comparisons=[c for c in comparisons if '/class -> ' in c and c.endswith('/class')]
    matrix=np.full((len(MAP_TYPES),len(comparisons)),np.nan)
    counts=np.zeros(matrix.shape,int)
    for i,kind_name in enumerate(MAP_TYPES):
        for j,comparison in enumerate(comparisons):
            row=associations[(associations.comparison==comparison)&(associations.map_type==kind_name)&
                (associations.feature=='normalized_l1')&(associations.target==target)&
                (associations.gt_bin=='all')&(associations.method=='spearman')&
                (associations.adjustment==adjustment)]
            if len(row)==1:
                value=row.iloc[0]
                matrix[i,j]=value.r if pd.notna(value.r) else np.nan
                counts[i,j]=value.n
    return comparisons,matrix,counts


def aggregate_plots(output,results,tables,qa):
    import matplotlib.pyplot as plt
    colors=('#386cb0','#ef8a62','#4daf4a')
    bins_colors=('#377eb8','#4daf4a','#ff7f00','#984ea3')
    destination=output/'figures'
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    limit=max(results.pred_count.max(),results.gt_count.max())*1.05
    for i,prompt in enumerate(PROMPTS):
        for j,model in enumerate(MODELS):
            data=results[(results.model==model)&(results.prompt_type==prompt)]
            ax=axes[i,j]
            ax.scatter(data.gt_count,data.pred_count,s=13,alpha=.45,color=colors[j])
            ax.plot([0,limit],[0,limit],'k--',lw=.8)
            ax.set(xscale='symlog',yscale='symlog',xlim=(0,limit),ylim=(min(0,results.pred_count.min()*1.05),limit),
                   xlabel='GT count',ylabel='Predicted count',title=model+' / '+prompt+' (n=230)')
    fig.suptitle('Observed counting responses; symlog axes, no dot localization GT')
    save_figure(fig,destination/'count_vs_gt.png',qa)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,prompt in enumerate(PROMPTS):
        data=[results[(results.model==m)&(results.prompt_type==prompt)] for m in MODELS]
        for j,key in enumerate(('abs_error','signed')):
            arrays=[d.abs_error if key=='abs_error' else d.pred_count-d.gt_count for d in data]
            axes[i,j].boxplot(arrays,labels=MODELS,showfliers=True)
            axes[i,j].set_yscale('symlog',linthresh=1)
            axes[i,j].set(title=prompt+' / '+('absolute error' if j==0 else 'signed error'),ylabel='Count error (symlog)')
            axes[i,j].axhline(0,color='gray',lw=.7)
    fig.suptitle('Error distributions; all outliers retained')
    save_figure(fig,destination/'error_distributions.png',qa)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,prompt in zip(axes,PROMPTS):
        for model,color in zip(MODELS,colors):
            values=np.sort(results[(results.model==model)&(results.prompt_type==prompt)].pred_count)
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=model,color=color)
        ax.set(xscale='symlog',xlabel='Predicted count (symlog)',ylabel='Empirical CDF',title=prompt)
        ax.legend(fontsize=8)
    fig.suptitle('Prediction magnitude distributions (n=230 per checkpoint/prompt)')
    save_figure(fig,destination/'prediction_distributions.png',qa)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    paired=tables['paired_counting'];class_pairs=paired[paired.prompt_type=='class']
    for (reference,candidate),data in class_pairs.groupby(['reference','candidate'],sort=False):
        for ax,key in zip(axes,('error_gain','delta_pred')):
            values=np.sort(data[key])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=reference+' -> '+candidate)
            ax.axvline(0,color='gray',lw=.7)
            ax.set_xscale('symlog',linthresh=1)
            ax.set_ylabel('Empirical CDF')
    axes[0].set_xlabel('Reference error - candidate error (positive = gain)')
    axes[1].set_xlabel('Candidate prediction - reference prediction')
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle('Paired class-name counting changes; 230 same-image pairs')
    save_figure(fig,destination/'paired_counting_gains.png',qa)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    counts=results[(results.model==MODELS[0])&(results.prompt_type=='class')].gt_bin.value_counts()
    for j,model in enumerate(MODELS):
        data=tables['counting_summary'][(tables['counting_summary'].model==model)&
             (tables['counting_summary'].prompt_type=='class')].set_index('gt_bin')
        for ax,key in zip(axes,('pred_mean','signed_bias')):
            ax.bar(np.arange(4)+(j-1)*.25,[data.loc[b,key] for b in BINS],width=.24,label=model,color=colors[j])
    for ax in axes:
        ax.set_xticks(range(4));ax.set_xticklabels([b+'\nn='+str(counts.get(b,0)) for b in BINS])
        ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean predicted count');axes[1].set_ylabel('Mean prediction - GT')
    fig.suptitle('Class-name prediction magnitude and signed bias by GT bin')
    save_figure(fig,destination/'gt_bin_magnitude_bias.png',qa)
    associations=tables['spatial_associations']
    for kind,target,filename in [('checkpoint','error_gain','spatial_association_matrix.png'),
                                 ('prompt','delta_pred','prompt_association_matrix.png')]:
        fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
        for ax,adjustment in zip(axes,('none','log1p_gt_and_reference_error')):
            comparisons,matrix,valid_n=_matrix(associations,kind,target,adjustment)
            rendered=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
            labels=[c.replace('/class','').replace('/detailed','').replace(' -> ','\n-> ') for c in comparisons]
            if kind=='prompt': labels=[c.split('/')[0] for c in comparisons]
            ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,fontsize=9)
            ax.set_yticks(range(len(MAP_TYPES)));ax.set_yticklabels(MAP_TYPES,fontsize=9)
            ax.set_title('Raw Spearman' if adjustment=='none' else 'Partial Spearman: GT + reference error',fontsize=10)
            for i,j in itertools.product(range(len(MAP_TYPES)),range(len(labels))):
                label=('{:.2f}'.format(matrix[i,j]) if np.isfinite(matrix[i,j]) else 'NA')+'\nn='+str(valid_n[i,j])
                ax.text(j,i,label,ha='center',va='center',fontsize=9)
            fig.colorbar(rendered,ax=ax,label='Correlation (not causal)',shrink=.8)
        fig.suptitle('Normalized map L1 association with '+('class counting gain' if kind=='checkpoint' else 'within-model detailed-minus-class count change'))
        save_figure(fig,destination/filename,qa)
    joined=tables['spatial_joined']
    comparisons=list(dict.fromkeys(joined[(joined.left_prompt=='class')&(joined.right_prompt=='class')].comparison))
    fig,axes=plt.subplots(3,2,figsize=(15,11),constrained_layout=True)
    short=('fused','HSCM2','HSCM1','density','attn12','attn24','attn48')
    for i,comparison in enumerate(comparisons):
        data=joined[joined.comparison==comparison]
        for j,metric in enumerate(('normalized_l1','pearson')):
            ax=axes[i,j]
            for offset,outcome,color in [(-.18,'better','#4daf4a'),(.18,'worse','#e78ac3')]:
                sign=data.error_gain>TIE_TOLERANCE if outcome=='better' else data.error_gain<-TIE_TOLERANCE
                arrays=[data[(data.map_type==kind)&sign][metric].dropna().to_numpy() for kind in MAP_TYPES]
                arrays=[a if len(a) else np.array([np.nan]) for a in arrays]
                box=ax.boxplot(arrays,positions=np.arange(7)+offset,widths=.3,patch_artist=True,showfliers=True)
                for patch in box['boxes']: patch.set_facecolor(color)
            ax.set_xticks(range(7));ax.set_xticklabels(short,fontsize=8)
            ax.set_title(comparison.replace('/class','')+' / '+metric,fontsize=10)
    fig.suptitle('Class map distributions: green=candidate better, pink=worse; ties in tables, undefined r excluded')
    save_figure(fig,destination/'spatial_group_distributions.png',qa)
    # Fixed comparison panels, never choose a pair/map because its correlation is large.
    for index,comparison in enumerate(comparisons):
        fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
        data=joined[joined.comparison==comparison]
        for ax,kind_name in zip(axes.flat,MAP_TYPES):
            subset=data[data.map_type==kind_name]
            for label,color in zip(BINS,bins_colors):
                part=subset[subset.gt_bin==label]
                ax.scatter(part.normalized_l1,part.error_gain,s=13,alpha=.55,color=color,label=label+' n='+str(len(part)))
            ax.axhline(0,color='gray',lw=.6)
            ax.set(xlabel='Normalized map L1',ylabel='Counting error gain',title=kind_name)
        axes.flat[-1].axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles,labels,loc='center',fontsize=10)
        fig.suptitle(comparison+' | class-name association; points are images, not localization labels')
        save_figure(fig,destination/'spatial_scatter_pair{}.png'.format(index+1),qa)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,model in zip(axes,MODELS):
        data=tables['prompt_sensitivity'][tables['prompt_sensitivity'].model==model]
        for label,color in zip(BINS,bins_colors):
            part=data[data.gt_bin==label]
            ax.scatter(part.delta_pred,part.delta_abs_error,s=14,alpha=.55,color=color,label=label)
        ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
        ax.set(xscale='symlog',yscale='symlog',xlabel='Detailed - class predicted count',
               ylabel='Detailed - class absolute error',title=model)
        ax.legend(fontsize=8)
    fig.suptitle('Within-checkpoint prompt sensitivity (symlog axes)')
    save_figure(fig,destination/'prompt_count_changes.png',qa)
    shape=tables['density_shape_pairs']
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for (left,right),data in shape[(shape.left_prompt=='class')&(shape.right_prompt=='class')].groupby(['left_model','right_model']):
        valid=data[data.shape_valid]
        ax.scatter(valid.mass_ratio,valid.total_variation,label=left+' -> '+right+' n='+str(len(valid)),s=28,alpha=.7)
    ax.axvline(1,color='gray',lw=.7)
    ax.set(xlabel='Candidate/reference total density mass',ylabel='Total variation of D/sum(D)',
           title='Original 20 only: magnitude versus normalized shape (no localization GT)')
    ax.legend(fontsize=9)
    save_figure(fig,destination/'density_mass_vs_shape.png',qa)


def qualitative_plots(root,output,manifest,results,images,qa):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, PowerNorm
    records={r['image']:r for r in manifest['selected_images']}
    lookup=results.set_index(['image','model','prompt_type'])
    raw_max,raw_min=0.,0.
    for image,model in itertools.product(ORIGINAL_IMAGES,MODELS):
        with np.load(Path(root)/'maps'/archive_name(image,model,'class'),allow_pickle=False) as data:
            raw_max=max(raw_max,float(data['density_full'].max()))
            raw_min=min(raw_min,float(data['density_full'].min()))
    raw_max=raw_max or 1.
    for image in ORIGINAL_IMAGES:
        logging.getLogger('diag002_cpu').info('Qualitative panels: %s',image)
        record=records[image]
        with Image.open(images[image]) as im: rgb=np.asarray(im.convert('RGB'))
        arrays={}
        for model in MODELS:
            with np.load(Path(root)/'maps'/archive_name(image,model,'class'),allow_pickle=False) as data:
                arrays[model]={k:data[k] for k in ('density_full','fused_attention','sim_x2','sim_x1')}
        probabilities={m:normalize_density(a['density_full'][0,0]) for m,a in arrays.items()}
        pmax=max([float(p.max()) for p,status in probabilities.values() if p is not None] or [1.])
        fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
        for i,model in enumerate(MODELS):
            row=lookup.loc[(image,model,'class')]
            density=arrays[model]['density_full'][0,0]
            axes[i,0].imshow(rgb)
            axes[i,0].set_title('{} | GT={:g}\npred={:.2f}, abs error={:.2f}'.format(model,row.gt_count,row.pred_count,row.abs_error),fontsize=10)
            norm=PowerNorm(.5,vmin=0,vmax=raw_max) if raw_min>=0 else Normalize(vmin=raw_min,vmax=raw_max)
            raw=axes[i,1].imshow(density,cmap='magma' if raw_min>=0 else 'coolwarm',norm=norm)
            axes[i,1].set_title('Raw D, mass={:.2f}\nshared all-20 scale; {}'.format(density.sum(dtype=np.float64),
                                'sqrt display' if raw_min>=0 else 'linear, negatives retained'),fontsize=9)
            p,status=probabilities[model]
            if p is None:
                axes[i,2].text(.5,.5,'P unavailable:\n'+status,ha='center',va='center',transform=axes[i,2].transAxes)
            else:
                shape=axes[i,2].imshow(p,cmap='viridis',vmin=0,vmax=pmax)
                fig.colorbar(shape,ax=axes[i,2],label='Probability / pixel',shrink=.65)
            axes[i,2].set_title('P = D / sum(D)\nshared across these three conditions',fontsize=9)
            fig.colorbar(raw,ax=axes[i,1],label='Count / pixel',shrink=.65)
            for ax in axes[i]: ax.axis('off')
        fig.suptitle('{} | target: {} | class-name inference\nMagnitude and normalized shape are separate; no valid target dots'.format(image,record['class']))
        save_figure(fig,output/'figures'/('qual_'+Path(image).stem+'_density.png'),qa)
        for index,(x,y,w,h) in enumerate(record['patch_xywh']):
            fig,axes=plt.subplots(3,4,figsize=(14,9),constrained_layout=True)
            for i,model in enumerate(MODELS):
                row=lookup.loc[(image,model,'class')]
                for j,kind in enumerate(('fused_attention','sim_x2','sim_x1')):
                    low,high,cmap=(0,1,'viridis') if j==0 else (-1,1,'coolwarm')
                    rendered=axes[i,j].imshow(arrays[model][kind][index,0],vmin=low,vmax=high,cmap=cmap,
                                              extent=(x,x+w,y+h,y))
                    axes[i,j].set_title(model+' / '+kind,fontsize=9)
                    fig.colorbar(rendered,ax=axes[i,j],shrink=.65)
                    axes[i,j].set_xlabel('Original image x',fontsize=8)
                    axes[i,j].set_ylabel('Original image y',fontsize=8)
                axes[i,3].imshow(rgb[y:y+h,x:x+w])
                axes[i,3].set_title('GT={:g}, full-image pred={:.2f}\nabs error={:.2f}'.format(row.gt_count,row.pred_count,row.abs_error),fontsize=9)
                axes[i,3].axis('off')
            fig.suptitle('{} | {} | patch {} xywh={}\nNative 48x48 intermediate maps placed in original patch coordinates; no reconstruction/localization claim'.format(image,record['class'],index,(x,y,w,h)))
            save_figure(fig,output/'figures'/('qual_'+Path(image).stem+'_patch{:02d}.png'.format(index)),qa)
    return {'raw_density_global_min':raw_min,'raw_density_global_max':raw_max,'raw_display_gamma':.5 if raw_min>=0 else None,
            'raw_scale_scope':'all 20 class-prompt images x three models',
            'normalized_shape_scale':'shared three conditions within each image, probability/pixel'}


def markdown_table(frame,columns):
    def display(value):
        if value is None or pd.isna(value): return 'NA'
        if isinstance(value,(float,np.floating)): return '{:.4f}'.format(value)
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(display(row[column]) for column in columns)+' |'
                      for row in frame.to_dict('records')])


def research_report(output,tables,statuses,manifest,config):
    counts=tables['counting_summary'];overall=counts[counts.gt_bin=='all']
    class_counts=overall[overall.prompt_type=='class'].set_index('model')
    pairs=tables['paired_counting_summary'];class_pairs=pairs[(pairs.prompt_type=='class')&(pairs.gt_bin=='all')]
    prompt=tables['prompt_sensitivity']
    prompt_summary=[]
    for model,data in prompt.groupby('model',sort=True):
        prompt_summary.append({'model':model,'n':len(data),'mean_delta_pred':data.delta_pred.mean(),
            'median_delta_pred':data.delta_pred.median(),'mean_delta_abs_error':data.delta_abs_error.mean(),
            'detailed_worse_n':int((data.delta_abs_error>TIE_TOLERANCE).sum())})
    prompt_summary=pd.DataFrame(prompt_summary)
    shape=tables['density_shape_pairs']
    shape_class=shape[(shape.left_prompt=='class')&(shape.right_prompt=='class')]
    shape_summary=[]
    for (reference,candidate),data in shape_class.groupby(['left_model','right_model']):
        valid=data[data.shape_valid]
        valid_r=valid.shape_pearson.dropna()
        valid_ratio=data.mass_ratio.dropna()
        shape_summary.append({'reference':reference,'candidate':candidate,'n':len(data),'valid_shape_n':len(valid),
            'median_mass_ratio':valid_ratio.median() if len(valid_ratio) else None,
            'median_TV':valid.total_variation.median() if len(valid) else None,
            'TV_min':valid.total_variation.min() if len(valid) else None,
            'TV_max':valid.total_variation.max() if len(valid) else None,
            'shape_pearson_valid_n':len(valid_r),
            'median_shape_pearson':valid_r.median() if len(valid_r) else None})
    shape_summary=pd.DataFrame(shape_summary)
    associations=tables['spatial_associations']
    key_associations=associations[(associations.kind=='checkpoint')&
        associations.comparison.str.contains('/class -> ',regex=False)&associations.comparison.str.endswith('/class')&
        (associations.feature=='normalized_l1')&(associations.gt_bin=='all')&
        (associations.method=='spearman')&(associations.adjustment=='log1p_gt_and_reference_error')]
    base_to_nine=class_pairs[(class_pairs.reference=='BASE-002')&(class_pairs.candidate=='RP-009')].iloc[0]
    invalid=sum(row['normalization_status']!='valid' for row in statuses)
    gt_bins=counts[(counts.model=='BASE-002')&(counts.prompt_type=='class')&(counts.gt_bin!='all')][['gt_bin','n']]
    sections=[
        '# DIAG-002 - CPU post-hoc research report (descriptive; researcher review required)',
        '\nThis report uses already-generated DIAG-001 artifacts only. No model/checkpoint was loaded, no inference or training was run. Class-name inference is primary. Detailed prompting is a diagnostic perturbation. Dot localization accuracy is unavailable.',
        '\n## Coverage and provenance',
        '\n230 images; 1,380 exact image/model/prompt records; 14,490 keyed spatial comparisons; 120 native-map NPZs on the original 20. Input hashes and source-code hash are in analysis_manifest.json. Source DIAG-001 commit: `'+str(manifest.get('source',{}).get('commit','unrecorded'))+'`.',
        '\nGT bin sizes (the high-count group is small; inspect n rather than interpreting its mean as stable):\n',
        markdown_table(gt_bins,['gt_bin','n']),
        '\n## Finding A - Counting calibration',
        '\n**Direct evidence.** Full-230 class-name counting summaries:',
        markdown_table(class_counts.reset_index(),['model','n','mae','rmse','signed_bias','median_absolute_error','over_n','over_pct','under_n','under_pct','pred_mean']),
        '\nPaired candidate wins/losses and conditional image-bootstrap mean error-gain intervals:',
        markdown_table(class_pairs,['reference','candidate','n','candidate_wins','candidate_loses','ties','mean_error_gain','gain_ci_low','gain_ci_high','mean_delta_pred','prediction_lower_n','mean_over_error_gain','mean_under_error_gain']),
        '\n**Arithmetic decomposition, not causal attribution.** BASE-002 -> RP-009: mean error gain = {:.4f}; reduction of positive overcount error = {:.4f}; reduction of undercount error = {:.4f}. The two signed contributions sum to the gain. RP-009 lowers predictions on {}/230 images.'.format(
            base_to_nine.mean_error_gain,base_to_nine.mean_over_error_gain,base_to_nine.mean_under_error_gain,int(base_to_nine.prediction_lower_n)),
        '\n**Association.** prediction_distributions.png, paired_counting_gains.png and gt_bin_magnitude_bias.png separate magnitude, signed error and GT strata. Full distributions/per-image values remain in the tables.',
        '\n**Plausible interpretation.** Reduced overcount burden is consistent with a calibration/magnitude component. It does not demonstrate improved object recognition. The density-shape evidence below tests whether a pure multiplicative description is sufficient on the saved subset.',
        '\n**Alternatives and uncertainty.** Training/procedure/embedding changes, scene composition and the small high-count bin can affect both predictions and error. Bootstrap intervals resample images with seed {} ({} repetitions); they condition on this dataset and these single trained checkpoints, not checkpoint/seed uncertainty or unrelated datasets.'.format(config['seed'],config['bootstrap_repetitions']),
        '\n## Finding B - Spatial redistribution',
        '\n**Direct evidence (original 20 only).** D is the saved evaluator-reassembled density. P=D/sum(D) is calculated only for finite nonnegative D with mass > {}. No clipping, smoothing or pseudo annotations. Invalid normalization conditions among all 120 archives: {}.'.format(EPSILON,invalid),
        markdown_table(shape_summary,['reference','candidate','n','valid_shape_n','median_mass_ratio','median_TV','TV_min','TV_max','median_shape_pearson']),
        '\n**Association.** density_mass_vs_shape.png plots mass ratios separately from total variation. TV=0 denotes identical normalized distributions; differences above floating-point tolerance cannot be produced by scalar multiplication alone.',
        '\n**Plausible interpretation.** Similar normalized shape with lower mass is consistent with magnitude change; changed normalized shape indicates redistribution. Neither establishes that mass moved toward the requested target.',
        '\n**Alternatives and uncertainty.** Shape changes may suppress or amplify distractors. The 20 images were selected before inference with count/class coverage and are not frequency-representative of all 230. Raw maps are unavailable for the other 210, so no raw-shape reconstruction or extrapolated localization claim is made.',
        '\n## Finding C - U-Net versus HSCM',
        '\n**Direct evidence / statistical association.** Partial Spearman between normalized map L1 and class-name counting gain, adjusting ranked variables for log1p(GT) and log1p(reference absolute error):',
        markdown_table(key_associations,['comparison','map_type','n','r','status']),
        '\nRaw Pearson/Spearman, 1-minus-spatial-Pearson features, GT strata and valid counts are in spatial_associations.csv. Better/worse/tie distributions are in spatial_groups.csv. Undefined spatial Pearson values are omitted only from calculations requiring them and their valid n is reported; normalized L1 still uses all valid rows.',
        '\n**Plausible interpretation.** Consistently stronger HSCM associations across pairs/adjustment would be compatible with an HSCM-related hypothesis. Compare both similarity stages against every attention resolution; the pipeline does not select the largest coefficient as a proven mechanism.',
        '\n**Alternatives and uncertainty.** GT and reference error can confound associations; reference error also appears algebraically in error_gain, so residualization does not create causal identification. Same-map minmax L1 is not raw magnitude. Attention/HSCM/density are not interchangeable physical scores. Associations are exploratory, correlated and numerous; no p-value screening or causal model ranking is performed.',
        '\n## Finding D - Prompt sensitivity',
        '\n**Direct evidence.** Within-checkpoint detailed minus class changes:',
        markdown_table(prompt_summary,['model','n','mean_delta_pred','median_delta_pred','mean_delta_abs_error','detailed_worse_n']),
        '\n**Statistical association.** prompt_association_matrix.png relates within-model count change to map changes, raw and adjusted. spatial_associations.csv separately includes delta_abs_error as a target. These joins compare each image with itself under its exact saved prompt; no cross-checkpoint result is substituted for a prompt effect.',
        '\n**Plausible interpretation.** Count increases accompanied by attention/HSCM/density changes are compatible with prompt-sensitive behavior. Descriptions change content, token count and embeddings together; no claim isolates text length or identifies why a particular internal stage causes failure.',
        '\n**Alternatives and uncertainty.** Token-mean attention for different sequences has different semantic content. Dataset-specific targets, prompt composition and training provenance remain alternative explanations. The FSC147-S ranking reversal does not establish that generic templates are generally superior; standard FSC147 test evidence differs.',
        '\n## Researcher decision table (no automatic selection)',
        '\n| Future direction | Evidence relevant to review | Still needed before a scientific decision |\n| --- | --- | --- |\n| Density calibration / counting objective | Bias, overcount decomposition, mass-versus-shape plots | A separately approved controlled intervention and held-out counting evaluation |\n| HSCM semantic fusion | Both-stage associations before/after adjustment | Provenance-matched causal control; no conclusion from correlation alone |\n| U-Net conditioning / cross-attention | Resolution-specific and prompt-paired associations | Token-semantic controls and a separately approved intervention |\n| No architectural change yet | Single checkpoints, confounding, missing target spatial GT | Researcher review, replication/provenance and additional evidence |',
        '\nThe final decision belongs to the researcher. PROJECT_STATE.md and EXPERIMENT_LEDGER.md are not edited by this pipeline.'
    ]
    (output/'research_report.md').write_text('\n'.join(sections)+'\n',encoding='utf-8')
    return {'class_counting':class_counts.reset_index().to_dict('records'),
            'class_paired':class_pairs.to_dict('records'),'prompt_sensitivity':prompt_summary.to_dict('records'),
            'density_shape_class20':shape_summary.to_dict('records'),'normalization_invalid_n':invalid,
            'gt_bins':gt_bins.to_dict('records')}


def input_inventory(root):
    root=Path(root)
    paths=[root/name for name in ('manifest.json','summary.json','results.csv','map_comparisons.csv')]
    paths+=sorted((root/'maps').glob('*.npz'))
    return {p.relative_to(root).as_posix():{'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in paths}


def _code_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def _source_commit():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def run_analysis(input_dir,output_dir=None,image_root=None,bootstrap_repetitions=2000,seed=3407,
                 reuse=False,validate_only=False):
    if seed!=3407:
        raise ValueError('Protocol seed is fixed at 3407')
    if bootstrap_repetitions<1:
        raise ValueError('bootstrap_repetitions must be positive')
    root=Path(input_dir).resolve()
    output=Path(output_dir).resolve() if output_dir else root/'analysis_cpu_v1'
    if root==output or output in root.parents or root/'maps'==output or root/'figures'==output:
        raise ValueError('Analysis output must not replace an input directory')
    log=logging.getLogger('diag002_cpu');log.setLevel(logging.INFO)
    console=logging.StreamHandler(sys.stdout);log.addHandler(console)
    file_handler=None
    state=None
    config={'seed':seed,'bootstrap_repetitions':bootstrap_repetitions,
            'density_mass_epsilon':EPSILON,'tie_tolerance':TIE_TOLERANCE,
            'confounders':['log1p(GT)','log1p(reference absolute error)']}
    try:
        log.info('DIAG-002: reading immutable saved artifacts (CPU only, no model imports)')
        before=input_inventory(root)
        manifest,source_summary,results,maps=validate_tables(root)
        log.info('Validated keyed table coverage: 230 images / 1380 predictions / 14490 map pairs')
        statuses,images=validate_npz(root,manifest,results,image_root)
        image_fingerprints={image:{'path':str(path),'sha256':sha256(path)} for image,path in images.items()}
        log.info('Validated 120 native NPZ schemas, density mass/scaling and 20 matching RGB images')
        if validate_only:
            return {'status':'validated','images':230,'conditions':1380,'map_comparisons':14490,
                    'archives':120,'normalization_statuses':statuses}
        if output.exists():
            if not reuse:
                raise FileExistsError('Analysis output exists; choose a new name or explicitly use --reuse: '+str(output))
            try:
                saved=json.loads((output/'analysis_manifest.json').read_text(encoding='utf-8'))
                if saved['status']!='complete' or saved['input_files']!=before or saved['config']!=config or saved['code_sha256']!=_code_hash():
                    raise ValueError('Existing report is incomplete or input/config/source hash differs')
                for relative,info in saved['output_files'].items():
                    if sha256(output/relative)!=info['sha256']:
                        raise ValueError('Existing output changed: '+relative)
                if input_inventory(root)!=before:
                    raise ValueError('Inputs changed during reuse validation')
            except (OSError,ValueError,KeyError) as exc:
                raise ValueError('Cannot reuse safely; preserve this directory and choose a new output: '+str(exc)) from exc
            log.info('Reused complete verified analysis; no files overwritten: %s',output)
            return json.loads((output/'summary.json').read_text(encoding='utf-8'))
        output.mkdir(parents=True,exist_ok=False)
        (output/'figures').mkdir()
        file_handler=logging.FileHandler(output/'run.log',encoding='utf-8')
        log.addHandler(file_handler)
        state={'schema_version':SCHEMA,'status':'running','created_at':datetime.now(timezone.utc).isoformat(),
               'input_dir':str(root),'output_dir':str(output),'config':config,'input_files':before,
               'images':image_fingerprints,'code_sha256':_code_hash(),'source_commit':_source_commit(),
               'source_commit_note':'Base checkout only; analysis implementation is identified by its own SHA256',
               'execution':{'device':'CPU','model_loading':False,'checkpoint_loading':False,'additional_inference':False},
               'environment':{'python':sys.version,'packages':{name:importlib.metadata.version(name)
                    for name in ('numpy','pandas','scipy','matplotlib','Pillow')}}}
        json_write(output/'analysis_manifest.json',state)
        json_write(output/'input_validation.json',{'status':'passed','images':230,'conditions':len(results),
                    'map_comparisons':len(maps),'archives':120,'normalization':statuses,
                    'join_keys':['image','exact model','exact prompt'],'input_repair':False})
        log.info('Computing paired counting, conditional bootstrap and spatial associations')
        tables=counting_tables(results,bootstrap_repetitions)
        tables.update(spatial_tables(results,maps))
        tables['density_normalization']=pd.DataFrame(statuses)
        tables['density_shape_pairs']=density_shapes(root,statuses)
        for name,data in tables.items(): csv_write(output/(name+'.csv'),data)
        # Prompt length is not manipulated or interpreted causally.
        qa=[]
        import matplotlib
        matplotlib.use('Agg',force=True)
        aggregate_plots(output,results,tables,qa)
        log.info('Rendering original-20 qualitative panels with native patch coordinates')
        display=qualitative_plots(root,output,manifest,results,images,qa)
        json_write(output/'figure_qa.json',qa)
        summary=research_report(output,tables,statuses,manifest,config)
        summary.update({'schema_version':SCHEMA,'status':'complete','evaluation_images':230,
                        'raw_map_images':20,'tables':{k:len(v) for k,v in tables.items()},'figure_count':len(qa),
                        'display':display,'interpretation':'Descriptive associations; no causal or localization-accuracy conclusion. Researcher decision required.'})
        json_write(output/'summary.json',summary)
        if input_inventory(root)!=before or any(sha256(p['path'])!=p['sha256'] for p in image_fingerprints.values()):
            raise RuntimeError('Input artifact changed during analysis; outputs not marked complete')
        state['input_immutability_verified']=True
        log.info('Completed CPU analysis; all %d figures rendered/checked; inputs unchanged',len(qa))
        log.removeHandler(file_handler);file_handler.close();file_handler=None
        state['output_files']={p.relative_to(output).as_posix():{'sha256':sha256(p),'size_bytes':p.stat().st_size}
                               for p in sorted(output.rglob('*')) if p.is_file() and p.name!='analysis_manifest.json'}
        state['status']='complete'
        json_write(output/'analysis_manifest.json',state)
        return summary
    except BaseException:
        if state is not None:
            state['status']='failed';json_write(output/'analysis_manifest.json',state)
        log.exception('DIAG-002 failed; inputs are not repaired and partial outputs remain for inspection')
        raise
    finally:
        if file_handler is not None:
            log.removeHandler(file_handler);file_handler.close()
        log.removeHandler(console);console.close()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--image-root',help='Existing original RGB directory; every image must match manifest SHA256')
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--reuse',action='store_true',help='Read a completed matching output after validating every output hash')
    parser.add_argument('--validate-only',action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    summary=run_analysis(args.input_dir,args.output_dir,args.image_root,args.bootstrap_repetitions,
                         args.seed,args.reuse,args.validate_only)
    print(json.dumps({'status':summary['status'],'output':str(Path(args.output_dir) if args.output_dir else Path(args.input_dir)/'analysis_cpu_v1')},indent=2))


if __name__=='__main__':
    main()
