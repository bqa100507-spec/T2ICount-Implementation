"""Synthetic CPU fixtures only; never initialize a model or open checkpoints."""
import ast
import builtins
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from tools import analyze_spatial_diagnostic as analysis


@pytest.fixture
def artifacts(tmp_path):
    root=tmp_path/'diag001';(root/'maps').mkdir(parents=True)
    images=tmp_path/'rgb';images.mkdir()
    ids=list(analysis.ORIGINAL_IMAGES)+['fixture_{:03d}.jpg'.format(i) for i in range(210)]
    records,rows=[],[]
    for i,image in enumerate(ids):
        gt=(1,3,10,35)[i%4]
        path=images/image
        if image in analysis.ORIGINAL_IMAGES:
            Image.new('RGB',(384,384),'white').save(path)
        record={'image':image,'class':'target','gt_count':gt,'image_hw':[384,384],
                'patch_xywh':[[0,0,384,384]],
                'image_file':{'path':str(path),'sha256':analysis.sha256(path) if path.exists() else 'not-read'},
                'prompts':{p:{'prompt':'target' if p=='class' else 'red target','input_ids':[1,10]+[2]*75,
                    'tokens':['token']*77,'mask':[0,1]+[0]*75,'active_token_indices':[1]} for p in analysis.PROMPTS}}
        records.append(record)
        for model,prompt in __import__('itertools').product(analysis.MODELS,analysis.PROMPTS):
            pred=gt+(5,3,1)[analysis.MODELS.index(model)]+(8 if prompt=='detailed' else 0)
            rows.append({'image':image,'class':'target','model':model,'prompt_type':prompt,
                         'prompt':record['prompts'][prompt]['prompt'],'gt_count':gt,'pred_count':pred,
                         'abs_error':pred-gt,'patch_count':1})
            if image in analysis.ORIGINAL_IMAGES:
                arrays={}
                for size in (12,24,48):
                    grid=np.linspace(0,1,size*size,dtype=np.float32).reshape(1,1,size,size)
                    arrays['tokens_'+str(size)]=grid
                    arrays['attention_'+str(size)]=grid.copy()
                for kind in ('fused_attention','sim_x2','sim_x1'):
                    arrays[kind]=arrays['tokens_48'].copy()
                arrays['density_native']=np.full((1,1,48,48),pred*60/2304,dtype=np.float32)
                arrays['density_full']=np.full((1,1,384,384),pred/147456,dtype=np.float32)
                arrays['patch_xywh']=np.array([[0,0,384,384]],dtype=np.int32)
                np.savez_compressed(root/'maps'/analysis.archive_name(image,model,prompt),**arrays)
    results=pd.DataFrame(rows)
    comparisons=[{'image':image,'left_model':left[0],'left_prompt':left[1],
                  'right_model':right[0],'right_prompt':right[1],'map_type':kind,
                  'normalized_l1':0.,'pearson':None if kind=='density_native' else 1.}
                 for image in ids for left,right in analysis.condition_pairs() for kind in analysis.MAP_TYPES]
    manifest={'schema_version':'fsc147s-spatial-v2','status':'complete','selected_images':records,
              'visualization':{'count':20,'selected_images':list(analysis.ORIGINAL_IMAGES)},
              'config':{'models':list(analysis.MODELS),'prompts':list(analysis.PROMPTS)},
              'inference':{'patch_size':384,'stride':384},
              'source':{'commit':'synthetic CPU fixture'},
              'capture_equivalence':[{'model':m,'image':'3319.jpg','prompt':p,'bitwise_equal':True}
                                    for m,p in __import__('itertools').product(analysis.MODELS,analysis.PROMPTS)]}
    summary={'status':'complete','counting':[
        {'model':m,'prompt_type':p,'n':230,'mae':float(d.abs_error.mean()),'rmse':float(np.sqrt((d.abs_error**2).mean()))}
        for (m,p),d in results.groupby(['model','prompt_type'])],
        'map_comparisons':[{'left_model':left[0],'left_prompt':left[1],'right_model':right[0],
            'right_prompt':right[1],'map_type':kind,'n':230,'mean_normalized_l1':0.,
            'mean_pearson':None if kind=='density_native' else 1.,'pearson_valid_n':0 if kind=='density_native' else 230}
            for left,right in analysis.condition_pairs() for kind in analysis.MAP_TYPES]}
    analysis.json_write(root/'manifest.json',manifest);analysis.json_write(root/'summary.json',summary)
    analysis.csv_write(root/'results.csv',rows);analysis.csv_write(root/'map_comparisons.csv',comparisons)
    return root,images,manifest


def test_exact_coverage_raw_schema_and_keyed_joins_ignore_row_order(artifacts):
    root,images,manifest=artifacts
    original=analysis.input_inventory(root)
    pd.read_csv(root/'results.csv').sample(frac=1,random_state=12).to_csv(root/'results.csv',index=False)
    pd.read_csv(root/'map_comparisons.csv').sample(frac=1,random_state=13).to_csv(root/'map_comparisons.csv',index=False)
    m,s,results,maps=analysis.validate_tables(root)
    statuses,paths=analysis.validate_npz(root,m,results,images)
    assert (results.image.nunique(),len(results),len(maps),len(statuses),len(paths))==(230,1380,14490,120,20)
    assert all(r['normalization_status']=='valid' for r in statuses)
    tables=analysis.counting_tables(results,20)
    gains=tables['paired_counting']
    assert len(gains)==1380
    selected=gains[(gains.reference=='BASE-002')&(gains.candidate=='RP-009')]
    assert np.all(selected.error_gain==4)
    assert np.allclose(selected.error_gain,selected.over_error_gain+selected.under_error_gain)
    assert len(analysis.density_shapes(root,statuses))==180


@pytest.mark.parametrize('fault',['duplicate','missing','nonfinite','prompt','map_summary','map_duplicates','npz_shape','npz_missing','mass','rgb'])
def test_input_corruption_is_reported_without_repair(artifacts,fault):
    root,images,_=artifacts
    if fault in ('duplicate','missing','nonfinite','prompt'):
        data=pd.read_csv(root/'results.csv')
        if fault=='duplicate': data.iloc[-1]=data.iloc[0]
        elif fault=='missing': data=data.iloc[:-1]
        elif fault=='nonfinite': data.loc[0,'pred_count']=np.inf
        else: data.loc[0,'prompt']='invented prompt'
        data.to_csv(root/'results.csv',index=False)
    elif fault=='map_summary':
        doc=json.loads((root/'summary.json').read_text());doc['map_comparisons'][0]['mean_normalized_l1']=.9
        analysis.json_write(root/'summary.json',doc)
    elif fault=='map_duplicates':
        data=pd.read_csv(root/'map_comparisons.csv');data.iloc[-1]=data.iloc[0]
        data.to_csv(root/'map_comparisons.csv',index=False)
    elif fault=='npz_missing':
        (root/'maps'/analysis.archive_name('3319.jpg','BASE-002','class')).unlink()
    elif fault=='rgb':
        Image.new('RGB',(384,384),'black').save(images/'3319.jpg')
    else:
        path=root/'maps'/analysis.archive_name('3319.jpg','BASE-002','class')
        with np.load(path,allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
        if fault=='npz_shape': arrays['tokens_12']=arrays['tokens_12'][:,:,:6,:6]
        else: arrays['density_full']*=2
        np.savez_compressed(path,**arrays)
    before=analysis.input_inventory(root)
    with pytest.raises(analysis.ValidationError):
        m,_,results,_=analysis.validate_tables(root)
        analysis.validate_npz(root,m,results,images)
    assert analysis.input_inventory(root)==before


def test_density_normalization_separates_mass_from_shape_and_rejects_invalid():
    density=np.array([[1.,2.],[3.,4.]])
    p,status=analysis.normalize_density(density)
    q,_=analysis.normalize_density(density*5)
    assert status=='valid' and p.sum()==1 and np.allclose(p,q)
    assert analysis.normalize_density(np.zeros((2,2)))[1]=='zero_or_near_zero_mass'
    negative=density.copy();negative[0,0]=-1
    assert analysis.normalize_density(negative)[1]=='negative_values'
    assert analysis.normalize_density(np.array([np.nan]))[1]=='nonfinite'
    assert density[0,0]==1  # Input never clipped or normalized in place.


def test_correlations_constant_missing_ranks_and_confounder_adjustment():
    assert analysis.correlation([1,1,1],[1,2,3])['r'] is None
    assert analysis.correlation([1,np.nan,3,4],[2,5,6,8])['n']==3
    assert analysis.correlation([1,1,2,3],[4,4,8,12],'spearman')['r']==pytest.approx(1)
    rng=np.random.default_rng(123);z=rng.normal(size=1000)
    x=z+.15*rng.normal(size=1000);y=z+.15*rng.normal(size=1000)
    assert analysis.correlation(x,y)['r']>.9
    adjusted=analysis.correlation(x,y,controls=z[:,None])
    assert abs(adjusted['r'])<.15 and adjusted['control_rank']==2
    assert analysis.bootstrap_mean([1,2,3],100)==analysis.bootstrap_mean([1,2,3],100)


def test_figure_qa_ignores_unrendered_ticks_but_rejects_clipped_text(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(constrained_layout=True)
    ax.plot([0,1],[0,1]);ax.set_yticks([0,.5,1,1.2]);ax.set_ylim(-.05,1.05)
    qa=[];analysis.save_figure(fig,tmp_path/'visible.png',qa)
    assert not qa[0]['clipped_text']
    fig,ax=plt.subplots();fig.text(-.8,.5,'clipped')
    with pytest.raises(ValueError,match='Clipped'):
        analysis.save_figure(fig,tmp_path/'clipped.png',[])


def test_cpu_pipeline_safe_reuse_hashes_and_no_model_imports(artifacts,tmp_path,monkeypatch):
    root,images,_=artifacts;output=root/'analysis_cpu_v1'
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','models','ldm','diffusers'):
            raise AssertionError('Forbidden model/GPU import: '+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    def simple_plot(output,results,tables,qa):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(constrained_layout=True);ax.plot([1,2],[1,2])
        analysis.save_figure(fig,output/'figures/synthetic.png',qa)
    monkeypatch.setattr(analysis,'aggregate_plots',simple_plot)
    monkeypatch.setattr(analysis,'qualitative_plots',lambda *args,**kwargs:{'fixture':'CPU unit test'})
    before=analysis.input_inventory(root)
    summary=analysis.run_analysis(root,image_root=images,bootstrap_repetitions=20)
    assert summary['status']=='complete' and summary['evaluation_images']==230
    assert analysis.input_inventory(root)==before
    state=json.loads((output/'analysis_manifest.json').read_text())
    assert state['input_immutability_verified'] and not state['execution']['model_loading']
    report=(output/'research_report.md').read_text()
    assert all('Finding '+letter in report for letter in 'ABCD')
    assert 'not causal attribution' in report
    assert all(r['shape_pearson_valid_n']==0 for r in summary['density_shape_class20'])
    snapshot={p:analysis.sha256(output/p) for p in state['output_files']}
    analysis.run_analysis(root,image_root=images,bootstrap_repetitions=20,reuse=True)
    assert snapshot=={p:analysis.sha256(output/p) for p in state['output_files']}
    with pytest.raises(FileExistsError): analysis.run_analysis(root,image_root=images,bootstrap_repetitions=20)
    (output/'research_report.md').write_text('modified outside analysis')
    with pytest.raises(ValueError,match='Cannot reuse safely'):
        analysis.run_analysis(root,image_root=images,bootstrap_repetitions=20,reuse=True)


def test_source_and_notebook_are_cpu_only_and_cells_compile():
    source=Path(analysis.__file__).read_text(encoding='utf-8')
    tree=ast.parse(source)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    assert not any(name and name.split('.')[0] in ('torch','models','ldm','diffusers') for name in imports)
    root=Path(analysis.__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/spatial_diagnostic_analysis.ipynb').read_text(encoding='utf-8'))
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            compile(cell['source'],str(i),'exec')
            assert cell['outputs']==[] and cell['execution_count'] is None


def test_notebook_executes_in_order_with_embedded_source_on_cpu(artifacts,tmp_path,monkeypatch):
    import atexit
    root,images,_=artifacts
    repo=Path(analysis.__file__).resolve().parents[1]
    notebook=json.loads((repo/'notebooks/spatial_diagnostic_analysis.ipynb').read_text(encoding='utf-8'))
    namespace={};calls=[]
    monkeypatch.setitem(sys.modules,'IPython.display',type('Display',(),{
        'display':staticmethod(lambda *a:None),'Image':staticmethod(lambda **k:None),
        'Markdown':staticmethod(lambda *a:None)}))
    def fake_run(*args,**kwargs):
        calls.append((args,kwargs));return {'status':'validated'}
    try:
        for index,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code': continue
            exec(compile(cell['source'],str(index),'exec'),namespace)
            if index==1:
                namespace.update(INPUT_DIR=root,OUTPUT_DIR=tmp_path/'new-analysis',IMAGE_ROOT=images,
                    LOCAL_ANALYSIS_SCRIPT=tmp_path/'not-synced.py',INSTALL_MISSING_CPU_PACKAGES=False)
            if index==5:
                monkeypatch.setattr(namespace['analysis'],'run_analysis',fake_run)
        assert len(calls)==1 and calls[0][1]['seed']==3407
        assert namespace['source_manifest']['status']=='complete'
        assert len(namespace['normalization_statuses'])==120
        assert namespace['analysis'].__file__.startswith(namespace['_cpu_source_dir'].name)
    finally:
        if '_cpu_source_dir' in namespace:
            atexit.unregister(namespace['_cpu_source_dir'].cleanup)
            namespace['_cpu_source_dir'].cleanup()


def test_notebook_rejects_stale_local_helper(tmp_path):
    repo=Path(analysis.__file__).resolve().parents[1]
    notebook=json.loads((repo/'notebooks/spatial_diagnostic_analysis.ipynb').read_text(encoding='utf-8'))
    namespace={};exec(notebook['cells'][1]['source'],namespace)
    stale=tmp_path/'stale.py';stale.write_text('raise RuntimeError("stale")')
    namespace['LOCAL_ANALYSIS_SCRIPT']=stale
    with pytest.raises(AssertionError,match='Local helper differs'):
        exec(notebook['cells'][5]['source'],namespace)
