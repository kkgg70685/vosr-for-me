"""Formal float-native paired benchmark, including fixed-scale comparison panels.

Manifest envelope: phase, rows, optional pairs. Each row has id,parent,lr,hr?
Pairs have id,parent,view0,view1,H_0_to_1. Ground truth is evaluation-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from . import VERSION
from .model import Config,AreaCamera
from .runtime import load_base,load_camera,generate_native
from .infer import source_identity
from .evaluate import evaluate_rows,parent_summary
from .matching import evaluate_pairs
from .artifacts import write_json, json_ready, write_report


def validate_manifest(manifest,root):
    if manifest.get('phase') not in ('smoke','dev','test'):raise ValueError('Declare smoke/dev/test phase')
    rows=manifest['rows'];ids={r['id'] for r in rows}
    if not rows or len(ids)!=len(rows):raise ValueError('Unique nonempty rows required')
    for r in rows:
        if not r.get('parent'):raise ValueError('Physical parent identity required')
        if not r.get('lr'):raise ValueError('LR input required')
        for key in ('lr','hr'):
            if r.get(key) and not (root/r[key]).is_file():raise FileNotFoundError(root/r[key])
    if manifest['phase']=='test':
        excluded=set(manifest.get('excluded_dev_parents',[]))
        if not excluded:raise ValueError('Test needs explicit historical/dev exclusion list')
        if excluded & {r['parent'] for r in rows}:raise ValueError('Development parent leaked into test')
        if any(not r.get('hr') for r in rows):raise ValueError('Reference images required for formal paired IQA')
    if manifest.get('task')=='known_homography' and not manifest.get('pairs'):raise ValueError('Homography benchmark needs pairs')
    if len({p['id'] for p in manifest.get('pairs',[])})!=len(manifest.get('pairs',[])):raise ValueError('Duplicate pair id')
    for pair in manifest.get('pairs',[]):
        for key in ('view0','view1'):
            if pair[key] not in ids:raise ValueError('Unknown view in matching pair')
            view=next(r for r in rows if r['id']==pair[key])
            if view['parent']!=pair['parent']:raise ValueError('Pair/physical parent mismatch')
    return rows


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--runtime-root',default=None, help='Asset directory; defaults to VOSR_RUNTIME_ROOT')
    p.add_argument('--config',default=str(Path(__file__).parents[2]/'configs/observable_transport_v25.json'))
    p.add_argument('--camera',choices=('deployed','synthetic-area'),default='deployed')
    p.add_argument('--device',default='cuda');p.add_argument('--perceptual',action='store_true')
    p.add_argument('--expected-manifest-sha',help='Required in test; use SHA frozen before inference')
    p.add_argument('--expected-protocol-sha',help='Required in test; use SHA frozen before inference')
    a=p.parse_args();manifest_path=Path(a.manifest);dest=Path(a.output_dir)
    if dest.exists():raise FileExistsError(dest)
    manifest=json.loads(manifest_path.read_text());rows=validate_manifest(manifest,manifest_path.parent)
    if manifest.get('camera')!=a.camera:raise ValueError('Manifest camera must match explicit CLI camera')
    if a.device not in ('cuda','cuda:0'):raise ValueError('Retained frozen VOSR runtime requires CUDA device 0')
    protocol_path=Path(a.config);protocol=json.loads(protocol_path.read_text())
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if manifest['phase']=='test':
        if not a.perceptual:raise ValueError('Formal test requires LPIPS and DISTS')
        if a.expected_manifest_sha!=manifest_sha or a.expected_protocol_sha!=protocol_sha:raise ValueError('Test lock SHA mismatch')
        if len({r['parent'] for r in rows})<protocol['evaluation']['min_test_parents']:raise ValueError('Too few independent test parents')
    cfg=Config(**protocol['model']);metrics={}
    if a.perceptual:
        import pyiqa
        metrics={n:pyiqa.create_metric(n.lower(),device=a.device) for n in ('LPIPS','DISTS')}
    base=load_base(a.runtime_root)
    camera=AreaCamera(cfg.scale) if a.camera=='synthetic-area' else load_camera(a.runtime_root,a.device)
    dest.mkdir(parents=True,exist_ok=False)
    inputs={str(manifest_path.parent/r[k]):hashlib.sha256((manifest_path.parent/r[k]).read_bytes()).hexdigest()
            for r in rows for k in ('lr','hr') if r.get(k)}
    identity={'version':VERSION,'protocol_sha256':protocol_sha,'manifest_sha256':manifest_sha,
              'source_sha256':source_identity(),'inputs':inputs,'phase':manifest['phase'],'camera':a.camera,
              'seed':protocol['seed'],'guidance':-.5,'steps':25,'torch':torch.__version__,
              'base_weight_sha256':hashlib.sha256(base.weight_path.read_bytes()).hexdigest()}
    write_json(dest/'identity.json',identity)
    started=time.perf_counter()
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    records=evaluate_rows(rows,manifest_path.parent,camera,cfg,a.device,metrics,dest,
                          native_provider=lambda y:generate_native(base,y,protocol['seed']))
    result={'identity':identity,'summary':parent_summary(records),'records':records,
            'elapsed_seconds':time.perf_counter()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
    write_json(dest/'evaluation.json',result)
    matching=None
    if manifest.get('pairs'):
        lookup={r['id']:{} for r in rows}
        for r in records:lookup[r['id']][r['method']]=r['image']
        pairs=[dict(id=p['id'],parent=p['parent'],H_0_to_1=p['H_0_to_1'],
                    methods={m:[lookup[p['view0']][m],lookup[p['view1']][m]] for m in lookup[p['view0']]}) for p in manifest['pairs']]
        matching=evaluate_pairs(pairs,dest)
        write_json(dest/'pairs.json',pairs);write_json(dest/'matching.json',matching)
    write_report(dest,result,matching)
    print(json.dumps(json_ready({'phase':manifest['phase'],'output':str(dest),'summary':result['summary']}),allow_nan=False),flush=True)


if __name__=='__main__':main()
