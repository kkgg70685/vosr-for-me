"""Build known-homography views on the server, before any SR is generated.

Sources JSON: [{"parent":"physical-scene-id", "hr":"/server/path.png"}].
No dataset names, labels, learned selector, or metric-based sample selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .artifacts import write_json


def view_transforms(height, width):
    """Three 512-pixel views; a common 1.2x center zoom avoids padded corners."""
    if min(height,width)<512:raise ValueError('Source must have at least 512 pixels on each side')
    scale=512*1.2/min(height,width)
    center=np.array([[1,0,-255.5],[0,1,-255.5],[0,0,1.]])
    m0=np.array([[scale,0,255.5-scale*(width-1)/2],
                 [0,scale,255.5-scale*(height-1)/2],[0,0,1.]])
    maps=[m0]
    for sign in (1,-1):
        angle=np.deg2rad(3*sign);co,si=np.cos(angle),np.sin(angle)
        local=np.array([[co,-si,8*sign],[si,co,-6*sign],[4e-5*sign,-3e-5*sign,1.]])
        maps.append(np.linalg.inv(center)@local@center@m0)
    corners=np.array([[0,0,1],[511,0,1],[511,511,1],[0,511,1]]).T
    for matrix in maps:
        q=np.linalg.inv(matrix)@corners;q=q[:2]/q[2:]
        if not ((q[0]>=2).all() and (q[0]<=width-3).all() and (q[1]>=2).all() and (q[1]<=height-3).all()):
            raise ValueError('Transform would use invalid source pixels')
    return maps


def main():
    import cv2
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--sources',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--phase',choices=('smoke','dev','test'),required=True)
    p.add_argument('--excluded-parents',help='JSON list of every previously exposed physical parent; mandatory for test')
    a=p.parse_args();source_file=Path(a.sources);dest=Path(a.output_dir)
    if dest.exists():raise FileExistsError(dest)
    sources=json.loads(source_file.read_text());excluded=json.loads(Path(a.excluded_parents).read_text()) if a.excluded_parents else []
    parents=[r['parent'] for r in sources]
    if not parents or len(set(parents))!=len(parents):raise ValueError('Unique physical parents required')
    if set(parents)&set(excluded):raise ValueError('Previously exposed parent in proposed split')
    if a.phase=='test' and not excluded:raise ValueError('Declare historical/development exclusions before test')
    # Validate the entire inventory before producing images.
    hashes=set();prepared=[]
    for row in sources:
        path=source_file.parent/row['hr'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:raise ValueError('Duplicate source bytes with different parent IDs')
        hashes.add(digest);rgb=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if rgb is None:raise ValueError('Unreadable HR source')
        transforms=view_transforms(*rgb.shape[:2]);prepared.append((row,path,digest,transforms))
    dest.mkdir(parents=True,exist_ok=False);rows=[];pairs=[];provenance=[]
    for index,(source,path,digest,transforms) in enumerate(prepared):
        rgb=cv2.imread(str(path),cv2.IMREAD_COLOR)
        for v,matrix in enumerate(transforms):
            name=f'{index:06d}_view{v}';hr=cv2.warpPerspective(rgb,matrix,(512,512),flags=cv2.INTER_CUBIC)
            # Area camera followed by documented 8-bit observation quantization.
            lr=np.rint(hr.reshape(128,4,128,4,3).mean((1,3))).clip(0,255).astype(np.uint8)
            if not cv2.imwrite(str(dest/(name+'_hr.png')),hr) or not cv2.imwrite(str(dest/(name+'_lr.png')),lr):
                raise OSError('Failed to write benchmark input')
            rows.append(dict(id=name,parent=source['parent'],lr=name+'_lr.png',hr=name+'_hr.png'))
            if v:pairs.append(dict(id=f'{index:06d}_pair{v}',parent=source['parent'],view0=f'{index:06d}_view0',view1=name,
                                   H_0_to_1=(matrix@np.linalg.inv(transforms[0])).tolist()))
        provenance.append(dict(parent=source['parent'],hr=str(path),sha256=digest,source_to_views=[m.tolist() for m in transforms]))
    manifest=dict(phase=a.phase,task='known_homography',camera='synthetic-area',rows=rows,pairs=pairs,
                  excluded_dev_parents=excluded,sources=provenance,
                  construction='HR warp with cubic sampling -> uint8 HR -> area4 -> round uint8 LR; 1.2x center zoom; +/-3 degree plus fixed perspective; valid source corners')
    write_json(dest/'manifest.json',manifest)
    print(json.dumps({'manifest':str(dest/'manifest.json'),'sha256':hashlib.sha256((dest/'manifest.json').read_bytes()).hexdigest(),
                      'parents':len(parents),'views':len(rows),'pairs':len(pairs)}))


if __name__=='__main__':main()
