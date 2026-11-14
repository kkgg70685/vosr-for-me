"""Verify v2.5 source identity, syntax, JSON and diagram structure; no assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
paths=[]
for pattern in ('src/observable_transport_v25/*.py','src/observable_transport_v25/*.md',
                'src/observable_transport_v25/*.txt','configs/observable_transport_v25.json',
                'tests/test*25.py','tests/test_v25*.py','docs/v25/*.md',
                'docs/architecture_v25/*.drawio','docs/architecture_v25/*.svg',
                'docs/architecture_v25/*.md','tools/*v25*.py'):
    paths.extend(root.glob(pattern))
paths=sorted(set(paths))
p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
records=[]
for path in paths:
    text=path.read_text(encoding='utf-8')
    if path.suffix=='.py':ast.parse(text,filename=str(path))
    if path.suffix=='.json':json.loads(text)
    if path.suffix=='.drawio':
        tree=ET.fromstring(text)
        assert len(tree.findall('diagram'))==3
        for page in tree.findall('diagram'):
            cells=page.findall('.//mxCell');ids=[c.attrib['id'] for c in cells]
            assert len(ids)==len(set(ids))
            for c in cells:
                if c.get('edge')=='1':
                    assert c.get('source') in ids and c.get('target') in ids
    records.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
target=root/'provenance_v25.json'
if args.write:
    target.write_bytes((json.dumps({'version':'2.5.0-prototype','base_main':'5b116bd5f8f998f708c3b265f822b52a40cb668a',
                                  'files':records},indent=2)+'\n').encode('utf-8'))
else:
    expected=json.loads(target.read_text(encoding='utf-8'))['files']
    assert expected==records,'v2.5 source inventory/hash differs; review before --write'
print('Verified',len(records),'v2.5 files; syntax, identity and 3-page diagram structure')
