"""Verify the unmodified source snapshot without loading model assets."""
from pathlib import Path
import ast, hashlib, json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    p=root/record['path'];blob=p.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==record['sha256'],record['path']
    if p.suffix=='.py': ast.parse(blob.decode('utf-8-sig'),filename=str(p))
    if p.suffix=='.json': json.loads(blob.decode('utf-8-sig'))
print('Verified',len(manifest['files']),'source/config files; hashes and syntax match.')
