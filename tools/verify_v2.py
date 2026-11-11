"""Verify the new architecture independently of the retained v1.5 publication."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance_v2.json"


def paths():
    selected = [ROOT / "README.md", ROOT / "configs/structure_observation_v2.json",
                ROOT / "tests/test_structure_v2.py", ROOT / "tools/draw_structure_v2.py",
                ROOT / "tools/verify_v2.py", ROOT / ".github/workflows/structure-v2.yml"]
    for directory in ("src/structure_observation_v2", "docs/architecture_v2"):
        selected.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh intentional new-version hashes")
    args = parser.parse_args()
    if args.write:
        manifest = {"version": "structure-observation-v2.0",
                    "base_commit": "5b116bd5f8f998f708c3b265f822b52a40cb668a",
                    "policy": "New architecture; original provenance.json publication bytes are retained.",
                    "gpu_experiments_performed": False,
                    "files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                               "bytes": p.stat().st_size} for p in paths()]}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = json.loads(MANIFEST.read_text())
    expected = {r["path"] for r in manifest["files"]}
    assert expected == {str(p.relative_to(ROOT)) for p in paths()}, "Unrecorded or missing v2 file"
    for record in manifest["files"]:
        path = ROOT / record["path"]; blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == record["sha256"], record["path"]
        if path.suffix == ".py":
            ast.parse(blob, filename=record["path"])
        elif path.suffix == ".json":
            json.loads(blob)
        elif path.suffix in (".drawio", ".svg"):
            tree = ET.fromstring(blob)
            if path.suffix == ".drawio":
                assert len(tree.findall("diagram")) == 5
                for page in tree.findall("diagram"):
                    cells = page.findall(".//mxCell"); ids = {c.get("id") for c in cells}
                    assert len(ids) == len(cells), "Duplicate draw.io id"
                    for cell in cells:
                        if cell.get("edge"):
                            assert cell.get("source") in ids and cell.get("target") in ids
    print(f"Verified {len(manifest['files'])} v2 files; hashes, syntax and vector references match.")


if __name__ == "__main__":
    main()
