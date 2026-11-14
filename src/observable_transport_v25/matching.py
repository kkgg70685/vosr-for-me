"""Downstream homography benchmark using a fixed, untrained SIFT matcher.

Input: id, parent, H_0_to_1, methods:{native:[image0,image1], ...}.
H is evaluation-only. No matches, H, labels or HR enter SR inference.
Failure cases remain in the AUC denominator. This is NOT a 3D pose benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .metrics import error_auc, homography_corner_error


def estimate_homography(a, b):
    import cv2
    cv2.setRNGSeed(6)
    sift = cv2.SIFT_create(nfeatures=2048)
    ka, da = sift.detectAndCompute(a, None); kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(da) < 4 or len(db) < 4:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75*pair[1].distance]
    if len(matches) < 4: return None, len(matches)
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    h, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3., maxIters=2000, confidence=.995)
    return h, len(matches)


def evaluate_pairs(rows, root):
    import cv2
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty manifest and unique pair ids required")
    names = set(rows[0]["methods"])
    if "native" not in names or len(names) < 2:
        raise ValueError("Include native and at least one paired control")
    records = []
    for row in rows:
        if not row.get("parent") or set(row["methods"]) != names:
            raise ValueError("Same methods and explicit parents required for all pairs")
        reference = np.asarray(row["H_0_to_1"], dtype=float)
        if reference.shape != (3,3) or not np.isfinite(reference).all() or np.linalg.matrix_rank(reference) != 3:
            raise ValueError("Invalid reference homography")
        shape = None
        for method, paths in row["methods"].items():
            if len(paths) != 2: raise ValueError("Each method requires two images")
            a, b = [cv2.imread(str(root/p), cv2.IMREAD_GRAYSCALE) for p in paths]
            if a is None or b is None: raise FileNotFoundError(paths)
            if shape is None: shape = (a.shape, b.shape)
            if shape != (a.shape, b.shape): raise ValueError("All controls must share HR pixel coordinates")
            estimated, count = estimate_homography(a, b)
            error = homography_corner_error(estimated, reference, *a.shape)
            records.append(dict(id=row["id"], parent=row["parent"], method=method,
                                error=error, matches=count, failed=not np.isfinite(error)))
    summary = {}
    for name in sorted(names):
        rr = [r for r in records if r["method"] == name]
        parents = sorted({r["parent"] for r in rr})
        macro = [error_auc([r["error"] for r in rr if r["parent"] == parent]) for parent in parents]
        summary[name] = {"pair_auc": error_auc([r["error"] for r in rr]),
                         "parent_macro_auc": {key: sum(m[key] for m in macro)/len(macro) for key in macro[0]},
                         "failed_pairs": sum(r["failed"] for r in rr), "pairs": len(rr), "parents": len(parents)}
    return {"summary": summary, "records": records,
            "matcher": "OpenCV SIFT-2048; ratio=.75; RANSAC=3px,2000,.995; RNG=6",
            "opencv_version": cv2.__version__, "threshold_unit": "HR pixels"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); manifest = Path(a.manifest)
    result = evaluate_pairs(json.loads(manifest.read_text(encoding="utf-8")), manifest.parent)
    result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for record in result["records"]:
        if not np.isfinite(record["error"]): record["error"] = None
    dest = Path(a.output_dir); dest.mkdir(parents=True, exist_ok=False)
    (dest/"matching.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__": main()
