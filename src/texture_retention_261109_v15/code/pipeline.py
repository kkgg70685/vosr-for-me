"""Complete registered tasks and exercise the final single-image command on the server."""
import fcntl
import json
import subprocess
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

CODE=Path(__file__).resolve().parent
RUN=CODE.parent
ROOT=RUN.parent


def stage(lock_name,marker,args):
    if (RUN/marker).exists():return
    with (RUN/lock_name).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    if not (RUN/marker).exists():
        subprocess.run([sys.executable,"-u",*args],cwd=CODE,check=True)
    if not (RUN/marker).exists():raise RuntimeError("Missing output: "+marker)


def cli_test():
    sys.path.insert(0,str(CODE))
    from balanced import pools
    sample=pools()[0][0]
    dest=RUN/"cli_smoke.png"
    subprocess.run([sys.executable,str(CODE/"balanced.py"),"restore","--input",str(ROOT/sample["lr"]),
                    "--output",str(dest),"--seed","6"],cwd=CODE,check=True)
    a=np.asarray(Image.open(dest)).astype("float32")
    b=np.asarray(Image.open(RUN/"balanced_check.png")).astype("float32")
    if a.shape!=(512,512,3):raise ValueError("Unexpected CLI output")
    diff=np.abs(a-b)
    receipt=dict(shape=list(a.shape),max_u8_difference=float(diff.max()),mean_u8_difference=float(diff.mean()))
    if diff.mean()>.2 or diff.max()>3:raise ValueError("CLI diverged: "+repr(receipt))
    (RUN/"cli_smoke.json").write_text(json.dumps(receipt))


if __name__=="__main__":
    with (RUN/"pipeline.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stage("study.lock","release.json",["v15_study.py","study"])
        stage("calibration.lock","v15_release.json",["release_study.py","study"])
        stage("balanced.lock","balanced_release.json",["balanced.py","study"])
        stage("check.lock","final_gallery.json",["check_release.py"])
        if not (RUN/"cli_smoke.json").exists():cli_test()
        print("Release ready",flush=True)
