"""Server entry for selection, paired tests and model release across the three runs."""
import fcntl
import subprocess
import sys
from pathlib import Path

ROOT=Path("/home/zhangxu/hx/HX001")
DETAIL=ROOT/"material_detail_projection_261108_v1"
MIX=ROOT/"visual_trajectory_mix_261108_v1"
sys.path.append(str(DETAIL/"code"))
from workflow import stage


if __name__=="__main__":
    with (MIX/"pipeline.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run([sys.executable,"-u","code/workflow.py"],cwd=DETAIL,check=True)
        (DETAIL/"factorial").mkdir(exist_ok=True)
        stage(DETAIL,"factorial/lock","factorial/complete.json",["code/factorial.py"])
        stage(MIX,"lock","release.json",["code/mix_study.py","study"])
        print("Released:",MIX/"release.json",flush=True)
