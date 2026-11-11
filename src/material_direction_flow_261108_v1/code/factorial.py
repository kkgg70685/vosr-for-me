"""Matched terminal corrections on the native, local-only and misplaced trajectories."""
import fcntl
import hashlib
import statistics
import time
from refine_study import (RUN,CODE,UP,Camera,FrozenVOSR,F,cases,coefficients,delta,finish,group,
                          json,np,refine,save_image,sha,start,texture_score,torch,write_json)

OUT=RUN/"factorial"


def evaluate(base,camera,domain):
    real=domain!="synth"
    preset="real" if real else domain
    sel=json.loads((RUN/preset/"choice.json").read_text())["selected"]
    flow=json.loads((UP/preset/"choice.json").read_text())["selected"]
    rows=[]
    for name,seed,lr,hr,truth in cases(domain,"confirm"):
        fname=hashlib.sha256(name.encode()).hexdigest()[:20]+f"_{seed}.json"
        old=json.loads((RUN/domain/"confirm"/fname).read_text())
        file=OUT/domain/fname
        if not file.exists():
            st=start(base,lr,seed,-.5 if real else .5)
            sensor=camera.sensor if real else lambda x:F.avg_pool2d(x,4)
            scores={"selected":old["scores"][sel["key"]]}
            for mode in ("native","local","misplaced"):
                z=finish(base,st,lr,-.5 if real else .5,mode,flow["scale"],flow["weight"],sensor)
                with torch.no_grad():
                    source=camera.pipeline(base,z,lr) if sel["source"].endswith("camera") else base.decode(z)
                    field_mode="visual" if mode=="native" else mode
                    x=refine(source,lr,coefficients(lr,st[1],field_mode),sel["steps"],not real)
                    scores[mode]=camera.scores(x,hr,lr) if real else texture_score(x,lr,hr,*truth)
            write_json(file,dict(scene=name,seed=seed,scores=scores))
        rows.append(json.loads(file.read_text()))
        print(domain,len(rows),flush=True)
    means={k:group([r["scores"][k] for r in rows]) for k in rows[0]["scores"]}
    fields=[k for k in means["selected"] if k!="dc_drift"]
    effects={k:{f:delta(rows,"selected",k,f) for f in fields} for k in ("native","local","misplaced")}
    write_json(OUT/domain/"summary.json",dict(cases=len(rows),means=means,effects=effects))


def latency(base,camera):
    from refine_study import restore
    _,seed,y,_,_=next(cases("real","dev"))
    def render(kind):
        if kind=="release":return restore(base,camera,"real",y,seed)
        with torch.no_grad():
            cond,sem,_=base.conditions(y)
            z=base.sample(cond,sem,y,seed=seed,cfg=-.5,return_latent=True)
            return camera.pipeline(base,z,y) if kind=="v1.3" else base.decode(z)
    results={}
    for k in ("native","v1.3","release"):
        render(k)
        durations=[];peaks=[]
        for _ in range(5):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            render(k)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-t)
            peaks.append(torch.cuda.max_memory_allocated()/1024**2)
        results[k]={"median_s":statistics.median(durations),"peak_allocated_mib":max(peaks)}
    write_json(OUT/"latency.json",dict(gpu=torch.cuda.get_device_name(),repeats=5,results=results))


if __name__=="__main__":
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/"lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=FrozenVOSR();base.use_checkpoint=False
        camera=None
        for domain in ("synth","real","dreal"):
            if domain=="real":camera=Camera()
            evaluate(base,camera,domain)
        latency(base,camera)
        write_json(OUT/"complete.json",dict(code_sha=sha(CODE/"factorial.py")))
