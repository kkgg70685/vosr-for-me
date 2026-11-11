"""Use the locked visual flow model on one native low-resolution observation."""
import argparse
from PIL import Image
from study import Camera, FrozenVOSR, P, RUN, F, Path, image, json, start, finish, torch


def restore(source, dest, domain, seed, use_camera_head):
    choice=json.loads((RUN/domain/"choice.json").read_text())["selected"]
    model=FrozenVOSR()
    model.use_checkpoint=False
    camera=Camera() if domain=="real" else None
    if use_camera_head and camera is None:
        raise ValueError("The published camera head requires the real preset")
    lr=image(source)[None].cuda().float()/255
    if lr.shape!=(1,3,128,128):raise ValueError("Expected a native 128 × 128 RGB image")
    sensor=camera.sensor if camera else lambda x:F.avg_pool2d(x,4)
    cfg=P["cfg_"+domain]
    method="native" if not choice["scale"] else "visual"
    state=start(model,lr,seed,cfg)
    z=finish(model,state,lr,cfg,method,choice["scale"],choice["weight"],sensor)
    with torch.no_grad():
        rgb=camera.pipeline(model,z,lr) if use_camera_head else model.decode(z)
    data=rgb[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    dest=Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(data).save(dest)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--domain",choices=("synth","real"),default="real")
    parser.add_argument("--seed",type=int,default=6)
    parser.add_argument("--camera-head",action="store_true")
    args=parser.parse_args()
    restore(args.input,args.output,args.domain,args.seed,args.camera_head)
