"""Run the released vision-only SR model on one native 128 x 128 RGB image."""
import argparse
import json
import torch
from semantic_study import PUB, OLD, ROOT, base_camera, load_heads, inputs, references, projected, image, save_png, compose
from evidence import LocalSR
from balanced import corrections
from upstream import start, finish, coefficients, refine, camera_stages
from v15_study import load_heads as load_texture_head


@torch.no_grad()
def restore(lr, seed, preset, base, camera, assets):
    lrnet, visual, pca, texture, ridge, choice = assets
    state = start(base, lr, seed, -.5)
    sem = state[1]
    coeff = coefficients(lr, sem, "visual")
    native_z = finish(base, state, lr, -.5, "native", .04, 8, camera.sensor)
    raw, _, _, calibrated = camera_stages(camera, base, native_z, lr)
    guided_z = finish(base, state, lr, -.5, "visual", .04, 8, camera.sensor)
    guided = camera.pipeline(base, guided_z, lr)
    v14 = torch.lerp(refine(calibrated, lr, coeff, 24, False), refine(guided, lr, coeff, 24, False), .5)
    data = {"raw":raw, "v14":v14, "lr":lr}
    fine, middle, reg = corrections(data, texture, ridge)
    anchor = v14 + fine + (.25 if preset=="balanced" else 0.)*middle + .5*reg
    bank = references(lr, sem)
    t = torch.nn.functional.interpolate(projected(sem,pca),size=lr.shape[-2:],mode="bilinear",align_corners=False)
    b,t = inputs(lr,bank,t,"vision")
    source = visual(lr,lrnet(lr),b,t)
    return compose(data,anchor,{"vision":source},choice)


def dependencies():
    p=PUB.parent
    lrnet=LocalSR().cuda().eval()
    lrnet.load_state_dict(torch.load(p/"localnet.pt",weights_only=True,map_location="cuda"))
    from semantic_model import EvidenceHead
    visual=EvidenceHead().cuda().eval()
    visual.load_state_dict(torch.load(PUB/"vision.pt",weights_only=True,map_location="cuda"))
    choice=json.loads((PUB/"release.json").read_text())["choice"]
    if choice["selected"].split("/")[0]!="vision":raise RuntimeError("Unexpected published model")
    return (lrnet,visual,torch.load(PUB/"pca.pt",weights_only=True,map_location="cuda"),
            load_texture_head("learned")["learned"],torch.load(OLD/"ridge.pt",weights_only=True,map_location="cuda"),choice)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=6)
    p.add_argument("--preset",choices=("balanced","fidelity"),default="balanced")
    args=p.parse_args();torch.set_num_threads(4)
    y=image(args.input)[None].cuda().float()/255
    if y.shape!=(1,3,128,128):raise ValueError("Expected a native 128 x 128 RGB LR image")
    base,camera=base_camera()
    save_png(restore(y,args.seed,args.preset,base,camera,dependencies()),args.output)
