"""Build editable draw.io pages and matching standalone SVGs from one vector spec."""
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"blue": ("#6089ce", "#355786"), "orange": ("#e99557", "#9e572c"),
          "pink": ("#efb9be", "#9c646f"), "green": ("#b8dbc7", "#4e8267"),
          "gray": ("#e2e3e5", "#777d85"), "white": ("#ffffff", "#355786")}


class Page:
    def __init__(self, name, title, subtitle):
        self.name, self.title, self.subtitle = name, title, subtitle
        self.nodes, self.edges = [], []

    def box(self, key, x, y, w, h, text, color="blue", dashed=False):
        self.nodes.append(dict(key=key, x=x, y=y, w=w, h=h, text=text, color=color, dashed=dashed))
        return key

    def arrow(self, source, target, text="", color="blue", dashed=False, route=None, label_at=None):
        self.edges.append(dict(source=source, target=target, text=text, color=color, dashed=dashed,
                               route=route, label_at=label_at))


def pages():
    p = Page("01-overall", "VOSR v2 · Structure and observation", "Frozen content / material generation + explicit graph hypotheses + native-default output")
    p.box("lr", 90, 100, 180, 65, "LR image\n128 × 128 RGB", "white")
    p.box("vae", 50, 225, 260, 70, "Frozen VAE / spatial encoder")
    p.box("vision", 380, 225, 260, 70, "Frozen vision encoder", "orange")
    p.box("graph", 1040, 100, 470, 80, "LR StructureHead\noccupancy · endpoints · corners · links · period", "green")
    p.box("hyp", 1040, 225, 470, 80, "Whole-object graph alternatives\nposition / connection / repeated phase", "green")
    p.box("obs", 1040, 350, 470, 100, "One shared camera + one appearance fit\nfinite nuisance covariance / paired LR evidence\nkeep ambiguous candidates", "green")
    p.box("draw", 1040, 500, 470, 80, "One candidate per object and seed\nspatial graph + 3 relation strengths", "green")
    p.box("cond", 50, 350, 260, 70, "Spatial latent condition")
    p.box("sem", 380, 350, 260, 70, "Visual semantic tokens", "orange")
    p.box("prefix", 210, 485, 470, 80, "Frozen VOSR velocity · Euler steps 1–16\nshared condition, noise and latent state", "pink")
    p.box("native", 70, 640, 310, 80, "Native suffix · steps 17–25\nnative velocity + condition", "blue")
    p.box("revised", 550, 640, 360, 80, "Revised suffix · steps 17–25\nv + spatial velocity adapter", "pink")
    p.box("raw", 70, 780, 310, 70, "Frozen decoder\nnative RGB + native texture")
    p.box("rev", 550, 780, 360, 70, "Same frozen decoder\nrevised RGB + attached texture", "pink")
    p.box("gate", 1040, 655, 470, 100, "Atomic object acceptance\nrelation evidence + actual RGB reprojection\ncopy whole object; blend only outside", "green")
    p.box("out", 1040, 820, 470, 90, "Native-default SR\nbounded coarse photometric feedback\n512 × 512 RGB", "white")
    for a, b in (("lr", "vae"), ("vae", "cond"), ("cond", "prefix"), ("prefix", "native"), ("native", "raw")):
        p.arrow(a, b)
    p.arrow("lr", "vision", color="orange", dashed=True); p.arrow("vision", "sem", color="orange", dashed=True)
    p.arrow("sem", "prefix", color="orange", dashed=True)
    p.arrow("lr", "graph", color="green", dashed=True)
    for a, b in (("graph", "hyp"), ("hyp", "obs"), ("obs", "draw"), ("draw", "revised")):
        p.arrow(a, b, color="green", dashed=True)
    p.arrow("prefix", "revised", "same latent", "pink")
    p.arrow("revised", "rev", color="pink"); p.arrow("rev", "gate", color="pink")
    p.arrow("raw", "gate", "native default", "blue", route=[(225,850),(225,925),(990,925),(990,735),(1040,735)], label_at=(580,917))
    p.arrow("gate", "out", color="green")
    yield p

    p = Page("02-structure", "Explicit structure · Complete alternatives", "structure.py / model.py: graph identity is the decision unit; texture is not forced into a graph")
    p.box("lr", 70, 130, 260, 70, "LR RGB", "white")
    p.box("head", 70, 265, 260, 110, "StructureHead\n3 × 3 convolution stack\nsubpixel ×4 output", "green")
    p.box("maps", 430, 190, 430, 160, "9 dense maps\nink / endpoint / corner / boundary\nx-y affinity / repeat likelihood\nperiod vector (dx, dy)", "green")
    p.box("graph", 1030, 190, 440, 160, "Threshold → thin → explicit graph\nvertices at HR pixel centres\nedges encode actual connectivity\nendpoint / turn consistency from degree", "green")
    p.box("budget", 1030, 420, 440, 110, "Merge nearby fragments and halos\nreject dense / oversized / colliding objects\nno arbitrary tiling of a stroke", "gray")
    p.box("position", 80, 620, 310, 110, "POSITION\nreference + ±x / ±y translation\nmove the entire graph", "blue")
    p.box("topology", 490, 620, 310, 110, "CONNECTION\njoin nearest eligible endpoints\nor open a chain segment", "pink")
    p.box("phase", 900, 620, 310, 110, "REPETITION PHASE\noptional half-period translation\nkeep all repeated edges coherent", "orange")
    p.box("complete", 450, 815, 710, 90, "At most 8 complete graphs per object\nreference + 4 position + 2 connection + 1 phase; no untested combinations", "white")
    p.arrow("lr", "head", color="green"); p.arrow("head", "maps", color="green")
    p.arrow("maps", "graph", color="green"); p.arrow("graph", "budget", color="green")
    for name, color in (("position", "blue"), ("topology", "pink"), ("phase", "orange")):
        x = {"position": 235, "topology": 645, "phase": 1055}[name]
        p.arrow("budget", name, color=color, dashed=True, route=[(1250,530),(1250,575),(x,575),(x,620)])
        p.arrow(name, "complete", color=color)
    yield p

    p = Page("03-evidence", "Difference-specific observation evidence", "observation.py: shared image formation; finite camera uncertainty; abstention when a proxy cannot explain the LR")
    p.box("candidates", 70, 140, 370, 100, "Complete graph candidates\ncommon foreground / background plane\nfitted once to the mean occupancy", "green")
    p.box("camera", 610, 140, 370, 100, "One frozen HR → LR sensor\ncommon blur + registration bank\nidentical parameters for every candidate", "blue")
    p.box("real", 1160, 140, 330, 100, "Observed LR\nfixed noise floor\nobject + PSF footprint", "white")
    p.box("pair", 610, 340, 370, 105, "Paired counterfactual difference\nΔ = Dθ(candidate B) − Dθ(candidate A)\nresidual measured against pair midpoint", "green")
    p.box("nuisance", 70, 530, 410, 130, "Separate nuisance explanations\nproject out RGB constant / slow planes\nwhiten with finite camera covariance\nlarge moves are not erased", "gray")
    p.box("tests", 610, 530, 370, 130, "Separate tests for EACH difference\nwhitened separation ≥ 3\npaired preference ≥ 2\nabsolute fit RMS ≤ 4 noise units", "green")
    p.box("retain", 1140, 530, 350, 130, "Keep plausible whole candidates\nseed chooses one coherent graph\nposition / topology / phase strengths\nnever one universal confidence", "orange")
    p.box("limit", 430, 805, 760, 100, "Conservative finite-bank criterion, not a calibrated posterior\nPoor model fit → no evidence. Different factors are not conflated.\nAn inference candidate is never selected using HR.", "white")
    p.arrow("candidates", "camera", color="green"); p.arrow("camera", "pair")
    p.arrow("real", "pair", color="blue", dashed=True)
    p.arrow("pair", "nuisance", color="gray"); p.arrow("nuisance", "tests", color="gray")
    p.arrow("pair", "tests", color="green"); p.arrow("tests", "retain", color="green")
    p.arrow("retain", "limit", color="orange", dashed=True)
    yield p

    p = Page("04-generation-output", "Velocity conditioning · Native-default acceptance", "model.py: frozen backbone, zero-initialised residual, exact interior copy, explicit coarse-only feedback")
    p.box("state", 60, 145, 300, 80, "Shared state z16\nspatial latent + visual tokens", "blue")
    p.box("field", 620, 145, 420, 80, "12-channel spatial condition\n6 graph maps + 3 strengths + support + period", "green")
    p.box("native", 60, 365, 300, 110, "Native velocity\nv = weak + cfg(full − weak)\n9 Euler updates\nfrozen decoder → N", "blue")
    p.box("adapter", 620, 315, 420, 125, "SpatialVelocityAdapter\n[z, condition, LR, graph, time]\nzero-output convolution head\nbounded residual × object support", "pink")
    p.box("rev", 620, 505, 420, 90, "Revised velocity v + δv\n9 Euler updates\nfrozen decoder → R", "pink")
    p.box("gate", 1170, 315, 340, 140, "Acceptance for the ACTUAL composite\n1. intended relation improves\n2. paired RGB evidence supports it\n3. local + global LR error improves\nincluding the exterior seam", "green")
    p.box("copy", 1170, 555, 340, 110, "Copy R inside accepted object\nkeep N inside rejected object\nblend only exterior background", "green")
    p.box("photo", 620, 760, 420, 115, "Bounded coarse RGB lattice\n4 × 4 → bilinear field\nshared-camera line search\nno geometric or fine-frequency correction", "gray")
    p.box("out", 1170, 800, 340, 75, "Final SR\nplus decision diagnostics", "white")
    p.arrow("state", "native"); p.arrow("state", "adapter", "same latent", "blue", True)
    p.arrow("field", "adapter", color="green", dashed=True); p.arrow("adapter", "rev", color="pink")
    p.arrow("native", "gate", "N = default", "blue", route=[(360,420),(480,420),(480,690),(1100,690),(1100,425),(1170,425)], label_at=(780,683))
    p.arrow("rev", "gate", color="pink")
    p.arrow("gate", "copy", color="green"); p.arrow("copy", "photo", color="gray")
    p.arrow("photo", "out", color="gray")
    yield p

    p = Page("05-training-experiments", "Training and experiments · What is and is not established", "train.py / evaluate.py: train new modules only; isolate parents and cameras; retain failed readouts in denominators")
    p.box("data", 70, 150, 400, 100, "Training-only paired data\nLR + HR + explicit structure masks\nsynthetic primitives for initialisation", "white")
    p.box("structure", 610, 150, 380, 120, "Stage A: train StructureHead\nclass-balanced 7-map supervision\nperiod regression + link incidence\nendpoint / graph proposal qualification", "green")
    p.box("adapter", 1140, 150, 370, 145, "Stage B: train velocity adapter\nfrozen VOSR + frozen structure head\nHR selects a compatible training candidate\nsample a native suffix state\ndecode clean-prediction surrogate", "pink")
    p.box("loss", 1140, 410, 370, 105, "Loss\nobject RGB + signed gradients\nexterior identity + residual penalty", "pink")
    p.box("dev", 610, 410, 380, 105, "Parent / camera-isolated development\nqualify graph and relation readout first\nlock weights, masks, thresholds, protocol", "gray")
    p.box("controls", 70, 410, 400, 145, "Matched-seed comparisons\nraw native / locked v1.5 / full v2\nuniform / misplaced graph\nno acceptance / no feedback\n8, 12, 16, 20 shared-step interventions", "blue")
    p.box("test", 450, 705, 740, 150, "Held-out evaluation\nmissing + unstable endpoint rates; connection error bounds\nboundary / skeleton / phase; PSNR / LPIPS / DISTS\naverage seeds within parent → paired parent bootstrap\nreport every device, worst group, acceptance coverage, runtime and memory", "white")
    p.arrow("data", "structure", color="green"); p.arrow("structure", "adapter", color="green")
    p.arrow("data", "adapter", "HR is training-only", "gray", True,
            route=[(270,150),(270,115),(1325,115),(1325,150)], label_at=(790,108))
    p.arrow("adapter", "loss", color="pink"); p.arrow("loss", "dev", color="pink")
    p.arrow("dev", "controls", color="blue"); p.arrow("controls", "test", color="blue")
    p.arrow("dev", "test", "locked protocol", "gray", True)
    yield p


def drawio(all_pages):
    file = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17", type="device")
    for page in all_pages:
        diagram = ET.SubElement(file, "diagram", id=page.name, name=page.name)
        model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="1000", grid="1", gridSize="10", page="1", pageWidth="1600", pageHeight="1000")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0"); ET.SubElement(root, "mxCell", id="1", parent="0")
        for key, y, text, size in (("title", 20, page.title, 29), ("subtitle", 65, page.subtitle, 16)):
            c = ET.SubElement(root, "mxCell", id=key, value=text, style=f"text;html=0;fontSize={size};fontFamily=Georgia;align=center;", vertex="1", parent="1")
            ET.SubElement(c, "mxGeometry", x="20", y=str(y), width="1560", height="35", **{"as": "geometry"})
        for node in page.nodes:
            fill, stroke = COLORS[node["color"]]
            style = f"rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontColor=#172236;fontFamily=Georgia;fontSize=18;arcSize=12;shadow=1;spacing=8;"
            if node["dashed"]:
                style += "dashed=1;"
            c = ET.SubElement(root, "mxCell", id=node["key"], value=node["text"], style=style, vertex="1", parent="1")
            ET.SubElement(c, "mxGeometry", x=str(node["x"]), y=str(node["y"]),
                          width=str(node["w"]), height=str(node["h"]), **{"as": "geometry"})
        for i, edge in enumerate(page.edges):
            stroke = COLORS[edge["color"]][1]
            style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;strokeWidth=2;strokeColor={stroke};fontSize=14;fontFamily=Georgia;labelBackgroundColor=#ffffff;"
            if edge["dashed"]:
                style += "dashed=1;dashPattern=5 4;"
            route = edge["route"]
            if route:
                lookup = {node["key"]: node for node in page.nodes}
                source, target = lookup[edge["source"]], lookup[edge["target"]]
                style += f"exitX={(route[0][0]-source['x'])/source['w']};exitY={(route[0][1]-source['y'])/source['h']};exitPerimeter=0;"
                style += f"entryX={(route[-1][0]-target['x'])/target['w']};entryY={(route[-1][1]-target['y'])/target['h']};entryPerimeter=0;"
            c = ET.SubElement(root, "mxCell", id=f"e{i}", value=edge["text"], style=style, edge="1", parent="1", source=edge["source"], target=edge["target"])
            geometry = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
            if route:
                points = ET.SubElement(geometry, "Array", **{"as": "points"})
                for x, y in route[1:-1]:
                    ET.SubElement(points, "mxPoint", x=str(x), y=str(y))
    ET.indent(file)
    return ET.tostring(file, encoding="unicode", xml_declaration=True)


def svg(page):
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">',
         '<defs><filter id="shadow" x="-10%" y="-10%" width="130%" height="140%"><feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.15"/></filter>']
    for key, (_, stroke) in COLORS.items():
        s.append(f'<marker id="{key}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9Z" fill="{stroke}"/></marker>')
    s.append('</defs><rect width="1600" height="1000" fill="white"/>')
    s.append(f'<text x="800" y="49" text-anchor="middle" font-family="Georgia,serif" font-size="29" fill="#172236">{escape(page.title)}</text>')
    s.append(f'<text x="800" y="86" text-anchor="middle" font-family="Georgia,serif" font-size="16" fill="#556174">{escape(page.subtitle)}</text>')
    lookup = {n["key"]: n for n in page.nodes}
    for edge in page.edges:
        a, b = lookup[edge["source"]], lookup[edge["target"]]
        ax, ay = a["x"]+a["w"]/2, a["y"]+a["h"]/2
        bx, by = b["x"]+b["w"]/2, b["y"]+b["h"]/2
        if abs(bx-ax) > abs(by-ay) * 1.5:
            direction = 1 if bx > ax else -1
            ax += direction*a["w"]/2; bx -= direction*b["w"]/2
            mid = (ax+bx)/2
            points = f"{ax},{ay} {mid},{ay} {mid},{by} {bx},{by}"
        else:
            direction = 1 if by > ay else -1
            ay += direction*a["h"]/2; by -= direction*b["h"]/2
            mid = (ay+by)/2
            points = f"{ax},{ay} {ax},{mid} {bx},{mid} {bx},{by}"
        if edge["route"]:
            points = " ".join(f"{x},{y}" for x, y in edge["route"])
        dash = ' stroke-dasharray="8 6"' if edge["dashed"] else ""
        s.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[edge["color"]][1]}" stroke-width="2" marker-end="url(#{edge["color"]})"{dash}/>')
        if edge["text"]:
            lx, ly = edge["label_at"] or ((ax+bx)/2, (ay+by)/2-7)
            s.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="Georgia,serif" font-size="14" fill="#355786">{escape(edge["text"])}</text>')
    for node in page.nodes:
        fill, stroke = COLORS[node["color"]]
        x, y, w, h = [node[k] for k in ("x", "y", "w", "h")]
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#shadow)"/>')
        lines = node["text"].splitlines()
        for i, line in enumerate(lines):
            yy = y+h/2+(i-(len(lines)-1)/2)*24+6
            s.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" font-family="Georgia,serif" font-size="18" fill="#172236">{escape(line)}</text>')
    s.append('<text x="800" y="970" text-anchor="middle" font-family="Georgia,serif" font-size="16" fill="#556174">Blue: spatial / native · Orange: visual / ambiguity · Green: structure / evidence · Pink: velocity · Gray: nuisance / training</text></svg>')
    return "\n".join(s)


if __name__ == "__main__":
    target = ROOT / "docs/architecture_v2"; target.mkdir(parents=True, exist_ok=True)
    all_pages = list(pages())
    (target / "structure-observation-v2.drawio").write_text(drawio(all_pages), encoding="utf-8")
    for page in all_pages:
        (target / f"{page.name}.svg").write_text(svg(page), encoding="utf-8")
    print(f"Wrote {len(all_pages)} editable pages and SVG previews to {target.relative_to(ROOT)}")
