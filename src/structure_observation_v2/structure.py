"""LR-only structural proposals, explicit pixel graphs, and whole-graph alternatives.

The graph is deliberately bounded. Unstable/dense texture is left to native VOSR.
Graph pixels are vertices; undirected edges carry connectivity, including gaps.
No image-dependent threshold is chosen using an evaluation HR image.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = ("ink", "endpoint", "corner", "boundary", "affinity_x", "affinity_y",
            "repeat", "period_x", "period_y")


class StructureHead(nn.Module):
    """9 HR maps from LR; semantic tokens remain on the original VOSR path."""
    def __init__(self, width=48):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, 9 * 16, 1), nn.PixelShuffle(4))

    def forward(self, lr):
        raw = self.body(lr - 0.5)
        return torch.cat((raw[:, :7].sigmoid(), raw[:, 7:].tanh()), 1)


def thin(mask):
    """Zhang-Suen thinning with a zero exterior; deterministic CPU graph extraction."""
    x = np.pad(np.asarray(mask, dtype=bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [x[:-2, 1:-1], x[:-2, 2:], x[1:-1, 2:], x[2:, 2:],
                 x[2:, 1:-1], x[2:, :-2], x[1:-1, :-2], x[:-2, :-2]]
            count = sum(a.astype(np.uint8) for a in p)
            turns = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                keep = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                keep = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            delete = x[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (turns == 1) & keep
            changed |= bool(delete.any())
            x[1:-1, 1:-1][delete] = False
        if not changed:
            return x[1:-1, 1:-1]


@dataclass
class Graph:
    xy: np.ndarray  # [N, 2], HR pixel-centre coordinates
    edges: np.ndarray  # [E, 2], undirected vertex indices
    period: tuple = (0.0, 0.0)
    repeat: float = 0.0

    @property
    def degree(self):
        return np.bincount(self.edges.reshape(-1), minlength=len(self.xy))

    @property
    def endpoints(self):
        return np.flatnonzero(self.degree <= 1)

    @property
    def corners(self):
        # Turns and junctions, independent of foreground polarity.
        out = self.degree > 2
        neighbours = [[] for _ in self.xy]
        for a, b in self.edges:
            neighbours[a].append(b)
            neighbours[b].append(a)
        for i, ns in enumerate(neighbours):
            if len(ns) == 2:
                u, v = self.xy[ns] - self.xy[i]
                out[i] = np.dot(u, v) > -0.85 * np.linalg.norm(u) * np.linalg.norm(v)
        return np.flatnonzero(out)

    def moved(self, dx, dy):
        return replace(self, xy=self.xy + np.array([dx, dy], dtype=np.float32))


@dataclass
class Hypothesis:
    name: str
    family: str
    graph: Graph


def extract_graphs(maps, cfg):
    """Join nearby fragments into ONE decision object before making alternatives.

    Connected groups whose uncertainty halos overlap are merged, so an object
    cannot be partly accepted through a competing neighbour. Budget overflow
    declines the object, instead of cutting it into arbitrary image tiles.
    """
    if maps.shape[0] != 1 or maps.shape[1] != 9:
        raise ValueError("Graph extraction expects [1,9,H,W]")
    a = maps.detach().float().cpu().numpy()[0]
    skeleton = thin(a[0] >= cfg.proposal_threshold)
    xy = np.argwhere(skeleton)[:, ::-1].astype(np.float32)
    if len(xy) == 0:
        return []
    # Explicit lattice edges; diagonal shortcuts across orthogonal corners are omitted.
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(xy)}
    edges = []
    for (x, y), i in lookup.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            j = lookup.get((x + dx, y + dy))
            if j is None:
                continue
            if dx and dy and ((x + dx, y) in lookup or (x, y + dy) in lookup):
                continue
            affinity = a[4 if dx else 5, y, x]
            if affinity >= 0.25:
                edges.append((i, j))
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    # Candidate displacement + feather + camera blur must not create overlap.
    halo = max(cfg.max_gap, 2 * (cfg.radius + cfg.feather + cfg.shift + 2))
    # Linear image memory, even for a dense texture map. Materialising all
    # neighbouring vertex pairs would be quadratic before the object budget.
    expanded = ndimage.distance_transform_edt(~skeleton) <= halo / 2
    labels, _ = ndimage.label(expanded, structure=np.ones((3, 3)))
    groups = {}
    for i, (x, y) in enumerate(xy.astype(int)):
        groups.setdefault(int(labels[y, x]), []).append(i)
    result = []
    for ids in sorted(groups.values(), key=lambda z: (-len(z), z[0])):
        if not cfg.min_pixels <= len(ids) <= cfg.max_pixels:
            continue
        ids = np.array(ids)
        remap = np.full(len(xy), -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        local_edges = remap[edges]
        local_edges = local_edges[(local_edges >= 0).all(1)]
        xs, ys = xy[ids].astype(int).T
        repeat = float(np.median(a[6, ys, xs]))
        period = tuple(np.median(a[7:9, ys, xs], axis=1) * cfg.max_period)
        result.append(Graph(xy[ids], local_edges, period, repeat))
        if len(result) == cfg.max_objects:
            break
    return result


def hypotheses(graph, cfg):
    """At most nine COMPLETE graphs, each counterfactual changes a named relation.

    Alternatives are always relative to the LR graph. No per-pixel candidate
    sampling and no product of untested position/topology combinations.
    """
    bank = [Hypothesis("lr_graph", "reference", graph)]
    for dx, dy in ((cfg.shift, 0), (-cfg.shift, 0), (0, cfg.shift), (0, -cfg.shift)):
        bank.append(Hypothesis(f"move_{dx:g}_{dy:g}", "position", graph.moved(dx, dy)))
    ends = graph.endpoints
    pairs = []
    edge_set = {tuple(sorted(e)) for e in graph.edges.tolist()}
    for ii, jj in sorted(cKDTree(graph.xy[ends]).query_pairs(cfg.max_gap)):
        i, j = ends[ii], ends[jj]
        d = np.linalg.norm(graph.xy[i] - graph.xy[j])
        if 1.5 < d and tuple(sorted((i, j))) not in edge_set:
            pairs.append((d, int(i), int(j)))
    if pairs:
        _, i, j = min(pairs)
        bank.append(Hypothesis("connect", "topology", replace(
            graph, edges=np.concatenate((graph.edges, [[i, j]])))))
    # Remove a short central section of a degree-2 chain, not one invisible edge.
    degree = graph.degree
    eligible = [i for i, (u, v) in enumerate(graph.edges) if degree[u] == degree[v] == 2]
    if eligible:
        centers = graph.xy[graph.edges[eligible]].mean(1)
        chosen = eligible[int(np.linalg.norm(centers - graph.xy.mean(0), axis=1).argmin())]
        center = graph.xy[graph.edges[chosen]].mean(0)
        keep_nodes = np.linalg.norm(graph.xy - center, axis=1) > cfg.radius + 1
        remap = np.cumsum(keep_nodes) - 1
        keep_edges = keep_nodes[graph.edges].all(1)
        if keep_edges.any():
            opened = replace(graph, xy=graph.xy[keep_nodes], edges=remap[graph.edges[keep_edges]])
            bank.append(Hypothesis("open", "topology", opened))
    if graph.repeat > 0.6 and 3 <= np.linalg.norm(graph.period) <= cfg.max_period:
        dx, dy = np.array(graph.period) * 0.5
        bank.append(Hypothesis("half_period", "phase", graph.moved(dx, dy)))
    return bank


def raster(graph, size, device, radius=2):
    """Render occupancy, endpoint, turn/junction, boundary, links and repeat phase."""
    h, w = size
    out = np.zeros((6, h, w), dtype=np.float32)

    def mark(channel, points):
        p = np.rint(points).astype(int)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)]
        out[channel, p[:, 1], p[:, 0]] = 1

    mark(0, graph.xy)
    for u, v in graph.edges:
        a, b = graph.xy[[u, v]]
        points = np.linspace(a, b, max(2, int(np.linalg.norm(a - b) * 2) + 1))
        mark(0, points)
        mark(4, points)
    mark(1, graph.xy[graph.endpoints])
    mark(2, graph.xy[graph.corners])
    t = torch.from_numpy(out)[None].to(device)
    t[:, :3] = F.max_pool2d(t[:, :3], 2 * radius + 1, stride=1, padding=radius)
    # Boundary of the ribbon and its repetition orientation/phase carrier.
    inner = -F.max_pool2d(-t[:, :1], 3, stride=1, padding=1)
    t[:, 3:4] = t[:, :1] - inner
    t[:, 5:6] = graph.repeat * t[:, :1]
    return t


def soft_targets(ink, period=(0.0, 0.0), max_period=32.0):
    """Targets from a supplied stroke mask (synthetic truth or human annotation)."""
    m = np.asarray(ink, dtype=np.float32).clip(0, 1)
    s = thin(m > 0.5)
    n = ndimage.convolve(s.astype(int), np.ones((3, 3), int), mode="constant") - s
    endpoint = ndimage.maximum_filter((s & (n == 1)).astype(float), size=5)
    # Junction/turn target includes curvature from a local skeleton covariance.
    corner = s & (n >= 3)
    for y, x in np.argwhere(s & (n == 2)):
        yy, xx = np.where(s[max(0, y-1):y+2, max(0, x-1):x+2])
        pts = np.stack((xx + max(0, x-1) - x, yy + max(0, y-1) - y), 1)
        pts = pts[(pts != 0).any(1)]
        corner[y, x] = len(pts) == 2 and np.dot(*pts) > -0.85 * np.prod(np.linalg.norm(pts, axis=1))
    boundary = ndimage.maximum_filter(m, size=3) - ndimage.minimum_filter(m, size=3)
    ax = np.zeros_like(m); ay = np.zeros_like(m)
    ax[:, :-1] = np.minimum(m[:, :-1], m[:, 1:])
    ay[:-1] = np.minimum(m[:-1], m[1:])
    repeated = float(np.linalg.norm(period) > 0)
    return np.stack((m, endpoint, ndimage.maximum_filter(corner.astype(float), size=5),
                     boundary, ax, ay, m * repeated,
                     m * period[0] / max_period, m * period[1] / max_period)).astype(np.float32)
