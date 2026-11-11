"""CPU invariants and adverse cases; no frozen assets or claimed SR benchmark."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from structure_observation_v2.config import Config
from structure_observation_v2.data import read_manifest, synthetic
from structure_observation_v2.model import (ObjectProposal, SpatialVelocityAdapter,
    coarse_feedback, compose, condition, decode_native, outer_alpha, restore, shared_prefix, suffix)
from structure_observation_v2.observation import (SharedCamera, choose_hypothesis,
    compare, relation_movement, shared_render, translate)
from structure_observation_v2.runtime import load_checkpoint, save_checkpoint
from structure_observation_v2.structure import Graph, Hypothesis, StructureHead, hypotheses, raster, soft_targets, thin
from structure_observation_v2.train import structure_loss, velocity_loss


class FakeBase:
    """Small differentiable API double; deliberately has no HR encode method."""
    def __init__(self):
        self.calls = 0

    def conditions(self, lr):
        return F.avg_pool2d(lr, 2), torch.ones(lr.shape[0], 4, 8), None

    def velocity(self, cond, z, t, sem):
        self.calls += 1
        return .2 * z - .1 * cond + .001 * sem.mean()

    def decoder_prefix(self, z):
        return z

    def decoder_suffix(self, z):
        return F.interpolate(z, scale_factor=8, mode="bilinear", align_corners=False).sigmoid()


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(32)
        self.cfg = Config()

    def test_shared_prefix_zero_adapter_exact(self):
        lr = torch.rand(1, 3, 16, 16)
        base = FakeBase(); state = shared_prefix(base, lr, 6, self.cfg)
        initial = state[2].clone()
        native = suffix(base, state, lr, self.cfg)
        adapter = SpatialVelocityAdapter(3, width=8)
        spatial = torch.rand(1, 12, 64, 64)
        revised = suffix(base, state, lr, self.cfg, adapter, spatial)
        self.assertTrue(torch.equal(native, revised))
        self.assertTrue(torch.equal(initial, state[2]))
        self.assertEqual(base.calls, 2 * (16 + 9 + 9))

    def test_same_seed_and_native_convention(self):
        base = FakeBase(); lr = torch.rand(1, 3, 16, 16)
        a = suffix(base, shared_prefix(base, lr, 7, self.cfg), lr, self.cfg)
        b = suffix(base, shared_prefix(base, lr, 7, replace(self.cfg, shared_steps=0)),
                   lr, replace(self.cfg, shared_steps=0))
        self.assertTrue(torch.equal(a, b))

    def test_velocity_gradients_and_local_support(self):
        base = FakeBase(); lr = torch.rand(1, 3, 16, 16)
        adapter = SpatialVelocityAdapter(3, width=8)
        spatial = torch.zeros(1, 12, 64, 64); spatial[..., 16:48, 16:48] = 1
        loss = velocity_loss(base, adapter, lr, torch.ones(1, 3, 64, 64), spatial, self.cfg, 9, 20)
        loss.backward()
        self.assertGreater(float(adapter.body[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            adapter.body[-1].bias.fill_(1)
        z = torch.ones(1, 3, 8, 8)
        delta = adapter(z, z, lr, spatial, torch.tensor(.2))
        self.assertTrue(torch.equal(delta[..., :2, :], torch.zeros_like(delta[..., :2, :])))

    def test_structure_head_supervision(self):
        lr, hr, target = synthetic("unit-test/3", self.cfg, size=16)
        head = StructureHead(8)
        pred = head(lr[None]); loss = structure_loss(pred, target[None]); loss.backward()
        self.assertEqual(pred.shape, (1, 9, 64, 64))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.body[0].weight.grad.abs().sum()), 0)

    def test_identical_hypotheses_never_invent_evidence(self):
        a = torch.rand(6, 1, 3, 8, 8); lr = a[0]
        ev = compare(a, a, lr, torch.ones(1, 1, 8, 8), self.cfg)
        self.assertEqual(ev.distinguish, 0)
        self.assertFalse(ev.reliable)

    def test_specific_local_difference_detectable(self):
        a = torch.full((6, 1, 3, 8, 8), .5); b = a.clone()
        b[..., 3:5, 3:5] += .1
        ev = compare(a, b, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        self.assertTrue(ev.reliable)
        self.assertGreater(ev.preference, self.cfg.preference_z)
        reverse = compare(b, a, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        self.assertLess(reverse.preference, 0)

    def test_brightness_is_not_structural_evidence(self):
        a = torch.full((6, 1, 3, 8, 8), .5); b = a + .05
        ev = compare(a, b, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        self.assertLess(ev.distinguish, .01)

    def test_actual_brightness_cannot_impersonate_graph_movement(self):
        a = torch.full((6, 1, 3, 8, 8), .5); b = a.clone()
        b[..., 3:5, 3:5] += .1
        movement = relation_movement(a, b, a, a+.1, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        self.assertLess(abs(movement), .01)

    def test_nuisance_reduces_confidence_without_erasing_large_moves(self):
        pattern = torch.zeros(1, 3, 8, 8); pattern[..., 2:5, 3:5] = 1
        a = torch.full((6, 1, 3, 8, 8), .5)
        b = a + .03 * pattern
        clean = compare(a, b, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        jitter = torch.tensor([0, .02, -.02, .01, -.01, 0])[:, None, None, None, None] * pattern
        uncertain = compare(a+jitter, b+jitter, b[0], torch.ones(1, 1, 8, 8), self.cfg)
        large = compare(a+jitter, a+.3*pattern+jitter, (a+.3*pattern)[0], torch.ones(1, 1, 8, 8), self.cfg)
        self.assertLess(uncertain.distinguish, clean.distinguish)
        self.assertGreater(large.distinguish, uncertain.distinguish)

    def graph(self):
        xy = np.array([[x, 16] for x in range(8, 24)], dtype=np.float32)
        return Graph(xy, np.array([[i, i+1] for i in range(15)], dtype=np.int64))

    def test_graph_open_connect_and_no_wrap(self):
        graph = self.graph(); bank = hypotheses(graph, self.cfg)
        self.assertIn("open", [h.name for h in bank])
        opened = next(h.graph for h in bank if h.name == "open")
        connected = hypotheses(opened, self.cfg)
        self.assertIn("connect", [h.name for h in connected])
        r = raster(graph, (32, 32), "cpu")
        self.assertEqual(r.shape, (1, 6, 32, 32))
        left = torch.zeros(1, 1, 8, 8); left[..., 2, 0] = 1
        moved = translate(left, -2, 0)
        self.assertEqual(float(moved[..., -1].sum()), 0)

    def test_common_renderer_and_ambiguous_seed_choice(self):
        graph = self.graph(); bank = hypotheses(graph, self.cfg)[:2]
        rasters = [raster(h.graph, (32, 32), "cpu") for h in bank]
        lr = torch.full((1, 3, 8, 8), .5); camera = SharedCamera()
        images, support, mask = shared_render(lr, rasters, camera)
        self.assertEqual(images.shape, (2, 3, 32, 32))
        projection = camera.project(images)
        a = choose_hypothesis(bank, projection, lr, mask, self.cfg, torch.Generator().manual_seed(7))
        b = choose_hypothesis(bank, projection, lr, mask, self.cfg, torch.Generator().manual_seed(7))
        self.assertEqual(a[0], b[0]); self.assertEqual(a[1], [0, 1])

    def test_atomic_interior_and_native_fallback(self):
        graph = self.graph(); bank = hypotheses(graph, self.cfg)[:1]
        r = raster(graph, (32, 32), "cpu")
        support = r[:, :1]; camera = SharedCamera()
        native = torch.rand(1, 3, 32, 32); revised = 1 - native
        obj = ObjectProposal(bank, [r], support, F.avg_pool2d(support, 4), 0, [0],
                             dict(position=0., topology=0., phase=0.), [], camera.project(native))
        lr = camera.nominal(native)
        out, decisions = compose(native, revised, [obj], lr, camera, self.cfg)
        self.assertTrue(torch.equal(out, native)); self.assertFalse(decisions[0]["accepted"])
        forced, _ = compose(native, revised, [obj], lr, camera, self.cfg, bypass_acceptance=True)
        inside = support.expand_as(native).bool()
        self.assertTrue(torch.equal(forced[inside], revised[inside]))
        alpha = outer_alpha(support, 2)
        self.assertTrue(torch.equal(alpha[support.bool()], torch.ones_like(alpha[support.bool()])))

    def test_supported_relation_is_actually_accepted(self):
        camera = SharedCamera(sigma=0)
        camera.nuisance = ((0., 0., 0.),) * 6
        native = torch.full((1, 3, 32, 32), .5); revised = native.clone()
        revised[..., 12:20, 12:20] = .7
        support = torch.zeros(1, 1, 32, 32); support[..., 8:24, 8:24] = 1
        graph = self.graph()
        bank = [Hypothesis("base", "reference", graph), Hypothesis("move", "position", graph)]
        projection = camera.project(torch.cat((native, revised)))
        obj = ObjectProposal(bank, [], support, F.avg_pool2d(support, 4), 1, [1],
                             dict(position=1., topology=0., phase=0.), [], projection)
        out, decisions = compose(native, revised, [obj], camera.nominal(revised), camera, self.cfg)
        self.assertTrue(decisions[0]["accepted"])
        self.assertTrue(torch.equal(out, revised))

    def test_missing_endpoints_are_not_counted_as_disconnected(self):
        from structure_observation_v2.evaluate import read_connections, parent_bootstrap
        relations = [{"a": [2, 2], "b": [8, 2], "connected": False}]
        self.assertEqual(read_connections(np.zeros((12, 12), bool), relations), [-1])
        records = [dict(dataset="x", parent="one", scores={"full": {"psnr": i+1}, "native": {"psnr": i}})
                   for i in range(8)]
        self.assertEqual(parent_bootstrap(records)["parents"], 1)
        self.assertIsNone(parent_bootstrap(records)["ci95"])

    def test_feedback_only_coarse_and_monotone(self):
        rgb = torch.full((1, 3, 32, 32), .4); lr = torch.full((1, 3, 8, 8), .5)
        camera = SharedCamera()
        out, step = coarse_feedback(rgb, lr, camera, self.cfg)
        self.assertGreater(step, 0)
        self.assertLessEqual(float((out-rgb).abs().max()), self.cfg.feedback_bound + 1e-6)
        self.assertTrue(bool(((camera.project(out)-lr).square().mean((1,2,3,4)) <
                              (camera.project(rgb)-lr).square().mean((1,2,3,4))).all()))

    def test_manifest_parent_leakage_rejected(self):
        row = dict(id="a", parent="p", dataset="d", camera="c", lr="l.png", hr="h.png", split="train")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"manifest.json"
            path.write_text(json.dumps([row, dict(row, id="b", split="test")]))
            with self.assertRaisesRegex(ValueError, "Parent leakage"):
                read_manifest(path)

    def test_checkpoint_strict_and_missing_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.pt"; cfg = replace(self.cfg, width=8)
            head = StructureHead(8); save_checkpoint(path, head, cfg)
            with self.assertRaisesRegex(ValueError, "no trained velocity"):
                load_checkpoint(path)
            save_checkpoint(path, head, cfg, adapter=SpatialVelocityAdapter(3, 8))
            _, adapter, restored, _ = load_checkpoint(path)
            self.assertEqual(restored, cfg); self.assertEqual(adapter.latent_channels, 3)

    def test_full_size_pipeline_with_a_graph_and_zero_adapter(self):
        class FixedHead(torch.nn.Module):
            def forward(self, lr):
                maps = lr.new_zeros(1, 9, 512, 512)
                maps[:, 0, 250:254, 100:180] = 1
                maps[:, 4:6, 250:254, 100:180] = 1
                return maps
        lr = torch.full((1, 3, 128, 128), .5)
        base = FakeBase(); head = FixedHead(); adapter = SpatialVelocityAdapter(3, width=8)
        out, stages = restore(base, head, adapter, SharedCamera(), lr, 6, self.cfg, control="no_feedback")
        self.assertEqual(out.shape, (1, 3, 512, 512))
        self.assertGreater(len(stages["decisions"]), 0)
        self.assertTrue(torch.equal(out, stages["native"]))
        self.assertTrue(torch.equal(stages["native"], stages["revised"]))

    def test_empty_structure_has_native_identity_path(self):
        class EmptyHead(torch.nn.Module):
            def forward(self, lr):
                return lr.new_zeros(1, 9, 512, 512)
        lr = torch.full((1, 3, 128, 128), .5); base = FakeBase()
        out, stages = restore(base, EmptyHead(), SpatialVelocityAdapter(3, 8), SharedCamera(),
                              lr, 6, self.cfg, control="no_feedback")
        self.assertEqual(stages["decisions"], [])
        self.assertTrue(torch.equal(out, stages["native"]))
        self.assertEqual(base.calls, 50)


if __name__ == "__main__":
    unittest.main()
