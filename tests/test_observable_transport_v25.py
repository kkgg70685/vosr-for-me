import unittest
import numpy as np
import torch
from observable_transport_v25.geometry import transport, jacobian_determinant, cosine_basis, field
from observable_transport_v25.model import Config, AreaCamera, ObservableTransport, profiled_residual
from observable_transport_v25.metrics import error_auc, homography_corner_error
from observable_transport_v25.runtime import generate_native


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def image(self):
        y, x = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
        return torch.stack([.5+.2*torch.sin(x*.3)+.1*torch.cos(y*.23),
                            .5+.2*torch.cos(x*.19)+.1*torch.sin(y*.31),
                            .5+.2*torch.sin((x+y)*.15)])[None]

    def test_identity_bit_exact(self):
        x = self.image()
        out, _ = transport(x, torch.zeros(1, 2, 64, 64))
        self.assertTrue(torch.equal(x, out))

    def test_subpixel_sign_and_gradient(self):
        x = torch.arange(64.).view(1, 1, 1, 64).expand(1, 3, 64, 64)
        for shift in (-.25, .25, 1.):
            v = torch.zeros(1, 2, 64, 64); v[:, 0] = shift
            v.requires_grad_()
            out, d = transport(x, v)
            torch.testing.assert_close(out[..., 5:-5, 5:-5], (x-shift)[..., 5:-5, 5:-5], atol=2e-5, rtol=0)
            self.assertGreater(float(jacobian_determinant(d).min().detach()), .999)
            out.sum().backward()
            self.assertTrue(torch.isfinite(v.grad).all())
            self.assertGreater(float(v.grad.abs().sum()), 0)

    def test_flat_ambiguity_returns_native(self):
        x = torch.full((1, 3, 64, 64), .5)
        out = ObservableTransport(AreaCamera()).restore(AreaCamera()(x), x)
        self.assertTrue(torch.equal(out["final"], x))
        self.assertEqual(out["diagnostics"]["accepted_steps"], 0)

    def test_known_translation_automatic_lr_recovery(self):
        x = self.image(); v = torch.zeros(1, 2, 64, 64); v[:, 0] = .75
        y = AreaCamera()(transport(x, v)[0])
        cfg = Config(modes=1, observation_floor=.0001, damping=1e-7, iterations=4)
        out = ObservableTransport(AreaCamera(), cfg).restore(y, x)
        self.assertLess(out["diagnostics"]["final_loss"], out["diagnostics"]["initial_loss"]*.05)
        self.assertAlmostEqual(float(out["velocity"][:, 0].mean()), .75, delta=.08)

    def test_color_nuisance_is_not_geometry(self):
        x = self.image(); y = AreaCamera()(x)*.85+.07
        r = profiled_residual(AreaCamera()(x), y)
        self.assertLess(float(r.square().mean()), 1e-12)
        out = ObservableTransport(AreaCamera()).restore(y, x)
        self.assertTrue(torch.equal(out["final"], x))

    def test_basis_whole_domain_and_identity_jacobian(self):
        x = self.image(); b = cosine_basis(x, 1)
        v = field(torch.tensor([1., -1.]), b)
        self.assertTrue(torch.equal(v[:, 0], torch.ones_like(v[:, 0])))
        self.assertTrue(torch.equal(jacobian_determinant(v), torch.ones(1, 63, 63)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Config(observation_floor=0)
        with self.assertRaises(ValueError): Config(modes=1.5)
        with self.assertRaises(ValueError): Config(max_motion=float('nan'))
        with self.assertRaises(ValueError): ObservableTransport(AreaCamera()).restore(torch.zeros(1,3,4,4), self.image())

    def test_auc_failure_denominator(self):
        self.assertEqual(error_auc([0., float('inf')])["3.0"], .5)
        self.assertEqual(error_auc([float('inf')])["3.0"], 0.)
        with self.assertRaises(ValueError): error_auc([])
        self.assertEqual(homography_corner_error(np.eye(3), np.eye(3), 64, 64), 0.)

    def test_native_api_and_seed(self):
        class Base:
            def conditions(self, x): return x, torch.zeros_like(x), None
            def velocity(self, cond, z, t, sem): return .1*z + cond
            def decoder_prefix(self, z): return z
            def decoder_suffix(self, z): return z
        x = torch.ones(1,3,4,4)
        a = generate_native(Base(), x)
        torch.testing.assert_close(a, generate_native(Base(), x), rtol=0, atol=0)
        self.assertFalse(torch.equal(a, generate_native(Base(), x, seed=7)))


if __name__ == "__main__": unittest.main()
