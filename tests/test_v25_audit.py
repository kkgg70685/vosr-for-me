import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from observable_transport_v25.model import profiled_residual, Config, ObservableTransport, AreaCamera
from observable_transport_v25.metrics import homography_corner_error
from observable_transport_v25.artifacts import json_ready, panels, highpass, texture_metrics
from observable_transport_v25.benchmark import validate_manifest
from observable_transport_v25.prepare import view_transforms


class AuditTests(unittest.TestCase):
    def test_inverted_texture_cannot_be_fitted_away(self):
        x=torch.linspace(.1,.9,64).view(1,1,8,8).expand(1,3,8,8)
        self.assertGreater(float(profiled_residual(x,1-x).square().mean()),.05)

    def test_gain_bounds_and_flat_model(self):
        with self.assertRaises(ValueError):Config(gain_min=-1)
        x=torch.full((1,3,32,32),.5)
        self.assertLess(float(profiled_residual(x,x+.1).abs().max()),1e-6)
        self.assertTrue(torch.isfinite(profiled_residual(x,torch.rand_like(x))).all())

    def test_json_does_not_rewrite_ids(self):
        record=json_ready({'id':'Infinity case','a':float('inf'),'b':float('-inf'),'c':float('nan')})
        result=json.loads(json.dumps(record,allow_nan=False))
        self.assertEqual(result['id'],'Infinity case');self.assertIsNone(result['c'])

    def test_bad_estimate_is_counted_as_failure(self):
        self.assertEqual(homography_corner_error(np.zeros((3,3)),np.eye(3),32,32),float('inf'))
        self.assertEqual(homography_corner_error(np.full((3,3),np.nan),np.eye(3),32,32),float('inf'))

    def test_common_scale_panel_and_roi(self):
        x=torch.rand(1,3,32,32);lr=AreaCamera()(x)
        with patch('PIL.Image.Image.save') as save, patch('pathlib.Path.mkdir'):
            metadata=panels(lr,{'native':x,'observable_transport':x},x,Path('not_written'))
        self.assertEqual(save.call_count,2);self.assertEqual(metadata['HF_display_gain'],4)
        self.assertEqual(texture_metrics(x,x)['HF_RMSE'],0)
        self.assertLess(float(highpass(torch.ones_like(x)).abs().max()),1e-6)

    def test_manifest_leakage_rejected(self):
        m={'phase':'test','excluded_dev_parents':['p'],
           'rows':[dict(id='a',parent='p',lr='a.png')]}
        with patch('pathlib.Path.is_file',return_value=True):
            with self.assertRaises(ValueError):validate_manifest(m,Path('.'))

    def test_rejected_motion_retains_image(self):
        x=torch.rand(1,3,64,64)
        result=ObservableTransport(AreaCamera(),Config(max_motion=1e-12,observation_floor=1e-5)).restore(AreaCamera()(x.roll(1,-1)),x)
        self.assertEqual(result['diagnostics']['accepted_steps'],0)
        self.assertTrue(torch.equal(result['final'],x))

    def test_known_views_have_valid_corners_and_nonidentity_relation(self):
        for shape in ((512,512),(700,1200)):
            m=view_transforms(*shape)
            h=m[1]@np.linalg.inv(m[0])
            self.assertGreater(homography_corner_error(h,np.eye(3),512,512),10)
            point=np.array([200.,230.,1.]);a=m[0]@point;b=m[1]@point
            np.testing.assert_allclose(h@a,b,atol=1e-10)


if __name__=='__main__':unittest.main()
