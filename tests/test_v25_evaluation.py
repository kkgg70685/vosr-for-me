import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from observable_transport_v25.evaluate import evaluate_rows, parent_summary
from observable_transport_v25.matching import estimate_homography, evaluate_pairs
from observable_transport_v25.metrics import homography_corner_error
from observable_transport_v25.model import Config, AreaCamera


class EvaluationTests(unittest.TestCase):
    def test_all_controls_missing_reference_denominator(self):
        x = torch.full((1,3,64,64), .5); y = AreaCamera()(x)
        with patch('observable_transport_v25.evaluate.read_image', side_effect=[y,x]):
            rows = evaluate_rows([dict(id='x', parent='p', lr='lr', native='native')], Path('.'), AreaCamera(), Config())
        report = parent_summary(rows)
        self.assertEqual(len(report), 4)
        for group in report.values():
            self.assertEqual(group['metrics']['RGB_PSNR']['parents_available'], 0)
            self.assertEqual(group['metrics']['RGB_PSNR']['parents_total'], 1)

    def test_matching_identity(self):
        a = np.random.default_rng(6).integers(0,256,(128,128),dtype=np.uint8)
        h, count = estimate_homography(a, a)
        self.assertGreaterEqual(count, 4)
        self.assertLess(homography_corner_error(h, np.eye(3), 128, 128), .01)

    def test_failed_pairs_retained(self):
        rows = [dict(id='pair', parent='p', H_0_to_1=np.eye(3).tolist(),
                     methods={'native':['a','b'], 'observable_transport':['c','d']})]
        with patch('cv2.imread', return_value=np.zeros((64,64),np.uint8)):
            report = evaluate_pairs(rows, Path('.'))
        for group in report['summary'].values():
            self.assertEqual(group['failed_pairs'], 1)
            self.assertEqual(group['pair_auc']['3.0'], 0.)


if __name__ == '__main__': unittest.main()
