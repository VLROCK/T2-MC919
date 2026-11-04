import unittest

import numpy as np

from src.scale_marker import Calibration, Observation, estimate_scale


class ScaleMarkerTest(unittest.TestCase):
    def test_known_square_recovers_metric_scale(self):
        corners = np.array([[-1,-1,4],[1,-1,4],[1,1,4],[-1,1,4]], dtype=float)
        cal = Calibration(np.array([[500,0,320],[0,500,240],[0,0,1]],dtype=float),np.zeros(5))
        observations = []
        for i,x in enumerate((-1,-.5,0,.5,1)):
            center = np.array([x,0,0],dtype=float)
            normalized = (corners-center)[:,:2]/(corners-center)[:,2,None]
            pixels = normalized*np.array([500,500])+np.array([320,240])
            observations.append(Observation(f'frame_{i:04}.jpg',pixels,normalized,
                                            np.eye(3),-center,cal))
        result = estimate_scale(observations,marker_size_m=.4)
        self.assertAlmostEqual(result['scale_m_per_colmap_unit'],.2,places=7)
        self.assertEqual(len(result['images_used']),5)
        self.assertLess(result['reprojection_rmse_px_approx'],1e-8)


if __name__ == '__main__':
    unittest.main()
