from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.analysis import analyze, detect_planes, metric_scale, observation_reprojection_errors, wall_dimensions
from src.model import Camera, Image, Model, Point, load_model
from src.pipeline import camera_intrinsics_json
from src.visualize import make_video


class ModelAndAnalysisTest(unittest.TestCase):
    def test_intrinsics_export_for_pinhole_has_no_empty_distortion_vector(self):
        camera = Camera(1,'PINHOLE',640,480,(500,500,320,240))
        self.assertNotIn('dist_coeffs',camera_intrinsics_json(camera))

    def test_room_planes_on_controlled_geometry(self):
        rng = np.random.default_rng(1)
        pts = []
        for x in (0, 4):
            pts.extend([x,y,z] for y in np.linspace(0,3,15) for z in np.linspace(0,2.6,12))
        for y in (0, 3):
            pts.extend([x,y,z] for x in np.linspace(0,4,17) for z in np.linspace(0,2.6,12))
        for z in (0, 2.6):
            pts.extend([x,y,z] for x in np.linspace(0,4,12) for y in np.linspace(0,3,10))
        xyz = np.array(pts) + rng.normal(0,.005,size=(len(pts),3))
        dims = wall_dimensions(detect_planes(xyz,.055),np.array([[1,1,1.4],[2,2,1.4]]))
        self.assertAlmostEqual(dims['width_m'],3,delta=.06)
        self.assertAlmostEqual(dims['length_m'],4,delta=.06)
        self.assertAlmostEqual(dims['height_m'],2.6,delta=.06)

    def test_text_parser_scale_reprojection_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'cameras.txt').write_text('1 PINHOLE 640 480 500 500 320 240\n')
            # C1=(0,0,0), C2=(1,0,0); both see X=(0,0,4), X=(1,0,4).
            (root/'images.txt').write_text(
                '1 1 0 0 0 0 0 0 1 frame_0001.jpg\n'
                '320 240 10 445 240 11\n'
                '2 1 0 0 0 -1 0 0 1 frame_0002.jpg\n'
                '195 240 10 320 240 11\n', encoding='utf-8')
            (root/'points3D.txt').write_text(
                '10 0 0 4 255 0 0 0 1 0 2 0\n'
                '11 1 0 4 0 255 0 0 1 1 2 1\n', encoding='utf-8')
            model = load_model(root)
            self.assertEqual(len(model.images),2)
            self.assertTrue(np.allclose(model.images[1].center,[1,0,0]))
            errors, unsupported = observation_reprojection_errors(model)
            self.assertEqual(unsupported,0)
            self.assertTrue(np.allclose(errors,0))
            s,source = metric_scale(model, {'reference':{'point_ids':[10,11], 'distance_m':0.5}},None)
            self.assertAlmostEqual(s,0.5)
            out = root/'analysis'
            result = analyze(model,s,source,{},out,2)
            self.assertAlmostEqual(result['loop_delta'],0.5)
            self.assertAlmostEqual(result['reprojection_error_px_mean_per_observation'],0)
            self.assertTrue((out/'floorplan.png').is_file())
            self.assertTrue((out/'scene3d.png').is_file())
            self.assertTrue((out/'scaled_sparse.ply').is_file())
            self.assertTrue((out/'metrics.json').is_file())
            make_video(root,out/'metrics.json',out/'navigation.mp4',seconds=2,fps=2)
            self.assertGreater((out/'navigation.mp4').stat().st_size,1000)


if __name__ == '__main__':
    unittest.main()
