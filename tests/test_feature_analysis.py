"""A geometry smoke test with two rendered views of a fixed 3D point scene."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.feature_analysis import build_parser, read_image, run, select_pairs


def render_pair() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(919)
    width, height, focal = 960, 640, 850.0
    matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.045, -0.01]))
    translation = np.array([0.28, 0.0, 0.02])
    scene = np.column_stack(
        (rng.uniform(-1.8, 1.8, 320), rng.uniform(-1.1, 1.1, 320), rng.uniform(3.0, 6.0, 320))
    )
    images = [np.full((height, width, 3), 30, dtype=np.uint8) for _ in range(2)]
    for point in scene:
        patch = rng.integers(25, 245, (19, 19, 3), dtype=np.uint8)
        patch[9, :] = 230
        patch[:, 9] = 50
        for index, (rot, trans) in enumerate(((np.eye(3), np.zeros(3)), (rotation, translation))):
            camera_point = rot @ point + trans
            x, y = np.rint((matrix @ camera_point)[:2] / camera_point[2]).astype(int)
            if 12 <= x < width - 12 and 12 <= y < height - 12:
                images[index][y - 9 : y + 10, x - 9 : x + 10] = patch
    return images[0], images[1], {
        "fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2, "width": width, "height": height
    }


class FeatureAnalysisTest(unittest.TestCase):
    def test_read_image_preserves_colmap_pixel_orientation(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated_by_exif.jpg"
            exif = Image.Exif()
            exif[274] = 6  # Viewers rotate this 90 degrees; COLMAP keeps the stored grid.
            Image.new("RGB", (40, 20), "red").save(path, exif=exif)
            self.assertEqual(read_image(path).shape[:2], (20, 40))

    def test_pair_selection_keeps_closure(self) -> None:
        images = [Path(f"image{i:04}.jpg") for i in range(10)]
        pairs = select_pairs(images, 3)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[-1], (images[0], images[-1], "closure"))

    def test_synthetic_two_view_geometry_and_files(self) -> None:
        image_a, image_b, intrinsics = render_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            images_dir.mkdir()
            cv2.imwrite(str(images_dir / "image0000.png"), image_a)
            cv2.imwrite(str(images_dir / "image0001.png"), image_b)
            calibration = root / "intrinsics.json"
            calibration.write_text(json.dumps(intrinsics), encoding="utf-8")
            output = root / "analysis"
            args = build_parser().parse_args(
                ["--images", str(images_dir), "--output", str(output), "--intrinsics", str(calibration), "--methods", "SIFT", "--visualize-pairs", "1"]
            )
            summary = run(args)
            result = summary["pairs"][0]["results"]["SIFT"]
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["f_inliers"], 40)
            self.assertGreater(result["triangulated_positive_depth"], 20)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "comparison.csv").is_file())
            self.assertTrue((output / "two_view_sift.ply").is_file())
            self.assertTrue(any((output / "visualizations").glob("*_epipolar.png")))


if __name__ == "__main__":
    unittest.main()
