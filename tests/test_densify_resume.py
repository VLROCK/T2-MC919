"""A finished MVS run must be recoverable when texturing is unsupported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from src.densify import run


def test_finalize_existing_mesh_without_texturer(tmp_path: Path) -> None:
    work = tmp_path / "work"
    images = work / "dense" / "images"
    maps = work / "dense" / "stereo" / "depth_maps"
    images.mkdir(parents=True)
    maps.mkdir(parents=True)
    (images / "image.jpg").write_bytes(b"image")
    (maps / "image.jpg.geometric.bin").write_bytes(b"depth")
    (work / "dense" / "fused.ply").write_bytes(b"cloud")
    (work / "dense" / "meshed-delaunay.ply").write_bytes(b"mesh")
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps({"registered_images": 1}), encoding="utf-8")
    args = argparse.Namespace(work=work, colmap="colmap", mesh=True,
                              max_image_size=2000, existing_only=True)

    with patch("src.densify.execute", side_effect=AssertionError("MVS must not rerun")), \
         patch("src.densify.colmap_command_available", return_value=False):
        manifest = run(args)

    assert manifest["texture_status"] == "unavailable_in_colmap"
    assert manifest["dense_ply"] == str(work / "dense" / "fused.ply")
    assert manifest["mesh_ply"] == str(work / "dense" / "meshed-delaunay.ply")
    assert "textured_dir" not in manifest
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["mesh_ply"] == manifest["mesh_ply"]
