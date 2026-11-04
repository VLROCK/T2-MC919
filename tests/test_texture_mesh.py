"""Texturing reuses completed MVS outputs and records real artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.texture_mesh import texture


def prepare(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    dense = work / "dense"
    (dense / "images").mkdir(parents=True)
    (dense / "sparse").mkdir()
    (dense / "meshed-delaunay.ply").write_bytes(b"mesh")
    (work / "manifest.json").write_text("{}", encoding="utf-8")
    return work


def test_unsupported_colmap_fails_before_touching_results(tmp_path: Path) -> None:
    work = prepare(tmp_path)
    with patch("src.texture_mesh.colmap_command_available", return_value=False), \
         patch("src.texture_mesh.execute", side_effect=AssertionError("must not run")):
        with pytest.raises(SystemExit, match="COLMAP 4.0"):
            texture(work, "colmap3")
    assert json.loads((work / "manifest.json").read_text(encoding="utf-8")) == {}


def test_texture_only_records_mesh_and_atlas(tmp_path: Path) -> None:
    work = prepare(tmp_path)

    def fake_execute(command: list[str], log: Path) -> float:
        assert command[1] == "mesh_texturer"
        output = Path(command[-1])
        output.mkdir()
        (output / "mesh.ply").write_bytes(b"textured mesh")
        (output / "texture.png").write_bytes(b"atlas")
        return 7.5

    with patch("src.texture_mesh.colmap_command_available", return_value=True), \
         patch("src.texture_mesh.execute", side_effect=fake_execute):
        result = texture(work, "colmap4")

    assert result["texture_status"] == "complete"
    assert Path(result["mesh"]).is_file()
    assert Path(result["atlas"]).is_file()
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["timing_seconds"]["mesh_texturer_s"] == 7.5
    assert manifest["textured_dir"] == str(work / "dense" / "textured")
