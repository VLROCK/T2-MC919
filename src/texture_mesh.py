"""Texturiza a malha densa existente com COLMAP 4.0+ sem repetir o MVS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import colmap_command_available, execute


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def texture(work: Path, colmap: str) -> dict:
    dense = work / "dense"
    mesh = dense / "meshed-delaunay.ply"
    if not present(mesh):
        raise SystemExit(f"Malha ausente ou vazia: {mesh}")
    if not (dense / "images").is_dir() or not (dense / "sparse").is_dir():
        raise SystemExit(f"Workspace retificado incompleto: {dense}/images e sparse são necessários")

    manifest_path = work / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Manifesto ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = dense / "textured"
    mesh_output = output / "mesh.ply"
    atlas_output = output / "texture.png"
    if not (present(mesh_output) and present(atlas_output)):
        if not colmap_command_available(colmap, "mesh_texturer"):
            raise SystemExit(f"{colmap} não oferece mesh_texturer; use COLMAP 4.0+.")
        print(f"Texturizando {mesh} com {colmap}; acompanhe {work / 'commands.log'}", flush=True)
        timing = manifest.setdefault("timing_seconds", {})
        timing["mesh_texturer_s"] = execute([
            colmap, "mesh_texturer", "--workspace_path", str(dense),
            "--input_path", str(mesh), "--output_path", str(output),
        ], work / "commands.log")
        if not (present(mesh_output) and present(atlas_output)):
            raise RuntimeError(f"Texturização não gerou {mesh_output} e {atlas_output}")

    manifest["mesh_ply"] = str(mesh)
    manifest["textured_dir"] = str(output)
    manifest["texture_status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"mesh": str(mesh_output), "atlas": str(atlas_output),
            "texture_status": manifest["texture_status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work/gpu_compat"))
    parser.add_argument("--colmap", required=True, help="Executável COLMAP 4.0+ com mesh_texturer.")
    args = parser.parse_args()
    print(json.dumps(texture(args.work, args.colmap), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
