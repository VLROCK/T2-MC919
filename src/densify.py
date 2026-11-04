"""Produz ou registra nuvem densa e malha; textura apenas se o COLMAP suportar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import colmap_command_available, execute


def existing_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.work / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Execução esparsa ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dense = args.work / "dense"
    if args.existing_only and not dense.is_dir():
        raise SystemExit(f"Reconstrução densa ausente: {dense}")
    dense.mkdir(exist_ok=True)
    log = args.work / "commands.log"
    timing = manifest.setdefault("timing_seconds", {})

    undistorted = dense / "images"
    if not undistorted.is_dir() or not any(undistorted.iterdir()):
        if args.existing_only:
            raise SystemExit(f"Imagens retificadas ausentes: {undistorted}")
        timing["image_undistorter_s"] = execute([
            args.colmap, "image_undistorter", "--image_path", manifest["images_used"],
            "--input_path", manifest["selected_model"], "--output_path", str(dense),
            "--output_type", "COLMAP", "--max_image_size", str(args.max_image_size),
        ], log)

    depth_maps = dense / "stereo" / "depth_maps"
    expected_maps = int(manifest["registered_images"])
    actual_maps = sum(existing_file(path) for path in depth_maps.glob("*.geometric.bin"))
    if actual_maps < expected_maps:
        if args.existing_only:
            raise SystemExit(f"Mapas de profundidade incompletos: {actual_maps}/{expected_maps}")
        timing["patch_match_stereo_s"] = execute([
            args.colmap, "patch_match_stereo", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true",
        ], log)

    fused = dense / "fused.ply"
    if not existing_file(fused):
        if args.existing_only:
            raise SystemExit(f"Nuvem densa ausente: {fused}")
        timing["stereo_fusion_s"] = execute([
            args.colmap, "stereo_fusion", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP", "--input_type", "geometric",
            "--output_path", str(fused),
        ], log)
    manifest["dense_ply"] = str(fused)

    if args.mesh:
        mesh = dense / "meshed-delaunay.ply"
        if not existing_file(mesh):
            if args.existing_only:
                raise SystemExit(f"Malha ausente: {mesh}")
            timing["delaunay_mesher_s"] = execute([
                args.colmap, "delaunay_mesher", "--input_path", str(dense),
                "--output_path", str(mesh),
            ], log)
        manifest["mesh_ply"] = str(mesh)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        textured = dense / "textured"
        texture_complete = existing_file(textured / "mesh.ply") and existing_file(textured / "texture.png")
        if not texture_complete:
            manifest.pop("textured_dir", None)
        if texture_complete:
            manifest["textured_dir"] = str(textured)
            manifest["texture_status"] = "complete"
        elif colmap_command_available(args.colmap, "mesh_texturer"):
            if args.existing_only:
                manifest["texture_status"] = "available_but_not_run_existing_only"
            else:
                timing["mesh_texturer_s"] = execute([
                    args.colmap, "mesh_texturer", "--workspace_path", str(dense),
                    "--input_path", str(mesh), "--output_path", str(textured),
                ], log)
                if not (existing_file(textured / "mesh.ply") and existing_file(textured / "texture.png")):
                    raise RuntimeError(f"Texturização não gerou mesh.ply e texture.png em {textured}")
                manifest["textured_dir"] = str(textured)
                manifest["texture_status"] = "complete"
        else:
            manifest["texture_status"] = "unavailable_in_colmap"
            print(f"Aviso: {args.colmap} não oferece mesh_texturer; nuvem e malha foram preservadas.")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work/main"))
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--existing-only", action="store_true", help="Finaliza artefatos existentes sem executar MVS novamente.")
    manifest = run(parser.parse_args())
    print(json.dumps({key: manifest.get(key) for key in
                      ("dense_ply", "mesh_ply", "textured_dir", "texture_status")}, indent=2))


if __name__ == "__main__":
    main()
