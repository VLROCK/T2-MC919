"""Executa COLMAP de forma rastreável para a variante A.

Exemplo: python -m src.pipeline --images data/images --work work/main --matching sequential
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .model import load_model


def execute(command: list[str], log_file: Path) -> float:
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as out:
        out.write("\n$ " + " ".join(command) + "\n")
        out.flush()
        process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Comando falhou ({process.returncode}): {' '.join(command)}. Veja {log_file}")
    return elapsed


def cpu_option(colmap: str, command: str, choices: tuple[str, str]) -> list[str]:
    help_result = subprocess.run([colmap, command, "-h"], capture_output=True,
                                 text=True, check=False)
    help_text = help_result.stdout + help_result.stderr
    for option in choices:
        if option in help_text:
            return [option, "0"]
    raise RuntimeError(f"Não achei opção de CPU para {command}; confira `colmap {command} -h`.")


def camera_intrinsics_json(camera) -> dict | None:
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = []
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]; dist = []
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [p[3],0,0,0,0]
    elif camera.model == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [p[3],p[4],0,0,0]
    elif camera.model in {"OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = p[:4]; dist = list(p[4:])
    else:
        return None
    result = {"fx":fx,"fy":fy,"cx":cx,"cy":cy,"width":camera.width,
              "height":camera.height}
    if dist:
        result["dist_coeffs"] = dist
    return result


def images_in(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=Path("data/images"))
    p.add_argument("--work", type=Path, default=Path("work/main"))
    p.add_argument("--matching", choices=["sequential", "exhaustive"], default="sequential")
    p.add_argument("--loop-detection", action="store_true",
                   help="Matcher sequencial detecta laços; requer vocabulary tree e possivelmente download.")
    p.add_argument("--dense", action="store_true", help="Executa MVS (exige mais tempo e memória).")
    p.add_argument("--mesh", action="store_true", help="Gera malha e textura a partir do MVS.")
    p.add_argument("--cpu", action="store_true", help="Desabilita extração e matching na GPU.")
    p.add_argument("--colmap", default="colmap", help="Executável COLMAP ou caminho para COLMAP.bat.")
    p.add_argument("--max-images", type=int, default=None,
                   help="Amostra uniforme para estudo de ablação; use work separado.")
    p.add_argument("--feature-image-size", type=int, default=2000,
                   help="Lado máximo usado na extração SIFT; reduz RAM (padrão: 2000 px).")
    p.add_argument("--max-features", type=int, default=4096,
                   help="Máximo de descritores SIFT por foto (padrão: 4096).")
    p.add_argument("--feature-threads", type=int, default=4,
                   help="Threads da extração, para limitar memória (padrão: 4).")
    p.add_argument("--max-matches", type=int, default=8192,
                   help="Limite de correspondências por par no matcher (padrão: 8192).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    source = args.images.resolve()
    photos = images_in(source) if source.exists() else []
    if len(photos) < 3:
        raise SystemExit(f"São necessárias pelo menos 3 fotos em {source}; encontradas {len(photos)}.")
    if args.max_images is not None and args.max_images < 3:
        raise SystemExit("--max-images precisa ser >= 3")
    if min(args.feature_image_size, args.max_features, args.feature_threads, args.max_matches) < 1:
        raise SystemExit("Limites de imagem, features, threads e matches devem ser positivos.")
    if args.loop_detection and args.matching != "sequential":
        raise SystemExit("--loop-detection exige --matching sequential")
    if args.mesh and not args.dense:
        raise SystemExit("--mesh exige --dense")
    work = args.work.resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"Pasta de trabalho já contém arquivos: {work}. Use outra pasta para evitar mistura de experimentos.")
    work.mkdir(parents=True, exist_ok=True)
    image_path = source
    if args.max_images and args.max_images < len(photos):
        import numpy as np
        selected = [photos[i] for i in np.linspace(0, len(photos)-1, args.max_images, dtype=int)]
        image_path = work / "images"
        image_path.mkdir()
        for j, photo in enumerate(selected):
            # Prefixo preserva a ordem temporal exigida pelo matcher sequencial.
            shutil.copy2(photo, image_path / f"frame_{j:04d}{photo.suffix.lower()}")
        photos = selected

    try:
        subprocess.run([args.colmap, "help"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"COLMAP não encontrado em '{args.colmap}'. Instale e confira `colmap help`: {exc}")

    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir()
    log = work / "commands.log"
    timing: dict[str, float] = {}

    cmd = [args.colmap, "feature_extractor", "--database_path", str(db),
           "--image_path", str(image_path),
           "--FeatureExtraction.max_image_size", str(args.feature_image_size),
           "--FeatureExtraction.num_threads", str(args.feature_threads),
           "--SiftExtraction.max_num_features", str(args.max_features)]
    if args.cpu:
        cmd += cpu_option(args.colmap, "feature_extractor",
                          ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"))
    timing["feature_extractor_s"] = execute(cmd, log)

    cmd = [args.colmap, f"{args.matching}_matcher", "--database_path", str(db)]
    cmd += ["--FeatureMatching.max_num_matches", str(args.max_matches)]
    if args.loop_detection:
        cmd += ["--SequentialMatching.loop_detection", "1"]
    if args.cpu:
        cmd += cpu_option(args.colmap, f"{args.matching}_matcher",
                          ("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"))
    timing[f"{args.matching}_matcher_s"] = execute(cmd, log)

    timing["mapper_s"] = execute([args.colmap, "mapper", "--database_path", str(db),
                                   "--image_path", str(image_path), "--output_path", str(sparse)], log)

    candidates = []
    for folder in sparse.iterdir():
        if not folder.is_dir():
            continue
        text_dir = work / "models_txt" / folder.name
        text_dir.mkdir(parents=True)
        execute([args.colmap, "model_converter", "--input_path", str(folder),
                 "--output_path", str(text_dir), "--output_type", "TXT"], log)
        model = load_model(text_dir)
        candidates.append((len(model.images), len(model.points), folder, text_dir))
    if not candidates:
        raise RuntimeError(f"COLMAP não gerou modelo esparso. Veja {log}")
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    registered, npoints, model_bin, model_txt = candidates[0]
    if model_txt != work / "sparse_txt":
        shutil.copytree(model_txt, work / "sparse_txt")
    chosen = load_model(model_txt)
    used_camera_ids = {im.camera_id for im in chosen.images}
    if len(used_camera_ids) == 1:
        camera = chosen.cameras[next(iter(used_camera_ids))]
        intrinsics = camera_intrinsics_json(camera)
        if intrinsics is not None:
            (work / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")

    manifest = {
        "images_input": str(source), "images_used": str(image_path), "num_input_images": len(photos),
        "matching": args.matching, "loop_detection": args.loop_detection, "cpu": args.cpu,
        "feature_image_size": args.feature_image_size, "max_features": args.max_features,
        "feature_threads": args.feature_threads, "max_matches": args.max_matches,
        "selected_model": str(model_bin),
        "model_text": str(work / "sparse_txt"), "registered_images": registered,
        "sparse_points": npoints, "all_models": [
            {"path": str(c[2]), "registered_images": c[0], "sparse_points": c[1]} for c in candidates],
        "timing_seconds": timing,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dense:
        dense = work / "dense"
        dense.mkdir()
        timing["image_undistorter_s"] = execute([
            args.colmap, "image_undistorter", "--image_path", str(image_path),
            "--input_path", str(model_bin), "--output_path", str(dense),
            "--output_type", "COLMAP", "--max_image_size", "2000"], log)
        timing["patch_match_stereo_s"] = execute([
            args.colmap, "patch_match_stereo", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"], log)
        timing["stereo_fusion_s"] = execute([
            args.colmap, "stereo_fusion", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP", "--input_type", "geometric",
            "--output_path", str(dense / "fused.ply")], log)
        manifest["dense_ply"] = str(dense / "fused.ply")
        if args.mesh:
            mesh_path = dense / "meshed-delaunay.ply"
            timing["delaunay_mesher_s"] = execute([
                args.colmap, "delaunay_mesher", "--input_path", str(dense),
                "--output_path", str(mesh_path)], log)
            timing["mesh_texturer_s"] = execute([
                args.colmap, "mesh_texturer", "--workspace_path", str(dense),
                "--input_path", str(mesh_path), "--output_path", str(dense / "textured")], log)
            manifest["mesh_ply"] = str(mesh_path)
            manifest["textured_dir"] = str(dense / "textured")
        manifest["timing_seconds"] = timing
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
