"""Diagnostics for feature matching and two-view geometry.

Run with ``python -m src.feature_analysis --images data/images --output results/features``.
The SfM reconstruction itself is performed by COLMAP; this module makes the
feature, epipolar-geometry, pose and initial-triangulation stages inspectable.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def read_image(path: Path) -> np.ndarray:
    """Read Unicode paths on Windows as well as ordinary paths."""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Não foi possível codificar a imagem: {path}")
    encoded.tofile(str(path))


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Diretório de imagens inexistente: {directory}")
    images = sorted(
        (p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p.relative_to(directory)).lower(),
    )
    if len(images) < 2:
        raise ValueError("São necessárias pelo menos duas imagens.")
    return images


def select_pairs(images: list[Path], max_adjacent_pairs: int) -> list[tuple[Path, Path, str]]:
    """Sample adjacent pairs across the route and always inspect the closing pair."""
    n = len(images)
    if max_adjacent_pairs < 0:
        raise ValueError("--max-pairs deve ser 0 (todos) ou um inteiro positivo.")
    if max_adjacent_pairs == 0 or max_adjacent_pairs >= n - 1:
        indexes = list(range(n - 1))
    else:
        indexes = sorted(set(np.linspace(0, n - 2, max_adjacent_pairs, dtype=int).tolist()))
    pairs = [(images[i], images[i + 1], "adjacent") for i in indexes]
    if n > 2:
        pairs.append((images[0], images[-1], "closure"))
    return pairs


def load_intrinsics(path: Path | None, width: int, height: int, focal_ratio: float) -> tuple[np.ndarray, np.ndarray | None, str]:
    if path is None:
        # Only a provisional camera model for the didactic two-view diagnostic.
        focal = focal_ratio * max(width, height)
        return np.array(
            [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ), None, "estimated_from_image_size_not_calibrated"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "camera_matrix" in data:
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    else:
        matrix = np.array(
            [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Matriz intrínseca inválida; use fx, fy, cx, cy ou camera_matrix 3x3.")
    if "width" in data and int(data["width"]) != width:
        raise ValueError(f"Largura da calibração ({data['width']}) difere das imagens ({width}).")
    if "height" in data and int(data["height"]) != height:
        raise ValueError(f"Altura da calibração ({data['height']}) difere das imagens ({height}).")
    distortion = None
    if "dist_coeffs" in data:
        distortion = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        if len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("dist_coeffs deve conter 4, 5, 8, 12 ou 14 coeficientes OpenCV finitos.")
    return matrix, distortion, "provided_calibration"


def detector_for(name: str) -> tuple[Any, int]:
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    if name == "ORB":
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING
    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"Descritor desconhecido: {name}")


def ratio_matches(desc1: np.ndarray | None, desc2: np.ndarray | None, norm: int, ratio: float) -> list[cv2.DMatch]:
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    matches = []
    for neighbors in matcher.knnMatch(desc1, desc2, k=2):
        if len(neighbors) == 2 and neighbors[0].distance < ratio * neighbors[1].distance:
            matches.append(neighbors[0])
    return sorted(matches, key=lambda item: item.distance)


def sampson_errors(matrix: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Square-root Sampson distances in pixels, useful as a pair diagnostic."""
    x1 = np.column_stack((points1, np.ones(len(points1))))
    x2 = np.column_stack((points2, np.ones(len(points2))))
    fx1 = (matrix @ x1.T).T
    ftx2 = (matrix.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def draw_epipolar(image1: np.ndarray, image2: np.ndarray, points1: np.ndarray, points2: np.ndarray, f_matrix: np.ndarray) -> np.ndarray:
    """Draw corresponding inlier points and epipolar lines on both images."""
    count = min(len(points1), 12)
    if count == 0:
        return np.hstack((image1, image2))
    indexes = np.linspace(0, len(points1) - 1, count, dtype=int)
    p1, p2 = points1[indexes], points2[indexes]
    lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, f_matrix).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, f_matrix).reshape(-1, 3)
    canvas1, canvas2 = image1.copy(), image2.copy()
    rng = np.random.default_rng(919)
    for line1, line2, point1, point2 in zip(lines1, lines2, p1, p2):
        color = tuple(int(x) for x in rng.integers(50, 255, 3))
        for canvas, line in ((canvas1, line1), (canvas2, line2)):
            a, b, c = line
            width = canvas.shape[1]
            height = canvas.shape[0]
            if abs(b) > abs(a) and abs(b) > 1e-9:
                endpoints = ((0, int(round(-c / b))), (width - 1, int(round(-(c + a * (width - 1)) / b))))
            elif abs(a) > 1e-9:
                endpoints = ((int(round(-c / a)), 0), (int(round(-(c + b * (height - 1)) / a)), height - 1))
            else:
                continue
            cv2.line(canvas, endpoints[0], endpoints[1], color, 1, cv2.LINE_AA)
        cv2.circle(canvas1, tuple(np.rint(point1).astype(int)), 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas2, tuple(np.rint(point2).astype(int)), 5, color, -1, cv2.LINE_AA)
    return np.hstack((canvas1, canvas2))


def save_relative_ply(path: Path, points: np.ndarray, image: np.ndarray, pixels: np.ndarray) -> None:
    """Save first-camera relative triangulation; its scale is arbitrary."""
    if len(points) == 0:
        return
    coords = np.rint(pixels).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, image.shape[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image.shape[0] - 1)
    colors = image[coords[:, 1], coords[:, 0], ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment two-view triangulation; arbitrary scale\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(points, colors):
            file.write(f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def analyze_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    name: str,
    matrix: np.ndarray,
    distortion: np.ndarray | None,
    ratio: float,
    f_threshold: float,
    e_threshold: float,
    visual_prefix: Path | None = None,
    ply_path: Path | None = None,
) -> dict[str, Any]:
    """Run one descriptor through matching, F/E RANSAC, pose and triangulation."""
    started = time.perf_counter()
    detector, norm = detector_for(name)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    matches = ratio_matches(descriptors1, descriptors2, norm, ratio)
    detection_matching_s = round(time.perf_counter() - started, 4)
    result: dict[str, Any] = {
        "descriptor": name,
        "keypoints_1": len(keypoints1),
        "keypoints_2": len(keypoints2),
        "ratio_matches": len(matches),
        "f_inliers": 0,
        "f_inlier_ratio": None,
        "sampson_median_px": None,
        "e_inliers": 0,
        "pose_inliers": 0,
        "triangulated_positive_depth": 0,
        "detection_matching_s": detection_matching_s,
        "status": "insufficient_matches",
    }
    if visual_prefix is not None:
        keypoint_image = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        write_png(visual_prefix.with_name(visual_prefix.name + "_keypoints.png"), keypoint_image)
        match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        write_png(visual_prefix.with_name(visual_prefix.name + "_matches.png"), match_image)
    if len(matches) < 8:
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    f_matrix, f_mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, f_threshold, 0.999)
    if f_matrix is None or f_mask is None or f_matrix.shape != (3, 3):
        result["status"] = "fundamental_failed"
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result
    f_mask = f_mask.ravel().astype(bool)
    result["f_inliers"] = int(f_mask.sum())
    result["f_inlier_ratio"] = round(float(f_mask.mean()), 4)
    if f_mask.any():
        result["sampson_median_px"] = round(float(np.median(sampson_errors(f_matrix, points1[f_mask], points2[f_mask]))), 4)
    if visual_prefix is not None:
        inlier_matches = [match for match, accepted in zip(matches, f_mask) if accepted]
        image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        write_png(visual_prefix.with_name(visual_prefix.name + "_inliers.png"), image)
        if result["f_inliers"] >= 8:
            epipolar = draw_epipolar(image1, image2, points1[f_mask], points2[f_mask], f_matrix)
            write_png(visual_prefix.with_name(visual_prefix.name + "_epipolar.png"), epipolar)

    # Restrict E to F-verified correspondences, reducing the effect of bad matches.
    points1_f, points2_f = points1[f_mask], points2[f_mask]
    if len(points1_f) < 8:
        result["status"] = "insufficient_f_inliers"
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result
    if distortion is not None:
        points1_e = cv2.undistortPoints(points1_f.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)
        points2_e = cv2.undistortPoints(points2_f.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)
    else:
        points1_e, points2_e = points1_f, points2_f
    essential, e_mask = cv2.findEssentialMat(points1_e, points2_e, matrix, cv2.RANSAC, 0.999, e_threshold)
    if essential is None or e_mask is None:
        result["status"] = "essential_failed"
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result
    essential = essential[:3, :3]
    e_mask = e_mask.ravel().astype(bool)
    result["e_inliers"] = int(e_mask.sum())
    if e_mask.sum() < 5:
        result["status"] = "insufficient_e_inliers"
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result
    p1_e, p2_e = points1_e[e_mask], points2_e[e_mask]
    p1_raw, p2_raw = points1_f[e_mask], points2_f[e_mask]
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, p1_e, p2_e, matrix)
    pose_mask = pose_mask.ravel().astype(bool)
    result["pose_inliers"] = int(pose_mask.sum())
    result["rotation_matrix"] = np.round(rotation, 6).tolist()
    result["translation_direction"] = np.round(translation.ravel(), 6).tolist()
    if pose_mask.sum() < 2:
        result["status"] = "insufficient_pose_inliers"
        result["elapsed_s"] = round(time.perf_counter() - started, 4)
        return result
    p1_pose, p2_pose = p1_raw[pose_mask], p2_raw[pose_mask]
    norm1 = cv2.undistortPoints(p1_pose.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    norm2 = cv2.undistortPoints(p2_pose.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    proj1 = np.column_stack((np.eye(3), np.zeros(3)))
    proj2 = np.column_stack((rotation, translation))
    homogeneous = cv2.triangulatePoints(proj1, proj2, norm1.T, norm2.T)
    good_w = np.isfinite(homogeneous).all(axis=0) & (np.abs(homogeneous[3]) > 1e-10)
    cloud = np.zeros((len(p1_pose), 3), dtype=np.float64)
    cloud[good_w] = (homogeneous[:3, good_w] / homogeneous[3, good_w]).T
    depth_2 = (rotation @ cloud.T + translation).T[:, 2]
    valid = good_w & np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0) & (depth_2 > 0)
    result["triangulated_positive_depth"] = int(valid.sum())
    result["status"] = "ok" if valid.any() else "triangulation_failed"
    if ply_path is not None and valid.any():
        save_relative_ply(ply_path, cloud[valid], image1, p1_pose[valid])
    result["elapsed_s"] = round(time.perf_counter() - started, 4)
    return result


def aggregate_metrics(records: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    fields = ("keypoints_1", "keypoints_2", "ratio_matches", "f_inliers", "f_inlier_ratio", "sampson_median_px", "e_inliers", "pose_inliers", "triangulated_positive_depth", "detection_matching_s")
    table: list[dict[str, Any]] = []
    for method in methods:
        rows = [pair["results"][method] for pair in records]
        row: dict[str, Any] = {"descriptor": method, "pairs": len(rows), "successful_pairs": sum(r["status"] == "ok" for r in rows)}
        for field in fields:
            values = [r[field] for r in rows if r.get(field) is not None]
            row[f"median_{field}"] = round(float(np.median(values)), 4) if values else None
        table.append(row)
    return table


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.images:
        directory = Path(args.images)
        if Path(args.output).resolve().is_relative_to(directory.resolve()):
            raise ValueError("O diretório de saída deve ficar fora do diretório das fotos para evitar reprocessar os PNGs gerados.")
        images = list_images(directory)
        pairs = select_pairs(images, args.max_pairs)
        label = lambda path: str(path.relative_to(directory)).replace("\\", "/")
    else:
        if not args.image_a or not args.image_b:
            raise ValueError("Forneça --images ou as duas opções --image-a e --image-b.")
        a, b = Path(args.image_a), Path(args.image_b)
        images = [a, b]
        pairs = [(a, b, "selected")]
        label = lambda path: path.name
    if not 0 < args.ratio < 1:
        raise ValueError("--ratio deve estar entre 0 e 1.")
    if args.f_threshold <= 0 or args.e_threshold <= 0 or args.focal_ratio <= 0:
        raise ValueError("Limiar RANSAC e razão focal devem ser positivos.")
    methods = list(dict.fromkeys(method.upper() for method in args.methods))
    for method in methods:
        detector_for(method)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    first = read_image(images[0])
    height, width = first.shape[:2]
    matrix, distortion, intrinsic_source = load_intrinsics(Path(args.intrinsics) if args.intrinsics else None, width, height, args.focal_ratio)
    records: list[dict[str, Any]] = []
    ply_written: set[str] = set()
    for index, (path1, path2, kind) in enumerate(pairs):
        image1 = first if path1 == images[0] else read_image(path1)
        image2 = read_image(path2)
        if image1.shape[:2] != (height, width) or image2.shape[:2] != (height, width):
            raise ValueError(f"Todas as imagens devem ter a mesma resolução ({width}x{height}): {path1}, {path2}")
        row: dict[str, Any] = {"image_1": label(path1), "image_2": label(path2), "kind": kind, "results": {}}
        visualize = index < args.visualize_pairs or kind == "closure"
        for method in methods:
            prefix = output / "visualizations" / f"pair_{index:03d}_{kind}_{method.lower()}" if visualize else None
            ply = output / f"two_view_{method.lower()}.ply" if method not in ply_written else None
            result = analyze_pair(image1, image2, method, matrix, distortion, args.ratio, args.f_threshold, args.e_threshold, prefix, ply)
            if ply is not None and ply.exists():
                ply_written.add(method)
            row["results"][method] = result
        records.append(row)
        print(f"[{index + 1}/{len(pairs)}] {row['image_1']} -> {row['image_2']} ({kind})")
    aggregate = aggregate_metrics(records, methods)
    summary = {
        "image_count": len(images),
        "pair_count": len(pairs),
        "methods": methods,
        "intrinsics_source": intrinsic_source,
        "camera_matrix": matrix.tolist(),
        "dist_coeffs": distortion.tolist() if distortion is not None else None,
        "parameters": {"ratio": args.ratio, "f_ransac_threshold_px": args.f_threshold, "e_ransac_threshold_px": args.e_threshold, "max_adjacent_pairs": args.max_pairs},
        "notes": ["Two-view translation and PLY are at arbitrary scale.", "Estimated intrinsics are exploratory; use calibrated/COLMAP intrinsics for the reported E and pose."] if intrinsic_source != "provided_calibration" else ["Two-view translation and PLY are at arbitrary scale."],
        "aggregate": aggregate,
        "pairs": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara descritores e inspeciona geometria epipolar em imagens da sala.")
    parser.add_argument("--images", help="Diretório de imagens em ordem temporal (nomes sequenciais).")
    parser.add_argument("--image-a", help="Primeira imagem de um par específico.")
    parser.add_argument("--image-b", help="Segunda imagem de um par específico.")
    parser.add_argument("--output", required=True, help="Diretório para JSON, CSV, PLY e imagens de diagnóstico.")
    parser.add_argument("--intrinsics", help="JSON com fx, fy, cx, cy (e, opcionalmente, width/height e dist_coeffs OpenCV) ou camera_matrix 3x3.")
    parser.add_argument("--focal-ratio", type=float, default=1.2, help="Estimativa focal exploratória sem calibração (padrão: 1.2 x dimensão máxima).")
    parser.add_argument("--methods", nargs="+", default=["SIFT", "ORB", "AKAZE"], help="Descritores comparados.")
    parser.add_argument("--max-pairs", type=int, default=12, help="Número de pares adjacentes amostrados; 0 para todos. O laço é incluído sempre.")
    parser.add_argument("--visualize-pairs", type=int, default=2, help="Pares iniciais com PNGs; o par de fechamento é incluído sempre.")
    parser.add_argument("--ratio", type=float, default=0.75, help="Limiar do teste de razão de Lowe.")
    parser.add_argument("--f-threshold", type=float, default=1.5, help="Limiar de RANSAC para F em pixels.")
    parser.add_argument("--e-threshold", type=float, default=1.0, help="Limiar de RANSAC para E em pixels.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"Resultados salvos em {Path(args.output).resolve()}; {summary['pair_count']} pares analisados.")


if __name__ == "__main__":
    main()
