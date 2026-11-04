"""Métricas, escala, planta e planos do modelo COLMAP.

Exemplo:
python -m src.analysis --model work/main/sparse_txt --output work/main/analysis \
    --scale-json work/main/scale.json --measurements data/measurements.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from .model import Model, load_model


def observation_reprojection_errors(model: Model) -> tuple[list[float], int]:
    """Reprojeta cada observação 2D com os intrínsecos e poses finais."""
    from .scale_marker import _camera_calibration
    residuals: list[float] = []
    unsupported = 0
    for image in model.images:
        try:
            cal = _camera_calibration(model.cameras[image.camera_id])
        except ValueError:
            unsupported += 1
            continue
        observed = [(x,y,point_id) for x,y,point_id in image.points2d if point_id in model.points]
        if not observed:
            continue
        xyz = np.array([model.points[v[2]].xyz for v in observed], dtype=np.float64).reshape(-1,1,3)
        xy = np.array([[v[0],v[1]] for v in observed], dtype=np.float64)
        rvec,_ = cv2.Rodrigues(image.rotation)
        tvec = image.tvec.astype(np.float64).reshape(3,1)
        if cal.fisheye:
            projected,_ = cv2.fisheye.projectPoints(xyz,rvec,tvec,cal.matrix,cal.distortion)
        else:
            projected,_ = cv2.projectPoints(xyz,rvec,tvec,cal.matrix,cal.distortion)
        residuals.extend(np.linalg.norm(projected.reshape(-1,2)-xy,axis=1).tolist())
    return residuals, unsupported


def metric_scale(model: Model, measurements: dict, scale_json: Path | None) -> tuple[float | None, str]:
    if scale_json:
        data = json.loads(scale_json.read_text(encoding="utf-8"))
        scale = float(data["scale_m_per_colmap_unit"])
        if scale <= 0:
            raise ValueError("Escala ArUco deve ser positiva.")
        return scale, "ArUco triangulado"
    ref = measurements.get("reference")
    if ref:
        a, b = (int(v) for v in ref["point_ids"])
        actual = float(ref["distance_m"])
        if actual <= 0 or a not in model.points or b not in model.points:
            raise ValueError("Referência manual inválida: IDs devem existir e distância deve ser positiva.")
        d = np.linalg.norm(model.points[a].xyz - model.points[b].xyz)
        if d < 1e-9:
            raise ValueError("Pontos da referência manual coincidem.")
        return actual / d, f"pontos 3D {a} e {b}"
    return None, "sem referência métrica"


def upright_rotation(images) -> np.ndarray:
    # COLMAP: eixo Y local aponta para baixo. A média pressupõe câmera em posição habitual.
    ups = np.array([im.rotation.T @ np.array([0.0, -1.0, 0.0]) for im in images])
    up = np.median(ups, axis=0)
    up /= np.linalg.norm(up)
    return Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()


def _fit_plane_ransac(xyz: np.ndarray, threshold: float, rng: np.random.Generator,
                      iterations: int = 600) -> dict | None:
    if len(xyz) < 3:
        return None
    best = np.empty(0, dtype=int)
    for _ in range(iterations):
        ids = rng.choice(len(xyz), 3, replace=False)
        a, b, c = xyz[ids]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inliers = np.flatnonzero(np.abs((xyz-a) @ normal) < threshold)
        if len(inliers) > len(best):
            best = inliers
    if len(best) < max(12, int(len(xyz)*0.012)):
        return None
    selected = xyz[best]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected-center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    refined = np.flatnonzero(np.abs((xyz-center) @ normal) < threshold)
    return {"normal": normal, "offset": float(normal @ center),
            "inliers": refined, "centroid": xyz[refined].mean(axis=0)}


def detect_planes(xyz: np.ndarray, threshold: float, max_planes: int = 8) -> list[dict]:
    rng = np.random.default_rng(919)
    remaining = np.arange(len(xyz))
    planes = []
    for _ in range(max_planes):
        fit = _fit_plane_ransac(xyz[remaining], threshold, rng)
        if fit is None:
            break
        ids = remaining[fit["inliers"]]
        if len(ids) < max(15, int(len(xyz)*0.015)):
            break
        fit["count"] = int(len(ids))
        fit["kind"] = ("horizontal" if abs(fit["normal"][2]) > 0.85 else
                       "vertical" if abs(fit["normal"][2]) < 0.25 else "slanted")
        planes.append(fit)
        mask = np.ones(len(remaining), dtype=bool)
        mask[fit["inliers"]] = False
        remaining = remaining[mask]
        if len(remaining) < 20:
            break
    return planes


def wall_dimensions(planes: list[dict], cameras_xyz: np.ndarray) -> dict:
    walls = [p for p in planes if p["kind"] == "vertical"]
    candidates = []
    camera_center = np.median(cameras_xyz, axis=0)
    for a, b in combinations(walls, 2):
        na = a["normal"][:2] / np.linalg.norm(a["normal"][:2])
        nb = b["normal"][:2] / np.linalg.norm(b["normal"][:2])
        alignment = float(na @ nb)
        if abs(alignment) < np.cos(np.deg2rad(15)):
            continue
        da = a["offset"] / np.linalg.norm(a["normal"][:2])
        db = b["offset"] / np.linalg.norm(b["normal"][:2])
        if alignment < 0:
            db = -db
        separation = abs(da-db)
        camera_projection = na @ camera_center[:2]
        if not (min(da,db) - .1 <= camera_projection <= max(da,db) + .1):
            continue
        if separation < 0.5:
            continue
        candidates.append({"distance_m": float(separation), "axis": na,
                           "support": a["count"] + b["count"]})
    candidates.sort(key=lambda v: v["support"], reverse=True)
    chosen = []
    for candidate in candidates:
        if not chosen or abs(float(candidate["axis"] @ chosen[0]["axis"])) < 0.3:
            chosen.append(candidate)
        if len(chosen) == 2:
            break
    dimensions = sorted([c["distance_m"] for c in chosen])
    horizontals = [p for p in planes if p["kind"] == "horizontal"]
    height_candidates = []
    for a,b in combinations(horizontals,2):
        na,nb = a["normal"],b["normal"]
        if abs(float(na @ nb)) < np.cos(np.deg2rad(15)):
            continue
        # The plane's z-intercept is stable for approximately horizontal planes.
        za = a["offset"]/na[2]
        zb = b["offset"]/nb[2]
        camera_z = float(np.median(cameras_xyz[:,2]))
        if min(za,zb) < camera_z < max(za,zb) and abs(za-zb) > 1:
            height_candidates.append((a["count"]+b["count"],abs(za-zb)))
    height_candidates.sort(reverse=True)
    return {"detected_parallel_wall_distances_m": dimensions,
            "width_m": dimensions[0] if len(dimensions) == 2 else None,
            "length_m": dimensions[1] if len(dimensions) == 2 else None,
            "height_m": float(height_candidates[0][1]) if height_candidates else None,
            "candidate_pair_count": len(candidates)}


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write(f"element vertex {len(xyz)}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, color in zip(xyz, rgb):
            out.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for _ in range(200):
            line = stream.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                break
    return None


def analyze(model: Model, scale: float | None, source: str, measurements: dict,
            output: Path, input_count: int | None = None,
            dense_count: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    u = upright_rotation(model.images)
    s = scale if scale is not None else 1.0
    images = model.ordered_images
    camera_xyz = np.array([u @ (im.center*s) for im in images])
    camera_to_world = np.array([u @ im.rotation.T for im in images])
    point_list = list(model.points.values())
    xyz = np.array([u @ (p.xyz*s) for p in point_list]) if point_list else np.empty((0,3))
    rgb = np.array([p.rgb for p in point_list], dtype=np.uint8) if point_list else np.empty((0,3), dtype=np.uint8)
    errors, unsupported_cameras = observation_reprojection_errors(model)
    path_length = float(np.linalg.norm(np.diff(camera_xyz, axis=0), axis=1).sum()) if len(images)>1 else 0.0
    loop_delta = float(np.linalg.norm(camera_xyz[-1]-camera_xyz[0])) if len(images)>1 else None
    report: dict = {
        "units": "m" if scale else "COLMAP arbitrary units",
        "scale_m_per_colmap_unit": scale, "scale_source": source,
        "registered_images": len(images), "input_images": input_count,
        "registration_fraction": len(images)/input_count if input_count else None,
        "sparse_points": len(point_list),
        "dense_points": dense_count,
        "reprojection_error_px_mean_per_observation": float(np.mean(errors)) if errors else None,
        "reprojection_error_px_median_per_observation": float(np.median(errors)) if errors else None,
        "reprojection_observation_count": len(errors),
        "reprojection_unsupported_image_count": unsupported_cameras,
        "first_registered_image": images[0].name, "last_registered_image": images[-1].name,
        "path_length": path_length, "loop_delta": loop_delta,
        "loop_delta_percent_of_path": 100*loop_delta/path_length if path_length > 0 else None,
    }
    if scale and len(xyz) >= 30:
        planes = detect_planes(xyz, threshold=0.055)
        report["planes"] = [{"normal": p["normal"].tolist(), "offset_m": p["offset"],
                             "support_points": p["count"], "kind": p["kind"]} for p in planes]
        report["room_auto"] = wall_dimensions(planes, camera_xyz)
        walls = [(i,p["normal"][:2]) for i,p in enumerate(planes) if p["kind"] == "vertical"]
        angles = []
        for (i,a),(j,b) in combinations(walls, 2):
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),0,1)))
            if angle >= 60:
                angles.append({"plane_a":i,"plane_b":j,"angle_deg":float(angle),
                               "deviation_from_90_deg":float(abs(90-angle))})
        report["candidate_adjacent_wall_angles"] = angles
    else:
        report["planes"] = []
        report["room_auto"] = None
    room_errors = {}
    for key, actual in measurements.get("room_ground_truth_m", {}).items():
        if key not in {"width_m","length_m","height_m"}:
            continue
        actual = float(actual)
        if actual <= 0:
            raise ValueError(f"room_ground_truth_m.{key} deve ser positivo")
        estimated = report["room_auto"].get(key) if report["room_auto"] else None
        room_errors[key] = {"actual_m": actual, "estimated_m": estimated,
                            "absolute_error_cm": 100*abs(estimated-actual) if estimated else None,
                            "relative_error_percent": 100*abs(estimated-actual)/actual if estimated else None}
    report["room_auto_errors"] = room_errors
    if scale:
        gt = measurements.get("room_ground_truth_m", {})
        if gt.get("width_m") and gt.get("length_m"):
            area = float(gt["width_m"])*float(gt["length_m"])
            area_source = "trena"
        elif report["room_auto"] and report["room_auto"]["width_m"] and report["room_auto"]["length_m"]:
            area = report["room_auto"]["width_m"]*report["room_auto"]["length_m"]
            area_source = "planos RANSAC"
        else:
            area = None
            area_source = None
        if area:
            report["area_for_density_m2"] = area
            report["area_for_density_source"] = area_source
            report["sparse_density_points_per_m2_approx"] = len(xyz)/area
            if dense_count is not None:
                report["dense_density_points_per_m2_approx"] = dense_count/area
    measures = {}
    for name, spec in measurements.get("dimensions", {}).items():
        ids = [int(v) for v in spec["point_ids"]]
        if len(ids) != 2 or any(v not in model.points for v in ids):
            raise ValueError(f"Dimensão {name}: selecione dois IDs válidos de points3D.txt.")
        measured = float(np.linalg.norm(model.points[ids[0]].xyz-model.points[ids[1]].xyz))*s
        actual = float(spec["actual_m"])
        if actual <= 0:
            raise ValueError(f"Dimensão {name}: actual_m deve ser positivo.")
        measures[name] = {"model_m": measured, "actual_m": actual,
                          "absolute_error_cm": 100*abs(measured-actual) if scale else None,
                          "relative_error_percent": 100*abs(measured-actual)/actual if scale and actual > 0 else None}
    report["reference_dimension_errors"] = measures
    evidence = []
    for item in measurements.get("evidence", []):
        point_id = int(item["point_id"])
        if point_id not in model.points:
            raise ValueError(f"Ponto de evidência {point_id} ausente do modelo.")
        evidence.append({"label": str(item["label"]), "point_id": point_id,
                         "xyz": (u @ (model.points[point_id].xyz*s)).tolist()})
    report["evidence"] = evidence
    report["evidence_distances"] = [
        {"a": a["label"], "b": b["label"], "distance_m": float(np.linalg.norm(np.array(a["xyz"])-b["xyz"]))}
        for a,b in combinations(evidence,2)] if scale else []
    write_ply(output/"scaled_sparse.ply", xyz, rgb)
    write_ply(output/"camera_centers.ply", camera_xyz, np.tile([255,40,40], (len(camera_xyz),1)))
    pose_data = [{"image": im.name, "center": center.tolist(), "camera_to_world": rotation.tolist()}
                 for im,center,rotation in zip(images,camera_xyz,camera_to_world)]
    (output/"camera_poses.json").write_text(json.dumps(pose_data,indent=2),encoding="utf-8")
    (output/"metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    from .visualize import save_figures
    save_figures(output, xyz, rgb, camera_xyz, camera_to_world, evidence, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("work/main/sparse_txt"))
    p.add_argument("--output", type=Path, default=Path("work/main/analysis"))
    p.add_argument("--scale-json", type=Path)
    p.add_argument("--measurements", type=Path, default=Path("data/measurements.json"))
    p.add_argument("--manifest", type=Path, default=Path("work/main/manifest.json"))
    args = p.parse_args()
    model = load_model(args.model)
    measurements = json.loads(args.measurements.read_text(encoding="utf-8")) if args.measurements.exists() else {}
    scale, source = metric_scale(model, measurements, args.scale_json)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
    dense_path = Path(manifest["dense_ply"]) if manifest.get("dense_ply") else None
    dense_count = ply_vertex_count(dense_path) if dense_path else None
    report = analyze(model, scale, source, measurements, args.output,
                     manifest.get("num_input_images"), dense_count)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
