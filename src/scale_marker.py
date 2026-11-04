"""Estimate monocular SfM metric scale from a fixed, printed ArUco marker.

The physical marker side length is still an external measurement. This program
automates finding its corners and comparing their triangulated length with it.
It does not change the COLMAP reconstruction; the resulting JSON scale factor
can be applied by the project's model analysis step.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .model import load_model
except ImportError:  # Allow ``python src/scale_marker.py`` as well.
    from model import load_model


@dataclass
class Calibration:
    matrix: np.ndarray
    distortion: np.ndarray
    fisheye: bool = False


@dataclass
class Observation:
    image_name: str
    corners_px: np.ndarray
    corners_normalized: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    calibration: Calibration

    @property
    def camera_center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @property
    def projection(self) -> np.ndarray:
        return np.column_stack((self.rotation, self.translation))


def _camera_calibration(camera: Any) -> Calibration:
    """Convert common COLMAP camera models to OpenCV undistortion parameters."""
    name = str(camera.model).upper()
    p = np.asarray(camera.params, dtype=np.float64)
    if name == "SIMPLE_PINHOLE" and len(p) == 3:
        f, cx, cy = p
        fx = fy = f
        d = np.zeros(5)
    elif name == "PINHOLE" and len(p) == 4:
        fx, fy, cx, cy = p
        d = np.zeros(5)
    elif name == "SIMPLE_RADIAL" and len(p) == 4:
        f, cx, cy, k1 = p
        fx = fy = f
        d = np.array([k1, 0, 0, 0, 0])
    elif name == "RADIAL" and len(p) == 5:
        f, cx, cy, k1, k2 = p
        fx = fy = f
        d = np.array([k1, k2, 0, 0, 0])
    elif name == "OPENCV" and len(p) == 8:
        fx, fy, cx, cy = p[:4]
        d = p[4:]
    elif name == "FULL_OPENCV" and len(p) == 12:
        fx, fy, cx, cy = p[:4]
        d = p[4:]
    elif name == "OPENCV_FISHEYE" and len(p) == 8:
        fx, fy, cx, cy = p[:4]
        d = p[4:]
        return Calibration(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
            np.asarray(d, dtype=np.float64),
            fisheye=True,
        )
    else:
        raise ValueError(
            f"Modelo de camera COLMAP {name!r} nao suportado para escala "
            "automatica; use SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, "
            "OPENCV, FULL_OPENCV ou OPENCV_FISHEYE."
        )
    return Calibration(
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
        np.asarray(d, dtype=np.float64),
    )


def _normalized_corners(corners: np.ndarray, cal: Calibration) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    if cal.fisheye:
        normalized = cv2.fisheye.undistortPoints(points, cal.matrix, cal.distortion)
    else:
        normalized = cv2.undistortPoints(points, cal.matrix, cal.distortion)
    return normalized.reshape(-1, 2)


def _rotation_from_qvec(qvec: Any) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def _detect_markers(image: np.ndarray, dictionary_name: str) -> tuple[list[np.ndarray], np.ndarray | None]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Esta instalacao de OpenCV nao contem cv2.aruco.")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None or not dictionary_name.startswith("DICT_"):
        raise ValueError(f"Dicionario ArUco desconhecido: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    return corners, ids


def _triangulate(observations: list[Observation], corner_index: int) -> np.ndarray:
    rows = []
    for obs in observations:
        u, v = obs.corners_normalized[corner_index]
        projection = obs.projection
        rows.extend((u * projection[2] - projection[0], v * projection[2] - projection[1]))
    _, singular_values, vt = np.linalg.svd(np.asarray(rows))
    x = vt[-1]
    if abs(x[3]) < 1e-12 or not np.isfinite(singular_values).all():
        raise ValueError("Triangulacao degenerada: aumente o deslocamento entre fotos.")
    return x[:3] / x[3]


def _residuals_px(observations: list[Observation], corners_3d: np.ndarray) -> np.ndarray:
    """Return per-image, per-corner approximate reprojection errors in pixels."""
    residuals = np.empty((len(observations), 4), dtype=np.float64)
    for i, obs in enumerate(observations):
        camera_points = (obs.rotation @ corners_3d.T).T + obs.translation
        if np.any(camera_points[:, 2] <= 0):
            residuals[i, :] = np.inf
            continue
        predicted = camera_points[:, :2] / camera_points[:, 2, None]
        diff = predicted - obs.corners_normalized
        fx, fy = obs.calibration.matrix[0, 0], obs.calibration.matrix[1, 1]
        residuals[i, :] = np.linalg.norm(diff * [fx, fy], axis=1)
    return residuals


def _max_ray_angle_deg(observations: list[Observation], point: np.ndarray) -> float:
    rays = np.array([point - obs.camera_center for obs in observations])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    dots = np.clip(rays @ rays.T, -1, 1)
    return float(np.degrees(np.arccos(np.min(dots))))


def estimate_scale(
    observations: list[Observation],
    marker_size_m: float,
    max_reprojection_px: float = 5.0,
    min_ray_angle_deg: float = 1.0,
) -> dict[str, Any]:
    """Triangulate marker corners and return metres per COLMAP model unit."""
    if marker_size_m <= 0 or not math.isfinite(marker_size_m):
        raise ValueError("marker_size_m deve ser positivo e finito.")
    if len(observations) < 3:
        raise ValueError("O mesmo marcador precisa aparecer em >= 3 imagens registradas.")

    active = list(observations)
    rejected: list[str] = []
    while True:
        corners_3d = np.array([_triangulate(active, j) for j in range(4)])
        residuals = _residuals_px(active, corners_3d)
        per_view = np.median(residuals, axis=1)
        worst = int(np.argmax(per_view))
        if per_view[worst] <= max_reprojection_px:
            break
        if len(active) <= 3:
            raise ValueError(
                "Reprojecao dos cantos acima do limite mesmo com 3 vistas; "
                "verifique marcador, calibracao e reconstrução."
            )
        rejected.append(active.pop(worst).image_name)

    ray_angle = _max_ray_angle_deg(active, corners_3d.mean(axis=0))
    if ray_angle < min_ray_angle_deg:
        raise ValueError(
            f"Paralaxe insuficiente ({ray_angle:.2f} graus); "
            "capture o marcador de posicoes mais separadas."
        )
    edges = np.linalg.norm(corners_3d - np.roll(corners_3d, -1, axis=0), axis=1)
    side = float(np.median(edges))
    if not math.isfinite(side) or side <= 1e-12:
        raise ValueError("Lado reconstruido degenerado.")
    edge_cv = float(np.std(edges) / np.mean(edges))
    scale = float(marker_size_m / side)
    return {
        "scale_m_per_colmap_unit": scale,
        "marker_size_m": float(marker_size_m),
        "reconstructed_side_colmap_units": side,
        "side_lengths_colmap_units": [float(x) for x in edges],
        "side_length_cv": edge_cv,
        "triangulated_corners_colmap_units": corners_3d.tolist(),
        "images_used": [obs.image_name for obs in active],
        "images_rejected": rejected,
        "max_ray_angle_deg": ray_angle,
        "reprojection_rmse_px_approx": float(np.sqrt(np.mean(residuals**2))),
        "warning": (
            "Alta variacao entre lados do marcador; confira resolucao, plano "
            "impresso, calibracao e observacoes."
            if edge_cv > 0.10 else None
        ),
    }


def estimate_from_colmap(
    model_path: Path,
    images_path: Path,
    marker_size_m: float,
    marker_id: int | None = None,
    dictionary_name: str = "DICT_4X4_50",
    max_reprojection_px: float = 5.0,
) -> dict[str, Any]:
    model = load_model(model_path)
    candidates: dict[int, list[Observation]] = {}
    missing: list[str] = []
    unsupported: set[str] = set()
    for registered in model.images:
        path = images_path / registered.name
        if not path.is_file():
            missing.append(registered.name)
            continue
        # imdecode handles non-ASCII Windows paths more consistently than imread.
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            missing.append(registered.name)
            continue
        camera = model.cameras[registered.camera_id]
        if (image.shape[1], image.shape[0]) != (camera.width, camera.height):
            # COLMAP builds differ in whether EXIF Orientation is applied while reading.
            raw = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if raw is not None and (raw.shape[1], raw.shape[0]) == (camera.width, camera.height):
                image = raw
            else:
                raise ValueError(
                    f"Dimensoes da imagem {registered.name} nao coincidem com cameras.txt; "
                    "confira orientacao EXIF e evite redimensionar apos SfM."
                )
        corners, ids = _detect_markers(image, dictionary_name)
        if ids is None:
            continue
        try:
            cal = _camera_calibration(camera)
        except ValueError:
            unsupported.add(str(camera.model))
            continue
        rotation = _rotation_from_qvec(registered.qvec)
        translation = np.asarray(registered.tvec, dtype=np.float64).reshape(3)
        for detected_corners, detected_id in zip(corners, ids.flatten()):
            this_id = int(detected_id)
            if marker_id is not None and this_id != marker_id:
                continue
            xy = np.asarray(detected_corners, dtype=np.float64).reshape(4, 2)
            candidates.setdefault(this_id, []).append(
                Observation(
                    image_name=registered.name,
                    corners_px=xy,
                    corners_normalized=_normalized_corners(xy, cal),
                    rotation=rotation,
                    translation=translation,
                    calibration=cal,
                )
            )

    if not candidates:
        detail = f" Cameras nao suportadas: {sorted(unsupported)}." if unsupported else ""
        raise ValueError(f"Nenhum marcador ArUco correspondente foi detectado.{detail}")
    if marker_id is None:
        marker_id = max(candidates, key=lambda key: len(candidates[key]))
    observations = candidates.get(marker_id, [])
    if len(observations) < 3:
        raise ValueError(
            f"Marcador ID {marker_id} visto em {len(observations)} imagens; "
            "sao necessarias pelo menos 3 vistas registradas."
        )
    result = estimate_scale(observations, marker_size_m, max_reprojection_px)
    result.update(
        {
            "method": "multi_view_aruco_corner_triangulation",
            "marker_id": marker_id,
            "dictionary": dictionary_name,
            "registered_images_scanned": len(model.images),
            "missing_or_unreadable_images": missing,
            "unsupported_camera_models": sorted(unsupported),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pasta do modelo COLMAP em TXT")
    parser.add_argument("--images", type=Path, required=True, help="Pasta das imagens originais")
    parser.add_argument("--marker-size-m", type=float, required=True, help="Lado impresso e medido, em metros")
    parser.add_argument("--marker-id", type=int, help="ID do marcador; sem ele, usa o mais visto")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="Nome do dicionario cv2.aruco")
    parser.add_argument("--max-reprojection-px", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True, help="Arquivo JSON de saida")
    args = parser.parse_args(argv)
    try:
        result = estimate_from_colmap(
            args.model,
            args.images,
            args.marker_size_m,
            args.marker_id,
            args.dictionary,
            args.max_reprojection_px,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"Erro de escala ArUco: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scale_m_per_colmap_unit": result["scale_m_per_colmap_unit"],
                      "images_used": len(result["images_used"]),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
