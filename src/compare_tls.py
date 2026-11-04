"""Comparação opcional entre fotogrametria métrica e scanner laser terrestre.

Ambas as nuvens devem estar em metros. O alinhamento usa pontos correspondentes
nomeados e apenas rotação + translação: ajustar escala aqui esconderia erro
fotogramétrico. As distâncias de vizinho mais próximo são descritivas e não
constituem calibração ou validação forense de um instrumento.

Uso:
    python -m src.compare_tls --photogrammetry work/scene_scaled.ply \
        --tls data/tls/scan.ply --landmarks data/tls/landmarks.json \
        --output work/tls_comparison.json --aligned-ply work/photo_in_tls.ply
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from scipy.spatial import cKDTree


_PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _ply_header(stream: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    first = stream.readline()
    if first.strip() != b"ply":
        raise ValueError("Arquivo sem cabeçalho PLY ('ply').")
    fmt: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    data_before_vertices = False
    for line_number in range(2, 10002):
        raw = stream.readline()
        if not raw:
            raise ValueError("Cabeçalho PLY sem end_header.")
        parts = raw.decode("ascii", errors="strict").strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "end_header":
            break
        if parts[0] == "format" and len(parts) >= 3:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            count = int(parts[2])
            if count < 0:
                raise ValueError("Contagem negativa em elemento PLY.")
            if current_element == "vertex":
                if vertex_count is not None:
                    raise ValueError("PLY com dois elementos vertex.")
                vertex_count = count
            elif vertex_count is None and count > 0:
                data_before_vertices = True
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("Propriedade de vértice PLY com lista não é suportada.")
            vertex_properties.append((parts[2], parts[1]))
    else:
        raise ValueError("Cabeçalho PLY excede 10000 linhas.")
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Formato PLY não suportado: {fmt!r}.")
    if vertex_count is None or vertex_count == 0:
        raise ValueError("PLY sem vértices.")
    if data_before_vertices:
        raise ValueError("PLY com elemento de dados antes de vertex não é suportado.")
    names = [name for name, _ in vertex_properties]
    if len(names) != len(set(names)) or not {"x", "y", "z"} <= set(names):
        raise ValueError("PLY precisa de propriedades únicas x, y e z por vértice.")
    return fmt, vertex_count, vertex_properties


def iter_ply_xyz(path: str | Path) -> Iterator[tuple[float, float, float]]:
    """Lê ASCII ou binário PLY com propriedades escalares e vértices primeiro."""
    with Path(path).open("rb") as stream:
        fmt, count, properties = _ply_header(stream)
        indices = [next(i for i, (name, _) in enumerate(properties) if name == axis)
                   for axis in "xyz"]
        if fmt == "ascii":
            for i in range(count):
                raw = stream.readline()
                if not raw:
                    raise ValueError(f"PLY truncado no vértice {i + 1}/{count}.")
                parts = raw.decode("ascii", errors="strict").split()
                if len(parts) < len(properties):
                    raise ValueError(f"Vértice PLY {i + 1} tem poucas propriedades.")
                xyz = tuple(float(parts[j]) for j in indices)
                if np.isfinite(xyz).all():
                    yield xyz  # type: ignore[misc]
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            try:
                row = struct.Struct(endian + "".join(_PLY_TYPES[typ] for _, typ in properties))
            except KeyError as exc:
                raise ValueError(f"Tipo de propriedade PLY não suportado: {exc}.") from exc
            for i in range(count):
                raw = stream.read(row.size)
                if len(raw) != row.size:
                    raise ValueError(f"PLY truncado no vértice {i + 1}/{count}.")
                values = row.unpack(raw)
                xyz = tuple(float(values[j]) for j in indices)
                if np.isfinite(xyz).all():
                    yield xyz  # type: ignore[misc]


def load_sampled_cloud(path: str | Path, max_points: int, seed: int,
                       voxel_size: float) -> tuple[np.ndarray, dict]:
    """Reservoir sampling reproduzível, seguido por média dos pontos em cada voxel."""
    if max_points < 3 or voxel_size <= 0:
        raise ValueError("max_points deve ser >= 3 e voxel_size deve ser positivo.")
    rng = np.random.default_rng(seed)
    sample: list[tuple[float, float, float]] = []
    valid_count = 0
    for xyz in iter_ply_xyz(path):
        valid_count += 1
        if valid_count <= max_points:
            sample.append(xyz)
        else:
            index = int(rng.integers(valid_count))
            if index < max_points:
                sample[index] = xyz
    if len(sample) < 3:
        raise ValueError(f"Nuvem {path} precisa de ao menos 3 vértices finitos.")
    points = np.asarray(sample, dtype=np.float64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    downsampled = sums / counts[:, None]
    return downsampled, {
        "valid_vertices": valid_count,
        "reservoir_vertices": len(sample),
        "voxel_vertices": len(downsampled),
        "voxel_size_m": voxel_size,
        "seed": seed,
    }


def load_landmarks(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """JSON com landmarks: [{name, photogrammetry: [x,y,z], tls: [x,y,z]}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("landmarks") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("landmarks.json precisa de ao menos 3 pontos correspondentes.")
    names: list[str] = []
    photo: list[np.ndarray] = []
    tls: list[np.ndarray] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("Cada landmark deve ter nome e coordenadas.")
        name = row["name"].strip()
        if not name or name in names:
            raise ValueError(f"Nome de landmark vazio ou repetido: {name!r}.")
        names.append(name)
        for field, target in (("photogrammetry", photo), ("tls", tls)):
            xyz = np.asarray(row.get(field), dtype=np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"Landmark {name!r} precisa de {field} = [x, y, z] finito.")
            target.append(xyz)
    return names, np.stack(photo), np.stack(tls)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: encontra R,t em target ~= source @ R.T + t, sem escala."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Alinhamento requer 3+ pares de pontos 3D.")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_center, target - target_center
    if np.linalg.matrix_rank(centered_source, tol=1e-8) < 2 or \
       np.linalg.matrix_rank(centered_target, tol=1e-8) < 2:
        raise ValueError("Landmarks colineares: escolha pelo menos 3 pontos não alinhados.")
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _distances(source: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    distances = cKDTree(target).query(source, k=1, workers=-1)[0]
    hits = distances <= threshold
    return {
        "sampled_source_points": int(len(source)),
        "mean_m": float(distances.mean()),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "fraction_within_threshold": float(hits.mean()),
        "median_within_threshold_m": float(np.median(distances[hits])) if hits.any() else None,
    }


def compare(photo_cloud: np.ndarray, tls_cloud: np.ndarray, names: list[str],
            photo_marks: np.ndarray, tls_marks: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    """Compara nuvens após alinhamento rígido por marcos fixos em metros."""
    if threshold <= 0:
        raise ValueError("threshold deve ser positivo.")
    rotation, translation = rigid_alignment(photo_marks, tls_marks)
    aligned_marks = photo_marks @ rotation.T + translation
    residuals = np.linalg.norm(aligned_marks - tls_marks, axis=1)
    aligned_cloud = photo_cloud @ rotation.T + translation
    photo_pair_lengths = np.linalg.norm(photo_marks[:, None] - photo_marks[None, :], axis=2)
    tls_pair_lengths = np.linalg.norm(tls_marks[:, None] - tls_marks[None, :], axis=2)
    pairs = np.triu_indices(len(names), k=1)
    if np.any(photo_pair_lengths[pairs] <= 1e-9):
        raise ValueError("Landmarks fotogramétricos repetidos ou coincidentes.")
    scale_ratios = tls_pair_lengths[pairs] / photo_pair_lengths[pairs]
    return {
        "units": "m",
        "alignment": "rigid_rotation_translation_only_no_scale_fit",
        "rotation_photo_to_tls": rotation.tolist(),
        "translation_photo_to_tls_m": translation.tolist(),
        "landmark_residuals_m": {name: float(err) for name, err in zip(names, residuals)},
        "landmark_rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "landmark_pair_distance_ratio_tls_over_photo_median": float(np.median(scale_ratios)),
        "overlap_threshold_m": threshold,
        "photo_to_tls": _distances(aligned_cloud, tls_cloud, threshold),
        "tls_to_photo": _distances(tls_cloud, aligned_cloud, threshold),
        "interpretation": (
            "Distâncias de vizinho mais próximo não são erro metrológico: diferenças de "
            "densidade, amostragem, oclusão e superfícies fora da área comum afetam os "
            "resultados. TLS->fotogrametria indica sobretudo cobertura quando a nuvem "
            "fotogramétrica é esparsa. Comparar medidas físicas independentes e inspecionar "
            "sobreposição antes de tirar conclusões."
        ),
    }, aligned_cloud


def write_ascii_ply(path: str | Path, points: np.ndarray) -> None:
    """Exporta a amostra alinhada para inspeção visual."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photogrammetry", type=Path, required=True,
                        help="Nuvem fotogramétrica já escalada em metros (PLY).")
    parser.add_argument("--tls", type=Path, required=True, help="Nuvem TLS da mesma cena em metros (PLY).")
    parser.add_argument("--landmarks", type=Path, required=True,
                        help="JSON com 3+ marcos 3D nomeados em ambas as nuvens.")
    parser.add_argument("--output", type=Path, help="Gravar métricas e transformação em JSON.")
    parser.add_argument("--aligned-ply", type=Path, help="Gravar amostra fotogramétrica alinhada (PLY ASCII).")
    parser.add_argument("--max-points", type=int, default=250_000,
                        help="Máximo por nuvem antes da redução em voxels (padrão: 250000).")
    parser.add_argument("--voxel-size", type=float, default=0.03,
                        help="Lado do voxel em metros para ambas as nuvens (padrão: 0.03).")
    parser.add_argument("--overlap-threshold", type=float, default=0.10,
                        help="Distância em metros para fração de sobreposição (padrão: 0.10).")
    parser.add_argument("--seed", type=int, default=0, help="Semente da amostragem (padrão: 0).")
    args = parser.parse_args()
    names, photo_marks, tls_marks = load_landmarks(args.landmarks)
    photo, photo_meta = load_sampled_cloud(args.photogrammetry, args.max_points,
                                           args.seed, args.voxel_size)
    tls, tls_meta = load_sampled_cloud(args.tls, args.max_points,
                                      args.seed, args.voxel_size)
    metrics, aligned = compare(photo, tls, names, photo_marks, tls_marks,
                               args.overlap_threshold)
    metrics["input"] = {
        "photogrammetry": str(args.photogrammetry), "tls": str(args.tls),
        "landmarks": str(args.landmarks), "photogrammetry_sample": photo_meta,
        "tls_sample": tls_meta,
    }
    result = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    if args.aligned_ply:
        write_ascii_ply(args.aligned_ply, aligned)
    print(result)


if __name__ == "__main__":
    main()
