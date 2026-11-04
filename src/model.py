"""Leitura do formato texto de reconstruções esparsas do COLMAP.

Formato: https://colmap.github.io/format.html
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class Image:
    id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    points2d: tuple[tuple[float, float, int], ...]

    @property
    def rotation(self) -> np.ndarray:
        q = self.qvec / np.linalg.norm(self.qvec)
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec


@dataclass(frozen=True)
class Point:
    id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float
    track_length: int


@dataclass(frozen=True)
class Model:
    cameras: dict[int, Camera]
    images: list[Image]
    points: dict[int, Point]

    @property
    def ordered_images(self) -> list[Image]:
        return sorted(self.images, key=lambda im: im.name)


def _data_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Modelo COLMAP sem {path}. Converta com model_converter --output_type TXT.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]


def load_model(directory: str | Path) -> Model:
    directory = Path(directory)
    cameras: dict[int, Camera] = {}
    for line in _data_lines(directory / "cameras.txt"):
        if not line:
            continue
        p = line.split()
        camera = Camera(int(p[0]), p[1], int(p[2]), int(p[3]), tuple(map(float, p[4:])))
        cameras[camera.id] = camera

    images: list[Image] = []
    lines = _data_lines(directory / "images.txt")
    if len(lines) % 2:
        raise ValueError("images.txt deve conter duas linhas por imagem, inclusive a linha de observações vazia.")
    for i in range(0, len(lines), 2):
        head = lines[i].split(maxsplit=9)
        if not head:
            continue
        if len(head) != 10:
            raise ValueError(f"Cabeçalho inválido em images.txt: {lines[i]}")
        obs = lines[i+1].split()
        if len(obs) % 3:
            raise ValueError(f"Observações inválidas da imagem {head[9]}")
        points2d = tuple((float(obs[j]), float(obs[j+1]), int(obs[j+2]))
                         for j in range(0, len(obs), 3))
        images.append(Image(int(head[0]), head[9], int(head[8]),
                            np.array(list(map(float, head[1:5]))),
                            np.array(list(map(float, head[5:8]))), points2d))

    points: dict[int, Point] = {}
    for line in _data_lines(directory / "points3D.txt"):
        if not line:
            continue
        p = line.split()
        if len(p) < 8 or (len(p)-8) % 2:
            raise ValueError(f"Linha inválida em points3D.txt: {line[:80]}")
        point = Point(int(p[0]), np.array(list(map(float, p[1:4]))),
                      tuple(map(int, p[4:7])), float(p[7]), (len(p)-8)//2)
        points[point.id] = point
    if not images:
        raise ValueError("Nenhuma imagem registrada no modelo.")
    return Model(cameras, images, points)
