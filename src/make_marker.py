"""Gera marcador ArUco SVG com dimensões físicas declaradas.

Imprima em 100%, sem 'ajustar à página', e meça o lado preto com régua.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def make_svg(marker_id: int, side_mm: float, dictionary_name: str,
             margin_mm: float = 15) -> str:
    if side_mm <= 0 or margin_mm < 0:
        raise ValueError("side_mm deve ser positivo e margin_mm não negativo")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None or not dictionary_name.startswith("DICT_"):
        raise ValueError(f"Dicionário ArUco inválido: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    bit_count = int(dictionary.markerSize) + 2
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, bit_count*100)
    cell = side_mm/bit_count
    total = side_mm+2*margin_mm
    rects = []
    for row in range(bit_count):
        for col in range(bit_count):
            if image[row*100+50,col*100+50] < 128:
                x = margin_mm+col*cell
                y = margin_mm+row*cell
                rects.append(f'<rect x="{x:.6f}" y="{y:.6f}" width="{cell:.6f}" height="{cell:.6f}" fill="black"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total:.4f}mm" '
            f'height="{total:.4f}mm" viewBox="0 0 {total:.4f} {total:.4f}">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            + "\n".join(rects) + "\n</svg>\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--side-mm", type=float, default=160.0)
    p.add_argument("--margin-mm", type=float, default=15.0)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--output", type=Path, default=Path("data/marker_0.svg"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_svg(args.id,args.side_mm,args.dictionary,args.margin_mm), encoding="utf-8")
    print(f"{args.output} | lado nominal {args.side_mm} mm. Confira o lado impresso com régua.")


if __name__ == "__main__":
    main()
