"""Consolida execuções do COLMAP em uma tabela de ablação.

Exemplo::

    python -m src.compare_runs --glob "work/*" --output work/comparison

Cada execução deve ter ``manifest.json`` e, se a análise foi feita,
``analysis/metrics.json``. Resultados ausentes permanecem vazios; o programa
não inventa medidas nem trata unidades arbitrárias como centímetros.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Any


COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "Execução"),
    ("matching", "Matcher"),
    ("loop_detection", "Laço"),
    ("cpu", "CPU"),
    ("input_images", "Fotos"),
    ("registered_images", "Registradas"),
    ("registration_percent", "Registro (%)"),
    ("sparse_points", "Pontos esparsos"),
    ("reprojection_mean_px", "Reprojeção média (px)"),
    ("reprojection_median_px", "Reprojeção mediana (px)"),
    ("metric_error_mean_cm", "Erro métrico médio (cm)"),
    ("metric_error_max_cm", "Erro métrico máximo (cm)"),
    ("loop_delta_cm", "Deriva do laço (cm)"),
    ("loop_delta_percent", "Deriva do trajeto (%)"),
    ("room_width_m", "Largura automática (m)"),
    ("room_length_m", "Comprimento automático (m)"),
    ("runtime_s", "Tempo total (s)"),
    ("dense_generated", "Densa"),
    ("metric_scale", "Escala métrica"),
    ("analysis_status", "Análise"),
    ("input_source", "Fonte das fotos"),
)


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _float_or_none(value: Any, digits: int = 4) -> float | None:
    number = finite_number(value)
    return round(number, digits) if number is not None else None


def read_run(run: Path) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Execução sem manifest.json: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = run / "analysis" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    input_count = manifest.get("num_input_images")
    registered = metrics.get("registered_images", manifest.get("registered_images"))
    fraction = metrics.get("registration_fraction")
    if fraction is None and finite_number(input_count) and finite_number(registered) and float(input_count) > 0:
        fraction = float(registered) / float(input_count)
    timing = manifest.get("timing_seconds") or {}
    stage_times = [n for item in timing.values() if (n := finite_number(item)) is not None]
    runtime = sum(stage_times) if stage_times else None
    scale = finite_number(metrics.get("scale_m_per_colmap_unit"))
    metric = scale is not None and scale > 0 and metrics.get("units") == "m"
    errors = [
        number
        for spec in (metrics.get("reference_dimension_errors") or {}).values()
        if (number := finite_number(spec.get("absolute_error_cm"))) is not None
    ]
    errors.extend(
        number
        for spec in (metrics.get("room_auto_errors") or {}).values()
        if (number := finite_number(spec.get("absolute_error_cm"))) is not None
    )
    room = metrics.get("room_auto") or {}
    loop_delta = finite_number(metrics.get("loop_delta")) if metric else None
    fraction_number = finite_number(fraction)
    return {
        "run": run.name,
        "matching": manifest.get("matching"),
        "loop_detection": bool(manifest.get("loop_detection", False)),
        "cpu": bool(manifest.get("cpu", False)),
        "input_images": input_count,
        "registered_images": registered,
        "registration_percent": _float_or_none(100 * fraction_number if fraction_number is not None else None, 2),
        "sparse_points": metrics.get("sparse_points", manifest.get("sparse_points")),
        "reprojection_mean_px": _float_or_none(metrics.get("reprojection_error_px_mean_per_observation")),
        "reprojection_median_px": _float_or_none(metrics.get("reprojection_error_px_median_per_observation")),
        "metric_error_mean_cm": _float_or_none(statistics.mean(errors) if errors else None, 2),
        "metric_error_max_cm": _float_or_none(max(errors) if errors else None, 2),
        "loop_delta_cm": _float_or_none(100 * loop_delta if loop_delta is not None else None, 2),
        "loop_delta_percent": _float_or_none(metrics.get("loop_delta_percent_of_path") if metric else None, 2),
        "room_width_m": _float_or_none(room.get("width_m")),
        "room_length_m": _float_or_none(room.get("length_m")),
        "runtime_s": _float_or_none(runtime, 2),
        "dense_generated": bool(manifest.get("dense_ply")),
        "metric_scale": metric,
        "analysis_status": "complete" if metrics else "metrics_missing",
        "input_source": manifest.get("images_input"),
    }


def discover_runs(pattern: str, explicit: list[Path], output: Path) -> list[Path]:
    candidates = [Path(path) for path in glob.glob(pattern)] + explicit
    unique: dict[Path, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == output.resolve() or not candidate.is_dir() or not (candidate / "manifest.json").is_file():
            continue
        unique[resolved] = candidate
    return sorted(unique.values(), key=lambda path: str(path).lower())


def markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "sim" if value else "não"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    keys = [key for key, _ in COLUMNS]
    with (output / "ablation.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    overview = (
        ("run", "Execução"), ("matching", "Matcher"), ("loop_detection", "Laço"),
        ("input_images", "Fotos"), ("registered_images", "Registradas"),
        ("registration_percent", "Registro (%)"), ("sparse_points", "Pontos"),
        ("reprojection_mean_px", "Reproj. (px)"), ("runtime_s", "Tempo (s)"),
    )
    geometry = (
        ("run", "Execução"), ("metric_scale", "Escala"),
        ("metric_error_mean_cm", "Erro médio (cm)"),
        ("metric_error_max_cm", "Erro máx. (cm)"),
        ("loop_delta_cm", "Deriva (cm)"),
        ("loop_delta_percent", "Deriva (%)"),
        ("room_width_m", "Largura (m)"), ("room_length_m", "Comprimento (m)"),
        ("dense_generated", "Densa"),
    )

    def table(fields: tuple[tuple[str, str], ...]) -> list[str]:
        headers = [label for _, label in fields]
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(markdown_cell(row.get(key)) for key, _ in fields) + " |" for row in rows)
        return lines

    markdown = ["# Comparação das execuções", "", "## Reconstrução", "", *table(overview),
                "", "## Medidas e geometria", "", *table(geometry)]
    markdown += [
        "",
        "Valores ausentes indicam análise não executada ou métrica indisponível.",
        "A deriva em centímetros e os erros métricos só aparecem quando a execução tem escala métrica válida.",
        "O erro métrico médio resume dimensões independentes anotadas e dimensões automáticas de paredes disponíveis.",
        "Compare execuções com a mesma sala, câmera, referências de escala e critérios de medição.",
        "Uma taxa de registro alta, muitas observações coerentes e erro métrico baixo devem ser avaliados em conjunto; não existe um único indicador suficiente.",
        "",
    ]
    (output / "ablation.md").write_text("\n".join(markdown), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="work/*", help="Padrão de pastas de execução (padrão: work/*).")
    parser.add_argument("--runs", nargs="*", type=Path, default=[], help="Pastas explícitas com manifest.json.")
    parser.add_argument("--output", type=Path, default=Path("work/comparison"), help="Diretório dos arquivos ablation.csv e ablation.md.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = discover_runs(args.glob, args.runs, args.output)
    if not runs:
        raise SystemExit("Nenhuma execução com manifest.json encontrada. Use --glob ou --runs.")
    rows = [read_run(run) for run in runs]
    write_outputs(rows, args.output)
    print(f"{len(rows)} execuções comparadas em {args.output.resolve()}")
    for row in rows:
        print(f"  {row['run']}: {row['registered_images']}/{row['input_images']} imagens, {row['sparse_points']} pontos, análise {row['analysis_status']}")


if __name__ == "__main__":
    main()
