from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import demand


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        year, value = part.split(":", 1)
        year = year.strip()
        try:
            weights[year] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Peso inválido para {year}: {value}"
            ) from None
    return weights or None


def _parse_quantiles(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return (0.33, 0.66)
    try:
        values = tuple(float(x.strip()) for x in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Cuantiles inválidos") from exc
    if not values:
        raise argparse.ArgumentTypeError("Debe indicar al menos un cuantil")
    if any(q <= 0.0 or q >= 1.0 for q in values):
        raise argparse.ArgumentTypeError("Los cuantiles deben estar en (0,1)")
    if tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError(
            "Los cuantiles deben venir ordenados de menor a mayor"
        )
    return values


def _parse_labels(raw: str | None, expected: int) -> tuple[str, ...]:
    if not raw:
        return ("pesimista", "regular", "optimista")
    labels = tuple(part.strip() for part in raw.split(","))
    if len(labels) != expected:
        raise argparse.ArgumentTypeError(
            f"Se esperaban {expected} etiquetas y se recibieron {len(labels)}"
        )
    if any(not label for label in labels):
        raise argparse.ArgumentTypeError("Las etiquetas no pueden estar vacías")
    return labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segmenta escenarios de demanda en categorías percentiles",
    )
    parser.add_argument(
        "input", type=Path, help="Archivo CSV con columnas scenario,2026-2030"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV de salida (default: mismo nombre con sufijo _segmented)",
    )
    parser.add_argument(
        "--method",
        choices=["geometric", "arithmetic"],
        default="geometric",
        help="Método para calcular el puntaje agregado",
    )
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        help="Pesos por año, formato 2026:1,2027:1.5",
    )
    parser.add_argument(
        "--quantiles",
        type=_parse_quantiles,
        help="Puntos de corte en (0,1), formato 0.2,0.8",
    )
    parser.add_argument(
        "--labels",
        help="Etiquetas separadas por coma para cada segmento",
    )
    parser.add_argument(
        "--years",
        help="Columnas de años a usar, formato 2026,2027,...",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    year_columns = None
    if args.years:
        year_columns = tuple(
            part.strip() for part in args.years.split(",") if part.strip()
        )

    quantiles = args.quantiles or (0.33, 0.66)
    labels = _parse_labels(args.labels, len(quantiles) + 1)

    try:
        segmentation = demand.segment_projection(
            args.input,
            year_columns=year_columns,
            method=args.method,
            weights=args.weights,
            quantiles=quantiles,
            labels=labels,
        )
    except demand.DemandProjectionError as exc:
        parser.error(str(exc))
        return 1

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(args.input.stem + "_segmented.csv")

    segmentation.data.to_csv(output_path, index=False)
    print(f"Archivo segmentado guardado en {output_path}")

    summary = segmentation.summary()
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print("\nResumen por segmento:\n")
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
