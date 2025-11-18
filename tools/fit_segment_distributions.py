from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_DISTS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "beta": stats.beta,
    "kde_gaussian": ("kde", "gaussian"),
    "kde_epanechnikov": ("kde", "epanechnikov"),
    "kde_exponential": ("kde", "exponential"),
    "kde_cosine": ("kde", "cosine"),
    "kde_linear": ("kde", "linear"),
    "kde_tophat": ("kde", "tophat"),
}


@dataclass
class FitResult:
    segment: str
    year: str
    distribution: str
    params: tuple[float, ...]
    ks_stat: float
    ks_pvalue: float
    sample_size: int
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        param_cols = {f"param_{idx}": value for idx, value in enumerate(self.params)}
        return {
            "segment": self.segment,
            "year": self.year,
            "distribution": self.distribution,
            "sample_size": self.sample_size,
            "ks_stat": self.ks_stat,
            "ks_pvalue": self.ks_pvalue,
            **param_cols,
            **self.extra,
        }


def _parse_distributions(raw: str | None) -> list[str]:
    if not raw:
        return list(SUPPORTED_DISTS.keys())
    dists = [name.strip().lower() for name in raw.split(",") if name.strip()]
    invalid = [name for name in dists if name not in SUPPORTED_DISTS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Distribuciones no soportadas: {', '.join(sorted(set(invalid)))}"
        )
    # preserve order but remove duplicates
    seen = set()
    ordered = []
    for name in dists:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _load_segment_year(df: pd.DataFrame, segment: str, year: str) -> np.ndarray:
    subset = df[df["segment"].str.lower() == segment.lower()]
    if subset.empty:
        raise ValueError(f"Segmento '{segment}' no encontrado en la proyección")
    if year not in df.columns:
        raise ValueError(
            f"Columna de año '{year}' no encontrada en el archivo segmentado"
        )
    values = subset[year].to_numpy(dtype=float, copy=False)
    cleaned = values[np.isfinite(values)]
    if cleaned.size == 0:
        raise ValueError(
            f"Segmento '{segment}' en año {year} no contiene valores numéricos válidos"
        )
    return cleaned


def _silverman_bandwidth(sample: np.ndarray) -> float:
    n = sample.size
    if n <= 1:
        return 1.0
    std = float(np.std(sample, ddof=1))
    if not np.isfinite(std) or std <= 0:
        iqr = float(np.subtract(*np.percentile(sample, [75, 25])))
        std = iqr / 1.349 if iqr > 0 else 1.0
    return 1.06 * std * n ** (-1 / 5)


def _fit_distribution(
    sample: np.ndarray,
    dist_name: str,
    *,
    bandwidth_search: bool = False,
    bandwidth_grid: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], float, float, dict[str, object]]:
    spec = SUPPORTED_DISTS[dist_name]
    if dist_name in {"lognormal", "gamma", "weibull", "beta"} and np.any(sample <= 0):
        raise ValueError("Las distribuciones seleccionadas requieren valores positivos")

    extra: dict[str, object] = {}

    if isinstance(spec, tuple) and spec[0] == "kde":
        kernel = spec[1]
        if bandwidth_search and bandwidth_grid:
            best_bandwidth = None
            best_score = float("-inf")
            data = sample.reshape(-1, 1)
            for bw in bandwidth_grid:
                bw = max(float(bw), 1e-4)
                kde_model = KernelDensity(kernel=kernel, bandwidth=bw)
                kde_model.fit(data)
                score = kde_model.score(data)
                if score > best_score:
                    best_score = score
                    best_bandwidth = bw
            bandwidth = (
                best_bandwidth
                if best_bandwidth is not None
                else _silverman_bandwidth(sample)
            )
        else:
            bandwidth = _silverman_bandwidth(sample)
        bandwidth = max(bandwidth, 1e-4)
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(sample.reshape(-1, 1))
        grid = np.linspace(sample.min(), sample.max(), 2048)
        log_pdf = kde.score_samples(grid.reshape(-1, 1))
        pdf_vals = np.exp(log_pdf)
        cdf_vals = np.cumsum(pdf_vals)
        if cdf_vals[-1] <= 0:
            raise ValueError("Integración degenerada para KDE")
        cdf_vals = cdf_vals / cdf_vals[-1]

        def cdf_fn(x):
            arr = np.atleast_1d(x).astype(float)
            values = np.interp(arr, grid, cdf_vals, left=0.0, right=1.0)
            if np.isscalar(x):
                return float(values[0])
            return values

        ks_stat, ks_pvalue = stats.kstest(sample, cdf_fn)
        params = (bandwidth,)
        extra = {
            "kde_kernel": kernel,
            "kde_bandwidth": bandwidth,
            "kde_grid": json.dumps(grid.tolist()),
            "kde_pdf": json.dumps(pdf_vals.tolist()),
            "kde_cdf": json.dumps(cdf_vals.tolist()),
        }
    else:
        dist = spec
        if dist_name == "beta":
            # Escalamos al [0,1] antes de ajustar beta
            min_v, max_v = float(sample.min()), float(sample.max())
            if math.isclose(max_v, min_v):
                raise ValueError("Datos constantes, no se puede ajustar beta")
            scaled = (sample - min_v) / (max_v - min_v)
            eps = 1e-6
            scaled = np.clip(scaled, eps, 1.0 - eps)
            params = dist.fit(scaled, floc=0, fscale=1)
            shape_params = params[:-2]
            loc = min_v
            scale = max_v - min_v
            fitted_dist = stats.beta(*shape_params, loc=loc, scale=scale)
            ks_stat, ks_pvalue = stats.kstest(sample, fitted_dist.cdf)
            params = (*shape_params, loc, scale)
            extra = {
                "beta_loc": loc,
                "beta_scale": scale,
            }
        else:
            params = dist.fit(sample)
            fitted_dist = dist(*params)
            ks_stat, ks_pvalue = stats.kstest(sample, fitted_dist.cdf)
            extra = {}
    return (
        params,
        float(ks_stat),
        float(ks_pvalue),
        extra,
    )


def fit_segments(
    df: pd.DataFrame,
    *,
    segments: Iterable[str],
    years: Sequence[str],
    distributions: Sequence[str],
    bandwidth_search: bool = False,
    bandwidth_grid: Sequence[float] | None = None,
) -> list[FitResult]:
    results: list[FitResult] = []
    for segment in segments:
        for year in years:
            try:
                sample = _load_segment_year(df, segment, year)
            except Exception as exc:
                print(f"[WARN] Segmento {segment} año {year}: {exc}")
                continue
            for dist_name in distributions:
                try:
                    params, ks_stat, ks_pvalue, extra = _fit_distribution(
                        sample,
                        dist_name,
                        bandwidth_search=bandwidth_search,
                        bandwidth_grid=bandwidth_grid,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] No se pudo ajustar {dist_name} para segmento {segment} año {year}: {exc}"
                    )
                    continue
                results.append(
                    FitResult(
                        segment=segment,
                        year=str(year),
                        distribution=dist_name,
                        params=params,
                        ks_stat=ks_stat,
                        ks_pvalue=ks_pvalue,
                        sample_size=int(sample.size),
                        extra=extra,
                    )
                )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ajusta distribuciones a los multiplicadores por segmento y calcula KS"
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="CSV segmentado con columnas scenario,2026-2030,score,segment",
    )
    parser.add_argument(
        "--segments",
        help="Segmentos a procesar, formato pesimista,regular,optimista",
    )
    parser.add_argument(
        "--distributions",
        help=(
            "Distribuciones a probar (normal,lognormal,gamma,weibull,beta,kde). "
            "Default: todas"
        ),
    )
    parser.add_argument(
        "--years",
        help="Columnas de años a incluir (default: todas las disponibles)",
    )
    parser.add_argument(
        "--kde-search",
        action="store_true",
        help="Realiza búsqueda de bandwidth (score máximo) en KDE",
    )
    parser.add_argument(
        "--kde-bandwidths",
        help="Lista de bandwidths a evaluar para KDE (ej: 0.01,0.02,0.05)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Archivo CSV de resultados (default: *_fit_summary.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"Archivo no encontrado: {args.input}")
        return 1

    segmented = pd.read_csv(args.input)

    if "segment" not in segmented.columns:
        parser.error("El archivo no contiene columna 'segment'")
        return 1

    if args.segments:
        segments = [seg.strip() for seg in args.segments.split(",") if seg.strip()]
    else:
        segments = [str(seg) for seg in segmented["segment"].dropna().unique()]
    if not segments:
        parser.error("No se encontraron segmentos en el archivo")
        return 1

    year_columns = (
        [col.strip() for col in args.years.split(",") if col.strip()]
        if args.years
        else [c for c in segmented.columns if c.isdigit()]
    )
    if not year_columns:
        parser.error("No se encontraron columnas de año para analizar")
        return 1
    missing_years = [col for col in year_columns if col not in segmented.columns]
    if missing_years:
        parser.error(
            "Columnas no encontradas en el archivo: " + ", ".join(missing_years)
        )
        return 1

    distributions = _parse_distributions(args.distributions)

    bandwidth_search = bool(args.kde_search)
    bandwidth_grid = None
    if args.kde_bandwidths:
        try:
            bandwidth_grid = [
                float(val.strip())
                for val in args.kde_bandwidths.split(",")
                if val.strip()
            ]
        except ValueError as exc:
            parser.error(f"Bandwidths inválidos: {args.kde_bandwidths}")
            return 1
        if not bandwidth_grid:
            bandwidth_grid = None
        else:
            bandwidth_search = True

    results = fit_segments(
        segmented,
        segments=segments,
        years=year_columns,
        distributions=distributions,
        bandwidth_search=bandwidth_search,
        bandwidth_grid=bandwidth_grid,
    )

    if not results:
        parser.error("No se pudo ajustar ninguna distribución")
        return 1

    results_df = pd.DataFrame([res.to_dict() for res in results])

    best_df = (
        results_df.sort_values(
            ["segment", "year", "ks_pvalue", "ks_stat"],
            ascending=[True, True, False, True],
        )
        .groupby(["segment", "year"], as_index=False)
        .first()
    )

    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + "_fit_summary.csv")
    best_df.to_csv(output, index=False)

    print("Ajustes (mejor por segmento y año) registrados en:", output)
    print()
    print(best_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
