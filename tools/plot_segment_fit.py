from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SUPPORTED_DISTS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "beta": stats.beta,
}


def _validate_year(df: pd.DataFrame, year: str) -> str:
    if not year:
        raise ValueError("Debe especificar un año a graficar")
    year = year.strip()
    if year not in df.columns:
        available = ", ".join(sorted(col for col in df.columns if col.isdigit()))
        raise ValueError(f"Año '{year}' no encontrado. Disponibles: {available}")
    return year


def _load_segment_series(
    segmented: pd.DataFrame, segment: str, year: str
) -> np.ndarray:
    mask = segmented["segment"].str.lower() == segment.lower()
    if not mask.any():
        available = ", ".join(sorted(segmented["segment"].dropna().unique()))
        raise ValueError(
            f"Segmento '{segment}' no encontrado. Disponibles: {available}"
        )
    values = segmented.loc[mask, year].to_numpy(dtype=float, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(
            "El segmento no contiene valores numéricos válidos para ese año"
        )
    return values


def _extract_params(row: pd.Series) -> tuple[float, ...]:
    param_cols = sorted(
        [c for c in row.index if c.startswith("param_") and pd.notna(row[c])],
        key=lambda name: int(name.split("_")[1]),
    )
    return tuple(float(row[c]) for c in param_cols)


def _build_distribution(row: pd.Series):
    dist_name = row["distribution"]
    params = _extract_params(row)
    if dist_name.startswith("kde_"):
        grid = np.array(json.loads(row["kde_grid"]), dtype=float)
        pdf = np.array(json.loads(row["kde_pdf"]), dtype=float)
        cdf = np.array(json.loads(row["kde_cdf"]), dtype=float)
        return dist_name, params, grid, pdf, cdf
    dist = SUPPORTED_DISTS.get(dist_name)
    if dist is None:
        raise ValueError(f"Distribución '{dist_name}' no soportada para graficar")
    return dist_name, params, None, None, None


def _evaluate_pdf_cdf(
    dist_name: str, params: tuple[float, ...], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if dist_name == "beta":
        pdf_vals = stats.beta.pdf(x, *params)
        cdf_vals = stats.beta.cdf(x, *params)
    else:
        dist = SUPPORTED_DISTS[dist_name]
        pdf_vals = dist.pdf(x, *params)
        cdf_vals = dist.cdf(x, *params)
    return pdf_vals, cdf_vals


def plot_segment_fit(
    segmented_path: Path,
    summary_path: Path,
    segment: str,
    year: str,
    *,
    bins: int = 30,
    output: Path | None = None,
    show: bool = False,
) -> Path:
    seg_df = pd.read_csv(segmented_path)
    year = _validate_year(seg_df, year)
    data = _load_segment_series(seg_df, segment, year)

    summary_df = pd.read_csv(summary_path)
    row = summary_df.loc[
        (summary_df["segment"].str.lower() == segment.lower())
        & (summary_df["year"].astype(str) == year)
    ]
    if row.empty:
        available = ", ".join(
            sorted(
                f"{seg} ({yr})"
                for seg, yr in summary_df[["segment", "year"]]
                .dropna()
                .itertuples(index=False)
            )
        )
        raise ValueError(
            f"Segmento '{segment}' año {year} no está en el resumen. Disponibles: {available}"
        )
    row = row.iloc[0]

    dist_name, params, grid, pdf_grid, cdf_grid = _build_distribution(row)

    fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_pdf.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.4,
        color="tab:blue",
        label="Histograma (densidad)",
    )

    xs = np.linspace(data.min(), data.max(), 512)

    if dist_name.startswith("kde_") and grid is not None:
        pdf_vals = np.interp(xs, grid, pdf_grid)
        cdf_vals = np.interp(xs, grid, cdf_grid)
        label_pdf = f"KDE ({row.get('kde_kernel', dist_name)})"
    else:
        pdf_vals, cdf_vals = _evaluate_pdf_cdf(dist_name, params, xs)
        label_pdf = f"{dist_name}"

    ax_pdf.plot(xs, pdf_vals, color="tab:red", label=label_pdf)
    ax_pdf.set_ylabel("densidad")
    ax_pdf.legend()
    ax_pdf.grid(True, alpha=0.2)

    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, data_sorted.size + 1) / data_sorted.size
    ax_cdf.step(
        data_sorted, empirical_cdf, where="post", color="tab:blue", label="CDF empírica"
    )
    ax_cdf.plot(xs, cdf_vals, color="tab:red", label="CDF modelo")
    ax_cdf.set_xlabel("factor de demanda")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.2)

    title_segment = row["segment"]
    ax_pdf.set_title(
        f"Segmento: {title_segment} | Año: {year} | Distribución: {label_pdf}"
    )

    fig.tight_layout()

    if output is None:
        safe_segment = str(segment).replace("/", "_").replace("\\", "_")
        output = segmented_path.with_name(f"{safe_segment}_{year}_fit_plot.png")
    fig.savefig(output, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grafica histograma, PDF y CDF ajustada para un segmento"
    )
    parser.add_argument("segmented", type=Path, help="CSV segmentado (escenarios)")
    parser.add_argument("summary", type=Path, help="CSV con el resumen de ajustes")
    parser.add_argument(
        "--segment", required=True, help="Nombre del segmento a graficar"
    )
    parser.add_argument(
        "--year", required=True, help="Año a graficar (columna del CSV)"
    )
    parser.add_argument("--bins", type=int, default=30, help="Bins del histograma")
    parser.add_argument("--output", type=Path, help="Ruta del archivo PNG de salida")
    parser.add_argument(
        "--show", action="store_true", help="Mostrar la figura tras guardarla"
    )
    args = parser.parse_args(argv)

    try:
        output_path = plot_segment_fit(
            args.segmented,
            args.summary,
            args.segment,
            year=args.year,
            bins=args.bins,
            output=args.output,
            show=args.show,
        )
    except Exception as exc:
        parser.error(str(exc))
        return 1

    print(f"Figura guardada en {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
