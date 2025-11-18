from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_YEAR_COLUMNS: tuple[str, ...] = ("2026", "2027", "2028", "2029", "2030")


class DemandProjectionError(RuntimeError):
    """Errores relacionados con la carga o segmentación de demanda."""


def _normalize_year_columns(columns: Iterable[str]) -> list[str]:
    known = {c for c in DEFAULT_YEAR_COLUMNS}
    return [str(c).strip() for c in columns if str(c).strip() in known]


@dataclass(frozen=True)
class DemandSegmentationResult:
    data: pd.DataFrame
    scores: pd.Series
    categories: pd.Series

    def summary(self) -> pd.DataFrame:
        years = [c for c in DEFAULT_YEAR_COLUMNS if c in self.data.columns]
        agg_map = {"scenario": "count"}
        agg_map.update({year: ["mean", "std", "min", "max"] for year in years})
        grouped = (
            self.data.assign(segment=self.categories)
            .groupby("segment", dropna=False)
            .agg(agg_map)
        )
        grouped.columns = [
            "_".join(col).strip("_") for col in grouped.columns.to_flat_index()
        ]
        grouped = grouped.rename(columns={"scenario_count": "scenario_count"})
        return grouped.sort_index()


def load_projection(
    path: str | Path, *, year_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DemandProjectionError(f"Archivo no encontrado: {path}")
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        raise DemandProjectionError(
            "La columna 'scenario' es obligatoria en la proyección"
        )
    if year_columns is None:
        year_columns = [c for c in DEFAULT_YEAR_COLUMNS if c in df.columns]
    else:
        year_columns = _normalize_year_columns(year_columns)
    if not year_columns:
        raise DemandProjectionError(
            "No se encontraron columnas de años válidas en el archivo"
        )
    df = df[["scenario", *year_columns]].copy()
    return df


def compute_scores(
    df: pd.DataFrame,
    *,
    method: str = "geometric",
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    years = [c for c in df.columns if c != "scenario"]
    values = df[years].to_numpy(dtype=float, copy=False)
    if weights:
        weights_vec = np.array(
            [float(weights.get(year, 0.0)) for year in years], dtype=float
        )
        if np.all(weights_vec == 0.0):
            weights_vec = None
    else:
        weights_vec = None

    if weights_vec is None:
        weights_vec = np.full(values.shape[1], 1.0 / values.shape[1])
    else:
        weights_vec = np.maximum(weights_vec, 0.0)
        total = weights_vec.sum()
        if total == 0.0:
            weights_vec = np.full(values.shape[1], 1.0 / values.shape[1])
        else:
            weights_vec = weights_vec / total

    if method not in {"geometric", "arithmetic"}:
        raise DemandProjectionError(
            "Método de score desconocido. Use 'geometric' o 'arithmetic'."
        )

    if method == "geometric":
        if np.any(values <= 0):
            raise DemandProjectionError(
                "La media geométrica requiere factores positivos."
            )
        log_vals = np.log(values)
        scores = np.exp(log_vals @ weights_vec)
    else:
        scores = values @ weights_vec

    return pd.Series(scores, index=df.index, name="score")


def assign_categories(
    scores: pd.Series,
    *,
    quantiles: Sequence[float] = (0.33, 0.66),
    labels: Sequence[str] = ("pesimista", "regular", "optimista"),
) -> pd.Series:
    if len(labels) != len(quantiles) + 1:
        raise DemandProjectionError("La cantidad de etiquetas debe ser quantiles+1")
    try:
        categories = pd.qcut(
            scores, q=[0.0, *quantiles, 1.0], labels=labels, duplicates="drop"
        )
    except ValueError as exc:
        raise DemandProjectionError(str(exc)) from exc
    return categories.astype(str)


def attach_segmentation(
    df: pd.DataFrame,
    scores: pd.Series,
    categories: pd.Series,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["score"] = scores.values
    enriched["segment"] = categories.values
    return enriched


def segment_projection(
    path: str | Path,
    *,
    year_columns: Sequence[str] | None = None,
    method: str = "geometric",
    weights: Mapping[str, float] | None = None,
    quantiles: Sequence[float] = (0.33, 0.66),
    labels: Sequence[str] = ("pesimista", "regular", "optimista"),
) -> DemandSegmentationResult:
    df = load_projection(path, year_columns=year_columns)
    scores = compute_scores(df, method=method, weights=weights)
    categories = assign_categories(scores, quantiles=quantiles, labels=labels)
    enriched = attach_segmentation(df, scores, categories)
    return DemandSegmentationResult(data=enriched, scores=scores, categories=categories)


_DIST_MAP = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "beta": stats.beta,
}


def _kde_expectation(grid_json: str, pdf_json: str) -> float | None:
    try:
        grid = np.array(json.loads(grid_json), dtype=float)
        pdf = np.array(json.loads(pdf_json), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if grid.size == 0 or pdf.size == 0 or grid.size != pdf.size:
        return None
    pdf = np.maximum(pdf, 0.0)
    total = pdf.sum()
    if total <= 0:
        return None
    return float(np.dot(grid, pdf) / total)


def _distribution_expectation(name: str, params: Sequence[float]) -> float | None:
    dist = _DIST_MAP.get(name.lower())
    if dist is None:
        return None
    try:
        return float(dist.mean(*params))
    except Exception:
        return None


def load_segment_expectations(
    segmented_path: str | Path,
    fit_summary_path: str | Path | None = None,
    *,
    segments: Sequence[str] | None = None,
    years: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    seg_path = Path(segmented_path)
    seg_df = pd.read_csv(seg_path)

    if "segment" not in seg_df.columns:
        raise DemandProjectionError(
            "El archivo segmentado debe contener columna 'segment'"
        )

    if years is None:
        years = [c for c in DEFAULT_YEAR_COLUMNS if c in seg_df.columns]
    else:
        years = [c for c in years if c in seg_df.columns]
    if not years:
        raise DemandProjectionError(
            "No se encontraron columnas de años válidas en el archivo segmentado"
        )

    if segments is None:
        segments = sorted(seg_df["segment"].dropna().unique().tolist())

    expectations: dict[str, dict[str, float]] = {
        str(segment): {} for segment in segments
    }

    summary_df: pd.DataFrame | None = None
    if fit_summary_path is not None and Path(fit_summary_path).exists():
        summary_df = pd.read_csv(fit_summary_path)
        if "segment" not in summary_df.columns or "year" not in summary_df.columns:
            summary_df = None

    for segment in segments:
        segment_mask = seg_df["segment"].str.lower() == str(segment).lower()
        subset = seg_df.loc[segment_mask]
        if subset.empty:
            continue
        for year in years:
            base_mean = float(subset[year].mean())
            estimate = base_mean
            if summary_df is not None:
                row = summary_df[
                    (summary_df["segment"].str.lower() == str(segment).lower())
                    & (summary_df["year"].astype(str) == str(year))
                ]
                if not row.empty:
                    row = row.iloc[0]
                    dist_name = str(row.get("distribution", "")).lower()
                    if (
                        dist_name.startswith("kde")
                        and pd.notna(row.get("kde_grid"))
                        and pd.notna(row.get("kde_pdf"))
                    ):
                        value = _kde_expectation(
                            row.get("kde_grid"), row.get("kde_pdf")
                        )
                        if value is not None:
                            estimate = value
                    elif dist_name in _DIST_MAP:
                        params = [
                            float(row[c])
                            for c in row.index
                            if c.startswith("param_") and pd.notna(row[c])
                        ]
                        value = _distribution_expectation(dist_name, params)
                        if value is not None:
                            estimate = value
            expectations[str(segment)][str(year)] = float(estimate)

    return expectations
