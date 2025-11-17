#!/usr/bin/env python
"""
Reconstruye el modelo de tiempos de servicio a partir de los outputs teoricos.

Pasos:
1. Ajusta una regresion lineal con items y dummies para perfil, prioridad,
   medio de pago, tipo de caja y tipo de dia.
2. Calcula residuales y marca como outliers a los que superen un umbral absoluto
   definido por un percentil configurable.
3. Ajusta distribuciones para residuales inlier/outlier usando KS + AIC para
   escoger la mejor alternativa disponible.
4. Exporta un JSON con coeficientes y modelos de residuales para uso directo
   dentro del simulador.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DAY_TYPE_BY_DAYNUM = {
    1: "tipo_1",
    2: "tipo_1",
    3: "tipo_2",
    4: "tipo_1",
    5: "tipo_2",
    6: "tipo_2",
    7: "tipo_3",
}

CAT_COLS = ["profile", "priority", "payment_method", "lane_type", "day_type"]
CANDIDATE_DISTS = ["norm", "t", "lognorm", "gamma"]
POSITIVE_SUPPORT_DISTS = {"lognorm", "gamma"}
LANE_NORMALIZATION = {"sco": "self_checkout", "self_checkout": "self_checkout"}


def iter_customer_files(root: Path) -> Iterable[Tuple[Path, str]]:
    pattern = re.compile(r"Day-(\d+)", re.IGNORECASE)
    for folder in sorted(root.glob("Week-*-Day-*")):
        match = pattern.search(folder.name)
        if not match:
            continue
        day_num = int(match.group(1))
        day_type = DAY_TYPE_BY_DAYNUM.get(day_num, "desconocido")
        csv_path = folder / "customers.csv"
        if csv_path.exists():
            yield csv_path, day_type


def load_dataset(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = [
        "profile",
        "priority",
        "payment_method",
        "lane_type",
        "items",
        "service_time_s",
        "outcome",
    ]
    for csv_path, day_type in iter_customer_files(root):
        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as exc:
            print(f"Advertencia: no se pudo leer {csv_path} ({exc})")
            continue
        df = df[df["outcome"] == "served"].copy()
        if df.empty:
            continue
        df["day_type"] = day_type
        df["profile"] = df["profile"].astype(str).str.strip().str.lower()
        df["priority"] = df["priority"].astype(str).str.strip().str.lower()
        df["payment_method"] = df["payment_method"].astype(str).str.strip().str.lower()
        df["lane_type"] = (
            df["lane_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(LANE_NORMALIZATION)
        )
        df["items"] = pd.to_numeric(df["items"], errors="coerce")
        df["service_time_s"] = pd.to_numeric(df["service_time_s"], errors="coerce")
        df = df[np.isfinite(df["items"]) & np.isfinite(df["service_time_s"])]
        if df.empty:
            continue
        frames.append(df[CAT_COLS + ["items", "service_time_s"]])
    if not frames:
        raise SystemExit("No se hallaron registros validos en outputs_teoricos")
    return pd.concat(frames, ignore_index=True)


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, dict]]:
    cat_info: Dict[str, dict] = {}
    X = pd.DataFrame({"items": df["items"].astype(float)})
    for col in CAT_COLS:
        cats = sorted(df[col].unique())
        baseline = cats[0]
        dummies = pd.get_dummies(df[col], prefix=col)
        base_col = f"{col}_{baseline}"
        if base_col in dummies:
            dummies = dummies.drop(columns=base_col)
        X = pd.concat([X, dummies], axis=1)
        cat_info[col] = {"baseline": baseline, "levels": cats}
    X = sm.add_constant(X)
    X = X.astype(float)
    return X, cat_info


def build_coefficients(params: pd.Series, cat_info: dict) -> dict:
    coeffs = {
        "intercept": float(params.get("const", 0.0)),
        "items": float(params.get("items", 0.0)),
        "categories": {},
    }
    for col, info in cat_info.items():
        entry = {"baseline": info["baseline"], "coeffs": {}}
        for level in info["levels"]:
            if level == info["baseline"]:
                continue
            col_name = f"{col}_{level}"
            if col_name in params:
                entry["coeffs"][level] = float(params[col_name])
        coeffs["categories"][col] = entry
    return coeffs


def fit_best_distribution(data: np.ndarray) -> dict:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"name": "norm", "params": [0.0, 1.0]}
    best = None
    for name in CANDIDATE_DISTS:
        if name in POSITIVE_SUPPORT_DISTS and np.any(data <= 0):
            continue
        try:
            dist = getattr(stats, name)
            params = dist.fit(data)
            ks_stat, ks_p = stats.kstest(data, name, params)
            logpdf = dist.logpdf(data, *params)
            if not np.all(np.isfinite(logpdf)):
                continue
            loglik = float(np.sum(logpdf))
            k = len(params)
            aic = 2 * k - 2 * loglik
        except Exception:
            continue
        score = (ks_p, -aic)
        if best is None or score > best[0]:
            best = (score, name, params, ks_p, aic)
    if best is None:
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=0) or 1.0)
        return {"name": "norm", "params": [mean, std]}
    _, name, params, ks_p, aic = best
    return {
        "name": name,
        "params": [float(p) for p in params],
        "ks_p": float(ks_p),
        "aic": float(aic),
    }


def analyze_residuals(df: pd.DataFrame, residual_pct: float) -> tuple[dict, dict]:
    models: dict[str, dict] = {}
    all_inliers: list[float] = []
    all_outliers: list[float] = []
    thresholds: list[float] = []
    total = 0
    total_out = 0
    grouped = df.groupby(["profile", "priority", "payment_method", "day_type", "lane_type"])
    for combo, subset in grouped:
        values = subset["residual"].to_numpy()
        if len(values) < 20:
            continue
        abs_values = np.abs(values)
        threshold = float(np.quantile(abs_values, residual_pct))
        mask = abs_values > threshold
        outliers = values[mask]
        inliers = values[~mask]
        all_inliers.extend(inliers.tolist())
        all_outliers.extend(outliers.tolist())
        total += len(values)
        total_out += len(outliers)
        thresholds.append(threshold)
        entry = {
            "prob_outlier": float(len(outliers) / len(values)) if len(values) else 0.0,
            "abs_threshold": threshold,
            "n_obs": int(len(values)),
            "inlier": fit_best_distribution(inliers),
            "outlier": fit_best_distribution(outliers) if len(outliers) >= 5 else None,
        }
        models["|".join(combo)] = entry
    defaults = {
        "prob_outlier": float(total_out / total) if total else 0.0,
        "abs_threshold": float(np.median(thresholds)) if thresholds else 0.0,
        "n_obs": int(total),
        "inlier": fit_best_distribution(np.asarray(all_inliers)),
        "outlier": fit_best_distribution(np.asarray(all_outliers)) if all_outliers else None,
    }
    return models, defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra el modelo de tiempos de servicio")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"))
    parser.add_argument("--output", type=Path, default=Path("service_time/service_time_model.json"))
    parser.add_argument("--residual-pct", type=float, default=0.99, help="Percentil absoluto para detectar outliers")
    args = parser.parse_args()

    df = load_dataset(args.root)
    X, cat_info = build_design_matrix(df)
    model = sm.OLS(df["service_time_s"], X).fit()
    coeffs = build_coefficients(model.params, cat_info)
    df["prediction"] = model.predict(X)
    df["residual"] = df["service_time_s"] - df["prediction"]
    residual_models, defaults = analyze_residuals(df, args.residual_pct)

    payload = {
        "coefficients": coeffs,
        "residual_models": residual_models,
        "defaults": defaults,
        "r2": float(model.rsquared),
        "n_obs": int(df.shape[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Modelo guardado en {args.output} (R2={model.rsquared:.4f}, N={df.shape[0]})")


if __name__ == "__main__":
    main()
