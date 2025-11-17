#!/usr/bin/env python
"""
Construye distribuciones de paciencia por combinacion de
(profile, priority, payment_method, day_type) usando los outputs teoricos.

Permite dos metodos:
- KDE: genera una malla (x_seconds, density)
- Parametrico: escoge la mejor distribucion entre un set de candidatas
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import json
import numpy as np
import pandas as pd
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


def iter_customer_files(root: Path) -> Iterable[tuple[Path, str]]:
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
    base_cols = ["profile", "priority", "payment_method", "patience_s", "outcome"]
    for csv_path, day_type in iter_customer_files(root):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] No se pudo leer {csv_path}: {exc}")
            continue
        missing = [col for col in base_cols if col not in df.columns]
        if missing:
            print(f"[WARN] {csv_path} no contiene columnas requeridas {missing}, se omite.")
            continue
        df = df[base_cols].copy()
        df["day_type"] = day_type
        df["profile"] = df["profile"].astype(str).str.strip().str.lower()
        df["priority"] = df["priority"].astype(str).str.strip().str.lower()
        df["payment_method"] = df["payment_method"].astype(str).str.strip().str.lower()
        df["patience_s"] = pd.to_numeric(df["patience_s"], errors="coerce")
        df = df[np.isfinite(df["patience_s"]) & (df["patience_s"] > 0)]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise SystemExit("No se encontraron registros validos en outputs_teoricos.")
    return pd.concat(frames, ignore_index=True)


def compute_kde(values: np.ndarray, grid_size: int = 256, bandwidth: str | float | None = None):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return None
    x_min = float(max(0.0, data.min() * 0.9))
    x_max = float(data.max() * 1.1)
    if x_max <= x_min:
        x_max = x_min + 1.0
    grid = np.linspace(x_min, x_max, grid_size)
    try:
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        density = kde(grid)
    except Exception:
        return None
    density = np.clip(density, 0, None)
    area = np.trapz(density, grid)
    if area > 0:
        density = density / area
    return grid, density


PARAMETRIC_DISTRIBUTIONS = {
    "exponential": {"dist": stats.expon, "fit_kwargs": {"floc": 0}},
    "gamma": {"dist": stats.gamma, "fit_kwargs": {"floc": 0}},
    "lognorm": {"dist": stats.lognorm, "fit_kwargs": {"floc": 0}},
    "weibull_min": {"dist": stats.weibull_min, "fit_kwargs": {"floc": 0}},
    "invgauss": {"dist": stats.invgauss, "fit_kwargs": {"floc": 0}},
    "genpareto": {"dist": stats.genpareto, "fit_kwargs": {"floc": 0}},
    "norm": {"dist": stats.norm, "fit_kwargs": {}},
}


def fit_parametric_distribution(values: np.ndarray, allowed: set[str] | None = None) -> dict | None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return None
    if allowed:
        candidates = {name: PARAMETRIC_DISTRIBUTIONS[name] for name in PARAMETRIC_DISTRIBUTIONS if name in allowed}
    else:
        candidates = PARAMETRIC_DISTRIBUTIONS
    best = None
    n = data.size
    for name, spec in candidates.items():
        dist = spec["dist"]
        fit_kwargs = spec.get("fit_kwargs", {})
        try:
            params = dist.fit(data, **fit_kwargs)
            cdf = dist.cdf(data, *params)
            if np.any(~np.isfinite(cdf)):
                continue
            stat, pval = stats.kstest(data, name, params)
            logpdf = dist.logpdf(data, *params)
            if np.any(~np.isfinite(logpdf)):
                continue
            loglik = float(np.sum(logpdf))
            k = len(params)
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
        except Exception:
            continue
        score = (pval, -aic)
        if best is None or score > best[0]:
            param_list = [float(p) for p in params]
            best = (
                score,
                {
                    "distribution": name,
                    "params": param_list,
                    "ks_pvalue": float(pval),
                    "ks_statistic": float(stat),
                    "aic": float(aic),
                    "bic": float(bic),
                    "log_likelihood": loglik,
                },
            )
    return best[1] if best else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye distribuciones de paciencia")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"), help="Carpeta con outputs teoricos")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("patience/patience_distribution_profile_priority_payment_day.csv"),
        help="Ruta del CSV de salida",
    )
    parser.add_argument("--mode", choices=["kde", "param", "auto"], default="auto", help="Tipo de distribucion a construir")
    parser.add_argument("--grid-size", type=int, default=256, help="Numero de puntos en la malla KDE (si aplica)")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimo de observaciones por combinacion")
    parser.add_argument("--bw", type=str, default="scott", help="Bandwidth para gaussian_kde (solo mode=kde)")
    parser.add_argument("--min-param-pvalue", type=float, default=0.05, help="P-valor minimo para aceptar distribucion parametrica en modo auto")
    parser.add_argument(
        "--param-dists",
        type=str,
        default="",
        help="Lista separada por comas de distribuciones parametrizas a probar (por defecto se usan todas).",
    )
    args = parser.parse_args()

    df = load_dataset(args.root)
    allowed_dists = {name.strip().lower() for name in args.param_dists.split(",") if name.strip()}
    invalid = allowed_dists - set(PARAMETRIC_DISTRIBUTIONS)
    if invalid:
        raise SystemExit(f"Distribuciones desconocidas en --param-dists: {', '.join(sorted(invalid))}")
    allowed = allowed_dists or None

    grouped = df.groupby(["profile", "priority", "payment_method", "day_type"])
    rows = []
    skipped = 0
    for (profile, priority, payment, day_type), subset in grouped:
        values = subset["patience_s"].to_numpy()
        if len(values) < args.min_samples:
            skipped += 1
            continue
        chosen_mode = args.mode
        best_param = None
        kde_result = None
        if args.mode in {"param", "auto"}:
            best_param = fit_parametric_distribution(values, allowed=allowed)
        if args.mode in {"kde", "auto"}:
            kde_result = compute_kde(values, grid_size=args.grid_size, bandwidth=args.bw)
        if args.mode == "auto":
            if best_param and best_param.get("ks_pvalue", 0.0) >= args.min_param_pvalue:
                chosen_mode = "param"
            elif kde_result is not None:
                chosen_mode = "kde"
            elif best_param:
                chosen_mode = "param"
            else:
                skipped += 1
                continue
        if chosen_mode == "param":
            entry = best_param if best_param else None
            if not entry:
                skipped += 1
                continue
            rows.append(
                {
                    "method": "param",
                    "profile": profile,
                    "priority": priority,
                    "payment_method": payment,
                    "day_type": day_type,
                    "n_obs": len(values),
                    "distribution": entry["distribution"],
                    "params": json.dumps(entry["params"]),
                    "ks_pvalue": entry["ks_pvalue"],
                    "ks_statistic": entry["ks_statistic"],
                    "aic": entry["aic"],
                    "bic": entry["bic"],
                    "log_likelihood": entry["log_likelihood"],
                    "x_seconds": None,
                    "density": None,
                }
            )
        else:
            if kde_result is None:
                skipped += 1
                continue
            grid, density = kde_result
            for x, d in zip(grid, density):
                rows.append(
                    {
                        "method": "kde",
                        "profile": profile,
                        "priority": priority,
                        "payment_method": payment,
                        "day_type": day_type,
                        "n_obs": len(values),
                        "distribution": None,
                        "params": None,
                        "ks_pvalue": None,
                        "ks_statistic": None,
                        "aic": None,
                        "bic": None,
                        "log_likelihood": None,
                        "x_seconds": float(x),
                        "density": float(d),
                    }
                )

    if not rows:
        raise SystemExit("No se genero ninguna distribucion (quizas min-samples muy alto).")
    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    combos = out_df[["profile", "priority", "payment_method", "day_type"]].drop_duplicates().shape[0]
    info = f"{combos} combinaciones, {len(out_df)} filas"
    if args.mode == "param":
        info += " (una fila por combinacion)"
    print(f"Distribuciones ({args.mode}) guardadas en {args.output}: {info}. Omitidas: {skipped}.")


if __name__ == "__main__":
    main()
