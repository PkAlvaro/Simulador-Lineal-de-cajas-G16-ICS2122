#!/usr/bin/env python
"""
Reconstruye el modelo de tiempos de servicio a partir de los outputs teoricos.

ENFOQUE HIBRIDO:
1. CAJAS REGULARES: Regresion Lineal Multivariada (Items + Priority + Payment).
   - Rationale: Cajeros entrenados, varianza constante, efectos aditivos claros.
   
2. SELF-CHECKOUT (SCO) y EXPRESS: Modelo de Tasa Estocastica (Stochastic Rate).
   - Rationale: Velocidad dictada por el cliente (SCO) o dominada por setup/pago variable (Express).
   - Metodo:
     a) Estimar Setup Time fijo (intercepto).
     b) Calcular 'Segundos por Item' para cada transaccion: (Time - Setup) / Items.
     c) Ajustar distribucion Lognormal a estos 'Segundos por Item'.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
        for col in ["profile", "priority", "payment_method"]:
            df[col] = df[col].astype(str).str.strip().str.lower()
            
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
        frames.append(df)
        
    if not frames:
        raise SystemExit("No se hallaron registros validos en outputs_teoricos")
    return pd.concat(frames, ignore_index=True)


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


def analyze_residuals(residuals: np.ndarray, residual_pct: float) -> dict:
    if len(residuals) < 20:
        return None
    abs_values = np.abs(residuals)
    threshold = float(np.quantile(abs_values, residual_pct))
    mask = abs_values > threshold
    outliers = residuals[mask]
    inliers = residuals[~mask]
    return {
        "prob_outlier": float(len(outliers) / len(residuals)) if len(residuals) else 0.0,
        "abs_threshold": threshold,
        "n_obs": int(len(residuals)),
        "inlier": fit_best_distribution(inliers),
        "outlier": fit_best_distribution(outliers) if len(outliers) >= 5 else None,
    }


def fit_stochastic_rate_model(subset: pd.DataFrame) -> dict:
    """
    Ajusta el modelo estocastico:
    Time = Setup + Items * Rate
    Rate ~ Lognormal
    """
    # 1. Estimar Setup Time (Intercepto robusto)
    X = sm.add_constant(subset["items"])
    pre_model = sm.OLS(subset["service_time_s"], X).fit()
    setup_time = max(5.0, pre_model.params.get("const", 15.0)) 
    
    # 2. Calcular tasas observadas (Seconds per Item)
    valid = subset[subset["items"] > 0].copy()
    valid["implied_rate"] = (valid["service_time_s"] - setup_time) / valid["items"]
    
    # Filtramos tasas negativas o irreales
    valid = valid[(valid["implied_rate"] > 0.1) & (valid["implied_rate"] < 120)]
    
    rates = valid["implied_rate"].values
    
    # 3. Ajustar Lognormal
    shape, loc, scale = stats.lognorm.fit(rates, floc=0)
    
    mu_log = np.log(scale)
    sigma_log = shape
    
    print(f"     [Stochastic] Setup={setup_time:.2f}s, Rate LogN(mu={mu_log:.2f}, sigma={sigma_log:.2f})")
    print(f"     Media Tasa: {np.mean(rates):.2f} s/item, Mediana: {np.median(rates):.2f} s/item")
    
    return {
        "model_method": "stochastic_rate",
        "params": {
            "setup_time": float(setup_time),
            "rate_dist": {
                "name": "lognorm",
                "mu": float(mu_log),
                "sigma": float(sigma_log)
            }
        },
        "stats": {
            "n_obs": int(len(rates)),
            "mean_rate": float(np.mean(rates))
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra el modelo de tiempos de servicio Hibrido")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"))
    parser.add_argument("--output", type=Path, default=Path("data/service_time/service_time_model.json"))
    parser.add_argument("--residual-pct", type=float, default=0.99)
    args = parser.parse_args()

    print(f"Cargando datos desde {args.root}...")
    df = load_dataset(args.root)
    
    models_by_lane = {}
    lane_types = df["lane_type"].unique()
    print(f"Tipos de caja encontrados: {lane_types}")
    
    for lane in lane_types:
        print(f"Ajustando modelo para: {lane}...")
        subset = df[df["lane_type"] == lane].copy()
        
        if len(subset) < 20:
            print(f"  -> Saltando {lane} por falta de datos")
            continue
            
        norm_lane = lane.lower().strip()
        is_sco = norm_lane in ["self_checkout", "sco", "autocaja"]
        is_express = "express" in norm_lane
        
        if is_sco or is_express:
            # --- MODELO ESTOCASTICO PARA SCO Y EXPRESS ---
            tipo = "SCO" if is_sco else "Express"
            print(f"  -> Detectado {tipo}: Usando ajuste de Tasa Estocastica")
            models_by_lane[lane] = fit_stochastic_rate_model(subset)
            
        else:
            # --- MODELO REGRESION MULTIVARIADA PARA REGULARES ---
            print("  -> Detectado Regular: Usando Regresion Multivariada")
            formula = "service_time_s ~ items + C(priority) + C(payment_method)"
            try:
                model = smf.ols(formula=formula, data=subset).fit()
            except Exception as e:
                print(f"  -> Error: {e}")
                continue
                
            coeffs = {name: float(val) for name, val in model.params.items()}
            r2 = float(model.rsquared)
            
            subset["prediction"] = model.predict(subset)
            subset["residual"] = subset["service_time_s"] - subset["prediction"]
            residual_model = analyze_residuals(subset["residual"].values, args.residual_pct)
            
            models_by_lane[lane] = {
                "model_method": "multivariate_regression",
                "coeffs": coeffs,
                "stats": {"r2": r2, "n_obs": int(len(subset))},
                "residual_model": residual_model
            }
            print(f"     R2={r2:.4f}")

    payload = {
        "type": "hybrid_multivariate_stochastic",
        "models": models_by_lane,
        "meta": {
            "source": str(args.root),
            "description": "Hibrido: SCO/Express=StochasticRate, Otros=MultivariateRegression"
        }
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    main()
