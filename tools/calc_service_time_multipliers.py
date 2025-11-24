#!/usr/bin/env python
"""
Calcula multiplicadores por (lane_type, profile) comparando los tiempos de servicio
teóricos (outputs_teoricos) con las predicciones del modelo actual (service_time_model.json).
El resultado se guarda en service_time/service_time_multipliers.csv y puede ser
usado por el simulador para escalar las muestras automáticamente.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.engine import (
    DAY_TYPE_BY_DAYNUM,
    ServiceTimeFactorModel,
    SERVICE_TIME_MODEL_JSON,
    LANE_NAME_NORMALIZATION,
)


def _norm_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _norm_lane(value) -> str:
    s = _norm_text(value)
    return LANE_NAME_NORMALIZATION.get(s, s)


def iter_customer_files(root: Path):
    pattern = re.compile(r"Day-(\d+)")
    for folder in sorted(root.glob("Week-*-Day-*")):
        match = pattern.search(folder.name)
        if not match:
            continue
        day_num = int(match.group(1))
        day_type = DAY_TYPE_BY_DAYNUM.get(day_num, "desconocido")
        csv_path = folder / "customers.csv"
        if csv_path.exists():
            yield csv_path, day_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula multiplicadores de tiempo de servicio usando outputs teóricos.")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"), help="Carpeta con Week-*-Day-* teóricos")
    parser.add_argument("--model", type=Path, default=SERVICE_TIME_MODEL_JSON, help="Ruta al service_time_model.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/service_time/service_time_multipliers.csv"),
        help="CSV de salida con los multiplicadores",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No existe el modelo de tiempos de servicio: {args.model}")

    model = ServiceTimeFactorModel(args.model)

    agg = defaultdict(lambda: {"actual_sum": 0.0, "pred_sum": 0.0, "count": 0})
    total_served_rows = 0
    total_used_rows = 0
    required_cols = ["profile", "priority", "payment_method", "lane_type", "items", "service_time_s", "outcome"]

    prediction_errors = 0

    for csv_path, day_type in iter_customer_files(args.root):
        try:
            df = pd.read_csv(csv_path, usecols=required_cols)
        except Exception as exc:
            print(f"Advertencia: no se pudo leer {csv_path}: {exc}")
            continue
        df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
        served = df[df["outcome"] == "served"].copy()
        if served.empty:
            continue
        served["profile"] = served["profile"].map(_norm_text)
        served["priority"] = served["priority"].map(_norm_text)
        served["payment_method"] = served["payment_method"].map(_norm_text)
        served["lane_type"] = served["lane_type"].map(_norm_lane)
        served["items"] = pd.to_numeric(served["items"], errors="coerce").fillna(0.0)
        served["service_time_s"] = pd.to_numeric(served["service_time_s"], errors="coerce")
        served = served[np.isfinite(served["service_time_s"])]
        served = served[served["lane_type"].astype(bool) & served["profile"].astype(bool)]
        total_served_rows += len(served)
        if served.empty:
            continue

        for _, row in served.iterrows():
            lane = row["lane_type"]
            profile = row["profile"]
            try:
                pred = model.expected_value(
                    profile=profile,
                    priority=row["priority"],
                    payment_method=row["payment_method"],
                    day_type=day_type,
                    lane_type=lane,
                    items=float(row["items"]),
                )
            except Exception as exc:
                prediction_errors += 1
                if prediction_errors <= 5:
                    print(f"[WARN] No se pudo predecir fila en {csv_path} ({exc})")
                continue
            key = (lane, profile)
            agg[key]["actual_sum"] += float(row["service_time_s"])
            agg[key]["pred_sum"] += float(pred)
            agg[key]["count"] += 1
            total_used_rows += 1

    if not agg:
        print(f"No se acumularon entradas válidas; filas_served={total_served_rows}, filas_usadas={total_used_rows}, errores_pred={prediction_errors}")
        raise SystemExit("No se encontraron registros válidos para calcular multiplicadores.")

    print(f"Filas servidas procesadas: {total_served_rows}, filas utilizadas: {total_used_rows}")

    rows = []
    for (lane, profile), stats in agg.items():
        if stats["pred_sum"] <= 0 or stats["count"] == 0:
            continue
        actual_mean = stats["actual_sum"] / stats["count"]
        pred_mean = stats["pred_sum"] / stats["count"]
        factor = actual_mean / pred_mean if pred_mean > 0 else 1.0
        rows.append(
            {
                "lane_type": lane,
                "profile": profile,
                "count": stats["count"],
                "service_time_mean_teo": actual_mean,
                "service_time_mean_pred": pred_mean,
                "multiplier": factor,
            }
        )

    if not rows:
        raise SystemExit("No se pudieron calcular multiplicadores (verifica los datos teóricos).")

    out_df = pd.DataFrame(rows).sort_values(["lane_type", "profile"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Multiplicadores guardados en {args.output} ({len(out_df)} combinaciones)")


if __name__ == "__main__":
    main()
