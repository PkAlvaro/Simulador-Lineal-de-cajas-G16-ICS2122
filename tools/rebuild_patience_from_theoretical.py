#!/usr/bin/env python
"""
Genera un archivo paramétrico de paciencia (`patience/patience_distribution_profile_priority_payment_day.csv`)
basado exclusivamente en los outputs teóricos. Usa distribuciones exponenciales simples
cuya media coincide con el tiempo de espera promedio observado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAY_TYPE_BY_DAYNUM = {
    1: "tipo_1",
    2: "tipo_1",
    3: "tipo_2",
    4: "tipo_1",
    5: "tipo_2",
    6: "tipo_2",
    7: "tipo_3",
}


def iter_customer_files(root: Path):
    for folder in sorted(root.glob("Week-*-Day-*")):
        name = folder.name
        try:
            day_num = int(name.split("-")[-1])
        except ValueError:
            continue
        day_type = DAY_TYPE_BY_DAYNUM.get(day_num, "tipo_1")
        csv_path = folder / "customers.csv"
        if csv_path.exists():
            yield csv_path, day_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra la distribución paramétrica de paciencia desde datos teóricos.")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"), help="Carpeta con Week-*-Day-*")
    parser.add_argument("--output", type=Path, default=Path("patience/patience_distribution_profile_priority_payment_day.csv"), help="CSV de salida")
    parser.add_argument("--min-mean", type=float, default=0.5, help="Media mínima (segundos) para evitar valores cero")
    args = parser.parse_args()

    records: list[pd.DataFrame] = []
    required = ["profile", "priority", "payment_method", "wait_time_s", "outcome"]

    for csv_path, day_type in iter_customer_files(args.root):
        try:
            df = pd.read_csv(csv_path, usecols=required)
        except Exception as exc:
            print(f"[WARN] No se pudo leer {csv_path}: {exc}")
            continue
        df["day_type"] = day_type
        records.append(df)

    if not records:
        raise SystemExit("No se encontraron customers.csv en la ruta indicada.")

    df = pd.concat(records, ignore_index=True)
    df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
    served = df[df["outcome"] == "served"].copy()
    served["wait_time_s"] = pd.to_numeric(served["wait_time_s"], errors="coerce")
    served = served[np.isfinite(served["wait_time_s"])]
    served["profile"] = served["profile"].astype(str).str.strip().str.lower()
    served["priority"] = served["priority"].astype(str).str.strip().str.lower()
    served["payment_method"] = served["payment_method"].astype(str).str.strip().str.lower()
    served["day_type"] = served["day_type"].astype(str).str.strip().str.lower()

    grouped = served.groupby(["profile", "priority", "payment_method", "day_type"])
    rows = []
    for combo, subset in grouped:
        waits = subset["wait_time_s"].to_numpy(dtype=float)
        if waits.size == 0:
            continue
        mean_wait = float(max(np.mean(waits), args.min_mean))
        params = [0.0, mean_wait]
        rows.append(
            {
                "method": "param",
                "profile": combo[0],
                "priority": combo[1],
                "payment_method": combo[2],
                "day_type": combo[3],
                "n_obs": int(waits.size),
                "distribution": "exponential",
                "params": str(params),
                "ks_pvalue": "",
                "aic": "",
                "x_seconds": "",
                "density": "",
            }
        )

    if not rows:
        raise SystemExit("No se generó ningún registro de paciencia.")

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Archivo de paciencia actualizado ({len(out_df)} combinaciones) en {args.output}")


if __name__ == "__main__":
    main()
