#!/usr/bin/env python
"""
Recalibra las tasas de llegada para el simulador a partir de los outputs teóricos.

Lee todos los `Week-*-Day-*` dentro de `outputs_teoricos`, toma los eventos
`arrival` del `time_log.csv` y construye series λ(t) por combinación de:

    - tipo de día (tipo_1, tipo_2, tipo_3)
    - perfil
    - prioridad
    - medio de pago
    - bucket de ítems (opcional)

El resultado se guarda como archivos .npz (uno por perfil) en `arrivals_npz/`.
Cada archivo contiene:
    * `bin_left_s`: vector con el inicio de cada bin (en segundos desde la apertura)
    * `bin_size_s`: tamaño del bin en segundos
    * `keys`: listado de combinaciones codificadas como strings
      `"{day_type}|{priority}|{payment}|{items_bucket}"`
    * `lambdas`: matriz (#keys x #bins) con λ por minuto para cada combinación.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

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

OPEN_S = 8 * 3600
CLOSE_S = 22 * 3600
JORNADA = CLOSE_S - OPEN_S


def parse_item_bins(bins: Optional[str]) -> Optional[list[float]]:
    if not bins:
        return None
    values: list[float] = []
    for token in bins.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise SystemExit(f"--item-bins contiene un valor no numérico: {token!r}")
    values = sorted(set(v for v in values if v > 0))
    return values or None


def bucket_label(items: float, edges: Optional[list[float]]) -> str:
    if not edges:
        return "all"
    lower = 0.0
    for edge in edges:
        if items < edge:
            return f"{int(lower)}-{int(edge)}"
        lower = edge
    return f"{int(edges[-1])}+"


def iter_time_logs(root: Path) -> Iterable[tuple[Path, int]]:
    pattern = re.compile(r"Day-(\d+)")
    for folder in sorted(root.glob("Week-*-Day-*")):
        match = pattern.search(folder.name)
        if not match:
            continue
        day_num = int(match.group(1))
        time_log = folder / "time_log.csv"
        if time_log.exists():
            yield time_log, day_num


def build_lambda_series(
    root: Path,
    bin_size_s: int,
    item_edges: Optional[list[float]],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, Counter[str], dict[tuple[str, str, str, str], float]]:
    bin_edges = np.arange(0, JORNADA + bin_size_s, bin_size_s, dtype=float)
    counts_dict: dict[tuple[str, str, str, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(bin_edges.size - 1, dtype=np.float64)
    )
    day_counts: Counter[str] = Counter()
    total_arrivals_profile_day: dict[tuple[str, str, str, str], float] = defaultdict(float)

    usecols = [
        "timestamp_s",
        "event_type",
        "profile",
        "priority",
        "items",
        "payment_method",
    ]

    for time_log, day_num in iter_time_logs(root):
        day_type = DAY_TYPE_BY_DAYNUM.get(day_num, "desconocido")
        day_counts[day_type] += 1
        try:
            df = pd.read_csv(time_log, usecols=usecols)
        except Exception as exc:
            print(f"Advertencia: no se pudo leer {time_log.name}: {exc}")
            continue

        arrivals = df[df["event_type"] == "arrival"].copy()
        if arrivals.empty:
            continue

        arrivals["timestamp_s"] = pd.to_numeric(arrivals["timestamp_s"], errors="coerce")
        arrivals = arrivals[np.isfinite(arrivals["timestamp_s"])]

        arrivals["profile"] = arrivals["profile"].astype(str).str.strip().str.lower()
        arrivals["priority"] = arrivals["priority"].astype(str).str.strip().str.lower()
        arrivals["payment_method"] = arrivals["payment_method"].astype(str).str.strip().str.lower()
        arrivals["items"] = pd.to_numeric(arrivals["items"], errors="coerce").fillna(0)
        arrivals["item_bucket"] = arrivals["items"].map(lambda x: bucket_label(x, item_edges))

        group_cols = ["profile", "priority", "payment_method", "item_bucket"]
        for combo, subset in arrivals.groupby(group_cols):
            times = subset["timestamp_s"].to_numpy(dtype=float)
            times = np.clip(times, 0, JORNADA)
            hist, _ = np.histogram(times, bins=bin_edges)
            key = combo + (day_type,)
            counts_dict[key] += hist
            total_arrivals_profile_day[(combo[0], combo[1], combo[2], day_type)] += float(hist.sum())

    results: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for (profile, priority, payment, bucket, day_type), counts in counts_dict.items():
        days = max(day_counts.get(day_type, 1), 1)
        lambdas = counts / days / (bin_size_s / 60.0)
        key = f"{day_type}|{priority}|{payment}|{bucket}"
        results[profile][key] = lambdas.astype(np.float32)

    return results, bin_edges[:-1].astype(np.float32), day_counts, total_arrivals_profile_day


def build_little_reference(root: Path, day_counts: Counter[str]) -> pd.DataFrame:
    pattern = re.compile(r"Day-(\d+)")
    served_accum: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wait_sum": 0.0})
    for folder in sorted(root.glob("Week-*-Day-*")):
        match = pattern.search(folder.name)
        if not match:
            continue
        day_num = int(match.group(1))
        day_type = DAY_TYPE_BY_DAYNUM.get(day_num, "desconocido")
        customers_path = folder / "customers.csv"
        if not customers_path.exists():
            continue
        try:
            df = pd.read_csv(customers_path, usecols=["profile", "priority", "payment_method", "wait_time_s", "outcome"])
        except Exception as exc:
            print(f"Advertencia: no se pudo leer {customers_path.name}: {exc}")
            continue
        served = df[df["outcome"] == "served"].copy()
        if served.empty:
            continue
        served["profile"] = served["profile"].astype(str).str.strip().str.lower()
        served["priority"] = served["priority"].astype(str).str.strip().str.lower()
        served["payment_method"] = served["payment_method"].astype(str).str.strip().str.lower()
        served["wait_time_s"] = pd.to_numeric(served["wait_time_s"], errors="coerce")
        for (profile, priority, payment), subset in served.groupby(["profile", "priority", "payment_method"]):
            waits = subset["wait_time_s"].dropna()
            if waits.empty:
                continue
            key = (day_type, profile, priority, payment)
            served_accum[key]["count"] += float(len(waits))
            served_accum[key]["wait_sum"] += float(waits.sum())

    rows = []
    for (day_type, profile, priority, payment), stats in served_accum.items():
        total = stats["count"]
        wait_sum = stats["wait_sum"]
        if total <= 0:
            continue
        wait_mean = wait_sum / total
        days = float(max(day_counts.get(day_type, 1), 1))
        served_per_day = total / days
        lambda_per_s = served_per_day / float(JORNADA or 1)
        L_value = lambda_per_s * wait_mean
        rows.append(
            {
                "dia_tipo": day_type,
                "profile": profile,
                "priority": priority,
                "payment_method": payment,
                "days_count": days,
                "served_total": total,
                "served_per_day": served_per_day,
                "wait_mean_s": wait_mean,
                "lambda_per_s": lambda_per_s,
                "L_value": L_value,
            }
        )
    return pd.DataFrame(rows)


def apply_little_scaling(
    results: dict[str, dict[str, np.ndarray]],
    totals_profile_day: dict[tuple[str, str, str, str], float],
    day_counts: Counter[str],
    little_df: pd.DataFrame,
) -> None:
    if little_df.empty:
        return
    jornada = float(JORNADA or 1)
    little_df = little_df.copy()
    little_df["profile"] = little_df["profile"].astype(str).str.strip().str.lower()
    little_df["dia_tipo"] = little_df["dia_tipo"].astype(str).str.strip().str.lower()
    for _, row in little_df.iterrows():
        profile = row["profile"]
        day_type = row["dia_tipo"]
        priority = row["priority"]
        payment = row["payment_method"]
        target_lambda = float(row.get("lambda_per_s", 0.0) or 0.0)
        if target_lambda <= 0:
            continue
        days = float(max(day_counts.get(day_type, 1), 1))
        current_total = float(totals_profile_day.get((profile, priority, payment, day_type), 0.0))
        current_lambda = (current_total / days) / jornada if current_total > 0 else 0.0
        if current_lambda <= 0:
            continue
        factor = target_lambda / current_lambda
        if not np.isfinite(factor) or factor <= 0:
            continue
        profile_entries = results.get(profile)
        if not profile_entries:
            continue
        for key, series in profile_entries.items():
            parts = key.split("|")
            if len(parts) < 4:
                continue
            day_key, prio_key, pay_key = parts[0], parts[1], parts[2]
            if day_key != day_type or prio_key != priority or pay_key != payment:
                continue
            new_series = (series.astype(np.float32) * factor).astype(np.float32)
            profile_entries[key] = new_series


def persist_npz(
    results: dict[str, dict[str, np.ndarray]],
    bin_left: np.ndarray,
    bin_size_s: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for profile, combo_dict in results.items():
        if not combo_dict:
            continue
        keys = np.array(list(combo_dict.keys()))
        lambda_matrix = np.vstack([combo_dict[k] for k in keys]).astype(np.float32)
        out_path = output_dir / f"lambda_{profile}.npz"
        np.savez_compressed(
            out_path,
            keys=keys,
            lambdas=lambda_matrix,
            bin_left_s=bin_left,
            bin_size_s=float(bin_size_s),
        )
        print(f"  -> {out_path} ({len(keys)} combinaciones)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra lambdas para el PPNH a partir de outputs teóricos.")
    parser.add_argument("--root", type=Path, default=Path("outputs_teoricos"), help="Carpeta raíz con Week-*-Day-*")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arrivals_npz"), help="Destino de los .npz")
    parser.add_argument("--bin-size", type=int, default=60, help="Tamaño del bin (segundos) para estimar lambda")
    parser.add_argument(
        "--item-bins",
        type=str,
        default="",
        help="Lista separada por comas con los límites superiores de cada bucket de ítems (ej. '10,20,40').",
    )
    parser.add_argument(
        "--apply-little",
        action="store_true",
        help="Escala las tasas por (perfil, dia_tipo) para que la lambda teórica (Little) se respete.",
    )
    args = parser.parse_args()

    item_edges = parse_item_bins(args.item_bins)
    if args.bin_size <= 0:
        raise SystemExit("--bin-size debe ser > 0")

    print(f"Procesando datos desde {args.root.resolve()}")
    results, bin_left, day_counts, totals_profile_day = build_lambda_series(args.root, args.bin_size, item_edges)
    if not results:
        raise SystemExit("No se encontraron llegadas en los archivos indicados.")

    little_df = build_little_reference(args.root, day_counts)
    if args.apply_little and not little_df.empty:
        print("Aplicando escalamiento por Ley de Little...")
        apply_little_scaling(results, totals_profile_day, day_counts, little_df)

    print(f"Generando archivos .npz en {args.output_dir.resolve()}")
    persist_npz(results, bin_left, args.bin_size, args.output_dir)
    if not little_df.empty:
        little_path = args.output_dir / "little_reference.csv"
        little_df.to_csv(little_path, index=False)
        print(f"Referencia de Little guardada en {little_path}")
    print("Listo.")


if __name__ == "__main__":
    main()
