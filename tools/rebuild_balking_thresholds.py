from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.balking.lane_state_logger import (
    build_lane_catalog,
    build_lane_log,
    infer_day_metadata,
)

LANE_TYPES = ("regular", "express", "priority", "self_checkout")


def _load_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["customer_id", "profile", "priority", "items", "payment_method"]
    arrivals = (
        df[df["event_type"] == "arrival"][cols]
        .drop_duplicates("customer_id")
        .set_index("customer_id")
    )
    return arrivals


def _eligible_lane_types(profile: str, priority: str, payment: str, items: int) -> List[str]:
    profile = (profile or "").strip().lower()
    priority = (priority or "").strip().lower()
    payment = (payment or "").strip().lower()
    eligible = ["regular", "priority"]
    if items <= 10:
        eligible.append("express")
    if (
        payment == "card"
        and priority != "reduced_mobility"
        and items <= 15
    ):
        eligible.append("self_checkout")
    # Remove duplicates while preserving order
    seen = set()
    filtered: List[str] = []
    for lane_type in eligible:
        if lane_type not in seen:
            filtered.append(lane_type)
            seen.add(lane_type)
    return filtered


def _build_lane_time_series(lane_log: pd.DataFrame) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Dict[str, List[str]],
]:
    per_lane: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    lanes_by_type: Dict[str, List[str]] = defaultdict(list)

    for lane_name, grp in lane_log.groupby("lane_name"):
        timestamps = grp["timestamp_s"].to_numpy(dtype=float)
        waiting = grp["waiting_after_event"].fillna(0).to_numpy(dtype=float)
        per_lane[lane_name] = (timestamps, waiting)
        lane_type = grp["lane_type"].iloc[0]
        lanes_by_type[lane_type].append(lane_name)

    return per_lane, lanes_by_type


def _min_waiting_for_type(
    lane_type: str,
    ts: float,
    per_lane: Dict[str, Tuple[np.ndarray, np.ndarray]],
    lanes_by_type: Dict[str, List[str]],
) -> Optional[float]:
    lane_names = lanes_by_type.get(lane_type, [])
    if not lane_names:
        return None
    waits: List[float] = []
    for lane_name in lane_names:
        timestamps, waiting = per_lane[lane_name]
        idx = np.searchsorted(timestamps, ts, side="right") - 1
        if idx < 0:
            waits.append(0.0)
        else:
            waits.append(float(max(0.0, waiting[idx])))
    return min(waits) if waits else None


def analyze_day(time_log_path: Path) -> List[dict]:
    df = pd.read_csv(time_log_path)
    balks = df[df["event_type"] == "balked"].copy()
    if balks.empty:
        return []
    arrivals = _load_arrivals(df)

    metadata = infer_day_metadata(time_log_path)
    day_type = metadata["day_type"]
    week = metadata["week"]
    day_label = metadata["day_label"]
    lane_catalog = build_lane_catalog(day_type)
    lane_log = build_lane_log(
        df[
            [
                "timestamp_s",
                "event_type",
                "customer_id",
                "lane_name",
                "lane_type",
                "effective_queue_length",
            ]
        ].copy(),
        lane_catalog,
    )
    per_lane, lanes_by_type = _build_lane_time_series(lane_log)

    records: List[dict] = []
    for row in balks.itertuples(index=False):
        customer_id = row.customer_id
        ts = float(row.timestamp_s)
        try:
            arrival = arrivals.loc[customer_id]
        except KeyError:
            continue
        profile = str(arrival["profile"]).strip().lower()
        priority = str(arrival["priority"]).strip().lower()
        payment = str(arrival["payment_method"]).strip().lower()
        items = int(arrival["items"])

        eligibles = _eligible_lane_types(profile, priority, payment, items)
        if not eligibles:
            continue

        best_len = math.inf
        best_type = None
        for lane_type in eligibles:
            q = _min_waiting_for_type(lane_type, ts, per_lane, lanes_by_type)
            if q is None:
                continue
            if q < best_len:
                best_len = q
                best_type = lane_type

        if best_type is None or not np.isfinite(best_len):
            continue

        records.append(
            {
                "week": week,
                "day_label": day_label,
                "timestamp_s": ts,
                "profile": profile,
                "priority": priority,
                "payment_method": payment,
                "items": items,
                "day_type": day_type,
                "lane_type": best_type,
                "min_queue_length": best_len,
            }
        )
    return records


def summarize(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    summary = (
        df.groupby(["profile", "priority", "payment_method", "day_type", "lane_type"])["min_queue_length"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min_queue="min",
            max_queue="max",
        )
        .reset_index()
    )
    summary["estimated_threshold"] = summary["median"].round(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate balking queue thresholds from theoretical outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs_teoricos"),
        help="Root folder containing Week-*/Day-*/time_log.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/balking_thresholds.csv"),
        help="Output CSV path for the aggregated thresholds.",
    )
    args = parser.parse_args()

    time_logs = sorted(args.root.glob("Week-*-Day-*/time_log.csv"))
    if not time_logs:
        raise FileNotFoundError(f"No time_log.csv files found under {args.root}")

    all_records: List[dict] = []
    for path in time_logs:
        all_records.extend(analyze_day(path))

    if not all_records:
        print("No balk events found; nothing to export.")
        return

    summary = summarize(all_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved balking thresholds to {args.output.resolve()}")


if __name__ == "__main__":
    main()
