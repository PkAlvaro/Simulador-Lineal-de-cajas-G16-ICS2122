#!/usr/bin/env python3

"""
Aggregate the queue-length tolerance (effective_queue_length) per customer
profile, payment method, and priority flag by scanning every `time_log.csv`
under `outputs_teoricos`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


Key = Tuple[str, str, str]
VALID_LANE_TYPES = {"regular", "express", "priority", "self_checkout"}
BALK_EVENTS = {"balk", "balked"}
BALK_REASONS = {"queue_too_long"}


class _Distribution:
    def __init__(self, values: List[float]) -> None:
        self.values = values

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        q = min(max(q, 0.0), 1.0)
        if len(self.values) == 1:
            return self.values[0]
        idx = (len(self.values) - 1) * q
        lower = int(idx)
        upper = min(lower + 1, len(self.values) - 1)
        frac = idx - lower
        lo = self.values[lower]
        hi = self.values[upper]
        return lo + (hi - lo) * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute queue-length tolerance statistics for every "
            "profile/payment/priority combination found in the time_log.csv files."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="outputs_teoricos",
        type=Path,
        help="Directory that contains the day folders with time_log.csv files.",
    )
    parser.add_argument(
        "--output",
        default="queue_tolerance_by_profile_payment_priority.csv",
        type=Path,
        help="Path of the CSV file to write with the aggregated statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every processed time_log.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list(iter_time_logs(args.base_dir))
    if not files:
        raise SystemExit(f"No time_log.csv files found under {args.base_dir}")

    stats = collect_queue_lengths(files, verbose=args.verbose)
    rows = summarize(stats)
    write_csv(args.output, rows)
    if args.verbose:
        print(f"Wrote {len(rows)} rows to {args.output}")


def iter_time_logs(base_dir: Path) -> Iterable[Path]:
    base_dir = base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory {base_dir} does not exist")

    for day_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        time_log = day_dir / "time_log.csv"
        if time_log.exists():
            yield time_log


def collect_queue_lengths(files: Iterable[Path], verbose: bool = False) -> Dict[Key, List[float]]:
    stats: Dict[Key, List[float]] = defaultdict(list)

    for time_log in files:
        if verbose:
            print(f"Processing {time_log}")
        for key, queue_len in iter_balk_queue_lengths(time_log):
            stats[key].append(queue_len)

    return stats


def iter_balk_queue_lengths(time_log: Path):
    lane_states: Dict[str, Dict[str, Any]] = {}
    lane_types: Dict[str, str] = {}
    customers: Dict[str, Dict[str, str | int]] = {}

    with time_log.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = (row.get("event_type") or "").strip()
            customer_id = (row.get("customer_id") or "").strip()
            if not customer_id:
                continue

            if event == "arrival":
                customers[customer_id] = {
                    "profile": normalize(row.get("profile"), default="unknown_profile"),
                    "payment_method": normalize(row.get("payment_method"), default="unknown_payment"),
                    "priority": normalize(row.get("priority"), default="no_priority"),
                    "items": safe_int(row.get("items")),
                }
            elif event == "queue_request":
                lane = (row.get("lane_name") or "").strip()
                ensure_lane(lane, row.get("lane_type"), lane_states, lane_types)
                if lane:
                    lane_states[lane]["waiting"].append(customer_id)
            elif event == "start_service":
                lane = (row.get("lane_name") or "").strip()
                state = ensure_lane(lane, row.get("lane_type"), lane_states, lane_types)
                if not lane or state is None:
                    continue
                waiting = state["waiting"]
                if waiting and waiting[0] == customer_id:
                    waiting.popleft()
                else:
                    try:
                        waiting.remove(customer_id)
                    except ValueError:
                        pass
                state["in_service"] = customer_id
            elif event == "end_service":
                lane = (row.get("lane_name") or "").strip()
                state = ensure_lane(lane, row.get("lane_type"), lane_states, lane_types)
                if lane and state is not None:
                    state["in_service"] = None
                customers.pop(customer_id, None)
            elif event == "abandon":
                lane = (row.get("lane_name") or "").strip()
                state = ensure_lane(lane, row.get("lane_type"), lane_states, lane_types)
                if lane and state is not None:
                    try:
                        state["waiting"].remove(customer_id)
                    except ValueError:
                        pass
                customers.pop(customer_id, None)
            elif event in BALK_EVENTS:
                reason = normalize(row.get("reason"), default="")
                if reason not in BALK_REASONS:
                    continue
                customer = customers.get(customer_id)
                if not customer:
                    continue
                queue_len = best_lane_queue_length(customer, lane_states, lane_types)
                if queue_len is None:
                    continue
                key = (
                    customer["profile"],
                    customer["payment_method"],
                    customer["priority"],
                )
                yield key, float(queue_len)
                customers.pop(customer_id, None)


def ensure_lane(
    lane_name: str,
    lane_type_value: str | None,
    lane_states: Dict[str, Dict[str, Any]],
    lane_types: Dict[str, str],
) -> Dict[str, Any] | None:
    lane_name = (lane_name or "").strip()
    if not lane_name:
        return None
    if lane_name not in lane_states:
        lane_states[lane_name] = {
            "waiting": deque(),
            "in_service": None,
        }
    if lane_name not in lane_types:
        lane_type_norm = normalize(lane_type_value, default="")
        if not lane_type_norm:
            lane_type_norm = infer_lane_type(lane_name)
        lane_types[lane_name] = lane_type_norm
    return lane_states[lane_name]


def infer_lane_type(lane_name: str) -> str:
    lane_name = lane_name.lower()
    if lane_name.startswith("regular"):
        return "regular"
    if lane_name.startswith("express"):
        return "express"
    if lane_name.startswith("priority"):
        return "priority"
    if lane_name.startswith("self_checkout"):
        return "self_checkout"
    return ""


def best_lane_queue_length(
    customer: Dict[str, str | int],
    lane_states: Dict[str, Dict[str, Any]],
    lane_types: Dict[str, str],
) -> float | None:
    best = None
    for lane_name, state in lane_states.items():
        lane_type = lane_types.get(lane_name, "")
        if lane_type not in VALID_LANE_TYPES:
            continue
        if not is_eligible(customer, lane_type):
            continue
        waiting_len = len(state["waiting"])
        in_service = 1 if state["in_service"] else 0
        queue_len = waiting_len + in_service
        candidate = (queue_len, lane_name)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return float(best[0])


def is_eligible(customer: Dict[str, str | int], lane_type: str) -> bool:
    priority = str(customer.get("priority", "")).lower()
    payment = str(customer.get("payment_method", "")).lower()
    items = int(customer.get("items", 0))

    if lane_type == "express" and items > 10:
        return False
    if lane_type == "priority" and priority == "no_priority":
        return False
    if lane_type == "self_checkout":
        if payment != "card":
            return False
        if priority == "reduced_mobility":
            return False
        if items > 15:
            return False
    return True


def summarize(stats: Dict[Key, List[float]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for (profile, payment, priority), values in sorted(stats.items()):
        values.sort()
        dist = _Distribution(values)
        count = len(values)
        mean = sum(values) / count if count else 0.0
        median = statistics.median(values) if count else 0.0
        stdev = statistics.pstdev(values) if count > 1 else 0.0
        rows.append(
            {
                "profile": profile,
                "payment_method": payment,
                "priority": priority,
                "observations": str(count),
                "mean_queue_length": f"{mean:.3f}",
                "median_queue_length": f"{median:.3f}",
                "std_queue_length": f"{stdev:.3f}",
                "min_queue_length": f"{values[0]:.3f}",
                "max_queue_length": f"{values[-1]:.3f}",
                "p90_queue_length": f"{dist.percentile(0.9):.3f}",
                "p95_queue_length": f"{dist.percentile(0.95):.3f}",
            }
        )

    return rows


def write_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path = output_path.expanduser()
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise SystemExit("No queue-length data found to write.")

    fieldnames = [
        "profile",
        "payment_method",
        "priority",
        "observations",
        "mean_queue_length",
        "median_queue_length",
        "std_queue_length",
        "min_queue_length",
        "max_queue_length",
        "p90_queue_length",
        "p95_queue_length",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(value: str | None, default: str) -> str:
    value = (value or "").strip()
    return value if value else default


def safe_int(value: str | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


if __name__ == "__main__":
    main()
