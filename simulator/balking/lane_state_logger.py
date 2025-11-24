from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

import pandas as pd


DAY_TYPE_BY_DAY_LABEL = {
    1: "tipo_1",
    2: "tipo_1",
    3: "tipo_2",
    4: "tipo_1",
    5: "tipo_2",
    6: "tipo_2",
    7: "tipo_3",
}

DEFAULT_LANE_COUNTS = {
    "tipo_1": {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    "tipo_2": {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    "tipo_3": {"regular": 15, "express": 3, "priority": 2, "self_checkout": 5},
}

STATEFUL_EVENTS = {"select_lane", "start_service", "end_service", "abandoned"}


@dataclass
class LaneState:
    waiting: int = 0
    in_service: int = 0
    lane_type: Optional[str] = None

    def snapshot(self) -> Dict[str, int]:
        return {
            "waiting_after_event": self.waiting,
            "in_service_after_event": self.in_service,
            "total_in_lane": self.waiting + self.in_service,
        }


def build_lane_catalog(day_type: str) -> Dict[str, str]:
    counts = DEFAULT_LANE_COUNTS.get(day_type)
    if counts is None:
        raise ValueError(f"Unsupported day_type {day_type!r}")

    catalog: Dict[str, str] = {}
    for lane_type, amount in counts.items():
        for idx in range(1, amount + 1):
            catalog[f"{lane_type}-{idx}"] = lane_type
    return catalog


def infer_day_metadata(time_log_path: Path) -> Dict[str, int]:
    """Extract week/day label from folder name such as Week-1-Day-3."""
    folder = time_log_path.parent.name
    parts = folder.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Expected folder name like 'Week-X-Day-Y', got '{folder}' instead"
        )
    try:
        week = int(parts[1])
        day_label = int(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid folder spec '{folder}'") from exc
    return {"week": week, "day_label": day_label, "day_type": DAY_TYPE_BY_DAY_LABEL[day_label]}


def normalize_lane_type(lane_name: str) -> str:
    return lane_name.split("-", 1)[0] if "-" in lane_name else lane_name


def update_lane_state(state: LaneState, event: str) -> None:
    if event == "select_lane":
        state.waiting += 1
    elif event == "start_service":
        if state.waiting > 0:
            state.waiting -= 1
        state.in_service = 1
    elif event == "end_service":
        state.in_service = max(0, state.in_service - 1)
    elif event == "abandoned":
        state.waiting = max(0, state.waiting - 1)


def build_lane_log(df: pd.DataFrame, lane_catalog: Dict[str, str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    states: MutableMapping[str, LaneState] = defaultdict(LaneState)

    initial_ts = float(df["timestamp_s"].min())
    for lane_name, lane_type in lane_catalog.items():
        lane_state = states[lane_name]
        lane_state.lane_type = lane_type
        records.append(
            {
                "timestamp_s": initial_ts,
                "event_type": "init_state",
                "lane_name": lane_name,
                "lane_type": lane_type,
                "customer_id": pd.NA,
                "waiting_after_event": 0,
                "in_service_after_event": 0,
                "total_in_lane": 0,
                "effective_queue_length": pd.NA,
            }
        )

    for row in df.itertuples(index=False):
        if row.event_type not in STATEFUL_EVENTS:
            continue
        lane_name = getattr(row, "lane_name", "") or None
        if not lane_name:
            continue
        lane_state = states[lane_name]
        lane_type = getattr(row, "lane_type", "") or lane_state.lane_type
        if not lane_type:
            lane_type = lane_catalog.get(lane_name) or normalize_lane_type(lane_name)
        lane_state.lane_type = lane_type

        update_lane_state(lane_state, row.event_type)

        records.append(
            {
                "timestamp_s": float(row.timestamp_s),
                "event_type": row.event_type,
                "lane_name": lane_name,
                "lane_type": lane_type,
                "customer_id": getattr(row, "customer_id", None),
                "effective_queue_length": getattr(row, "effective_queue_length", None)
                or pd.NA,
                **lane_state.snapshot(),
            }
        )

    log_df = pd.DataFrame.from_records(records)
    log_df.sort_values(["timestamp_s", "lane_name", "event_type"], inplace=True)
    log_df.reset_index(drop=True, inplace=True)
    return log_df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-lane queue length log from simulation time_log.csv"
    )
    parser.add_argument(
        "--time-log",
        required=True,
        type=Path,
        help="Path to the time_log.csv of a specific day",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults next to the source file)",
    )
    parser.add_argument(
        "--day-type",
        choices=DEFAULT_LANE_COUNTS.keys(),
        default=None,
        help="Override day_type inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    time_log_path: Path = args.time_log
    if not time_log_path.exists():
        raise FileNotFoundError(time_log_path)

    metadata = infer_day_metadata(time_log_path)
    day_type = args.day_type or metadata["day_type"]
    lane_catalog = build_lane_catalog(day_type)

    df = pd.read_csv(
        time_log_path,
        usecols=[
            "timestamp_s",
            "event_type",
            "customer_id",
            "lane_name",
            "lane_type",
            "effective_queue_length",
        ],
    ).sort_values("timestamp_s")

    lane_log = build_lane_log(df, lane_catalog)

    if args.output:
        output_path = args.output
    else:
        output_dir = time_log_path.parent
        output_path = output_dir / f"{time_log_path.parent.name}_lane_log.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lane_log.to_csv(output_path, index=False)
    print(f"Lane log saved to {output_path}")


if __name__ == "__main__":
    main()

