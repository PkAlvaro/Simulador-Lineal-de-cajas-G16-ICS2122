from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import reporting
from simulator import engine
from simulator.engine import (
    DayType,
    LaneType,
    reset_checkout_item_limits,
    set_checkout_item_limits,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No se encontró el archivo: {path}") from exc


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and "scenarios" in data:
        data = data["scenarios"]
    if not isinstance(data, list):
        raise ValueError("El archivo de escenarios debe contener una lista")
    scenarios: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        scenarios.append(
            {
                "label": label,
                "express_max_items": entry.get("express_max_items"),
                "sco_max_items": entry.get("sco_max_items"),
            }
        )
    if not scenarios:
        raise ValueError("El archivo de escenarios no contiene entradas válidas")
    return scenarios


def load_policy_config(path: Path) -> Dict[DayType, Dict[LaneType, int]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError("El archivo de políticas debe ser un JSON con claves por tipo de día")
    mapping: Dict[DayType, Dict[LaneType, int]] = {}
    for key, raw_counts in data.items():
        try:
            dt = DayType[key.upper()]
        except KeyError as exc:
            raise ValueError(f"Tipo de día desconocido en la política: {key}") from exc
        if not isinstance(raw_counts, dict):
            raise ValueError(f"Los conteos para {key} deben venir como objeto JSON")
        try:
            counts = {
                LaneType.REGULAR: int(raw_counts["regular"]),
                LaneType.EXPRESS: int(raw_counts["express"]),
                LaneType.PRIORITY: int(raw_counts["priority"]),
                LaneType.SCO: int(raw_counts["self_checkout"]),
            }
        except KeyError as exc:  # pragma: no cover - validación
            raise ValueError(
                f"Faltan conteos para {key}; se requieren regular, express, priority, self_checkout"
            ) from exc
        mapping[dt] = counts
    if not mapping:
        raise ValueError("El archivo de políticas no contiene configuraciones válidas")
    return mapping


def apply_policy_config(policy_map: Dict[DayType, Dict[LaneType, int]]) -> None:
    for dt, counts in policy_map.items():
        engine.CURRENT_LANE_POLICY[dt] = {
            LaneType.REGULAR: max(0, int(counts.get(LaneType.REGULAR, 0))),
            LaneType.EXPRESS: max(0, int(counts.get(LaneType.EXPRESS, 0))),
            LaneType.PRIORITY: max(0, int(counts.get(LaneType.PRIORITY, 0))),
            LaneType.SCO: max(0, int(counts.get(LaneType.SCO, 0))),
        }
    # Recalcular costos para reflejar la política instalada
    summary = engine._compute_lane_cost_summary(engine.CURRENT_LANE_POLICY, engine.LANE_COST_SPECS)  # type: ignore[attr-defined]
    engine.LANE_COST_SUMMARY = summary  # type: ignore[attr-defined]
    engine.LANE_COST_TOTAL_ANNUAL = float(summary.get("total_cost", 0.0))  # type: ignore[attr-defined]
    engine.LANE_COST_PER_WEEK = float(summary.get("per_week_cost", 0.0))  # type: ignore[attr-defined]
    engine.LANE_COST_BY_DAYTYPE = summary.get("by_day_type", {})  # type: ignore[attr-defined]
    engine.LANE_COST_BY_DAYTYPE_STR = {  # type: ignore[attr-defined]
        (dt.value if isinstance(dt, DayType) else str(dt)): float(cost)
        for dt, cost in engine.LANE_COST_BY_DAYTYPE.items()  # type: ignore[attr-defined]
    }


def run_sensitivity(
    *,
    scenarios_path: Path,
    policy_path: Path,
    weeks: int,
    outputs_root: Path,
    resultados_root: Path,
) -> None:
    scenarios = load_scenarios(scenarios_path)
    policy_config = load_policy_config(policy_path)
    apply_policy_config(policy_config)

    weeks = max(1, int(weeks))
    for scenario in scenarios:
        label = scenario["label"]
        print(f"\n[SENS] Escenario '{label}'...")
        express_limit = scenario.get("express_max_items")
        sco_limit = scenario.get("sco_max_items")
        if express_limit is None and sco_limit is None:
            reset_checkout_item_limits()
            print("[SENS] Límites de ítems por defecto.")
        else:
            set_checkout_item_limits(express_limit, sco_limit)
            exp_lim, sco_lim = engine.get_checkout_item_limits()
            print(f"[SENS] Límites aplicados: express={exp_lim}, SCO={sco_lim}")

        scenario_outputs = outputs_root / label
        scenario_resultados = resultados_root / label

        reporting.run_full_workflow(
            num_weeks_sample=weeks,
            output_root=scenario_outputs,
            resultados_root=scenario_resultados,
        )
        print(
            f"[SENS] Escenario '{label}' completado. "
            f"Resultados en: {scenario_resultados.resolve()}"
        )

    reset_checkout_item_limits()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba sensibilidad de KPIs variando límites de ítems y usando políticas fijas."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Archivo JSON con lista de escenarios (label, express_max_items, sco_max_items).",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        required=True,
        help="Archivo JSON con configuraciones de cajas por tipo de día.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=1,
        help="Semanas a simular por escenario (como en opción 1 del simulador).",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs_sensibilidad"),
        help="Carpeta base para los outputs de simulación.",
    )
    parser.add_argument(
        "--resultados-root",
        type=Path,
        default=Path("resultados_sensibilidad"),
        help="Carpeta base para los reportes KPI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_sensitivity(
        scenarios_path=args.scenarios,
        policy_path=args.policy_config,
        weeks=args.weeks,
        outputs_root=args.outputs_root,
        resultados_root=args.resultados_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
