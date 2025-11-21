from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import engine  # noqa: E402
from simulator.engine import DayType, reset_checkout_item_limits, set_checkout_item_limits  # noqa: E402
from simulator.policy_planner import (  # noqa: E402
    SequentialOptimizationResult,
    evaluate_base_performance,
    plan_multi_year_optimization,
)
from tools.export_plan_report import (  # noqa: E402
    build_initial_policies_dataframe,
    build_multipliers_dataframe,
    build_plan_detail_dataframe,
    export_excel_report,
    plot_objective_by_segment,
    plot_total_lanes,
)


def _parse_list(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    values = [part.strip() for part in arg.split(",") if part.strip()]
    return values or None


def _parse_day_types(values: Iterable[str] | None) -> list[DayType] | None:
    if not values:
        return None
    parsed: list[DayType] = []
    for raw in values:
        text = str(raw).strip()
        matched = None
        for dt_type in DayType:
            if text.lower() in {dt_type.name.lower(), dt_type.value.lower()}:
                matched = dt_type
                break
        if matched is None:
            raise argparse.ArgumentTypeError(f"No se reconoce DayType '{raw}'")
        parsed.append(matched)
    return parsed


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Archivo de escenarios no encontrado: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
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


def _export_report(
    *,
    result: SequentialOptimizationResult,
    output_dir: Path,
    compare_weeks: int,
    compare_reps: int,
) -> None:
    base_compare = evaluate_base_performance(
        result,
        num_weeks_sample=max(1, int(compare_weeks)),
        num_rep=max(1, int(compare_reps)),
    )
    detail_df = build_plan_detail_dataframe(result, base_compare=base_compare)
    multipliers_df = build_multipliers_dataframe(result)
    initial_df = build_initial_policies_dataframe(result)

    csv_path = output_dir / "plan_detalle.csv"
    detail_df.to_csv(csv_path, index=False)

    excel_path = output_dir / "plan_resumen.xlsx"
    export_excel_report(detail_df, multipliers_df, initial_df, excel_path)

    chart_obj = output_dir / "objetivo_por_segmento.png"
    plot_objective_by_segment(detail_df, chart_obj)

    chart_lanes = output_dir / "total_lanes_por_dia.png"
    plot_total_lanes(detail_df, chart_lanes)

    print(f"[SENS] Reporte guardado en {output_dir}")


def run_scenario(
    *,
    scenario: dict[str, Any],
    args: argparse.Namespace,
    segments: tuple[str, ...],
    years: list[str] | None,
    day_types: list[DayType] | None,
) -> None:
    label = scenario["label"]
    print(f"\n[SENS] Ejecutando escenario '{label}'...")
    express_limit = scenario.get("express_max_items")
    sco_limit = scenario.get("sco_max_items")
    if express_limit is not None or sco_limit is not None:
        set_checkout_item_limits(express_limit, sco_limit)
        print(
            f"[SENS] Límites aplicados: express={engine.get_checkout_item_limits()[0]}, "
            f"SCO={engine.get_checkout_item_limits()[1]}"
        )
    else:
        reset_checkout_item_limits()
        print("[SENS] Se utilizan límites por defecto.")

    report_root = Path(args.output_dir) / label
    timestamp = dt.datetime.now().strftime("plan_multi_year_%Y%m%d_%H%M%S")
    report_dir = report_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    optimizer_kwargs = json.loads(args.optimizer_kwargs) if args.optimizer_kwargs else {}

    result = plan_multi_year_optimization(
        segmented_path=args.segmented,
        fit_summary_path=args.fit_summary,
        years=years,
        segments=segments,
        day_types=day_types,
        max_workers=args.max_workers,
        optimizer_kwargs=optimizer_kwargs,
        single_investment=args.single_investment,
    )

    _export_report(
        result=result,
        output_dir=report_dir,
        compare_weeks=args.base_compare_weeks,
        compare_reps=args.base_compare_rep,
    )

    reset_checkout_item_limits()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta análisis de sensibilidad variando los límites de ítems."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Archivo JSON con lista de escenarios (label, express_max_items, sco_max_items).",
    )
    parser.add_argument(
        "--segmented",
        type=Path,
        default=Path("demand_projection_2026_2030_segmented.csv"),
        help="CSV segmentado con demanda (default: demand_projection_2026_2030_segmented.csv).",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=Path("demand_projection_2026_2030_segmented_fit_summary.csv"),
        help="CSV con ajustes estadísticos (default: *_fit_summary.csv).",
    )
    parser.add_argument(
        "--segments",
        help="Segmentos a incluir, formato pesimista,regular,optimista.",
    )
    parser.add_argument(
        "--years",
        help="Años a optimizar, formato 2026,2027,...",
    )
    parser.add_argument(
        "--day-types",
        help="Tipos de día a optimizar (TYPE_1,TYPE_2,TYPE_3).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Máximo de procesos en paralelo.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        help="JSON con parámetros extra para optimizar (se pasa directo a optimizador).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_sensibilidad"),
        help="Directorio base donde guardar los reportes.",
    )
    parser.add_argument(
        "--single-investment",
        action="store_true",
        help="Optimiza una única política válida para todos los años.",
    )
    parser.add_argument(
        "--base-compare-weeks",
        type=int,
        default=1,
        help="Semanas utilizadas al reevaluar la política base por escenario.",
    )
    parser.add_argument(
        "--base-compare-rep",
        type=int,
        default=2,
        help="Réplicas SAA al reevaluar la política base por escenario.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scenarios = _load_scenarios(Path(args.scenarios))
    segments = tuple(_parse_list(args.segments) or ("pesimista", "regular", "optimista"))
    years = _parse_list(args.years)
    day_types = _parse_day_types(_parse_list(args.day_types))

    for scenario in scenarios:
        run_scenario(
            scenario=scenario,
            args=args,
            segments=segments,
            years=years,
            day_types=day_types,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
