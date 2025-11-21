from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import engine  # noqa: E402
from simulator.engine import DayType  # noqa: E402
from simulator.policy_planner import (  # noqa: E402
    LANE_ORDER,
    SequentialOptimizationResult,
    plan_multi_year_optimization,
)
from optimizador_cajas import cost_anual_config, evaluate_policy_saa  # noqa: E402


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


def _lane_tuple_to_counts(x: tuple[int, int, int, int]) -> dict[str, int]:
    counts = {lane: int(x[idx]) for idx, lane in enumerate(LANE_ORDER)}
    return engine.enforce_lane_constraints(counts)


def _add_detail_row(
    rows: list[dict[str, Any]],
    *,
    year: str,
    segment: str,
    day_type: DayType,
    result: Any,
    factor: float,
    base_result: Any | None = None,
) -> None:
    if result is None or not hasattr(result, "x"):
        return
    try:
        x_tuple = tuple(int(v) for v in result.x)
    except Exception:
        return
    counts = _lane_tuple_to_counts(x_tuple)
    infra_cost = float(cost_anual_config(counts))
    profit_mean = float(getattr(result, "profit_mean", 0.0))
    profit_std = float(getattr(result, "profit_std", 0.0))
    objetivo_mean = float(
        getattr(result, "objetivo_mean", profit_mean - infra_cost)
    )
    objetivo_std = float(getattr(result, "objetivo_std", 0.0))
    n_rep = int(getattr(result, "n_rep", 0))
    row: dict[str, Any] = {
        "year": str(year),
        "segment": str(segment),
        "day_type": day_type.name,
        "demand_factor": float(factor),
        "regular": counts.get("regular", 0),
        "express": counts.get("express", 0),
        "priority": counts.get("priority", 0),
        "self_checkout": counts.get("self_checkout", 0),
        "total_lanes": sum(counts.values()),
        "profit_mean": profit_mean,
        "profit_std": profit_std,
        "objetivo_mean": objetivo_mean,
        "objetivo_std": objetivo_std,
        "infra_cost_clp": infra_cost,
        "objective_minus_profit_gap": objetivo_mean - (profit_mean - infra_cost),
        "n_rep": n_rep,
    }
    if base_result is not None:
        row["base_profit_mean"] = float(getattr(base_result, "profit_mean", 0.0))
        row["base_profit_std"] = float(getattr(base_result, "profit_std", 0.0))
        row["base_objetivo_mean"] = float(
            getattr(
                base_result,
                "objetivo_mean",
                row["base_profit_mean"] - infra_cost,
            )
        )
        row["base_objetivo_std"] = float(
            getattr(base_result, "objetivo_std", 0.0)
        )
        row["base_n_rep"] = int(getattr(base_result, "n_rep", 0))
        row["objetivo_delta_vs_base"] = (
            row["objetivo_mean"] - row["base_objetivo_mean"]
        )
        row["profit_delta_vs_base"] = (
            row["profit_mean"] - row["base_profit_mean"]
        )
    else:
        row["base_profit_mean"] = None
        row["base_profit_std"] = None
        row["base_objetivo_mean"] = None
        row["base_objetivo_std"] = None
        row["base_n_rep"] = None
        row["objetivo_delta_vs_base"] = None
        row["profit_delta_vs_base"] = None

    rows.append(row)


def build_plan_detail_dataframe(
    sequential: SequentialOptimizationResult,
    base_compare: dict[tuple[str, str, str], Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day_type, res in sequential.base_year.items():
        _add_detail_row(
            rows,
            year="2025",
            segment="base",
            day_type=day_type,
            result=res,
            factor=1.0,
            base_result=res,
        )
    for year, seg_map in sequential.yearly.items():
        for segment, dt_map in seg_map.items():
            for day_type, res in dt_map.items():
                factor = float(
                    sequential.multipliers.get(str(segment), {}).get(str(year), 1.0)
                )
                base_result = None
                if base_compare:
                    base_result = base_compare.get(
                        (str(year), str(segment), day_type.name)
                    )
                _add_detail_row(
                    rows,
                    year=str(year),
                    segment=str(segment),
                    day_type=day_type,
                    result=res,
                    factor=factor,
                    base_result=base_result,
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["year"] = df["year"].astype(str)
    df = df.sort_values(["year", "segment", "day_type"]).reset_index(drop=True)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def build_multipliers_dataframe(
    sequential: SequentialOptimizationResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, mapping in sequential.multipliers.items():
        for year, factor in mapping.items():
            rows.append(
                {"segment": str(segment), "year": str(year), "demand_factor": float(factor)}
            )
    return (
        pd.DataFrame(rows).sort_values(["segment", "year"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["segment", "year", "demand_factor"])
    )


def build_initial_policies_dataframe(
    sequential: SequentialOptimizationResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day_name, counts_tuple in sequential.initial_policies.items():
        try:
            x_tuple = tuple(int(v) for v in counts_tuple)
        except Exception:
            continue
        counts = _lane_tuple_to_counts(x_tuple)
        rows.append(
            {
                "day_type": day_name,
                "regular": counts.get("regular", 0),
                "express": counts.get("express", 0),
                "priority": counts.get("priority", 0),
                "self_checkout": counts.get("self_checkout", 0),
                "total_lanes": sum(counts.values()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("day_type").reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["day_type", *LANE_ORDER, "total_lanes"])
    )


def export_excel_report(
    detail_df: pd.DataFrame,
    multipliers_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    path: Path,
) -> None:
    summary_df = (
        detail_df.groupby(["year", "segment"], dropna=False)
        .agg(
            objetivo_mean=("objetivo_mean", "sum"),
            profit_mean=("profit_mean", "sum"),
            infra_cost_clp=("infra_cost_clp", "sum"),
            total_lanes=("total_lanes", "mean"),
        )
        .reset_index()
        if not detail_df.empty
        else pd.DataFrame(
            columns=["year", "segment", "objetivo_mean", "profit_mean", "infra_cost_clp", "total_lanes"]
        )
    )
    with pd.ExcelWriter(path) as writer:
        detail_df.to_excel(writer, index=False, sheet_name="plan_detalle")
        summary_df.to_excel(writer, index=False, sheet_name="resumen")
        multipliers_df.to_excel(writer, index=False, sheet_name="multipliers")
        initial_df.to_excel(writer, index=False, sheet_name="initial_policies")


def plot_objective_by_segment(detail_df: pd.DataFrame, path: Path) -> None:
    df = detail_df[detail_df["segment"] != "base"].copy()
    df = df.dropna(subset=["year_num"])
    if df.empty:
        return
    agg = (
        df.groupby(["segment", "year_num"], dropna=False)["objetivo_mean"]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for segment, seg_df in agg.groupby("segment"):
        seg_sorted = seg_df.sort_values("year_num")
        ax.plot(
            seg_sorted["year_num"],
            seg_sorted["objetivo_mean"],
            marker="o",
            label=str(segment),
        )
    ax.set_xlabel("Año")
    ax.set_ylabel("Objetivo anual agregado (CLP)")
    ax.set_title("Objetivo esperado por segmento")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_total_lanes(detail_df: pd.DataFrame, path: Path) -> None:
    if detail_df.empty:
        return
    day_types = sorted(detail_df["day_type"].unique())
    if not day_types:
        return
    df = detail_df.dropna(subset=["year_num"])
    if df.empty:
        return
    fig, axes = plt.subplots(
        1, len(day_types), figsize=(5 * len(day_types), 4), sharey=True
    )
    if len(day_types) == 1:
        axes = [axes]
    for ax, day in zip(axes, day_types):
        subset = df[df["day_type"] == day]
        for segment, seg_df in subset.groupby("segment"):
            seg_sorted = seg_df.sort_values("year_num")
            ax.plot(
                seg_sorted["year_num"],
                seg_sorted["total_lanes"],
                marker="o",
                label=str(segment),
            )
        ax.set_title(day)
        ax.set_xlabel("Año")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Total de cajas")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_base_counts(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError("base_counts debe ser un JSON con formato {day_type: counts}")


def evaluate_base_performance(
    sequential: SequentialOptimizationResult,
    *,
    num_weeks_sample: int,
    num_rep: int,
) -> dict[tuple[str, str, str], Any]:
    mapping: dict[tuple[str, str, str], Any] = {}
    eval_kwargs = {
        "num_weeks_sample": max(1, int(num_weeks_sample)),
        "num_rep": max(1, int(num_rep)),
    }

    for year, seg_map in sequential.yearly.items():
        for segment, dt_map in seg_map.items():
            factor = float(
                sequential.multipliers.get(str(segment), {}).get(str(year), 1.0)
            )
            for day_type in dt_map.keys():
                base_tuple = sequential.initial_policies.get(day_type.name)
                if base_tuple is None:
                    continue
                key = (str(year), str(segment), day_type.name)
                engine.set_demand_multiplier(factor)
                try:
                    result = evaluate_policy_saa(
                        tuple(int(v) for v in base_tuple),
                        day_type=day_type,
                        eval_context=f"BASE | {segment} | year={year}",
                        **eval_kwargs,
                    )
                finally:
                    engine.set_demand_multiplier(1.0)
                mapping[key] = result
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera reporte completo para la planificación multi-anual."
    )
    parser.add_argument(
        "segmented",
        type=Path,
        help="CSV segmentado con columnas scenario,2026-2030,score,segment",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        help="CSV con ajustes estadísticos por segmento/año (opcional).",
    )
    parser.add_argument(
        "--years",
        help="Años a optimizar, formato 2026,2027,... (default: todos).",
    )
    parser.add_argument(
        "--segments",
        help="Segmentos a incluir, formato pesimista,regular,optimista.",
    )
    parser.add_argument(
        "--day-types",
        help="Tipos de día a optimizar (TYPE_1,TYPE_2,TYPE_3).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Máximo de procesos paralelos.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        help="JSON con parámetros extra para optimizador_cajas.optimizar_cajas_grasp_saa.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_opt"),
        help="Directorio base para guardar el reporte.",
    )
    parser.add_argument(
        "--base-counts",
        help="Ruta a JSON con configuraciones iniciales por tipo de día (si se omite base).",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Saltar optimización del año base y usar configuraciones provistas.",
    )
    parser.add_argument(
        "--skip-base-compare",
        action="store_true",
        help="Omitir la reevaluación de las configuraciones base para cada año/segmento.",
    )
    parser.add_argument(
        "--base-compare-weeks",
        type=int,
        default=1,
        help="Número de semanas sample a usar al evaluar la configuración base por año (default: 1).",
    )
    parser.add_argument(
        "--base-compare-rep",
        type=int,
        default=3,
        help="Número de réplicas SAA para la evaluación base por año (default: 3).",
    )
    args = parser.parse_args(argv)

    years = _parse_list(args.years)
    segments = _parse_list(args.segments)
    day_types = _parse_day_types(_parse_list(args.day_types))
    optimizer_kwargs = json.loads(args.optimizer_kwargs) if args.optimizer_kwargs else {}
    base_counts = load_base_counts(args.base_counts)

    result = plan_multi_year_optimization(
        segmented_path=args.segmented,
        fit_summary_path=args.fit_summary,
        years=years,
        segments=segments or ("pesimista", "regular", "optimista"),
        day_types=day_types,
        max_workers=args.max_workers,
        optimizer_kwargs=optimizer_kwargs,
        optimize_base_year=not args.skip_base,
        base_counts=base_counts,
    )

    base_compare_map: dict[tuple[str, str, str], Any] | None = None
    if not args.skip_base_compare:
        base_compare_map = evaluate_base_performance(
            result,
            num_weeks_sample=args.base_compare_weeks,
            num_rep=args.base_compare_rep,
        )

    detail_df = build_plan_detail_dataframe(result, base_compare=base_compare_map)
    multipliers_df = build_multipliers_dataframe(result)
    initial_df = build_initial_policies_dataframe(result)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.output_dir) / f"plan_multi_year_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "plan_detalle.csv"
    detail_df.to_csv(csv_path, index=False)

    excel_path = report_dir / "plan_resumen.xlsx"
    export_excel_report(detail_df, multipliers_df, initial_df, excel_path)

    chart1_path = report_dir / "objetivo_por_segmento.png"
    plot_objective_by_segment(detail_df, chart1_path)

    chart2_path = report_dir / "total_lanes_por_dia.png"
    plot_total_lanes(detail_df, chart2_path)

    print("Reporte multi-anual generado:")
    print(f"  - Excel: {excel_path}")
    print(f"  - CSV detalle: {csv_path}")
    if chart1_path.exists():
        print(f"  - Gráfico objetivo/segmento: {chart1_path}")
    if chart2_path.exists():
        print(f"  - Gráfico cajas/día: {chart2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
