from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from . import demand, engine
from .engine import DayType

try:
    from optimizador_cajas import optimizar_cajas_grasp_saa
except ImportError:  # pragma: no cover
    optimizar_cajas_grasp_saa = None


@dataclass
class SequentialOptimizationResult:
    base_year: Dict[DayType, Any]
    yearly: Dict[str, Dict[str, Dict[DayType, Any]]]
    multipliers: Dict[str, Dict[str, float]]


def _optimizer_worker(task):
    (
        year,
        segment,
        day_value,
        factor,
        initial_solution,
        optimizer_kwargs,
    ) = task
    from simulator import engine as eng
    from optimizador_cajas import optimizar_cajas_grasp_saa as opt

    day_type = DayType(day_value)
    print(
        f"[PLAN][WORKER] Iniciando año {year} segmento {segment} tipo {day_type.name} con factor {factor:.4f}"
    )
    eng.set_demand_multiplier(factor)
    try:
        context_label = f"Año={year} | Segmento={segment} | Día={day_type.name}"
        kwargs = dict(optimizer_kwargs or {})
        kwargs["context_label"] = context_label
        result = opt(
            day_type=day_type,
            initial_solution=initial_solution,
            **kwargs,
        )
    finally:
        eng.set_demand_multiplier(1.0)
    print(
        f"[PLAN][WORKER] Finalizado año {year} segmento {segment} tipo {day_type.name}"
    )
    return year, segment, day_value, factor, result


def plan_multi_year_optimization(
    *,
    segmented_path: str | Path,
    fit_summary_path: str | Path | None = None,
    years: Sequence[str] | None = ("2026", "2027", "2028", "2029", "2030"),
    segments: Sequence[str] = ("pesimista", "regular", "optimista"),
    day_types: Sequence[DayType] | None = None,
    max_workers: Optional[int] = None,
    optimizer_kwargs: Optional[Mapping[str, object]] = None,
) -> SequentialOptimizationResult:
    if optimizar_cajas_grasp_saa is None:
        raise RuntimeError(
            "No se pudo importar optimizador_cajas.optimizar_cajas_grasp_saa"
        )

    if day_types is None:
        day_types = tuple(DayType)

    optimizer_kwargs = dict(optimizer_kwargs or {})

    expectations = demand.load_segment_expectations(
        segmented_path,
        fit_summary_path,
        segments=segments,
        years=years,
    )

    if years is None:
        year_sequence = sorted(
            {year for mapping in expectations.values() for year in mapping.keys()}
        )
    else:
        year_sequence = [str(year) for year in years]

    engine.set_demand_multiplier(1.0)
    base_year_results: Dict[DayType, Any] = {}
    print("\n[PLAN] Ejecutando optimización base (año 2025)...")
    for dt in day_types:
        context_label = f"Año=2025 | Segmento=base | Día={dt.name}"
        base_kwargs = dict(optimizer_kwargs)
        base_kwargs["context_label"] = context_label
        base_year_results[dt] = optimizar_cajas_grasp_saa(
            day_type=dt, **base_kwargs
        )

    previous_solutions: Dict[str, Dict[DayType, tuple[int, int, int, int]]] = {
        str(segment): {
            dt: tuple(int(v) for v in base_year_results[dt].x) for dt in day_types
        }
        for segment in segments
    }

    yearly_results: Dict[str, Dict[str, Dict[DayType, Any]]] = {}

    for year in year_sequence:
        print(f"\n[PLAN] Optimizando año {year}...")
        tasks = []
        for segment in segments:
            factor = float(expectations.get(str(segment), {}).get(str(year), 1.0))
            prev_map = previous_solutions.get(str(segment), {})
            for dt in day_types:
                initial = prev_map.get(dt)
                tasks.append(
                    (
                        str(year),
                        str(segment),
                        dt.value,
                        float(factor),
                        initial,
                        optimizer_kwargs,
                    )
                )

        year_results: Dict[str, Dict[DayType, Any]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_optimizer_worker, task): task for task in tasks}
            for fut in as_completed(futures):
                year_val, segment_name, day_value, factor, result = fut.result()
                dt = DayType(day_value)
                year_results.setdefault(segment_name, {})[dt] = result
                print(
                    f"[PLAN] Año {year_val} segmento {segment_name} tipo {dt.name}: "
                    f"factor {factor:.4f} -> objetivo {result.objetivo_mean:,.0f}"
                )
        yearly_results[year] = year_results

        for segment_name, dt_map in year_results.items():
            previous_solutions[segment_name] = {
                dt: tuple(int(v) for v in res.x) for dt, res in dt_map.items()
            }

    engine.set_demand_multiplier(1.0)

    return SequentialOptimizationResult(
        base_year=base_year_results,
        yearly=yearly_results,
        multipliers={
            str(segment): expectations.get(str(segment), {}) for segment in segments
        },
    )
