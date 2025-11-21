from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
    initial_policies: Dict[str, tuple[int, int, int, int]]


LANE_ORDER: tuple[str, ...] = (
    "regular",
    "express",
    "priority",
    "self_checkout",
)


def _normalize_lane_tuple(raw: Any) -> tuple[int, int, int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        seq = [raw.get(lane, raw.get(lane.upper(), 0)) for lane in LANE_ORDER]
    else:
        try:
            seq = list(raw)
        except TypeError:
            return None
    if len(seq) < len(LANE_ORDER):
        return None
    try:
        tuple_counts = tuple(int(max(0, seq[idx])) for idx in range(len(LANE_ORDER)))
    except (TypeError, ValueError):
        return None
    counts_dict = {lane: tuple_counts[idx] for idx, lane in enumerate(LANE_ORDER)}
    try:
        normalized = engine.enforce_lane_constraints(counts_dict.copy())
        return tuple(int(normalized.get(lane, 0)) for lane in LANE_ORDER)
    except AttributeError:
        return tuple_counts


def _seed_for_day(
    day_type: DayType, seeds: Mapping[str, Sequence[int]] | None
) -> tuple[int, int, int, int] | None:
    if not seeds:
        return None
    candidates = (
        seeds.get(day_type.name),
        seeds.get(day_type.name.upper()),
        seeds.get(day_type.name.lower()),
        seeds.get(day_type.value),
        seeds.get(day_type.value.upper()),
        seeds.get(day_type.value.lower()),
    )
    for candidate in candidates:
        normalized = _normalize_lane_tuple(candidate)
        if normalized is not None:
            return normalized
    return None


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
    optimize_base_year: bool = True,
    base_counts: Mapping[str, Sequence[int]] | None = None,
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
    base_seed_per_day: Dict[DayType, tuple[int, int, int, int]] = {}

    if optimize_base_year:
        print("\n[PLAN] Ejecutando optimización base (año 2025)...")
        for dt in day_types:
            context_label = f"Año=2025 | Segmento=base | Día={dt.name}"
            base_kwargs = dict(optimizer_kwargs)
            base_kwargs["context_label"] = context_label
            result = optimizar_cajas_grasp_saa(day_type=dt, **base_kwargs)
            base_year_results[dt] = result
            base_seed_per_day[dt] = tuple(int(v) for v in result.x)
    else:
        print(
            "\n[PLAN] Se omite la optimización del año base; se utilizarán configuraciones iniciales."
        )
        for dt in day_types:
            seed_tuple = _seed_for_day(dt, base_counts)
            if seed_tuple is None:
                default_counts = engine.DEFAULT_LANE_COUNTS.get(dt, {})
                seed_tuple = _normalize_lane_tuple(default_counts)
                print(
                    f"[PLAN] {dt.name}: se usa configuración por defecto {seed_tuple}."
                )
            else:
                print(
                    f"[PLAN] {dt.name}: se usa configuración proporcionada {seed_tuple}."
                )
            if seed_tuple is None:
                raise RuntimeError(
                    f"No se pudo determinar configuración inicial para {dt.name}"
                )
            base_seed_per_day[dt] = seed_tuple
            base_year_results[dt] = SimpleNamespace(
                x=seed_tuple,
                profit_mean=0.0,
                profit_std=0.0,
                objetivo_mean=0.0,
                objetivo_std=0.0,
                n_rep=0,
                day_type=dt,
            )

    for dt in day_types:
        if dt not in base_seed_per_day:
            default_counts = engine.DEFAULT_LANE_COUNTS.get(dt, {})
            fallback = _normalize_lane_tuple(default_counts)
            if fallback is None:
                raise RuntimeError(
                    f"No se pudo determinar configuración inicial para {dt.name}"
                )
            base_seed_per_day[dt] = fallback
            if dt not in base_year_results:
                base_year_results[dt] = SimpleNamespace(
                    x=fallback,
                    profit_mean=0.0,
                    profit_std=0.0,
                    objetivo_mean=0.0,
                    objetivo_std=0.0,
                    n_rep=0,
                    day_type=dt,
                )

    previous_solutions: Dict[str, Dict[DayType, tuple[int, int, int, int]]] = {
        str(segment): {dt: base_seed_per_day[dt] for dt in day_types}
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
            seg_prev = previous_solutions.setdefault(segment_name, {})
            for dt, res in dt_map.items():
                seg_prev[dt] = tuple(int(v) for v in res.x)

    engine.set_demand_multiplier(1.0)

    return SequentialOptimizationResult(
        base_year=base_year_results,
        yearly=yearly_results,
        multipliers={
            str(segment): expectations.get(str(segment), {}) for segment in segments
        },
        initial_policies={dt.name: base_seed_per_day[dt] for dt in day_types},
    )
