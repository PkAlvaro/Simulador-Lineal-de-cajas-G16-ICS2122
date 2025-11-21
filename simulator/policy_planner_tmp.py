from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from . import demand, engine
from .engine import DayType, _warn

try:
    from optimizador_cajas import evaluate_policy_saa, optimizar_cajas_grasp_saa
except ImportError:  # pragma: no cover
    optimizar_cajas_grasp_saa = None
    evaluate_policy_saa = None


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
        f"[PLAN][WORKER] Iniciando ano {year} segmento {segment} "
        f"tipo {day_type.name} con factor {factor:.4f}"
    )
    eng.set_demand_multiplier(factor)
    try:
        context_label = f"Ano={year} | Segmento={segment} | Dia={day_type.name}"
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
        f"[PLAN][WORKER] Finalizado ano {year} segmento {segment} "
        f"tipo {day_type.name}"
    )
    return year, segment, day_value, factor, result


def _optimize_daytype_worker(task):
    day_value, optimizer_kwargs, scenario_weights = task
    from optimizador_cajas import optimizar_cajas_grasp_saa as opt

    dt = DayType(day_value)
    result = opt(
        day_type=dt,
        scenario_weights=scenario_weights,
        **optimizer_kwargs,
    )
    return day_value, result


def plan_multi_year_optimization(
    *,
    segmented_path: str | Path,
    fit_summary_path: str | Path | None = None,
    years: Sequence[str] | None = ("2026", "2027", "2028", "2029", "2030"),
    segments: Sequence[str] = ("pesimista", "regular", "optimista"),
    day_types: Sequence[DayType] | None = None,
    max_workers: Optional[int] = None,
    optimizer_kwargs: Optional[Mapping[str, object]] = None,
    optimize_base_year: bool = True,  # kept for compatibility; siempre se reoptimiza
    base_counts: Mapping[str, Sequence[int]] | None = None,
    single_investment: bool = False,
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

    # --------------------------------------------------------------
    # OptimizaciÃ³n del aÃ±o base (2025) - siempre se reoptimiza
    # --------------------------------------------------------------
    engine.set_demand_multiplier(1.0)

    scenario_weights: list[tuple[str, float, float]] | None = None
    if single_investment:
        # Promedio del factor por segmento para usar en la optimizaciÃ³n base
        scenario_weights = []
        for segment in segments:
            seg_map = expectations.get(str(segment), {})
            factors = [float(seg_map.get(str(year), 1.0)) for year in year_sequence]
            if not factors:
                continue
            avg_factor = sum(factors) / len(factors)
            scenario_weights.append((str(segment), avg_factor, float(len(factors))))

    eval_weeks_default = int(optimizer_kwargs.get("num_weeks_sample_busqueda") or 1)
    eval_reps_default = int(optimizer_kwargs.get("num_rep_saa_busqueda") or 1)

    base_year_results: Dict[DayType, Any] = {}
    base_seed_per_day: Dict[DayType, tuple[int, int, int, int]] = {}

    if base_counts:
        _warn(
            "Se ignoran configuraciones iniciales proporcionadas; siempre se reoptimiza el ano base.",
            None,
        )

    print("\n[PLAN] Ejecutando optimizacion base (ano 2025)...")
    base_tasks = []
    for dt in day_types:
        context_label = f"Ano=2025 | Segmento=base | Dia={dt.name}"
        base_kwargs = dict(optimizer_kwargs)
        base_kwargs["context_label"] = context_label
        base_tasks.append(
            (
                dt.value,
                base_kwargs,
                scenario_weights if single_investment else None,
            )
        )

    base_results: dict[DayType, Any] = {}
    if len(base_tasks) == 1:
        day_value, res = _optimize_daytype_worker(base_tasks[0])
        base_results[DayType(day_value)] = res
    else:
        max_workers_base = max_workers or len(base_tasks)
        max_workers_base = max(1, min(max_workers_base, len(base_tasks)))
        with ProcessPoolExecutor(max_workers=max_workers_base) as pool:
            for day_value, res in pool.map(_optimize_daytype_worker, base_tasks):
                base_results[DayType(day_value)] = res

    for dt, result in base_results.items():
        if single_investment:
            if evaluate_policy_saa is None:
                raise RuntimeError(
                    "evaluate_policy_saa no disponible para evaluar ano base"
                )
            prev_factor = engine.get_demand_multiplier()
            engine.set_demand_multiplier(1.0)
            try:
                base_eval = evaluate_policy_saa(
                    tuple(int(v) for v in result.x),
                    day_type=dt,
                    num_weeks_sample=eval_weeks_default,
                    num_rep=eval_reps_default,
                    eval_context="Ano=2025 | Segmento=base | POLITICA UNICA",
                )
            finally:
                engine.set_demand_multiplier(prev_factor)
            base_year_results[dt] = base_eval
        else:
            base_year_results[dt] = result

        base_seed_per_day[dt] = tuple(int(v) for v in result.x)

    # Politicas iniciales heredadas por segmento/tipo de dÃ­a
    previous_solutions: Dict[str, Dict[DayType, tuple[int, int, int, int]]] = {
        str(segment): {dt: base_seed_per_day[dt] for dt in day_types}
        for segment in segments
    }

    yearly_results: Dict[str, Dict[str, Dict[DayType, Any]]] = {}

    # ------------------------------------------------------------------
    # Modo "single_investment":
    #   - Por cada segmento y tipo de dÃ­a:
    #       1) Optimizar cada aÃ±o por separado (Stage 1).
    #       2) Usar esas polÃ­ticas (mÃ¡s la base) como candidatas y
    #          elegir la que maximiza el objetivo acumulado a 5 aÃ±os
    #          evaluando aÃ±o por aÃ±o con evaluate_policy_saa (Stage 2).
    #   - yearly_results se llena siempre con la polÃ­tica fija elegida.
    # ------------------------------------------------------------------
    if single_investment:
        if evaluate_policy_saa is None:
            raise RuntimeError(
                "evaluate_policy_saa no disponible; no se puede evaluar politica fija"
            )

        eval_weeks = eval_weeks_default
        eval_reps = eval_reps_default

        per_segment_candidates: Dict[
            str, Dict[DayType, set[tuple[int, int, int, int]]]
        ] = {}

        # Stage 1: optimizar aÃ±os por separado
        for segment in segments:
            seg_name = str(segment)
            print(f"\n[PLAN] Optimizando anos por separado para segmento {seg_name}...")

            seg_prev = {
                dt: base_seed_per_day[dt] for dt in day_types if dt in base_seed_per_day
            }
            candidate_policies: Dict[
                DayType, set[tuple[int, int, int, int]]
            ] = {dt: set() for dt in day_types}

            # siempre incluir la polÃ­tica base como candidata
            for dt in day_types:
                base_policy = base_seed_per_day.get(dt)
                if base_policy is not None:
                    candidate_policies[dt].add(base_policy)

            tasks = []
            for year in year_sequence:
                factor = float(
                    expectations.get(seg_name, {}).get(str(year), 1.0)
                )
                for dt in day_types:
                    initial = seg_prev.get(dt)
                    tasks.append(
                        (
                            str(year),
                            seg_name,
                            dt.value,
                            float(factor),
                            initial,
                            optimizer_kwargs,
                        )
                    )

            if tasks:
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(_optimizer_worker, task): task for task in tasks
                    }
                    for fut in as_completed(futures):
                        year_val, segment_name, day_value, factor, result = fut.result()
                        dt = DayType(day_value)
                        seg_prev[dt] = tuple(int(v) for v in result.x)
                        candidate_policies[dt].add(tuple(int(v) for v in result.x))
                        print(
                            f"[PLAN] (single) Segmento {segment_name} ano {year_val} "
                            f"tipo {dt.name}: factor {factor:.4f} "
                            f"-> objetivo {getattr(result, 'objetivo_mean', float('nan')):,.0f}"
                        )

            per_segment_candidates[seg_name] = candidate_policies

        # Stage 2: elegir polÃ­tica fija por segmento y tipo de dÃ­a
        for segment in segments:
            seg_name = str(segment)
            seg_candidates = per_segment_candidates.get(seg_name, {})

            for dt in day_types:
                policies = list(seg_candidates.get(dt, set()))
                if not policies:
                    continue

                best_policy: Optional[tuple[int, int, int, int]] = None
                best_total_obj = float("-inf")
                best_year_evals: Dict[str, Any] = {}

                for policy in policies:
                    total_obj = 0.0
                    year_evals: Dict[str, Any] = {}
                    for year in year_sequence:
                        factor = float(
                            expectations.get(seg_name, {}).get(str(year), 1.0)
                        )
                        prev_factor = engine.get_demand_multiplier()
                        engine.set_demand_multiplier(factor)
                        try:
                            res = evaluate_policy_saa(
                                policy,
                                day_type=dt,
                                num_weeks_sample=eval_weeks,
                                num_rep=eval_reps,
                                eval_context=(
                                    f"Ano={year} | Segmento={seg_name} | "
                                    f"POLITICA-CANDIDATA {policy}"
                                ),
                            )
                        finally:
                            engine.set_demand_multiplier(prev_factor)

                        year_evals[str(year)] = res
                        total_obj += getattr(res, "objetivo_mean", float("-inf"))

                    if total_obj > best_total_obj:
                        best_total_obj = total_obj
                        best_policy = policy
                        best_year_evals = year_evals

                if best_policy is None:
                    continue

                print(
                    f"[PLAN] Segmento {seg_name} tipo {dt.name}: "
                    f"politica fija elegida {best_policy} "
                    f"con objetivo acumulado {best_total_obj:,.0f}"
                )

                for year, res in best_year_evals.items():
                    yearly_results.setdefault(str(year), {}).setdefault(seg_name, {})[
                        dt
                    ] = res

        engine.set_demand_multiplier(1.0)

        return SequentialOptimizationResult(
            base_year=base_year_results,
            yearly=yearly_results,
            multipliers={
                str(segment): expectations.get(str(segment), {})
                for segment in segments
            },
            initial_policies={
                dt.name: base_seed_per_day[dt] for dt in day_types
            },
        )

    # ------------------------------------------------------------------
    # Modo normal (no single_investment): optimiza secuencialmente por
    # segmento, aÃ±o y tipo de dÃ­a, heredando la soluciÃ³n anterior.
    # ------------------------------------------------------------------
    for segment in segments:
        seg_name = str(segment)
        print(f"\n[PLAN] Optimizando segmento {seg_name}...")
        tasks = []
        seg_prev = previous_solutions.get(seg_name, {})

        for year in year_sequence:
            factor = float(expectations.get(seg_name, {}).get(str(year), 1.0))
            for dt in day_types:
                initial = seg_prev.get(dt)
                tasks.append(
                    (
                        str(year),
                        seg_name,
                        dt.value,
                        float(factor),
                        initial,
                        optimizer_kwargs,
                    )
                )

        seg_year_results: Dict[str, Dict[DayType, Any]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_optimizer_worker, task): task for task in tasks}
            for fut in as_completed(futures):
                year_val, segment_name, day_value, factor, result = fut.result()
                dt = DayType(day_value)
                seg_year_results.setdefault(year_val, {})[dt] = result
                print(
                    f"[PLAN] Segmento {segment_name} ano {year_val} tipo {dt.name}: "
                    f"factor {factor:.4f} -> objetivo {result.objetivo_mean:,.0f}"
                )

        for year_val, dt_map in seg_year_results.items():
            yearly_results.setdefault(year_val, {})[seg_name] = dt_map
            seg_prev = previous_solutions.setdefault(seg_name, {})
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

