from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from . import demand, engine
from .engine import DayType, _warn

try:  # re-export helper used por reportes y sensibilidad
    from tools.export_plan_report import evaluate_base_performance as _evaluate_base_performance  # type: ignore
except Exception:  # pragma: no cover
    _evaluate_base_performance = None

try:
    from .optimizador_cajas import evaluate_policy_saa, optimizar_cajas_grasp_saa
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


def evaluate_base_performance(
    sequential: SequentialOptimizationResult,
    *,
    num_weeks_sample: int,
    num_rep: int,
) -> Dict[tuple[str, str, str], Any]:
    """
    Wrapper fino para mantener compatibilidad con herramientas externas.

    Delegamos en tools.export_plan_report.evaluate_base_performance para no
    duplicar lógica. Si no está disponible, levantamos un error claro.
    """
    if _evaluate_base_performance is None:
        raise RuntimeError(
            "evaluate_base_performance no disponible; "
            "verifica que tools/export_plan_report.py esté accesible"
        )
    return _evaluate_base_performance(
        sequential,
        num_weeks_sample=num_weeks_sample,
        num_rep=num_rep,
    )


def _normalize_lane_tuple(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        # If dict, convert to tuple (replicated for blocks if needed, but here we might just return base tuple
        # and let the optimizer expand it? No, we should probably expand it here if we want consistency)
        # But wait, raw dict usually means single block.
        # Let's assume if dict, it's a single block config.
        seq = [raw.get(lane, raw.get(lane.upper(), 0)) for lane in LANE_ORDER]
        # We need to expand it to full schedule length if we are in schedule mode.
        # But policy_planner doesn't know about TIME_BLOCKS directly unless we import.
        # Let's import TIME_BLOCKS from optimizador_cajas
        from simulator.optimizador_cajas import TIME_BLOCKS
        
        base = tuple(int(max(0, x)) for x in seq)
        return base * len(TIME_BLOCKS)
        
    else:
        try:
            seq = list(raw)
        except TypeError:
            return None

    if len(seq) == 0 or len(seq) % 4 != 0:
        return None

    # Enforce constraints per block
    from simulator import engine
    
    final_seq = []
    num_blocks = len(seq) // 4
    
    for i in range(num_blocks):
        block = seq[i*4 : (i+1)*4]
        counts_dict = {lane: int(max(0, block[idx])) for idx, lane in enumerate(LANE_ORDER)}
        try:
            normalized = engine.enforce_lane_constraints(counts_dict.copy())
            final_seq.extend(int(normalized.get(lane, 0)) for lane in LANE_ORDER)
        except AttributeError:
            final_seq.extend(int(max(0, x)) for x in block)
            
    return tuple(final_seq)


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
    from simulator.optimizador_cajas import optimizar_cajas_grasp_saa as opt

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
    from simulator.optimizador_cajas import optimizar_cajas_grasp_saa as opt

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
    # Optimización del año base (2025) - siempre se reoptimiza
    # usando SOLO demanda base (factor 1.0, sin escenarios).
    # --------------------------------------------------------------
    engine.set_demand_multiplier(1.0)

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
        base_tasks.append((dt.value, base_kwargs, None))

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

    # Politicas iniciales heredadas por segmento/tipo de día
    previous_solutions: Dict[str, Dict[DayType, tuple[int, int, int, int]]] = {
        str(segment): {dt: base_seed_per_day[dt] for dt in day_types}
        for segment in segments
    }

    yearly_results: Dict[str, Dict[str, Dict[DayType, Any]]] = {}

    # ------------------------------------------------------------------
    # Modo "single_investment":
    #   - Por cada segmento y tipo de día:
    #       1) Optimizar cada año por separado (Stage 1).
    #       2) Usar esas políticas (más la base) como candidatas y
    #          elegir la que maximiza el objetivo acumulado a 5 años
    #          evaluando año por año con evaluate_policy_saa (Stage 2).
    #   - yearly_results se llena siempre con la política fija elegida.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Modo "single_investment":
    #   - Tomamos la política óptima del año base (2025).
    #   - La evaluamos ("estresamos") contra las demandas proyectadas de los años futuros.
    #   - NO re-optimizamos. Solo medimos el desempeño de mantener la infraestructura fija.
    # ------------------------------------------------------------------
    if single_investment:
        if evaluate_policy_saa is None:
            raise RuntimeError(
                "evaluate_policy_saa no disponible; no se puede evaluar politica fija"
            )

        eval_weeks = eval_weeks_default
        eval_reps = eval_reps_default

        print("\n[PLAN] Evaluando política base fija frente a demanda futura (sin re-optimizar)...")

        for segment in segments:
            seg_name = str(segment)
            
            # Tareas para evaluar en paralelo si se desea
            eval_tasks = []
            
            for year in year_sequence:
                factor = float(expectations.get(seg_name, {}).get(str(year), 1.0))
                
                for dt in day_types:
                    # Política a evaluar: la óptima del año base
                    policy_to_eval = base_seed_per_day.get(dt)
                    if policy_to_eval is None:
                        continue
                        
                    # Preparamos la tarea (podríamos paralelizar, pero evaluate_policy_saa es rápido si reps es bajo)
                    # Para simplificar y evitar problemas de pickling con closures, lo hacemos secuencial o usamos un worker simple.
                    # Dado que evaluate_policy_saa puede ser costoso, usaremos el pool si max_workers > 1
                    
                    # Definimos una función helper global o usamos una lambda picklable? No.
                    # Usaremos un bloque try-except dentro del loop secuencial por robustez en Windows,
                    # a menos que el usuario pida explícitamente paralelismo masivo.
                    
                    # Ejecución Secuencial (más segura en Windows para llamadas directas)
                    prev_factor = engine.get_demand_multiplier()
                    engine.set_demand_multiplier(factor)
                    try:
                        res = evaluate_policy_saa(
                            policy_to_eval,
                            day_type=dt,
                            num_weeks_sample=eval_weeks,
                            num_rep=eval_reps,
                            eval_context=(
                                f"Ano={year} | Segmento={seg_name} | "
                                f"POLITICA-FIJA (Base 2025)"
                            ),
                        )
                        yearly_results.setdefault(str(year), {}).setdefault(seg_name, {})[dt] = res
                        
                        print(
                            f"[PLAN] Segmento {seg_name} ano {year} tipo {dt.name}: "
                            f"factor {factor:.4f} -> Profit {res.profit_mean*365:,.0f} | VPN {res.objetivo_mean:,.0f}"
                        )
                    except Exception as e:
                        print(f"[ERROR] Falló evaluación para {year}/{seg_name}/{dt.name}: {e}")
                    finally:
                        engine.set_demand_multiplier(prev_factor)

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
    # segmento, año y tipo de día, heredando la solución anterior.
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
