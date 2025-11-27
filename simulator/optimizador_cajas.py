from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import uuid
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import engine as sim
from simulator.reporting import load_customers_year

DECISION_LANES = ["regular", "express", "priority", "self_checkout"]

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 14
HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY

# Parámetros Financieros
TAX_RATE = 0.27  # 27% Impuesto Primera Categoría
DISCOUNT_RATE = 0.0854  # 8.54% Costo de Capital (WACC/Ke)
PROJECT_HORIZON_YEARS = 5

# Estructuras para guardar costos proyectados
@dataclass
class LaneCostData:
    capex: float
    maintenance_yearly: float
    opex_hourly: float
    wage_hourly: float
    useful_life: float

@dataclass
class AnnualFixedCosts:
    year: int
    total_fixed_clp: float


@dataclass
class ScenarioWeight:
    scenario_id: str
    weight: float

@dataclass
class AnnualVariableCosts:
    year: int
    lane_costs: dict[str, LaneCostData] # LaneCostData se reutiliza pero ahora por año

# Diccionarios globales para acceso rápido: {year: ...}
# Diccionarios globales para acceso rápido: {year: ...}
FIXED_COSTS_PROJECTION: dict[int, float] = {}
VARIABLE_COSTS_PROJECTION: dict[int, dict[str, LaneCostData]] = {}

# Variables globales para seguimiento
OPT_PROGRESS: list[dict] = []
EVAL_COUNT: int = 0
START_TIME: float | None = None
MAX_SECONDS: float | None = None
MAX_EVAL_COUNT: int | None = None


def _time_exceeded() -> bool:
    if MAX_SECONDS is None or START_TIME is None:
        return False
    return (time.time() - START_TIME) > MAX_SECONDS


def _eval_limit_reached() -> bool:
    if MAX_EVAL_COUNT is None:
        return False
    return EVAL_COUNT >= MAX_EVAL_COUNT

def _load_projections() -> None:
    """
    Carga las proyecciones de costos fijos y variables desde los CSVs.
    """
    global FIXED_COSTS_PROJECTION, VARIABLE_COSTS_PROJECTION
    
    # 1. Cargar Costos Fijos
    fijos_path = PROJECT_ROOT / "data/proyeccion_costos_fijos.csv"
    if not fijos_path.exists():
        raise FileNotFoundError(f"No se encontró {fijos_path}")
        
    df_fijos = pd.read_csv(fijos_path)
    # Sumamos todas las columnas de costos (excepto 'year') para obtener el total fijo anual
    cost_cols = [c for c in df_fijos.columns if c != "year"]
    
    for _, row in df_fijos.iterrows():
        year = int(row["year"])
        total = sum(float(row[c]) for c in cost_cols)
        FIXED_COSTS_PROJECTION[year] = total

    # 2. Cargar Costos Variables Unitarios
    var_path = PROJECT_ROOT / "data/proyeccion_costos_variables.csv"
    if not var_path.exists():
        raise FileNotFoundError(f"No se encontró {var_path}")
        
    df_var = pd.read_csv(var_path)
    # Columnas: year, lane_type, capex_clp, maintenance_yearly_clp, opex_hourly_clp, wage_hourly_clp
    
    # Agrupamos por año
    years = df_var["year"].unique()
    for year in years:
        year_int = int(year)
        df_year = df_var[df_var["year"] == year]
        lane_map = {}
        
        for _, row in df_year.iterrows():
            lane = str(row["lane_type"]).strip().lower()
            if lane not in DECISION_LANES:
                continue
                
            data = LaneCostData(
                capex=float(row["capex_clp"]), # CAPEX se usa solo en t=0 (2026), pero lo guardamos
                maintenance_yearly=float(row["maintenance_yearly_clp"]),
                opex_hourly=float(row["opex_hourly_clp"]),
                wage_hourly=float(row["wage_hourly_clp"]),
                useful_life=5.0 # Asumimos 5 años fijo por ahora
            )
            lane_map[lane] = data
            
        VARIABLE_COSTS_PROJECTION[year_int] = lane_map

# Cargar proyecciones al inicio
_load_projections()


TIME_BLOCKS = [(8, 13), (13, 18), (18, 22)]


def lane_dict_to_tuple(counts: dict[str, int]) -> tuple[int, ...]:
    # Replicate the single configuration for all time blocks
    base = tuple(int(counts.get(k, 0)) for k in DECISION_LANES)
    return base * len(TIME_BLOCKS)


def lane_tuple_to_schedule(
    x: tuple[int, ...]
) -> list[tuple[int, int, dict[str, int]]]:
    schedule = []
    n_lanes = len(DECISION_LANES)
    for i, (start, end) in enumerate(TIME_BLOCKS):
        offset = i * n_lanes
        block_counts = x[offset : offset + n_lanes]
        counts_dict = {
            DECISION_LANES[j]: int(block_counts[j]) for j in range(n_lanes)
        }
        schedule.append((start, end, counts_dict))
    return schedule


def lane_tuple_to_dict(x: tuple[int, ...]) -> dict[str, int]:
    # DEPRECATED/COMPATIBILITY: Returns max counts across blocks
    # This is used for reporting/logging where a single dict is expected
    schedule = lane_tuple_to_schedule(x)
    max_counts = {l: 0 for l in DECISION_LANES}
    for _, _, counts in schedule:
        for l, c in counts.items():
            max_counts[l] = max(max_counts[l], c)
    return max_counts


def calculate_financial_metrics(
    counts: dict[str, int] | list, annual_gross_profit: float
) -> dict[str, float]:
    """
    Calcula el VPN proyectado considerando el horizonte de datos disponibles (2025-2030).
    Ano Base Inversion: Inicios de 2025 (t=0).
    Flujos Operativos: Finales de 2025 (t=1) a Finales de 2030 (t=6).
    """
    # 1. Inversion Inicial (t=0, Inicios de 2025)
    start_year = 2025
    if start_year not in VARIABLE_COSTS_PROJECTION:
        # Fallback si no se cargo 2025, usar el menor ano disponible
        start_year = min(VARIABLE_COSTS_PROJECTION.keys())

    costs_base = VARIABLE_COSTS_PROJECTION[start_year]

    # Determine counts for CAPEX (max lanes) and hours for OPEX
    if isinstance(counts, list):
        # Schedule: list of (start, end, block_counts)
        schedule = counts
        counts_for_capex = {l: 0 for l in DECISION_LANES}
        daily_hours_per_lane = {l: 0.0 for l in DECISION_LANES}
        for start, end, block_counts in schedule:
            duration = end - start
            for lane, count in block_counts.items():
                counts_for_capex[lane] = max(counts_for_capex.get(lane, 0), count)
                daily_hours_per_lane[lane] += count * duration
    else:
        # Static dict
        counts_for_capex = counts
        daily_hours_per_lane = {
            l: c * HOURS_PER_DAY for l, c in counts.items()
        }

    total_capex = 0.0
    for lane, count in counts_for_capex.items():
        if count > 0:
            total_capex += costs_base[lane].capex * count

    vp_flujos = 0.0
    total_opex_acumulado = 0.0
    total_ebitda_acumulado = 0.0
    total_fcf_acumulado = 0.0

    # Determinamos el horizonte real basado en datos disponibles (ej. 2025 a 2030 = 6 anos)
    max_year_data = max(VARIABLE_COSTS_PROJECTION.keys())
    horizon_years = (max_year_data - start_year) + 1

    # 2. Flujos Anuales
    for t in range(1, horizon_years + 1):
        year = start_year + (t - 1)

        # Costos Fijos de la Empresa para este ano
        fixed_costs = FIXED_COSTS_PROJECTION.get(year, 0.0)

        # Costos Variables de las Cajas para este ano
        var_costs_map = VARIABLE_COSTS_PROJECTION.get(year)
        if not var_costs_map:
            # Fallback al ultimo ano disponible
            var_costs_map = VARIABLE_COSTS_PROJECTION[
                max(VARIABLE_COSTS_PROJECTION.keys())
            ]

        opex_cajas_year = 0.0
        depreciacion_year = 0.0

        for lane in DECISION_LANES:
            # Usamos counts_for_capex para iterar lanes relevantes
            max_count = counts_for_capex.get(lane, 0)
            if max_count <= 0:
                continue
            
            data = var_costs_map[lane]
            daily_hours = daily_hours_per_lane.get(lane, 0.0)

            # Costo Operativo (OPEX + Mantenimiento)
            # OPEX es por hora de operación real
            opex_anual_total = data.opex_hourly * daily_hours * DAYS_PER_YEAR
            
            # Mantenimiento es por lane instalado (anual)
            maint_total = data.maintenance_yearly * max_count

            # Costo Laboral (Sueldos)
            # Sueldos dependen de las horas operativas
            # Para SCO, la regla de supervisores aplica por bloque?
            # Simplificación: Calculamos wage por hora efectiva.
            # Si es SCO, necesitamos saber cuántos supervisores hubo cada hora.
            
            if lane == "self_checkout":
                # Recalcular horas de supervisores si es schedule
                if isinstance(counts, list):
                    wage_daily_total = 0.0
                    for start, end, block_counts in counts:
                        c = block_counts.get("self_checkout", 0)
                        if c > 0:
                            num_islas = math.ceil(c / 5.0)
                            num_supervisores = num_islas * 2
                            wage_daily_total += (
                                data.wage_hourly * num_supervisores * (end - start)
                            )
                    costo_laboral_anual = wage_daily_total * DAYS_PER_YEAR
                else:
                    # Static
                    num_islas = math.ceil(max_count / 5.0)
                    num_supervisores = num_islas * 2
                    wage_hora_total = data.wage_hourly * num_supervisores
                    costo_laboral_anual = wage_hora_total * daily_hours * DAYS_PER_YEAR
            else:
                # Cajas asistidas: 1 persona por caja -> wage * horas totales
                costo_laboral_anual = data.wage_hourly * daily_hours * DAYS_PER_YEAR

            opex_cajas_year += costo_laboral_anual + opex_anual_total + maint_total

            # Depreciacion (Usamos CAPEX base / vida util)
            if data.useful_life > 0:
                depreciacion_year += (
                    costs_base[lane].capex * max_count
                ) / data.useful_life

        # Calculo del Flujo del Ano
        total_opex_year = fixed_costs + opex_cajas_year
        ebitda_year = annual_gross_profit - total_opex_year
        ebit_year = ebitda_year - depreciacion_year

        nopat_year = ebit_year * (1 - TAX_RATE)
        fcf_year = nopat_year + depreciacion_year

        # Descuento al Valor Presente
        vp_flujos += fcf_year / ((1 + DISCOUNT_RATE) ** t)

        total_opex_acumulado += total_opex_year
        total_ebitda_acumulado += ebitda_year
        total_fcf_acumulado += fcf_year

    npv = -total_capex + vp_flujos

    return {
        "npv": npv,
        "capex": total_capex,
        "opex_anual_promedio": total_opex_acumulado / horizon_years,
        "ebitda_anual_promedio": total_ebitda_acumulado / horizon_years,
        "fcf_anual_promedio": total_fcf_acumulado / horizon_years,
        "gross_profit": annual_gross_profit,
    }


def _set_global_random_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    if hasattr(sim, "RNG_ITEMS"):
        sim.RNG_ITEMS = np.random.default_rng(seed + 1)
    if hasattr(sim, "RNG_PROFIT"):
        sim.RNG_PROFIT = np.random.default_rng(seed + 2)


def _apply_lane_config(x: tuple[int, ...]) -> list[tuple[int, int, dict[str, int]]]:
    schedule = lane_tuple_to_schedule(x)
    normalized_schedule = []
    for start, end, counts in schedule:
        norm_counts = sim.enforce_lane_constraints(counts)
        normalized_schedule.append((start, end, norm_counts))
    
    # We pass the schedule list to update_current_lane_policy
    # Note: sim.update_current_lane_policy expects a dict if we follow strict typing,
    # but we modified engine to handle list in _simular_dia_periodo.
    # However, update_current_lane_policy implementation in engine.py might need update
    # if it tries to merge dicts.
    # Let's check engine.py's update_current_lane_policy.
    # It calls build_uniform_policy(normalized).
    # If we pass a list, build_uniform_policy will fail.
    # So we should probably bypass update_current_lane_policy or update it.
    # For now, let's assume we update CURRENT_LANE_POLICY directly in engine via a new method
    # or just set it.
    # But wait, _simular_dia_periodo reads CURRENT_LANE_POLICY.
    # So we can just set it.
    
    # Actually, let's just update the specific day type entry in CURRENT_LANE_POLICY.
    # But _apply_lane_config doesn't know the day type.
    # It's called by _evaluate_policy_once which knows day_type.
    # But _apply_lane_config is called before _evaluate_policy_once uses day_type?
    # No, _evaluate_policy_once calls _apply_lane_config(x).
    
    # Issue: _apply_lane_config updates GLOBAL state in engine.
    # And it doesn't know which day type to update.
    # Previously it updated ALL day types with the same config.
    
    # We should probably change _apply_lane_config to NOT update global state,
    # and instead return the config, and let _evaluate_policy_once update it.
    # But _evaluate_policy_once calls sim.simulacion_periodos which runs the simulation.
    # sim.simulacion_periodos reads CURRENT_LANE_POLICY.
    
    # Workaround: Update ALL day types in CURRENT_LANE_POLICY with the schedule.
    for dt in sim.DayType:
        sim.CURRENT_LANE_POLICY[dt] = normalized_schedule
        
    return normalized_schedule


@dataclass
class EvalSAAResult:
    x: tuple[int, ...]
    day_type: sim.DayType
    profit_mean: float
    profit_std: float
    objetivo_mean: float
    objetivo_std: float
    n_rep: int
    elapsed_s: float
    kpis: dict[str, float] = None


def evaluate_policy_saa(
    x: tuple[int, ...],
    day_type: sim.DayType,
    num_weeks_sample: int = 1,
    num_rep: int = 3,
    keep_outputs: bool = False,
    eval_context: str | None = None,
    scenarios: Sequence[ScenarioWeight] | None = None,
    use_in_memory: bool = True,
    shared_output_root: Path | None = None,
) -> EvalSAAResult:
    global EVAL_COUNT
    EVAL_COUNT += 1
    
    start_t = time.time()
    
    # Apply policy (schedule)
    _apply_lane_config(x)
    
    # Setup simulation environment
    perfiles = [
        sim.CustomerProfile.DEAL_HUNTER,
        sim.CustomerProfile.FAMILY_CART,
        sim.CustomerProfile.WEEKLY_PLANNER,
        sim.CustomerProfile.SELF_CHECKOUT_FAN,
        sim.CustomerProfile.REGULAR,
        sim.CustomerProfile.EXPRESS_BASKET,
    ]
    cfg_by_prof = {p: sim.ProfileConfig(p) for p in perfiles}
    balk_model = sim.BALK_MODEL
    
    profits = []
    
    # Run replications
    # Note: For strict SAA, we should control seeds. 
    # Here we rely on the engine's randomness (which might be seeded globally).
    # If we want independent samples, we assume the engine advances RNG state.
    
    for i in range(num_rep):
        if keep_outputs:
            out_folder = (shared_output_root or PROJECT_ROOT / "outputs_opt") / f"eval_{EVAL_COUNT}_rep_{i}"
            out_folder.mkdir(parents=True, exist_ok=True)
        else:
            out_folder = Path("tmp_null")
            
        dia_config = (1, day_type, f"Opt Eval {EVAL_COUNT} Rep {i}")
        
        try:
            # Suppress stdout to avoid clutter during optimization
            # import contextlib, io
            # with contextlib.redirect_stdout(io.StringIO()):
            stats = sim._simular_dia_periodo(
                week_idx=1,
                dia_config=dia_config,
                perfiles=perfiles,
                cfg_by_prof=cfg_by_prof,
                balk_model=balk_model,
                output_folder=out_folder,
                write_outputs=keep_outputs
            )
            profits.append(stats["profit_total_clp"])
        except Exception as e:
            print(f"[ERROR] Simulation failed in eval {EVAL_COUNT}: {e}")
            profits.append(0.0)

    profit_mean = float(np.mean(profits)) if profits else 0.0
    profit_std = float(np.std(profits)) if profits else 0.0
    
    # Calculate Financial Metrics (NPV)
    schedule = lane_tuple_to_schedule(x)
    # Multiplicamos por 365 para proyectar el profit diario a anual
    metrics = calculate_financial_metrics(schedule, profit_mean * DAYS_PER_YEAR)
    
    # Calculate average KPIs from stats
    # stats is the result of the last replication, but we should ideally average across replications
    # For now, we will use the last replication's stats as a proxy or if we stored them all, average them.
    # Since we didn't store all stats objects, we can't average them here easily without refactoring.
    # Let's assume the user wants the KPIs of the last run or we can modify the loop to store them.
    # Given the constraints, let's use the last stats object if available.
    
    kpis = {}
    if 'stats' in locals() and stats:
        # Extract relevant KPIs from the stats dictionary
        # Assuming stats has keys like 'avg_wait_time', 'abandonment_rate', etc.
        # We need to check what _simular_dia_periodo returns exactly in 'stats'.
        # Based on previous context, it returns a dict with 'profit_total_clp' and likely others.
        # Let's extract what we can find or default to 0.
        kpis = {
            "avg_wait_time": stats.get("avg_wait_time", 0.0),
            "abandonment_rate": stats.get("abandonment_rate", 0.0),
            "avg_system_time": stats.get("avg_system_time", 0.0),
            "served_customers": stats.get("served_customers", 0),
        }

    return EvalSAAResult(
        x=x,
        day_type=day_type,
        profit_mean=profit_mean,
        profit_std=profit_std,
        objetivo_mean=metrics["npv"],
        objetivo_std=0.0, # Approximation
        n_rep=num_rep,
        elapsed_s=time.time() - start_t,
        kpis=kpis
    )


def generar_vecinos(
    x: tuple[int, ...],
    max_total_lanes: int | None = None,
) -> list[tuple[int, ...]]:
    if max_total_lanes is None:
        max_total_lanes = sim.MAX_TOTAL_LANES

    vecinos: set[tuple[int, ...]] = set()
    n_lanes = len(DECISION_LANES)

    for block_idx in range(len(TIME_BLOCKS)):
        offset = block_idx * n_lanes
        # Extract block tuple
        block_tuple = x[offset : offset + n_lanes]

        # Convert to dict for easier manipulation
        block_counts = {
            DECISION_LANES[j]: int(block_tuple[j]) for j in range(n_lanes)
        }

        # Enforce constraints on base block
        block_counts = sim.enforce_lane_constraints(block_counts)
        total_actual = sum(block_counts.values())

        for lane_name in DECISION_LANES:
            # Neighbor: -1
            if block_counts[lane_name] > 0:
                c_minus = block_counts.copy()
                c_minus[lane_name] = max(0, c_minus[lane_name] - 1)
                c_minus = sim.enforce_lane_constraints(c_minus)
                if sum(c_minus.values()) <= max_total_lanes:
                    # Construct new x
                    new_block_tuple = tuple(
                        int(c_minus.get(k, 0)) for k in DECISION_LANES
                    )
                    new_x = list(x)
                    new_x[offset : offset + n_lanes] = new_block_tuple
                    vecinos.add(tuple(new_x))

            # Neighbor: +1
            if total_actual < max_total_lanes:
                c_plus = block_counts.copy()
                c_plus[lane_name] = c_plus[lane_name] + 1
                c_plus = sim.enforce_lane_constraints(c_plus)
                if sum(c_plus.values()) <= max_total_lanes:
                    # Construct new x
                    new_block_tuple = tuple(
                        int(c_plus.get(k, 0)) for k in DECISION_LANES
                    )
                    new_x = list(x)
                    new_x[offset : offset + n_lanes] = new_block_tuple
                    vecinos.add(tuple(new_x))

    if x in vecinos:
        vecinos.remove(x)

    return list(vecinos)


def construir_solucion_inicial_grasp(
    day_type: sim.DayType,
    alpha: float = 0.5,
    num_weeks_sample: int = 1,
    num_rep_saa: int = 1,
    max_total_lanes: int | None = None,
    context_label: str | None = None,
    scenario_weights: Sequence[ScenarioWeight] | None = None,
    use_in_memory: bool = True,
    keep_outputs: bool = False,
    shared_output_root: Path | None = None,
) -> EvalSAAResult:
    if max_total_lanes is None:
        max_total_lanes = sim.MAX_TOTAL_LANES

    base_counts = sim.DEFAULT_LANE_COUNTS[day_type].copy()
    base_counts = sim.enforce_lane_constraints(base_counts)
    x_current = lane_dict_to_tuple(base_counts)
    base_context = context_label or f"Dia={day_type.name}"

    best_eval = evaluate_policy_saa(
        x_current,
        day_type=day_type,
        num_weeks_sample=num_weeks_sample,
        num_rep=num_rep_saa,
        eval_context=f"{base_context} | fase=GRASP | iter=0 (init)",
        scenarios=scenario_weights,
        use_in_memory=use_in_memory,
        keep_outputs=keep_outputs,
        shared_output_root=shared_output_root,
    )
    _print_iteration_progress("GRASP", 0, best_eval)

    improved = True
    iter_max = 10

    for step in range(1, iter_max + 1):
        if _time_exceeded() or _eval_limit_reached():
            print("Tiempo o limite de evaluaciones alcanzado durante la fase GRASP.")
            break
        if not improved:
            break
        improved = False

        vecinos = generar_vecinos(x_current, max_total_lanes=max_total_lanes)
        if not vecinos:
            break

        total_vecinos = len(vecinos)
        evals: list[EvalSAAResult] = []
        for idx, xv in enumerate(vecinos, start=1):
            if _time_exceeded() or _eval_limit_reached():
                break
            ev = evaluate_policy_saa(
                xv,
                day_type=day_type,
                num_weeks_sample=num_weeks_sample,
                num_rep=num_rep_saa,
                eval_context=(
                    f"{base_context} | fase=GRASP | iter={step}"
                    f" | vecino={idx}/{total_vecinos}"
                ),
                scenarios=scenario_weights,
                use_in_memory=use_in_memory,
                keep_outputs=keep_outputs,
                shared_output_root=shared_output_root,
            )
            evals.append(ev)

        if not evals:
            break

        evals.sort(key=lambda e: e.objetivo_mean, reverse=True)

        best_obj = evals[0].objetivo_mean
        worst_obj = evals[-1].objetivo_mean
        if best_obj <= best_eval.objetivo_mean:
            _print_iteration_progress("GRASP", step, best_eval)
            break

        if best_obj == worst_obj:
            rcl = evals
        else:
            threshold = best_obj - alpha * (best_obj - worst_obj)
            rcl = [e for e in evals if e.objetivo_mean >= threshold]
        if not rcl:
            break

        choice_idx = np.random.randint(len(rcl))
        chosen = rcl[choice_idx]

        if chosen.objetivo_mean > best_eval.objetivo_mean:
            best_eval = chosen
            x_current = chosen.x
            improved = True
        _print_iteration_progress("GRASP", step, best_eval)
        if not improved:
            break

    return best_eval


def busqueda_local(
    x0: tuple[int, int, int, int],
    day_type: sim.DayType,
    num_weeks_sample: int = 1,
    num_rep_saa: int = 3,
    max_iter: int = 20,
    max_total_lanes: int | None = None,
    use_sa: bool = False,
    T0: float = 1e6,
    cooling: float = 0.9,
    context_label: str | None = None,
    scenario_weights: Sequence[ScenarioWeight] | None = None,
    use_in_memory: bool = True,
    keep_outputs: bool = False,
    shared_output_root: Path | None = None,
) -> EvalSAAResult:
    if max_total_lanes is None:
        max_total_lanes = sim.MAX_TOTAL_LANES

    base_context = context_label or f"Dia={day_type.name}"

    current = evaluate_policy_saa(
        x0,
        day_type=day_type,
        num_weeks_sample=num_weeks_sample,
        num_rep=num_rep_saa,
        eval_context=f"{base_context} | fase=BUSQ | iter=0",
        scenarios=scenario_weights,
        use_in_memory=use_in_memory,
        keep_outputs=keep_outputs,
        shared_output_root=shared_output_root,
    )
    best = current
    T = float(T0)

    _print_iteration_progress("BUSQ", 0, current)

    for iter_idx in range(1, max_iter + 1):
        if _time_exceeded() or _eval_limit_reached():
            print(
                "Tiempo o limite de evaluaciones alcanzado durante la búsqueda local."
            )
            break

        vecinos = generar_vecinos(current.x, max_total_lanes=max_total_lanes)
        if not vecinos:
            break

        total_vecinos = len(vecinos)
        vec_evals: list[EvalSAAResult] = []
        for idx, xv in enumerate(vecinos, start=1):
            if _time_exceeded():
                break
            ev = evaluate_policy_saa(
                xv,
                day_type=day_type,
                num_weeks_sample=num_weeks_sample,
                num_rep=num_rep_saa,
                eval_context=(
                    f"{base_context} | fase=BUSQ | iter={iter_idx}"
                    f" | vecino={idx}/{total_vecinos}"
                ),
                scenarios=scenario_weights,
                use_in_memory=use_in_memory,
                keep_outputs=keep_outputs,
                shared_output_root=shared_output_root,
            )
            vec_evals.append(ev)

        if not vec_evals:
            break

        vec_evals.sort(key=lambda e: e.objetivo_mean, reverse=True)
        cand = vec_evals[0]

        delta = cand.objetivo_mean - current.objetivo_mean

        if delta > 0:
            current = cand
            if cand.objetivo_mean > best.objetivo_mean:
                best = cand
        else:
            if use_sa and T > 1e-6:
                prob = math.exp(delta / max(T, 1e-6))
                if np.random.random() < prob:
                    current = cand
        _print_iteration_progress("BUSQ", iter_idx, current)
        T *= cooling

    return best


def _fmt_clp(x: float) -> str:
    return f"{x:,.0f} CLP".replace(",", ".")


def _format_policy_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{lane}={counts.get(lane, 0)}" for lane in DECISION_LANES)


def _record_progress(label: str, iteration: int, res: EvalSAAResult) -> None:
    # Use the full schedule for accurate financial metrics
    schedule = lane_tuple_to_schedule(res.x)
    metrics = calculate_financial_metrics(schedule, res.profit_mean * DAYS_PER_YEAR)

    # For logging 'counts', we use the max counts (legacy compatibility)
    counts = sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))

    OPT_PROGRESS.append(
        {
            "phase": label,
            "iteration": iteration,
            "profit_gross": res.profit_mean * DAYS_PER_YEAR,
            "npv": res.objetivo_mean,  # Objetivo es VPN
            "capex": metrics["capex"],
            "ebitda": metrics["ebitda_anual_promedio"],
            "counts": counts.copy(),
            "x": res.x,
            "day_type": res.day_type.name,
            "elapsed_s": time.time() - (START_TIME or time.time()),
        }
    )


def _print_iteration_progress(label: str, iteration: int, res: EvalSAAResult) -> None:
    _record_progress(label, iteration, res)
    counts = OPT_PROGRESS[-1]["counts"]
    npv_val = OPT_PROGRESS[-1]["npv"]
    profit_val = OPT_PROGRESS[-1]["profit_gross"]
    counts_str = _format_policy_counts(counts)
    print(
        f"[{label}] Iter {iteration}: "
        f"x={res.x} ({counts_str}) | "
        f"ProfitBruto(Anual)={_fmt_clp(profit_val)} | "
        f"VPN={_fmt_clp(npv_val)}"
    )


def imprimir_resumen_resultado(res: EvalSAAResult, titulo: str = "Resultado") -> None:
    counts = sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))
    total_lanes = sum(counts.values())
    
    metrics = calculate_financial_metrics(counts, res.profit_mean * DAYS_PER_YEAR)

    print(f"\n--- {titulo} ({res.day_type.name}) ---")
    print(
        f"Configuración x = (regular={res.x[0]}, express={res.x[1]}, "
        f"priority={res.x[2]}, self_checkout={res.x[3]})"
    )
    print(f"Config normalizada: {counts}  (total {total_lanes} cajas)")

    
    print("\nEstructura Financiera (Estimada):")
    print(f"  Inversión Inicial (CAPEX) : {_fmt_clp(metrics['capex'])}")
    print(f"  OPEX Anual Promedio (Total): {_fmt_clp(metrics['opex_anual_promedio'])}")
    print(f"  Profit Bruto Anual (Sim)   : {_fmt_clp(res.profit_mean * DAYS_PER_YEAR)}")
    print(f"  EBITDA Anual Promedio      : {_fmt_clp(metrics['ebitda_anual_promedio'])}")
    print(f"  Flujo Caja Libre (FCF)    : {_fmt_clp(metrics['fcf_anual_promedio'])}")
    print("\nDesempeño (Objetivo = VPN):")
    print(
        f"  VPN (5 años, {DISCOUNT_RATE*100}%) : {_fmt_clp(res.objetivo_mean)} "
        f"(± {res.objetivo_std:,.0f})"
    )
    
    if total_lanes > 0:
        print(f"  VPN medio por caja      : {_fmt_clp(res.objetivo_mean / total_lanes)}")
    
    if res.kpis:
        print("\nKPIs Operativos (Estimados):")
        print(f"  Tiempo Espera Promedio  : {res.kpis.get('avg_wait_time', 0):.1f} seg")
        print(f"  Tasa de Abandono        : {res.kpis.get('abandonment_rate', 0)*100:.1f}%")
        print(f"  Clientes Atendidos (Día): {res.kpis.get('served_customers', 0)}")
        
    print(f"  Réplicas SAA usadas: {res.n_rep}")
    print("------------------------------")


def _export_progress_plot(day_type: sim.DayType) -> Path | None:
    entries = [e for e in OPT_PROGRESS if e["day_type"] == day_type.name]
    if not entries:
        return None
    out_root = PROJECT_ROOT / "resultados_opt"
    out_root.mkdir(parents=True, exist_ok=True)
    idx = range(len(entries))
    objectives = [e["npv"] for e in entries]
    profits = [e["profit_gross"] for e in entries]
    capex = [e["capex"] for e in entries]
    labels = [f"{e['phase']}#{e['iteration']}" for e in entries]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(idx, objectives, label="VPN (Objetivo)", marker="o", color="blue")
    ax.plot(idx, profits, label="Profit Bruto", marker="s", color="green", alpha=0.5)
    ax.plot(idx, capex, label="CAPEX", marker="^", color="red", alpha=0.5)
    ax.set_title(f"Progreso optimización {day_type.name} (VPN)")
    ax.set_xlabel("Evaluación")
    ax.set_ylabel("CLP (miles de millones)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{val / 1e6:.0f}M"))
    base_obj = entries[0]["npv"]
    ax.axhline(base_obj, linestyle="--", color="gray", label="VPN base")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    best_entry = max(entries, key=lambda e: e["npv"])
    counts_text = _format_policy_counts(best_entry["counts"])
    improvement_pct = (
        ((best_entry["npv"] - base_obj) / abs(base_obj)) * 100.0
        if base_obj != 0
        else 0.0
    )
    text = (
        f"Mejor política: {best_entry['x']} ({counts_text})\n"
        f"Profit Bruto={_fmt_clp(best_entry['profit_gross'])} | "
        f"CAPEX={_fmt_clp(best_entry['capex'])} | "
        f"VPN={_fmt_clp(best_entry['npv'])}\n"
        f"Mejora vs base: {improvement_pct:.2f}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.01, 0.01, text, fontsize=9, ha="left", va="bottom")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = out_root / f"optimizer_progress_{day_type.name}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Progreso guardado en {plot_path}")
    return plot_path


def _export_progress_csv(day_type: sim.DayType) -> Path | None:
    entries = [e for e in OPT_PROGRESS if e["day_type"] == day_type.name]
    if not entries:
        return None
    out_root = PROJECT_ROOT / "resultados_opt"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        counts_str = _format_policy_counts(e["counts"])
        rows.append(
            {
                "phase": e["phase"],
                "iteration": e["iteration"],
                "profit_gross_mean": e["profit_gross"],
                "capex": e["capex"],
                "ebitda": e["ebitda"],
                "npv_mean": e["npv"],
                "lane_counts": counts_str,
                "x_tuple": e["x"],
                "elapsed_s": e["elapsed_s"],
            }
        )
    df = pd.DataFrame(rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_root / f"optimizer_progress_{day_type.name}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Tabla de progreso guardada en {csv_path}")
    return csv_path


def optimizar_cajas_grasp_saa(
    day_type: sim.DayType,
    num_weeks_sample_construccion: int = 1,
    num_rep_saa_construccion: int = 1,
    num_weeks_sample_busqueda: int = 1,
    num_rep_saa_busqueda: int = 3,
    max_total_lanes: int | None = None,
    alpha: float = 0.5,
    use_sa: bool = False,
    max_seconds: float | None = None,
    max_eval_count: int | None = None,
    initial_solution: tuple[int, int, int, int] | None = None,
    context_label: str | None = None,
    scenario_weights: Sequence[ScenarioWeight] | None = None,
    use_in_memory: bool | None = None,
    keep_outputs_eval: bool = False,
) -> EvalSAAResult:
    global START_TIME, MAX_SECONDS, MAX_EVAL_COUNT, EVAL_COUNT
    START_TIME = time.time()
    MAX_SECONDS = max_seconds if max_seconds and max_seconds > 0 else None
    MAX_EVAL_COUNT = max_eval_count if max_eval_count and max_eval_count > 0 else None
    EVAL_COUNT = 0
    OPT_PROGRESS.clear()

    best: EvalSAAResult | None = None
    x0_eval: EvalSAAResult | None = None
    context_base = context_label or f"Dia={day_type.name}"

    if use_in_memory is None:
        use_in_memory = not keep_outputs_eval
    shared_tmp = None
    if not use_in_memory and not keep_outputs_eval:
        # Reutilizamos un directorio temporal para todas las rplicas/vecinos
        from tempfile import TemporaryDirectory

        shared_tmp_ctx = TemporaryDirectory()
        shared_tmp = Path(shared_tmp_ctx.name)

    try:
        print(f"\n=== Optimizacion para dia tipo {day_type.name} ===")

        if initial_solution is not None:
            tuple_init = tuple(int(max(0, v)) for v in initial_solution)
            normalized = lane_dict_to_tuple(
                sim.enforce_lane_constraints(lane_tuple_to_dict(tuple_init))
            )
            x0_eval = evaluate_policy_saa(
                normalized,
                day_type=day_type,
                num_weeks_sample=num_weeks_sample_busqueda,
                num_rep=num_rep_saa_busqueda,
                keep_outputs=keep_outputs_eval,
                eval_context=f"{context_base} | fase=INIT_HEREDADA",
                scenarios=scenario_weights,
                use_in_memory=use_in_memory,
                shared_output_root=shared_tmp,
            )
            imprimir_resumen_resultado(
                x0_eval,
                titulo=f"Solucion inicial (heredada) para {day_type.name}",
            )
        else:
            x0_eval = construir_solucion_inicial_grasp(
                day_type=day_type,
                alpha=alpha,
                num_weeks_sample=num_weeks_sample_construccion,
                num_rep_saa=num_rep_saa_construccion,
                max_total_lanes=max_total_lanes,
                context_label=context_base,
                scenario_weights=scenario_weights,
                use_in_memory=use_in_memory,
                keep_outputs=keep_outputs_eval,
                shared_output_root=shared_tmp,
            )
            imprimir_resumen_resultado(
                x0_eval,
                titulo=f"Solucion inicial (GRASP) para {day_type.name}",
            )

        if _time_exceeded():
            print(
                "Tiempo maximo alcanzado tras la fase GRASP. Devolviendo solucion inicial."
            )
            _export_progress_plot(day_type)
            _export_progress_csv(day_type)
            return x0_eval

        best = busqueda_local(
            x0_eval.x,
            day_type=day_type,
            num_weeks_sample=num_weeks_sample_busqueda,
            num_rep_saa=num_rep_saa_busqueda,
            max_iter=15,
            max_total_lanes=max_total_lanes,
            use_sa=use_sa,
            T0=1e6,
            cooling=0.85,
            context_label=context_base,
            scenario_weights=scenario_weights,
            use_in_memory=use_in_memory,
            keep_outputs=keep_outputs_eval,
            shared_output_root=shared_tmp,
        )

        imprimir_resumen_resultado(
            best,
            titulo=f"Mejor solucion encontrada para {day_type.name}",
        )

        _export_progress_plot(day_type)
        _export_progress_csv(day_type)
        return best
    except KeyboardInterrupt:
        print("\n[WARN] Optimizacion interrumpida por el usuario (Ctrl+C).")
        if best is not None:
            imprimir_resumen_resultado(best, titulo="Mejor solucion parcial")
            _export_progress_plot(day_type)
            _export_progress_csv(day_type)
            return best
        if x0_eval is not None:
            imprimir_resumen_resultado(x0_eval, titulo="Solucion inicial disponible")
            _export_progress_plot(day_type)
            _export_progress_csv(day_type)
            return x0_eval
        raise
    finally:
        # Limpia el directorio temporal compartido si se cre en modo rpido con disco.
        if shared_tmp is not None:
            try:
                shared_tmp_ctx.cleanup()
            except Exception:
                pass

@dataclass
class MultiDayResult:
    por_tipo: dict[sim.DayType, EvalSAAResult]
    dias_por_tipo: dict[sim.DayType, int]
    profit_bruto_global_anual: float
    capex_total: float
    npv_global: float


def combinar_resultados_por_tipo(
    res_por_tipo: dict[sim.DayType, EvalSAAResult],
    dias_por_tipo: dict[sim.DayType, int],
) -> MultiDayResult:
    total_days = float(sum(dias_por_tipo.values()))
    if total_days <= 0:
        raise ValueError("La suma de dias_por_tipo debe ser > 0.")

    frac_por_tipo: dict[sim.DayType, float] = {
        dt: dias_por_tipo[dt] / total_days for dt in dias_por_tipo
    }

    profit_global = 0.0
    for dt, res in res_por_tipo.items():
        frac = frac_por_tipo.get(dt, 0.0)
        profit_global += frac * res.profit_mean

    counts_por_tipo: dict[sim.DayType, dict[str, int]] = {}
    for dt, res in res_por_tipo.items():
        counts_por_tipo[dt] = sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))

    counts_max: dict[str, int] = {lane: 0 for lane in DECISION_LANES}
    for lane in DECISION_LANES:
        counts_max[lane] = max(counts_por_tipo[dt][lane] for dt in counts_por_tipo)

    # Calculamos el VPN global considerando la infraestructura máxima necesaria
    # y el profit promedio ponderado.
    metrics_global = calculate_financial_metrics(counts_max, profit_global)

    return MultiDayResult(
        por_tipo=res_por_tipo,
        dias_por_tipo=dias_por_tipo,
        profit_bruto_global_anual=profit_global,
        capex_total=metrics_global["capex"],
        npv_global=metrics_global["npv"],
    )


def imprimir_resumen_global_multi_dia(multi: MultiDayResult) -> None:
    dias_por_tipo = multi.dias_por_tipo
    total_days = float(sum(dias_por_tipo.values()))
    frac_por_tipo = {dt: dias_por_tipo[dt] / total_days for dt in dias_por_tipo}

    print("\n=== RESUMEN GLOBAL (todos los tipos de día) ===")
    print("Días por tipo:")
    for dt, n in dias_por_tipo.items():
        print(f"  {dt.name:10s}: {n:4d} días ({100.0*frac_por_tipo[dt]:5.1f} %)")

    print("\nResultados por tipo de día (VPN individual):")
    for dt, res in multi.por_tipo.items():
        print(
            f"- {dt.name:10s}: Profit Bruto ≈ {_fmt_clp(res.profit_mean)} "
            f"| VPN ≈ {_fmt_clp(res.objetivo_mean)}"
        )

    counts_por_tipo = {
        dt: sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))
        for dt, res in multi.por_tipo.items()
    }
    counts_max = {lane: 0 for lane in DECISION_LANES}
    for lane in DECISION_LANES:
        counts_max[lane] = max(counts_por_tipo[dt][lane] for dt in counts_por_tipo)

    print("\nInfraestructura requerida (configuración máxima entre tipos):")
    print(
        f"  Configuración máxima: {counts_max} "
        f"(total {sum(counts_max.values())} cajas)"
    )
    print(f"  CAPEX Total (Inversión t=0): {_fmt_clp(multi.capex_total)}")

    print("\nDesempeño Global (Ponderado):")
    print(f"  Profit Bruto Anual Promedio ≈ {_fmt_clp(multi.profit_bruto_global_anual)}")
    print(f"  VPN Global del Proyecto     ≈ {_fmt_clp(multi.npv_global)}")
    print("==============================================")


def run_optimizer_cli() -> None:
    dias_por_tipo = {
        sim.DayType.TYPE_1: 220,
        sim.DayType.TYPE_2: 100,
        sim.DayType.TYPE_3: 45,
    }

    res_por_tipo = {}
    for dt in [sim.DayType.TYPE_1, sim.DayType.TYPE_2, sim.DayType.TYPE_3]:
        res = optimizar_cajas_grasp_saa(
            day_type=dt,
            num_weeks_sample_construccion=1,
            num_rep_saa_construccion=1,
            num_weeks_sample_busqueda=1,
            num_rep_saa_busqueda=1,
            alpha=0.5,
            use_sa=False,
            max_seconds=600.0,
            max_eval_count=None,
            context_label=f"CLI-SCRIPT | Día={dt.name}",
        )
        res_por_tipo[dt] = res

    multi = combinar_resultados_por_tipo(res_por_tipo, dias_por_tipo)
    imprimir_resumen_global_multi_dia(multi)


if __name__ == "__main__":
    run_optimizer_cli()
