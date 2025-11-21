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

PROJECT_ROOT = Path(__file__).resolve().parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import engine as sim
from simulator.reporting import load_customers_year

DECISION_LANES = ["regular", "express", "priority", "self_checkout"]

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 14
HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY

WAGE_PER_HOUR = {
    "regular": 4500,
    "express": 4500,
    "priority": 4700,
    "self_checkout": 5000,
}

OPEX_PER_HOUR = {
    "regular": 400,
    "express": 420,
    "priority": 400,
    "self_checkout": 500,
}

CAPEX_CLP = {
    "regular": 8_000_000,
    "express": 8_000_000,
    "priority": 8_000_000,
    "self_checkout": 25_000_000,
}

MAINT_CLP_PER_YEAR = {
    "regular": 800_000,
    "express": 800_000,
    "priority": 800_000,
    "self_checkout": 1_500_000,
}

START_TIME: float | None = None
MAX_SECONDS: float | None = None
MAX_EVALS: int | None = None
EVAL_COUNT: int = 0
OPT_PROGRESS: list[dict] = []


def _time_exceeded() -> bool:
    if START_TIME is None or MAX_SECONDS is None:
        return False
    return (time.time() - START_TIME) >= MAX_SECONDS


def _eval_limit_reached() -> bool:
    return MAX_EVALS is not None and EVAL_COUNT >= MAX_EVALS


def _load_annual_costs_from_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    required_cols = {"lane_type", "supervisors", "useful_life_years"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {csv_path.name}: {', '.join(sorted(missing))}"
        )

    costs: dict[str, float] = {}
    for _, row in df.iterrows():
        lane = str(row["lane_type"]).strip().lower()
        if lane not in DECISION_LANES:
            continue

        life = float(row["useful_life_years"])
        supervisors = float(row["supervisors"])

        capex = float(CAPEX_CLP[lane])
        maint = float(MAINT_CLP_PER_YEAR[lane])

        if lane == "self_checkout":
            wage_h = WAGE_PER_HOUR["self_checkout"] * supervisors
        else:
            wage_h = WAGE_PER_HOUR[lane] * 1.0

        opex_h = float(OPEX_PER_HOUR[lane])

        annual_capex = capex / max(life, 1.0)
        annual_running = (wage_h + opex_h) * HOURS_PER_YEAR

        total_annual = annual_capex + maint + annual_running
        costs[lane] = total_annual

    for lane in DECISION_LANES:
        if lane not in costs:
            raise ValueError(
                f"No se encontró costo para lane_type='{lane}' en {csv_path.name}"
            )

    return costs


COSTS_CSV_PATH = PROJECT_ROOT / "costos_cajas.csv"
COST_PER_LANE_ANNUAL = _load_annual_costs_from_csv(COSTS_CSV_PATH)


def lane_dict_to_tuple(counts: dict[str, int]) -> tuple[int, int, int, int]:
    return tuple(int(counts.get(k, 0)) for k in DECISION_LANES)


def lane_tuple_to_dict(x: tuple[int, int, int, int]) -> dict[str, int]:
    r, e, p, s = x
    return {
        "regular": int(r),
        "express": int(e),
        "priority": int(p),
        "self_checkout": int(s),
    }


def cost_anual_config(counts: dict[str, int]) -> float:
    return float(
        sum(
            int(counts.get(k, 0)) * float(COST_PER_LANE_ANNUAL.get(k, 0.0))
            for k in DECISION_LANES
        )
    )


def _set_global_random_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    if hasattr(sim, "RNG_ITEMS"):
        sim.RNG_ITEMS = np.random.default_rng(seed + 1)
    if hasattr(sim, "RNG_PROFIT"):
        sim.RNG_PROFIT = np.random.default_rng(seed + 2)


def _apply_lane_config(x: tuple[int, int, int, int]) -> dict[str, int]:
    raw_counts = lane_tuple_to_dict(x)
    normalized = sim.enforce_lane_constraints(raw_counts)
    sim.update_current_lane_policy(normalized)
    return normalized


def _evaluate_policy_once(
    x: tuple[int, int, int, int],
    day_type: sim.DayType,
    num_weeks_sample: int = 1,
    run_id: str = "run",
    keep_outputs: bool = False,
    context: str | None = None,
) -> tuple[float, float]:
    counts_norm = _apply_lane_config(x)

    output_root = PROJECT_ROOT / "outputs_opt" / f"{run_id}"
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)

    if context:
        print(f"[SIM] {context} | semanas={num_weeks_sample} | run_id={run_id} | x={x}")
    else:
        print(
            f"[SIM] Dia={day_type.name} | semanas={num_weeks_sample} | run_id={run_id} | x={x}"
        )

    sim.simulacion_periodos(
        num_weeks=num_weeks_sample,
        output_root=str(output_root),
        include_timestamp=False,
        start_week_idx=1,
        titulo=f"OPT-EVAL {run_id} x={x}",
    )

    df_eval = load_customers_year(output_root)
    df_eval["total_profit_clp"] = pd.to_numeric(
        df_eval["total_profit_clp"], errors="coerce"
    ).fillna(0)

    df_dt = df_eval[df_eval["dia_tipo"].str.lower() == day_type.value]
    prof_muestra = df_dt.loc[
        df_dt["outcome_norm"] == "served", "total_profit_clp"
    ].sum()

    factor_anual = 52.0 / float(num_weeks_sample)
    profit_anual = float(prof_muestra * factor_anual)

    cost_anual = cost_anual_config(counts_norm)
    objetivo = profit_anual - cost_anual

    if not keep_outputs:
        shutil.rmtree(output_root, ignore_errors=True)

    if context:
        print(f"[SIM] {context} | completado | objetivo={objetivo:,.0f}")
    else:
        print(f"[SIM] Dia={day_type.name} | completado | objetivo={objetivo:,.0f}")

    return float(objetivo), float(profit_anual)


@dataclass
class EvalSAAResult:
    x: tuple[int, int, int, int]
    objetivo_mean: float
    objetivo_std: float
    profit_mean: float
    profit_std: float
    n_rep: int
    day_type: sim.DayType


_FITNESS_CACHE: dict[tuple[tuple[int, int, int, int], sim.DayType, float], EvalSAAResult] = {}


ScenarioWeight = tuple[str, float, float]


def evaluate_policy_saa(
    x: tuple[int, int, int, int],
    day_type: sim.DayType,
    num_weeks_sample: int = 1,
    num_rep: int = 3,
    keep_outputs: bool = False,
    eval_context: str | None = None,
    scenarios: Sequence[ScenarioWeight] | None = None,
) -> EvalSAAResult:
    if scenarios:
        return _evaluate_policy_multi_scenarios(
            x,
            day_type=day_type,
            num_weeks_sample=num_weeks_sample,
            num_rep=num_rep,
            keep_outputs=keep_outputs,
            eval_context=eval_context,
            scenarios=scenarios,
        )
    return _evaluate_policy_saa_single(
        x,
        day_type=day_type,
        num_weeks_sample=num_weeks_sample,
        num_rep=num_rep,
        keep_outputs=keep_outputs,
        eval_context=eval_context,
    )


def _evaluate_policy_saa_single(
    x: tuple[int, int, int, int],
    day_type: sim.DayType,
    num_weeks_sample: int = 1,
    num_rep: int = 3,
    keep_outputs: bool = False,
    eval_context: str | None = None,
) -> EvalSAAResult:
    x = tuple(int(v) for v in x)
    current_factor = float(sim.get_demand_multiplier())
    factor_key = round(current_factor, 9)
    key = (x, day_type, factor_key)

    if key in _FITNESS_CACHE:
        return _FITNESS_CACHE[key]

    objetivos: list[float] = []
    profits: list[float] = []

    for rep in range(num_rep):
        if _time_exceeded() or _eval_limit_reached():
            break

        seed = 12345 + 10000 * rep + 101 * x[0] + 17 * x[1] + 7 * x[2] + 3 * x[3]
        _set_global_random_seeds(seed)

        unique_tag = uuid.uuid4().hex[:8]
        run_id = (
            f"{day_type.name}_x_{x[0]}_{x[1]}_{x[2]}_{x[3]}_" f"rep{rep}_{unique_tag}"
        )
        run_context = eval_context or f"Dia={day_type.name}"
        obj, prof = _evaluate_policy_once(
            x,
            day_type=day_type,
            num_weeks_sample=num_weeks_sample,
            run_id=run_id,
            keep_outputs=keep_outputs,
            context=f"{run_context} | rep={rep + 1}/{num_rep}",
        )
        global EVAL_COUNT
        EVAL_COUNT += 1
        objetivos.append(obj)
        profits.append(prof)

    if not objetivos:
        res = EvalSAAResult(
            x=x,
            objetivo_mean=float("-inf"),
            objetivo_std=0.0,
            profit_mean=0.0,
            profit_std=0.0,
            n_rep=0,
            day_type=day_type,
        )
        return res

    objetivos_arr = np.asarray(objetivos, dtype=float)
    profits_arr = np.asarray(profits, dtype=float)

    res = EvalSAAResult(
        x=x,
        objetivo_mean=float(objetivos_arr.mean()),
        objetivo_std=float(
            objetivos_arr.std(ddof=1) if len(objetivos_arr) > 1 else 0.0
        ),
        profit_mean=float(profits_arr.mean()),
        profit_std=float(profits_arr.std(ddof=1) if len(profits_arr) > 1 else 0.0),
        n_rep=len(objetivos),
        day_type=day_type,
    )

    _FITNESS_CACHE[key] = res
    return res


def _evaluate_policy_multi_scenarios(
    x: tuple[int, int, int, int],
    day_type: sim.DayType,
    *,
    num_weeks_sample: int,
    num_rep: int,
    keep_outputs: bool,
    eval_context: str | None,
    scenarios: Sequence[ScenarioWeight],
) -> EvalSAAResult:
    prev_factor = float(sim.get_demand_multiplier())
    agg_profit = 0.0
    agg_profit_std = 0.0
    agg_obj = 0.0
    agg_obj_std = 0.0
    last_res: EvalSAAResult | None = None
    total_weight = 0.0

    try:
        for label, factor, weight in scenarios:
            weight_val = float(weight) if weight is not None else 1.0
            if weight_val <= 0:
                continue
            total_weight += weight_val
            sim.set_demand_multiplier(float(factor))
            res = _evaluate_policy_saa_single(
                x,
                day_type=day_type,
                num_weeks_sample=num_weeks_sample,
                num_rep=num_rep,
                keep_outputs=keep_outputs,
                eval_context=(
                    f"{eval_context} | escenario={label}" if eval_context else None
                ),
            )
            last_res = res
            agg_profit += weight_val * res.profit_mean
            agg_profit_std += weight_val * res.profit_std
            agg_obj += weight_val * res.objetivo_mean
            agg_obj_std += weight_val * res.objetivo_std
    finally:
        sim.set_demand_multiplier(prev_factor)

    if total_weight <= 0 or last_res is None:
        sim.set_demand_multiplier(prev_factor)
        return _evaluate_policy_saa_single(
            x,
            day_type=day_type,
            num_weeks_sample=num_weeks_sample,
            num_rep=num_rep,
            keep_outputs=keep_outputs,
            eval_context=eval_context,
        )

    return EvalSAAResult(
        x=x,
        objetivo_mean=float(agg_obj),
        objetivo_std=float(agg_obj_std),
        profit_mean=float(agg_profit),
        profit_std=float(agg_profit_std),
        n_rep=last_res.n_rep,
        day_type=day_type,
    )


def generar_vecinos(
    x: tuple[int, int, int, int],
    max_total_lanes: int | None = None,
) -> list[tuple[int, int, int, int]]:
    if max_total_lanes is None:
        max_total_lanes = sim.MAX_TOTAL_LANES

    vecinos: set[tuple[int, int, int, int]] = set()
    base_counts = lane_tuple_to_dict(x)
    base_counts = sim.enforce_lane_constraints(base_counts)
    base_tuple = lane_dict_to_tuple(base_counts)
    total_actual = sum(base_counts.values())

    for lane_name in DECISION_LANES:
        if base_counts[lane_name] > 0:
            c_minus = base_counts.copy()
            c_minus[lane_name] = max(0, c_minus[lane_name] - 1)
            c_minus = sim.enforce_lane_constraints(c_minus)
            if sum(c_minus.values()) <= max_total_lanes:
                vecinos.add(lane_dict_to_tuple(c_minus))

        if total_actual < max_total_lanes:
            c_plus = base_counts.copy()
            c_plus[lane_name] = c_plus[lane_name] + 1
            c_plus = sim.enforce_lane_constraints(c_plus)
            if sum(c_plus.values()) <= max_total_lanes:
                vecinos.add(lane_dict_to_tuple(c_plus))

    if base_tuple in vecinos:
        vecinos.remove(base_tuple)

    return list(vecinos)


def construir_solucion_inicial_grasp(
    day_type: sim.DayType,
    alpha: float = 0.5,
    num_weeks_sample: int = 1,
    num_rep_saa: int = 1,
    max_total_lanes: int | None = None,
    context_label: str | None = None,
    scenario_weights: Sequence[ScenarioWeight] | None = None,
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
    counts = sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))
    cost_cfg = cost_anual_config(counts)
    OPT_PROGRESS.append(
        {
            "phase": label,
            "iteration": iteration,
            "profit": res.profit_mean,
            "cost": cost_cfg,
            "objective": res.objetivo_mean,
            "counts": counts.copy(),
            "x": res.x,
            "day_type": res.day_type.name,
            "elapsed_s": time.time() - (START_TIME or time.time()),
        }
    )


def _print_iteration_progress(label: str, iteration: int, res: EvalSAAResult) -> None:
    _record_progress(label, iteration, res)
    counts = OPT_PROGRESS[-1]["counts"]
    cost_cfg = OPT_PROGRESS[-1]["cost"]
    counts_str = _format_policy_counts(counts)
    print(
        f"[{label}] Iter {iteration}: "
        f"x={res.x} ({counts_str}) | "
        f"Profit={_fmt_clp(res.profit_mean)} | "
        f"Costo={_fmt_clp(cost_cfg)} | "
        f"Objetivo={_fmt_clp(res.objetivo_mean)}"
    )


def imprimir_resumen_resultado(res: EvalSAAResult, titulo: str = "Resultado") -> None:
    counts = sim.enforce_lane_constraints(lane_tuple_to_dict(res.x))
    total_lanes = sum(counts.values())
    cost_config = cost_anual_config(counts)

    print(f"\n--- {titulo} ({res.day_type.name}) ---")
    print(
        f"Configuración x = (regular={res.x[0]}, express={res.x[1]}, "
        f"priority={res.x[2]}, self_checkout={res.x[3]})"
    )
    print(f"Config normalizada: {counts}  (total {total_lanes} cajas)")

    print("\nCostos:")
    print(f"  Costo anual configuración: {_fmt_clp(cost_config)}")
    print("\nDesempeño estimado (SAA):")
    print(
        f"  Profit anual medio      ≈ {_fmt_clp(res.profit_mean)} "
        f"(± {res.profit_std:,.0f})"
    )
    print(
        f"  Objetivo (profit - costo) medio ≈ {_fmt_clp(res.objetivo_mean)} "
        f"(± {res.objetivo_std:,.0f})"
    )
    if total_lanes > 0:
        print(f"  Profit medio por caja   ≈ {_fmt_clp(res.profit_mean / total_lanes)}")
        print(
            f"  Objetivo medio por caja ≈ {_fmt_clp(res.objetivo_mean / total_lanes)}"
        )
    print(f"  Réplicas SAA usadas: {res.n_rep}")
    print("------------------------------")


def _export_progress_plot(day_type: sim.DayType) -> Path | None:
    entries = [e for e in OPT_PROGRESS if e["day_type"] == day_type.name]
    if not entries:
        return None
    out_root = PROJECT_ROOT / "resultados_opt"
    out_root.mkdir(parents=True, exist_ok=True)
    idx = range(len(entries))
    objectives = [e["objective"] for e in entries]
    profits = [e["profit"] for e in entries]
    costs = [e["cost"] for e in entries]
    labels = [f"{e['phase']}#{e['iteration']}" for e in entries]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(idx, objectives, label="Objetivo", marker="o")
    ax.plot(idx, profits, label="Profit", marker="s")
    ax.plot(idx, costs, label="Costo", marker="^")
    ax.set_title(f"Progreso optimización {day_type.name}")
    ax.set_xlabel("Evaluación")
    ax.set_ylabel("CLP (miles de millones)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{val / 1e9:.2f}"))
    base_obj = entries[0]["objective"]
    ax.axhline(base_obj, linestyle="--", color="gray", label="Objetivo base")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    best_entry = max(entries, key=lambda e: e["objective"])
    counts_text = _format_policy_counts(best_entry["counts"])
    improvement_pct = (
        ((best_entry["objective"] - base_obj) / abs(base_obj)) * 100.0
        if base_obj not in (0, None)
        else 0.0
    )
    text = (
        f"Mejor política: {best_entry['x']} ({counts_text})\n"
        f"Profit={_fmt_clp(best_entry['profit'])} | "
        f"Costo={_fmt_clp(best_entry['cost'])} | "
        f"Objetivo={_fmt_clp(best_entry['objective'])}\n"
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
                "profit_mean": e["profit"],
                "cost_mean": e["cost"],
                "objective_mean": e["objective"],
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
) -> EvalSAAResult:
    global START_TIME, MAX_SECONDS, MAX_EVALS, EVAL_COUNT
    START_TIME = time.time()
    MAX_SECONDS = max_seconds if max_seconds and max_seconds > 0 else None
    MAX_EVALS = max_eval_count if max_eval_count and max_eval_count > 0 else None
    EVAL_COUNT = 0
    OPT_PROGRESS.clear()

    best: EvalSAAResult | None = None
    x0_eval: EvalSAAResult | None = None
    context_base = context_label or f"Dia={day_type.name}"

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
                keep_outputs=False,
                eval_context=f"{context_base} | fase=INIT_HEREDADA",
                scenarios=scenario_weights,
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


@dataclass
class MultiDayResult:
    por_tipo: dict[sim.DayType, EvalSAAResult]
    dias_por_tipo: dict[sim.DayType, int]
    profit_global: float
    cost_infra_anual: float
    objective_global: float


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

    cost_infra_anual = cost_anual_config(counts_max)
    objective_global = profit_global - cost_infra_anual

    return MultiDayResult(
        por_tipo=res_por_tipo,
        dias_por_tipo=dias_por_tipo,
        profit_global=profit_global,
        cost_infra_anual=cost_infra_anual,
        objective_global=objective_global,
    )


def imprimir_resumen_global_multi_dia(multi: MultiDayResult) -> None:
    dias_por_tipo = multi.dias_por_tipo
    total_days = float(sum(dias_por_tipo.values()))
    frac_por_tipo = {dt: dias_por_tipo[dt] / total_days for dt in dias_por_tipo}

    print("\n=== RESUMEN GLOBAL (todos los tipos de día) ===")
    print("Días por tipo:")
    for dt, n in dias_por_tipo.items():
        print(f"  {dt.name:10s}: {n:4d} días ({100.0*frac_por_tipo[dt]:5.1f} %)")

    print("\nResultados por tipo de día:")
    for dt, res in multi.por_tipo.items():
        print(
            f"- {dt.name:10s}: profit medio ≈ {_fmt_clp(res.profit_mean)} "
            f"(± {res.profit_std:,.0f}), objetivo ≈ {_fmt_clp(res.objetivo_mean)}"
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
    print(f"  Costo anual infraestructura: {_fmt_clp(multi.cost_infra_anual)}")

    print("\nObjetivo global (ponderado por frecuencia de tipos de día):")
    print(f"  Profit anual combinado ≈ {_fmt_clp(multi.profit_global)}")
    print(f"  Objetivo global        ≈ {_fmt_clp(multi.objective_global)}")
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
