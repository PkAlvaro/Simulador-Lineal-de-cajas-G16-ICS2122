from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import reporting

try:
    from optimizador_cajas import optimizar_cajas_grasp_saa
except ImportError:  # pragma: no cover
    optimizar_cajas_grasp_saa = None

from tools.export_plan_report import (
    build_initial_policies_dataframe,
    build_multipliers_dataframe,
    build_plan_detail_dataframe,
    evaluate_base_performance,
    export_excel_report,
    plot_objective_by_segment,
    plot_total_lanes,
)

def _prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Entrada no valida, usando el valor por defecto.")
        return default


# Valores permitidos para mantener consistencia (1 semana o multiplos de 4 hasta 52)
ALLOWED_WEEK_SAMPLES = [1] + [4 * i for i in range(1, 14)]  # 4..52
LANE_ORDER = ("regular", "express", "priority", "self_checkout")
DEFAULT_PLAN_REPORT_DIR = Path("resultados_opt")


def _normalize_weeks_choice(weeks: int) -> int:
    if weeks in ALLOWED_WEEK_SAMPLES:
        return weeks
    # elegir el valor permitido mas cercano
    closest = min(ALLOWED_WEEK_SAMPLES, key=lambda w: abs(w - weeks))
    print(
        f"Cantidad de semanas {weeks} no permitida; "
        f"se usara {closest} para mantener la consistencia (1 o multiplos de 4)."
    )
    return closest


def _format_lane_tuple(counts: tuple[int, int, int, int]) -> str:
    return ", ".join(f"{lane}={int(value)}" for lane, value in zip(LANE_ORDER, counts))


def _coerce_lane_counts(entry) -> tuple[int, int, int, int]:
    if isinstance(entry, dict):
        seq = [entry.get(lane, entry.get(lane.upper(), 0)) for lane in LANE_ORDER]
    else:
        try:
            seq = list(entry)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError("Formato de configuración inválido") from exc
    if len(seq) < len(LANE_ORDER):
        raise ValueError("Se requieren valores para regular, express, priority y self_checkout")
    try:
        return tuple(int(max(0, seq[idx])) for idx in range(len(LANE_ORDER)))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensivo
        raise ValueError("Los conteos de cajas deben ser números enteros") from exc


def _load_base_seed_file(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("El archivo debe contener un objeto JSON con claves por tipo de día")
    seeds: dict[str, tuple[int, int, int, int]] = {}
    for key, value in data.items():
        key_norm = str(key).strip().upper()
        if not key_norm:
            continue
        seeds[key_norm] = _coerce_lane_counts(value)
    if not seeds:
        raise ValueError("El archivo JSON no contiene configuraciones válidas")
    return seeds


def _generate_plan_report(
    result,
    *,
    report_root: Path,
    base_compare_weeks: int,
    base_compare_reps: int,
) -> None:
    report_root = report_root.resolve()
    timestamp = dt.datetime.now().strftime("plan_multi_year_%Y%m%d_%H%M%S")
    target_dir = report_root / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    base_compare_map = None
    try:
        print(
            "[PLAN] Re-evaluando configuraciones base para comparativo "
            f"({base_compare_weeks} semana(s), {base_compare_reps} réplica(s))..."
        )
        base_compare_map = evaluate_base_performance(
            result,
            num_weeks_sample=base_compare_weeks,
            num_rep=base_compare_reps,
        )
    except Exception as exc:  # pragma: no cover - análisis defensivo
        print(f"[WARN] No se pudo evaluar la configuración base: {exc}")

    detail_df = build_plan_detail_dataframe(result, base_compare=base_compare_map)
    multipliers_df = build_multipliers_dataframe(result)
    initial_df = build_initial_policies_dataframe(result)

    detail_path = target_dir / "plan_detalle.csv"
    detail_df.to_csv(detail_path, index=False)

    excel_path = target_dir / "plan_resumen.xlsx"
    export_excel_report(detail_df, multipliers_df, initial_df, excel_path)

    chart_obj_path = target_dir / "objetivo_por_segmento.png"
    plot_objective_by_segment(detail_df, chart_obj_path)

    chart_lanes_path = target_dir / "total_lanes_por_dia.png"
    plot_total_lanes(detail_df, chart_lanes_path)

    print("\n[PLAN] Reporte multi-anual generado automáticamente:")
    print(f"  - Carpeta base: {target_dir}")
    print(f"  - Detalle CSV: {detail_path.name}")
    print(f"  - Planilla Excel: {excel_path.name}")
    if chart_obj_path.exists():
        print(f"  - Gráfico objetivo/segmento: {chart_obj_path.name}")
    if chart_lanes_path.exists():
        print(f"  - Gráfico cajas/tipo de día: {chart_lanes_path.name}")


def run_sample_simulation() -> None:
    weeks = _prompt_int("Cuantas semanas quieres simular?", 1)
    weeks = _normalize_weeks_choice(max(1, weeks))
    root = input("Carpeta de salida [outputs_sample]: ").strip() or "outputs_sample"
    reporting.run_full_workflow(num_weeks_sample=weeks, output_root=Path(root))


def run_full_year() -> None:
    root = input("Carpeta de salida [outputs_anual]: ").strip() or "outputs_anual"
    reporting.run_full_workflow(num_weeks_sample=52, output_root=Path(root))


def run_optimizer() -> None:
    if optimizar_cajas_grasp_saa is None:
        print(
            "No se pudo importar optimizador_cajas.py. Asegurate de que este disponible."
        )
        return
    print("Ejecutando optimizador de cajas (GRASP + SAA)...")
    import simulator.engine as engine

    mode = (
        input("Modo 1=tipo especifico / 2=optimizar todos los tipos [1]: ").strip()
        or "1"
    )
    max_seconds = _prompt_int("Limite de tiempo en segundos (0 = sin limite)", 0)
    max_seconds = max_seconds if max_seconds > 0 else None
    max_evals = _prompt_int("Limite de evaluaciones (0 = sin limite)", 0)
    max_evals = max_evals if max_evals > 0 else None

    if mode == "2":
        day_values = [dt.value for dt in engine.DayType]
        with ProcessPoolExecutor(max_workers=len(day_values)) as pool:
            futures = {
                pool.submit(
                    _optimizer_worker, day_value, max_seconds, max_evals
                ): day_value
                for day_value in day_values
            }
            for fut in as_completed(futures):
                day_value = futures[fut]
                try:
                    result = fut.result()
                    _print_optimizer_summary(result)
                except Exception as exc:  # pragma: no cover
                    print(f"[ERROR] Optimización falló para {day_value}: {exc}")
    else:
        mapping = {str(i + 1): dt for i, dt in enumerate(engine.DayType)}
        print("Selecciona tipo de dia a optimizar:")
        for key, dt in mapping.items():
            print(f"  {key}) {dt.name} ({dt.value})")
        choice = input("Opcion [1]: ").strip() or "1"
        day = mapping.get(choice, engine.DayType.TYPE_1)
        result = optimizar_cajas_grasp_saa(
            day_type=day,
            max_seconds=max_seconds,
            max_eval_count=max_evals,
            context_label=f"CLI | Día={day.name}",
        )
        _print_optimizer_summary(result)


def run_sequential_policy_plan() -> None:
    if optimizar_cajas_grasp_saa is None:
        print(
            "No se pudo importar optimizador_cajas.py. Asegurate de que este disponible."
        )
        return

    try:
        from .policy_planner import plan_multi_year_optimization
    except ImportError as exc:  # pragma: no cover
        print(f"No se pudo importar policy_planner: {exc}")
        return

    import simulator.engine as engine

    default_segmented = "demand_projection_2026_2030_segmented.csv"
    segmented_input = (
        input(f"Archivo segmentado [{default_segmented}]: ").strip()
        or default_segmented
    )
    segmented_path = Path(segmented_input)
    if not segmented_path.exists():
        print(f"Archivo segmentado no encontrado: {segmented_path}")
        return

    default_summary = segmented_path.with_name(segmented_path.stem + "_fit_summary.csv")
    summary_input = input(
        f"Archivo de resumen de ajustes (enter para {default_summary.name} o 'none'): "
    ).strip()
    if not summary_input:
        summary_path: Path | None = (
            default_summary if default_summary.exists() else None
        )
        if summary_path is None:
            print("No se encontró resumen de ajustes, se usará media muestral.")
    elif summary_input.lower() in {"none", "ninguno"}:
        summary_path = None
    else:
        summary_path = Path(summary_input)
        if not summary_path.exists():
            print(f"Resumen no encontrado: {summary_path}. Se usará media muestral.")
            summary_path = None

    segments_raw = input(
        "Segmentos a considerar (coma, enter=default pesimista,regular,optimista): "
    ).strip()
    if segments_raw:
        segments = tuple(seg.strip() for seg in segments_raw.split(",") if seg.strip())
    else:
        segments = ("pesimista", "regular", "optimista")

    years_raw = input("Años a optimizar (coma, enter=todos): ").strip()
    years = (
        tuple(year.strip() for year in years_raw.split(",") if year.strip())
        if years_raw
        else None
    )

    max_workers = _prompt_int("Cantidad de procesos en paralelo (0 = auto)", 0)
    max_workers = max_workers if max_workers > 0 else None

    report_base_input = (
        input(
            "Carpeta base para exportar el reporte multi-anual "
            f"[{DEFAULT_PLAN_REPORT_DIR}]: "
        )
        .strip()
    )
    report_base_dir = (
        Path(report_base_input) if report_base_input else DEFAULT_PLAN_REPORT_DIR
    )

    weeks_busqueda = _prompt_int("Semanas por evaluación en búsqueda local", 1)
    weeks_busqueda = _normalize_weeks_choice(max(1, weeks_busqueda))

    reps_busqueda = _prompt_int("Repeticiones SAA por evaluación", 1)
    reps_busqueda = max(1, reps_busqueda)

    weeks_construccion = _prompt_int("Semanas por evaluación en fase GRASP", 1)
    weeks_construccion = _normalize_weeks_choice(max(1, weeks_construccion))

    reps_construccion = _prompt_int("Repeticiones SAA en fase GRASP", 1)
    reps_construccion = max(1, reps_construccion)

    max_seconds = _prompt_int("Límite de tiempo por corrida (0 = sin límite)", 0)
    max_seconds = max_seconds if max_seconds > 0 else None

    max_evals = _prompt_int("Límite de evaluaciones por corrida (0 = sin límite)", 0)
    max_evals = max_evals if max_evals > 0 else None

    recalc_base = (
        input("¿Recalcular optimización del año base 2025? [S/n]: ")
        .strip()
        .lower()
    )
    optimize_base_year = recalc_base not in {"n", "no", "0"}

    base_seed_map: dict[str, tuple[int, int, int, int]] | None = None
    if not optimize_base_year:
        seed_input = input(
            "Ruta JSON con configuraciones iniciales (enter = usar valores por defecto): "
        ).strip()
        if seed_input:
            seed_path = Path(seed_input)
            if not seed_path.exists():
                print(f"Archivo no encontrado: {seed_path}. Se usará la configuración por defecto.")
            else:
                try:
                    raw_seeds = _load_base_seed_file(seed_path)
                    base_seed_map = {}
                    for dt in engine.DayType:
                        for key in (dt.name.upper(), dt.value.upper()):
                            if key in raw_seeds:
                                base_seed_map[dt.name] = raw_seeds[key]
                                break
                    if not base_seed_map:
                        print(
                            "No se encontraron claves válidas en el JSON (TYPE_1/TYPE_2/TYPE_3 o tipo_1/...)."
                        )
                        base_seed_map = None
                except Exception as exc:
                    print(f"No se pudo cargar el archivo de configuraciones base: {exc}")
                    base_seed_map = None

    print("\n[PLAN] Iniciando planificación secuencial multi-año...")
    try:
        result = plan_multi_year_optimization(
            segmented_path=segmented_path,
            fit_summary_path=summary_path,
            segments=segments,
            years=years if years else None,
            max_workers=max_workers,
            optimizer_kwargs={
                "num_weeks_sample_busqueda": weeks_busqueda,
                "num_rep_saa_busqueda": reps_busqueda,
                "num_weeks_sample_construccion": weeks_construccion,
                "num_rep_saa_construccion": reps_construccion,
                "max_seconds": max_seconds,
                "max_eval_count": max_evals,
            },
            optimize_base_year=optimize_base_year,
            base_counts=base_seed_map,
        )
    except Exception as exc:
        print(f"[ERROR] No se pudo completar la planificación: {exc}")
        return

    print("\n=== RESULTADOS BASE (Año 2025) ===")
    for dt, res in result.base_year.items():
        if getattr(res, "n_rep", None) == 0:
            counts_str = _format_lane_tuple(tuple(int(v) for v in res.x))
            print(
                f"[RESUMEN {dt.name}] x={res.x} ({counts_str}) | Configuración base aplicada (sin reoptimizar)"
            )
        else:
            _print_optimizer_summary(res)

    for year, per_segment in result.yearly.items():
        print(f"\n=== RESULTADOS AÑO {year} ===")
        for segment, dt_map in per_segment.items():
            print(f"-- Segmento {segment} --")
            for dt, res in dt_map.items():
                _print_optimizer_summary(res)

    print("\n=== MULTIPLICADORES UTILIZADOS ===")
    for segment, mapping in result.multipliers.items():
        formatted = ", ".join(
            f"{year}:{value:.4f}" for year, value in sorted(mapping.items())
        )
        print(f"{segment}: {formatted}")

    if result.initial_policies:
        print("\nConfiguraciones iniciales utilizadas:")
        for day_name, counts in result.initial_policies.items():
            print(f"  {day_name}: {counts} ({_format_lane_tuple(counts)})")

    try:
        compare_weeks = max(1, weeks_busqueda)
        compare_reps = max(1, reps_busqueda)
        _generate_plan_report(
            result,
            report_root=report_base_dir,
            base_compare_weeks=compare_weeks,
            base_compare_reps=compare_reps,
        )
    except Exception as exc:  # pragma: no cover - defensivo
        print(f"[WARN] No se pudo generar el reporte automático: {exc}")

    print("\nPlanificación completada.")


def _optimizer_worker(day_value: str, max_seconds: int | None, max_evals: int | None):
    import simulator.engine as engine
    from optimizador_cajas import optimizar_cajas_grasp_saa as worker_opt

    day = next(dt for dt in engine.DayType if dt.value == day_value)
    return worker_opt(
        day_type=day,
        max_seconds=max_seconds,
        max_eval_count=max_evals,
        context_label=f"CLI|POOL | Día={day.name}",
    )


def _print_optimizer_summary(res):
    if not res:
        return
    lane_names = ["regular", "express", "priority", "self_checkout"]
    counts_str = ", ".join(f"{name}={value}" for name, value in zip(lane_names, res.x))
    profit_fmt = f"{res.profit_mean:,.0f}".replace(",", ".")
    objective_fmt = f"{res.objetivo_mean:,.0f}".replace(",", ".")
    print(
        f"[RESUMEN {res.day_type.name}] x={res.x} ({counts_str}) | Profit={profit_fmt} CLP | Objetivo={objective_fmt} CLP"
    )


def show_menu() -> None:
    print(
        """
=== SIMULADOR DE CAJAS ===
1) Simular X semanas y generar KPIs
2) Simular año completo (52 semanas)
3) Ejecutar optimizador de cajas (GRASP+SAA)
4) Planificación secuencial multi-año
5) Salir
"""
    )


def main() -> None:
    actions = {
        "1": run_sample_simulation,
        "2": run_full_year,
        "3": run_optimizer,
        "4": run_sequential_policy_plan,
    }
    while True:
        show_menu()
        choice = input("Selecciona una opcion: ").strip()
        if choice == "5" or choice.lower() in {"q", "quit", "salir"}:
            print("Hasta luego.")
            break
        action = actions.get(choice)
        if not action:
            print("Opcion invalida. Intenta nuevamente.")
            continue
        action()


if __name__ == "__main__":
    main()
