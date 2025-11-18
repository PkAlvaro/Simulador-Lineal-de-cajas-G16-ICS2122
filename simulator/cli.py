from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import reporting

try:
    from optimizador_cajas import optimizar_cajas_grasp_saa
except ImportError:  # pragma: no cover
    optimizar_cajas_grasp_saa = None


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
        print("No se pudo importar optimizador_cajas.py. Asegurate de que este disponible.")
        return
    print("Ejecutando optimizador de cajas (GRASP + SAA)...")
    import simulator.engine as engine

    mode = input("Modo 1=tipo especifico / 2=optimizar todos los tipos [1]: ").strip() or "1"
    max_seconds = _prompt_int("Limite de tiempo en segundos (0 = sin limite)", 0)
    max_seconds = max_seconds if max_seconds > 0 else None
    max_evals = _prompt_int("Limite de evaluaciones (0 = sin limite)", 0)
    max_evals = max_evals if max_evals > 0 else None

    if mode == "2":
        day_values = [dt.value for dt in engine.DayType]
        with ProcessPoolExecutor(max_workers=len(day_values)) as pool:
            futures = {
                pool.submit(_optimizer_worker, day_value, max_seconds, max_evals): day_value
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
        )
        _print_optimizer_summary(result)


def _optimizer_worker(day_value: str, max_seconds: int | None, max_evals: int | None):
    import simulator.engine as engine
    from optimizador_cajas import optimizar_cajas_grasp_saa as worker_opt

    day = next(dt for dt in engine.DayType if dt.value == day_value)
    return worker_opt(
        day_type=day,
        max_seconds=max_seconds,
        max_eval_count=max_evals,
    )


def _print_optimizer_summary(res):
    if not res:
        return
    lane_names = ["regular", "express", "priority", "self_checkout"]
    counts_str = ", ".join(f"{name}={value}" for name, value in zip(lane_names, res.x))
    profit_fmt = f"{res.profit_mean:,.0f}".replace(",", ".")
    objective_fmt = f"{res.objetivo_mean:,.0f}".replace(",", ".")
    print(f"[RESUMEN {res.day_type.name}] x={res.x} ({counts_str}) | Profit={profit_fmt} CLP | Objetivo={objective_fmt} CLP")


def show_menu() -> None:
    print(
        """
=== SIMULADOR DE CAJAS ===
1) Simular X semanas y generar KPIs
2) Simular año completo (52 semanas)
3) Ejecutar optimizador de cajas (GRASP+SAA)
4) Salir
"""
    )


def main() -> None:
    actions = {
        "1": run_sample_simulation,
        "2": run_full_year,
        "3": run_optimizer,
    }
    while True:
        show_menu()
        choice = input("Selecciona una opcion: ").strip()
        if choice == "4" or choice.lower() in {"q", "quit", "salir"}:
            print("Hasta luego.")
            break
        action = actions.get(choice)
        if not action:
            print("Opcion invalida. Intenta nuevamente.")
            continue
        action()


if __name__ == "__main__":
    main()
