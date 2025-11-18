from __future__ import annotations

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
    optimizar_cajas_grasp_saa(day_type=engine.DayType.TYPE_1)


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
