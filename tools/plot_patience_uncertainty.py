from __future__ import annotations

"""
Visualiza las distribuciones de paciencia usadas por el simulador, para detectar
fuentes de incertidumbre. Genera una grilla de densidades por perfil (filas)
y tipo de día (columnas), coloreando combinaciones de prioridad/medio de pago.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.engine import (  # noqa: E402
    CustomerProfile,
    DayType,
    PATIENCE_DISTRIBUTION_FILE,
    PaymentMethod,
    PatienceDistributionTable,
    PriorityType,
)


def _to_enum(value: str, enum_cls):
    """Convierte texto a enum, devolviendo None si viene vacío."""
    if value in (None, "", "nan"):
        return None
    value = str(value).strip().lower()
    for member in enum_cls:
        if member.value == value or member.name.lower() == value:
            return member
    raise ValueError(f"No se reconoce valor '{value}' para {enum_cls.__name__}")


def _enumerate_entries(table: PatienceDistributionTable) -> Iterable[Tuple[str, str, str, str]]:
    """Devuelve las claves disponibles en la tabla interna (strings normalizados)."""
    return table._entries.keys()  # type: ignore[attr-defined]


def _sample_distribution(
    table: PatienceDistributionTable,
    profile: CustomerProfile,
    priority: PriorityType | None,
    payment: PaymentMethod | None,
    day: DayType | None,
    n: int,
) -> np.ndarray:
    """Genera n muestras de paciencia en segundos para una combinación dada."""
    samples = [table.sample(profile, priority, payment, day) for _ in range(n)]
    return np.array(samples, dtype=float)


def plot_patience_grid(
    table: PatienceDistributionTable,
    *,
    samples: int = 2000,
    output_path: Path,
) -> Path:
    entries = list(_enumerate_entries(table))
    if not entries:
        raise RuntimeError("No se encontraron distribuciones de paciencia para graficar")

    # Agrupamos por (profile, day_type) y dentro agrupamos por (priority, payment)
    per_profile_day: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    for prof, prio, pay, day in entries:
        key = (prof, day or "desconocido")
        per_profile_day[key].append((prof, prio, pay, day))

    profiles = sorted({k[0] for k in per_profile_day})
    day_types = sorted({k[1] for k in per_profile_day})
    n_rows = len(profiles)
    n_cols = len(day_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False, sharey=False)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for i, prof in enumerate(profiles):
        for j, day in enumerate(day_types):
            ax = axes[i, j]
            combos = per_profile_day.get((prof, day), [])
            if not combos:
                ax.axis("off")
                continue
            for prof_s, prio_s, pay_s, day_s in combos:
                try:
                    prof_e = _to_enum(prof_s, CustomerProfile)
                    prio_e = _to_enum(prio_s, PriorityType) if prio_s else None
                    pay_e = _to_enum(pay_s, PaymentMethod) if pay_s else None
                    day_e = _to_enum(day_s, DayType) if day_s else None
                except ValueError:
                    continue
                color = colors[color_idx % len(colors)]
                color_idx += 1
                sample_arr = _sample_distribution(
                    table,
                    prof_e,
                    prio_e,
                    pay_e,
                    day_e,
                    n=samples,
                )
                if sample_arr.size == 0:
                    continue
                ax.hist(
                    sample_arr,
                    bins=40,
                    density=True,
                    alpha=0.35,
                    color=color,
                    label=f"{prio_s or 'any'}|{pay_s or 'any'} ({len(sample_arr)} samp)",
                )
            ax.set_title(f"{prof} | {day}")
            ax.set_xlabel("patience_s")
            ax.set_ylabel("densidad")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualiza distribuciones de paciencia en una grilla por perfil y tipo de día."
    )
    parser.add_argument(
        "--patience-file",
        type=Path,
        default=PATIENCE_DISTRIBUTION_FILE,
        help="CSV de distribuciones de paciencia (default: data/patience/...csv)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Muestras por combinación para estimar densidad (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados/patience_grid.png"),
        help="Ruta de salida de la imagen",
    )
    args = parser.parse_args()

    table = PatienceDistributionTable(source_file=args.patience_file)
    out = plot_patience_grid(table, samples=max(100, args.samples), output_path=args.output)
    print(f"Grilla de distribuciones exportada a {out}")


if __name__ == "__main__":
    main()
