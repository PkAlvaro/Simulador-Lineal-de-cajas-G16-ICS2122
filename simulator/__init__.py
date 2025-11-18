"""Simulador modularizado.

Este paquete agrupa el motor legacy del simulador en :mod:`simulator.engine`
para que otras herramientas (optimizador, runners, CLI) puedan importar
símbolos de forma estable sin acoplarse a un script monolítico.
"""

from .engine import (  # noqa: F401
    DayType,
    LaneType,
    MAX_TOTAL_LANES,
    CURRENT_LANE_POLICY,
    DEFAULT_LANE_COUNTS,
    RNG_ITEMS,
    RNG_PROFIT,
    enforce_lane_constraints,
    update_current_lane_policy,
    simulacion_periodos,
    simulacion_7_dias_completa,
    simulacion_anual_completa,
    set_demand_multiplier,
    get_demand_multiplier,
)
from .reporting import (  # noqa: F401
    load_customers_year,
    export_kpi_report,
    export_formatted_excel_report,
    run_full_workflow,
)

__all__ = [
    "DayType",
    "LaneType",
    "MAX_TOTAL_LANES",
    "CURRENT_LANE_POLICY",
    "DEFAULT_LANE_COUNTS",
    "RNG_ITEMS",
    "RNG_PROFIT",
    "enforce_lane_constraints",
    "update_current_lane_policy",
    "set_demand_multiplier",
    "get_demand_multiplier",
    "load_customers_year",
    "simulacion_periodos",
    "simulacion_7_dias_completa",
    "simulacion_anual_completa",
    "export_kpi_report",
    "export_formatted_excel_report",
    "run_full_workflow",
]
