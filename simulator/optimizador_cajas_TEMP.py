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

# Parametros Financieros
TAX_RATE = 0.27  # 27% Impuesto Primera Categoria
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
class AnnualVariableCosts:
    year: int
    lane_costs: dict[str, LaneCostData]

# Diccionarios globales para acceso rapido: {year: ...}
FIXED_COSTS_PROJECTION: dict[int, float] = {}
VARIABLE_COSTS_PROJECTION: dict[int, dict[str, LaneCostData]] = {}

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


def _load_projections() -> None:
    """
    Carga las proyecciones de costos fijos y variables desde los CSVs.
    """
    global FIXED_COSTS_PROJECTION, VARIABLE_COSTS_PROJECTION
    
    # 1. Cargar Costos Fijos
    fijos_path = PROJECT_ROOT / "data/proyeccion_costos_fijos.csv"
    if not fijos_path.exists():
        raise FileNotFoundError(f"No se encontro {fijos_path}")
        
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
        raise FileNotFoundError(f"No se encontro {var_path}")
        
    df_var = pd.read_csv(var_path)
    # Columnas: year, lane_type, capex_clp, maintenance_yearly_clp, opex_hourly_clp, wage_hourly_clp
    
    # Agrupamos por ano
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
                capex=float(row["capex_clp"]),
                maintenance_yearly=float(row["maintenance_yearly_clp"]),
                opex_hourly=float(row["opex_hourly_clp"]),
                wage_hourly=float(row["wage_hourly_clp"]),
                useful_life=5.0
            )
            lane_map[lane] = data
            
        VARIABLE_COSTS_PROJECTION[year_int] = lane_map

# Cargar proyecciones al inicio
_load_projections()


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


def calculate_financial_metrics(counts: dict[str, int], annual_gross_profit: float) -> dict[str, float]:
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
    total_capex = 0.0
    for lane, count in counts.items():
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
            var_costs_map = VARIABLE_COSTS_PROJECTION[max(VARIABLE_COSTS_PROJECTION.keys())]
            
        opex_cajas_year = 0.0
        depreciacion_year = 0.0
        
        for lane, count in counts.items():
            if count <= 0:
                continue
            data = var_costs_map[lane]
            
            # Costo Operativo (OPEX + Mantenimiento)
            opex_hora_total = data.opex_hourly * count
            maint_total = data.maintenance_yearly * count
            
            # Costo Laboral (Sueldos)
            if lane == "self_checkout":
                # Regla SCO: 1 Isla = 5 Kioscos. 2 Supervisores por Isla.
                num_islas = math.ceil(count / 5.0)
                num_supervisores = num_islas * 2
                wage_hora_total = data.wage_hourly * num_supervisores
            else:
                # Cajas asistidas: 1 persona por caja
                wage_hora_total = data.wage_hourly * count
            
            costo_laboral_anual = wage_hora_total * HOURS_PER_YEAR
            costo_opex_anual = opex_hora_total * HOURS_PER_YEAR
            
            opex_cajas_year += costo_laboral_anual + costo_opex_anual + maint_total
            
            # Depreciacion (Usamos CAPEX base / vida util)
            if data.useful_life > 0:
                depreciacion_year += (costs_base[lane].capex * count) / data.useful_life

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
        "gross_profit": annual_gross_profit
    }
