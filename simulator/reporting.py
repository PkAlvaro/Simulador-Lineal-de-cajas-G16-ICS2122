from __future__ import annotations

import datetime as _dt
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)

from . import engine

HV_THRESHOLD_DEFAULT = 40
HV_THRESHOLD_BY_PROFILE = {
    "deal_hunter": 20,
    "express_basket": 11,
    "family_cart": 99,
    "regular": 39,
    "self_checkout_fan": 13,
    "weekly_planner": 75,
}


def _high_volume_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prof_norm = df["profile"].astype(str).str.lower()
    thresholds = prof_norm.map(HV_THRESHOLD_BY_PROFILE).fillna(HV_THRESHOLD_DEFAULT)
    return df[df["items"] >= thresholds]


def _reset_outputs_folder(root: Path) -> None:
    """
    Limpia el directorio de outputs para evitar mezclar corridas anteriores.
    Elimina todo el contenido bajo root (subcarpetas Week-*/Day-* y archivos),
    manteniendo la carpeta raiz.
    """
    root = Path(root)
    if root.exists():
        for entry in root.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue
    root.mkdir(parents=True, exist_ok=True)


def load_customers_year(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("Week-*/Day-*/customers.csv"))
    if not files:
        files = sorted(root.glob("Week-*-Day-*/customers.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron customers.csv bajo {root.resolve()}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # extrae semana/dia y asegura tipos
        if "source_folder" in df.columns:
            df["week"] = df["source_folder"].str.extract(r"Week-0*(\d+)", expand=False)
            df["day"] = df["source_folder"].str.extract(r"Day-0*(\d+)", expand=False)
        else:
            folder = f.parent.name
            df["week"] = re.search(r"Week-0*(\d+)", folder).group(1) if re.search(r"Week-0*(\d+)", folder) else 0
            df["day"] = re.search(r"Day-0*(\d+)", folder).group(1) if re.search(r"Day-0*(\d+)", folder) else 0
        df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
        df["day"] = pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(int)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["month_4w"] = ((out["week"] - 1) // 4) + 1
    # normaliza outcomes a nomenclatura del KPI
    out["outcome_norm"] = out["outcome"].replace({"abandon": "abandoned", "balk": "balked"}).fillna(out["outcome"])
    # asegura numerico por si hay celdas vacias
    out["total_profit_clp"] = pd.to_numeric(out["total_profit_clp"], errors="coerce").fillna(0)
    out["wait_time_s"] = pd.to_numeric(out["wait_time_s"], errors="coerce")
    out["dia_tipo"] = (
        out["day"]
        .map(lambda d: engine.DAY_TYPE_BY_DAYNUM.get(int(d), "desconocido") if pd.notna(d) else "desconocido")
        .str.lower()
    )
    return out


def _compare_kpi_tables(
    df_est: pd.DataFrame,
    df_ref: pd.DataFrame,
    key_cols: list[str],
    numeric_cols: list[str],
    titulo: str,
) -> Optional[pd.DataFrame]:
    if df_est.empty or df_ref.empty:
        print(f"No hay datos para comparar {titulo}.")
        return None

    cols_est = key_cols + numeric_cols
    cols_ref = key_cols + numeric_cols
    merged = (
        df_est[cols_est]
        .merge(df_ref[cols_ref], on=key_cols, how="inner", suffixes=("_est", "_teo"))
        .copy()
    )
    for col in numeric_cols:
        est_col = f"{col}_est"
        teo_col = f"{col}_teo"
        err_col = f"{col}_rel_err_pct"
        merged[err_col] = np.where(
            merged[teo_col] != 0,
            (merged[est_col] - merged[teo_col]) / merged[teo_col] * 100.0,
            np.nan,
        )
    ordered_cols = key_cols[:]
    for col in numeric_cols:
        ordered_cols.extend([f"{col}_est", f"{col}_teo", f"{col}_rel_err_pct"])
    return merged[ordered_cols]


def _sanitize_sheet_name(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return (clean or "Sheet")[:30]


def _write_number_safe(worksheet, row: int, col: int, value, cell_format):
    """Escribe numeros manejando NaN/None para evitar errores de xlsxwriter."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        worksheet.write_blank(row, col, None, cell_format)
    else:
        worksheet.write(row, col, value, cell_format)


def export_kpi_report(export_root: Path, tables: dict[str, Optional[pd.DataFrame]]) -> Optional[Path]:
    valid_tables = {k: df for k, df in tables.items() if isinstance(df, pd.DataFrame) and not df.empty}
    if not valid_tables:
        return None

    export_root.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = export_root / f"kpi_resumen_{timestamp}.xlsx"

    excel_written = False
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            for name, df in valid_tables.items():
                sheet = _sanitize_sheet_name(name)
                df.to_excel(writer, sheet_name=sheet, index=False)
        excel_written = True
    except ModuleNotFoundError:
        print("xlsxwriter no esta instalado; se omite la generacion del archivo Excel.")
        excel_path = None

    return excel_path if excel_written else None


def export_formatted_excel_report(
    export_root: Path,
    profit_df: Optional[pd.DataFrame],
    profit_day_df: Optional[pd.DataFrame],
    wait_day_profile_df: Optional[pd.DataFrame],
    service_time_df: Optional[pd.DataFrame],
    tac_lane_df: Optional[pd.DataFrame],
    tac_df: Optional[pd.DataFrame],
    tmc_df: Optional[pd.DataFrame],
    hv_df: Optional[pd.DataFrame],
    client_counts_df: Optional[pd.DataFrame],
    little_df: Optional[pd.DataFrame],
    profit_anual_estimado: float = 0.0,
) -> Optional[Path]:
    try:
        import xlsxwriter  # noqa: F401
    except ModuleNotFoundError:
        print("xlsxwriter no esta instalado; se omite el reporte formateado.")
        return None

    required_tables = {
        "profit_df": profit_df,
        "profit_day_df": profit_day_df,
        "wait_day_profile_df": wait_day_profile_df,
        "service_time_df": service_time_df,
        "tac_lane_df": tac_lane_df,
    }
    missing = [
        name
        for name, table in required_tables.items()
        if not isinstance(table, pd.DataFrame) or table.empty
    ]
    if missing:
        print(
            "Datos insuficientes para generar el reporte formateado. "
            f"Tablas faltantes/vacias: {', '.join(missing)}"
        )
        return None

    import xlsxwriter

    export_root.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = export_root / f"kpi_formato_{timestamp}.xlsx"

    wb = xlsxwriter.Workbook(excel_path)

    fmt_header = wb.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#7FBFE7", "border": 1}
    )
    fmt_subheader = wb.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#D9E1F2", "border": 1}
    )
    fmt_number = wb.add_format({"num_format": "#,##0", "border": 1})
    fmt_decimal = wb.add_format({"num_format": "0.00", "border": 1})
    fmt_percent = wb.add_format({"num_format": "0.00", "border": 1})
    fmt_text = wb.add_format({"border": 1})

    # Hoja Profit
    ws_profit = wb.add_worksheet("Profit_Dia")
    ws_profit.merge_range(0, 0, 0, 5, "Resumen Profit", fmt_header)
    ws_profit.write_row(2, 0, ["Caso Base", "Simulacion cruda", "Simulacion degradada", "Costo cajas", "Error crudo (%)", "Error degradado (%)"], fmt_subheader)
    if profit_df is not None and not profit_df.empty:
        row_vals = profit_df.iloc[0]
        cols = ["case_base", "sim_crudo", "sim_degradado", "costos_cajas", "error_crudo_pct", "error_degradado_pct"]
        for idx, col in enumerate(cols):
            fmt = fmt_percent if "pct" in col else fmt_number
            _write_number_safe(ws_profit, 3, idx, row_vals.get(col), fmt)
    ws_profit.merge_range(6, 0, 6, 6, "Profit por tipo de dia", fmt_header)
    ws_profit.write_row(8, 0, ["Tipo de dia", "Caso Base", "Simulacion cruda", "Simulacion degradada", "Costo cajas", "Error crudo (%)", "Error degradado (%)"], fmt_subheader)
    for idx, row in profit_day_df.iterrows():
        r = 9 + idx
        ws_profit.write(r, 0, row.get("dia_tipo"), fmt_text)
        _write_number_safe(ws_profit, r, 1, row.get("profit_clp_teo"), fmt_number)
        _write_number_safe(ws_profit, r, 2, row.get("profit_clp_est"), fmt_number)
        _write_number_safe(ws_profit, r, 3, row.get("profit_clp_degradado"), fmt_number)
        _write_number_safe(ws_profit, r, 4, row.get("costo_cajas_clp"), fmt_number)
        _write_number_safe(ws_profit, r, 5, row.get("profit_clp_rel_err_pct"), fmt_percent)
        _write_number_safe(ws_profit, r, 6, row.get("profit_clp_degradado_rel_err_pct"), fmt_percent)

    # Hoja Wait
    ws_wait = wb.add_worksheet("Wait_Dia_Perfil")
    ws_wait.merge_range(0, 0, 0, 4, "Tiempo de espera por dia y perfil (s)", fmt_header)
    ws_wait.write_row(2, 0, ["Tipo de dia", "Perfil", "Caso Base (s)", "Simulacion (s)", "Error (%)"], fmt_subheader)
    for idx, row in wait_day_profile_df.iterrows():
        ws_wait.write(3 + idx, 0, row.get("dia_tipo"), fmt_text)
        ws_wait.write(3 + idx, 1, row.get("profile"), fmt_text)
        _write_number_safe(ws_wait, 3 + idx, 2, row.get("wait_time_s_teo"), fmt_decimal)
        _write_number_safe(ws_wait, 3 + idx, 3, row.get("wait_time_s_est"), fmt_decimal)
        _write_number_safe(ws_wait, 3 + idx, 4, row.get("wait_time_s_rel_err_pct"), fmt_percent)

    # Hoja Service Time
    ws_service = wb.add_worksheet("Service_Time")
    ws_service.merge_range(0, 0, 0, 4, "Tiempos de servicio por tipo de caja (s)", fmt_header)
    ws_service.write_row(2, 0, ["Tipo caja", "Perfil", "Caso Base (s)", "Simulacion (s)", "Error (%)"], fmt_subheader)
    for idx, row in service_time_df.iterrows():
        ws_service.write(3 + idx, 0, row.get("lane_type"), fmt_text)
        ws_service.write(3 + idx, 1, row.get("profile"), fmt_text)
        _write_number_safe(ws_service, 3 + idx, 2, row.get("service_time_mean_teo"), fmt_decimal)
        _write_number_safe(ws_service, 3 + idx, 3, row.get("service_time_mean_est"), fmt_decimal)
        _write_number_safe(ws_service, 3 + idx, 4, row.get("service_time_mean_rel_err_pct"), fmt_percent)

    if tac_df is not None and not tac_df.empty:
        ws_tac_profile = wb.add_worksheet("TAC_Perfil")
        ws_tac_profile.merge_range(0, 0, 0, 3, "TAC por perfil", fmt_header)
        ws_tac_profile.write_row(
            2,
            0,
            [
                "Perfil",
                "TAC Teorico (%)",
                "TAC Simulado (%)",
                "Error (%)",
            ],
            fmt_subheader,
        )
        
        # Calcular TAC usando la formula: TAC = (abandoned + balked) / (served + abandoned + balked) * 100
        for idx, row in tac_df.iterrows():
            ws_tac_profile.write(3 + idx, 0, row.get("profile"), fmt_text)
            
            # Obtener conteos simulados
            served_sim = row.get("served_est", row.get("served", 0))
            abandoned_sim = row.get("abandoned_est", row.get("abandoned", 0))
            balked_sim = row.get("balked_est", row.get("balked", 0))
            
            # Obtener conteos teoricos
            served_teo = row.get("served_teo", 0)
            abandoned_teo = row.get("abandoned_teo", 0)
            balked_teo = row.get("balked_teo", 0)
            
            # Calcular TAC simulado
            total_sim = served_sim + abandoned_sim + balked_sim
            tac_sim = ((abandoned_sim + balked_sim) / total_sim * 100.0) if total_sim > 0 else np.nan
            
            # Calcular TAC teorico
            total_teo = served_teo + abandoned_teo + balked_teo
            tac_teo = ((abandoned_teo + balked_teo) / total_teo * 100.0) if total_teo > 0 else np.nan
            
            # Calcular error
            tac_error = ((tac_sim - tac_teo) / tac_teo * 100.0) if (tac_teo and tac_teo != 0) else np.nan
            
            _write_number_safe(ws_tac_profile, 3 + idx, 1, tac_teo, fmt_percent)
            _write_number_safe(ws_tac_profile, 3 + idx, 2, tac_sim, fmt_percent)
            _write_number_safe(ws_tac_profile, 3 + idx, 3, tac_error, fmt_percent)
        counts_start = 4 + len(tac_df)
        if client_counts_df is not None and not client_counts_df.empty:
            ws_tac_profile.merge_range(counts_start, 0, counts_start, 6, "Conteo por perfil y tipo de dia", fmt_header)
            ws_tac_profile.write_row(
                counts_start + 2,
                0,
                [
                    "Tipo de dia",
                    "Perfil",
                    "Served (sim)",
                    "Served (teo)",
                    "Error % served",
                    "Abandoned (sim)",
                    "Abandoned (teo)",
                    "Error % abandoned",
                    "Balked (sim)",
                    "Balked (teo)",
                    "Error % balked",
                ],
                fmt_subheader,
            )
            for idx, row in client_counts_df.iterrows():
                r = counts_start + 3 + idx
                ws_tac_profile.write(r, 0, row.get("dia_tipo"), fmt_text)
                ws_tac_profile.write(r, 1, row.get("profile"), fmt_text)
                _write_number_safe(ws_tac_profile, r, 2, row.get("served_est"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 3, row.get("served_teo"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 4, row.get("served_rel_err_pct"), fmt_percent)
                _write_number_safe(ws_tac_profile, r, 5, row.get("abandoned_est"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 6, row.get("abandoned_teo"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 7, row.get("abandoned_rel_err_pct"), fmt_percent)
                _write_number_safe(ws_tac_profile, r, 8, row.get("balked_est"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 9, row.get("balked_teo"), fmt_number)
                _write_number_safe(ws_tac_profile, r, 10, row.get("balked_rel_err_pct"), fmt_percent)

    ws_tac_lane = wb.add_worksheet("TAC_Perfil_Caja")
    ws_tac_lane.merge_range(0, 0, 0, 10, "TAC por perfil y tipo de caja", fmt_header)
    ws_tac_lane.write_row(
        2,
        0,
        [
            "Tipo caja",
            "Perfil",
            "Served (sim %)",
            "Served (teo %)",
            "Error % served",
            "Abandoned (sim %)",
            "Abandoned (teo %)",
            "Error % abandoned",
            "Balked (sim %)",
            "Balked (teo %)",
            "Error % balked",
        ],
        fmt_subheader,
    )
    for idx, row in tac_lane_df.iterrows():
        ws_tac_lane.write(3 + idx, 0, row.get("lane_type"), fmt_text)
        ws_tac_lane.write(3 + idx, 1, row.get("profile"), fmt_text)
        _write_number_safe(ws_tac_lane, 3 + idx, 2, row.get("served_pct_est"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 3, row.get("served_pct_teo"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 4, row.get("served_pct_rel_err_pct"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 5, row.get("abandoned_pct_est"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 6, row.get("abandoned_pct_teo"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 7, row.get("abandoned_pct_rel_err_pct"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 8, row.get("balked_pct_est"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 9, row.get("balked_pct_teo"), fmt_percent)
        _write_number_safe(ws_tac_lane, 3 + idx, 10, row.get("balked_pct_rel_err_pct"), fmt_percent)

    if tmc_df is not None and not tmc_df.empty:
        ws_tmc = wb.add_worksheet("TMC_Perfil")
        ws_tmc.merge_range(0, 0, 0, 4, "TMC por perfil", fmt_header)
        ws_tmc.write_row(2, 0, ["Perfil", "Caso Base (s)", "Simulacion (s)", "Error (%)"], fmt_subheader)
        for idx, row in tmc_df.iterrows():
            ws_tmc.write(3 + idx, 0, row.get("profile"), fmt_text)
            teo_val = row.get("tmc_wait_time_s_teo")
            est_val = row.get("tmc_wait_time_s_est", row.get("tmc_wait_time_s"))
            err_val = row.get("tmc_wait_time_s_rel_err_pct")
            _write_number_safe(ws_tmc, 3 + idx, 1, teo_val, fmt_decimal)
            _write_number_safe(ws_tmc, 3 + idx, 2, est_val, fmt_decimal)
            _write_number_safe(ws_tmc, 3 + idx, 3, err_val, fmt_percent)

    if hv_df is not None and not hv_df.empty:
        ws_hv = wb.add_worksheet("TAC_Alto_Vol")
        ws_hv.merge_range(0, 0, 0, 4, "TAC alto volumen", fmt_header)
        ws_hv.write_row(2, 0, ["Perfil", "TAC HV (sim %)", "TAC HV (teo %)", "Error %"], fmt_subheader)
        for idx, row in hv_df.iterrows():
            ws_hv.write(3 + idx, 0, row.get("profile"), fmt_text)
            _write_number_safe(ws_hv, 3 + idx, 1, row.get("tac_hv_pct_est"), fmt_percent)
            _write_number_safe(ws_hv, 3 + idx, 2, row.get("tac_hv_pct_teo"), fmt_percent)
            _write_number_safe(ws_hv, 3 + idx, 3, row.get("tac_hv_pct_rel_err_pct"), fmt_percent)

    if client_counts_df is not None and not client_counts_df.empty:
        ws_clients = wb.add_worksheet("Clientes")
        ws_clients.merge_range(0, 0, 0, 10, "Clientes por perfil y tipo de dia", fmt_header)
        ws_clients.write_row(
            2,
            0,
            [
                "Tipo de dia",
                "Perfil",
                "Served (sim)",
                "Served (teo)",
                "Error % served",
                "Abandoned (sim)",
                "Abandoned (teo)",
                "Error % abandoned",
                "Balked (sim)",
                "Balked (teo)",
                "Error % balked",
            ],
            fmt_subheader,
        )
        for idx, row in client_counts_df.iterrows():
            r = 3 + idx
            ws_clients.write(r, 0, row.get("dia_tipo"), fmt_text)
            ws_clients.write(r, 1, row.get("profile"), fmt_text)
            _write_number_safe(ws_clients, r, 2, row.get("served_est"), fmt_number)
            _write_number_safe(ws_clients, r, 3, row.get("served_teo"), fmt_number)
            _write_number_safe(ws_clients, r, 4, row.get("served_rel_err_pct"), fmt_percent)
            _write_number_safe(ws_clients, r, 5, row.get("abandoned_est"), fmt_number)
            _write_number_safe(ws_clients, r, 6, row.get("abandoned_teo"), fmt_number)
            _write_number_safe(ws_clients, r, 7, row.get("abandoned_rel_err_pct"), fmt_percent)
            _write_number_safe(ws_clients, r, 8, row.get("balked_est"), fmt_number)
            _write_number_safe(ws_clients, r, 9, row.get("balked_teo"), fmt_number)
            _write_number_safe(ws_clients, r, 10, row.get("balked_rel_err_pct"), fmt_percent)

    if little_df is not None and not little_df.empty:
        ws_little = wb.add_worksheet("Little")
        ws_little.merge_range(0, 0, 0, 11, "Ley de Little (lambda*W vs L)", fmt_header)
        ws_little.write_row(
            2,
            0,
            [
                "Tipo de dia",
                "Perfil",
                "Servidos/dia (sim)",
                "Servidos/dia (teo)",
                "Error % serv/dia",
                "W (s) sim",
                "W (s) teo",
                "Error % W",
                "Lambda (1/s) sim",
                "Lambda (1/s) teo",
                "L = λW sim",
                "L = λW teo",
                "Error % L",
            ],
            fmt_subheader,
        )
        for idx, row in little_df.iterrows():
            r = 3 + idx
            ws_little.write(r, 0, row.get("dia_tipo"), fmt_text)
            ws_little.write(r, 1, row.get("profile"), fmt_text)
            _write_number_safe(ws_little, r, 2, row.get("served_per_day_est"), fmt_decimal)
            _write_number_safe(ws_little, r, 3, row.get("served_per_day_teo"), fmt_decimal)
            _write_number_safe(ws_little, r, 4, row.get("served_per_day_rel_err_pct"), fmt_percent)
            _write_number_safe(ws_little, r, 5, row.get("wait_mean_s_est"), fmt_decimal)
            _write_number_safe(ws_little, r, 6, row.get("wait_mean_s_teo"), fmt_decimal)
            _write_number_safe(ws_little, r, 7, row.get("wait_mean_s_rel_err_pct"), fmt_percent)
            _write_number_safe(ws_little, r, 8, row.get("lambda_per_s_est"), fmt_decimal)
            _write_number_safe(ws_little, r, 9, row.get("lambda_per_s_teo"), fmt_decimal)
            _write_number_safe(ws_little, r, 10, row.get("L_value_est"), fmt_decimal)
            _write_number_safe(ws_little, r, 11, row.get("L_value_teo"), fmt_decimal)
            _write_number_safe(ws_little, r, 12, row.get("L_value_rel_err_pct"), fmt_percent)

    # Nueva hoja: Costos Financieros
    try:
        from .optimizador_cajas import calculate_financial_metrics
        
        # Obtener configuración actual de cajas
        current_policy = engine.get_current_lane_policy()
        
        # Calcular métricas financieras usando profit bruto anual estimado
        metrics = calculate_financial_metrics(current_policy, profit_anual_estimado)
        
        ws_costos = wb.add_worksheet("Costos_Financieros")
        ws_costos.merge_range(0, 0, 0, 3, "Analisis Financiero del Proyecto", fmt_header)
        
        # Configuración de cajas
        ws_costos.merge_range(2, 0, 2, 3, "Configuracion de Cajas", fmt_subheader)
        ws_costos.write(3, 0, "Tipo de Caja", fmt_text)
        ws_costos.write(3, 1, "Cantidad", fmt_text)
        row_idx = 4
        for lane_type in ["regular", "express", "priority", "self_checkout"]:
            ws_costos.write(row_idx, 0, lane_type.replace("_", " ").title(), fmt_text)
            ws_costos.write(row_idx, 1, current_policy.get(lane_type, 0), fmt_number)
            row_idx += 1
        
        ws_costos.write(row_idx, 0, "Total Cajas", fmt_subheader)
        ws_costos.write(row_idx, 1, sum(current_policy.values()), fmt_number)
        row_idx += 2
        
        # Métricas Financieras
        ws_costos.merge_range(row_idx, 0, row_idx, 3, "Metricas Financieras (Horizonte 2025-2030)", fmt_subheader)
        row_idx += 1
        
        financial_items = [
            ("Inversion Inicial (CAPEX)", metrics["capex"]),
            ("OPEX Anual Promedio", metrics["opex_anual_promedio"]),
            ("Profit Bruto Anual (Simulado)", metrics["gross_profit"]),
            ("EBITDA Anual Promedio", metrics["ebitda_anual_promedio"]),
            ("FCF Anual Promedio", metrics["fcf_anual_promedio"]),
            ("", None),  # Línea en blanco
            ("VPN del Proyecto (5-6 anos, 8.54%)", metrics["npv"]),
        ]
        
        for label, value in financial_items:
            if value is None:
                row_idx += 1
                continue
            ws_costos.write(row_idx, 0, label, fmt_text)
            _write_number_safe(ws_costos, row_idx, 1, value, fmt_number)
            row_idx += 1
        
        # Parámetros usados
        row_idx += 1
        ws_costos.merge_range(row_idx, 0, row_idx, 3, "Parametros Financieros", fmt_subheader)
        row_idx += 1
        
        params = [
            ("Tasa de Impuesto (Primera Categoria)", "27%"),
            ("Tasa de Descuento (Costo de Capital)", "8.54%"),
            ("Horizonte de Proyeccion", "2025-2030"),
        ]
        
        for label, value in params:
            ws_costos.write(row_idx, 0, label, fmt_text)
            ws_costos.write(row_idx, 1, value, fmt_text)
            row_idx += 1
            
    except ImportError:
        # Si no se puede importar calculate_financial_metrics, omitir esta hoja
        pass

    wb.close()
    return excel_path



def run_full_workflow(
    num_weeks_sample: int = 52,
    output_root: str | Path = "outputs_sample",
    resultados_root: str | Path = "resultados",
) -> dict[str, Optional[Path]]:
    """Ejecuta la simulacion principal y genera los reportes/KPIs."""
    run_start = time.perf_counter()
    output_root_path = Path(output_root)
    _reset_outputs_folder(output_root_path)
    stats_muestra = engine.simulacion_periodos(
        num_weeks=num_weeks_sample,
        output_root=str(output_root_path),
        include_timestamp=False,
        start_week_idx=1,
        titulo=f"SIMULACION {num_weeks_sample} SEMANAS PARA ESTIMAR EL AÑO",
    )
    for registro in stats_muestra:
        print(registro)

    clientes_por_tipo: dict[str, list[float]] = defaultdict(list)
    for r in stats_muestra:
        dia_tipo = str(r.get("dia_tipo", "")).lower()
        clientes_por_tipo[dia_tipo].append(float(r.get("total_clientes", 0.0)))

    def _mean_or_zero(vals):
        return float(np.mean(vals)) if vals else 0.0

    mu_t1 = _mean_or_zero(clientes_por_tipo.get("tipo_1", []))
    mu_t2 = _mean_or_zero(clientes_por_tipo.get("tipo_2", []))
    mu_t3 = _mean_or_zero(clientes_por_tipo.get("tipo_3", []))

    clientes_semana_esperados = 3 * mu_t1 + 3 * mu_t2 + 1 * mu_t3
    clientes_anio_esperados = 52 * clientes_semana_esperados

    print(
        f"Clientes esperados por semana (E[3xTipo1 + 3xTipo2 + 1xTipo3]): "
        f"{clientes_semana_esperados:,.2f}"
    )
    print(f"Clientes esperados por ano (52 semanas): {clientes_anio_esperados:,.2f}")

    resultados_root = Path(resultados_root)
    df = load_customers_year(output_root_path)
    
    # Resumen general
    perfiles_presentes = sorted(df["profile"].dropna().unique().tolist())
    dias_simulados = df[["week","day"]].drop_duplicates().shape[0]
    print(f"Perfiles presentes en la muestra: {perfiles_presentes}")
    print(f"Dias simulados en outputs_sample: {dias_simulados}")
    
    df["dia_tipo_norm"] = (
        df["day"]
        .map(lambda d: engine.DAY_TYPE_BY_DAYNUM.get(int(d), "desconocido") if pd.notna(d) else "desconocido")
        .str.lower()
    )
    
    # 1) PROF_CLP (clientes servidos, muestra de semanas) y extrapolacion anual
    prof_clp_muestra = df.loc[df["outcome_norm"] == "served", "total_profit_clp"].sum()
    print(f"PROF_CLP (servidos, en {num_weeks_sample} semanas): {prof_clp_muestra:,.0f} CLP")
    
    factor_anual = 52.0 / float(num_weeks_sample)
    prof_clp_estimado_anual = prof_clp_muestra * factor_anual
    print(f"PROF_CLP estimado anual (escala lineal 52/{num_weeks_sample}): {prof_clp_estimado_anual:,.0f} CLP")
    lane_cost_sample = engine.LANE_COST_PER_WEEK * num_weeks_sample
    lane_cost_annual = engine.LANE_COST_TOTAL_ANNUAL
    profit_degradado_muestra = prof_clp_muestra - lane_cost_sample
    profit_degradado_anual = prof_clp_estimado_anual - lane_cost_annual
    if lane_cost_annual > 0:
        print(f"Costo de operacion estimado (muestra): {lane_cost_sample:,.0f} CLP")
        print(f"Costo de operacion estimado anual: {lane_cost_annual:,.0f} CLP")
        print(f"Profit degradado (muestra): {profit_degradado_muestra:,.0f} CLP")
        print(f"Profit degradado anual: {profit_degradado_anual:,.0f} CLP")
    else:
        profit_degradado_muestra = prof_clp_muestra
        profit_degradado_anual = prof_clp_estimado_anual
    
    profit_compare_df = pd.DataFrame(
        [
            {
                "case_base": np.nan,
                "sim_crudo": prof_clp_estimado_anual,
                "sim_degradado": profit_degradado_anual,
                "costos_cajas": lane_cost_annual,
                "error_crudo_pct": np.nan,
                "error_degradado_pct": np.nan,
            }
        ]
    )
    
    # 2) Profit mensual (bloques de 4 semanas de la muestra)
    monthly_prof = (
        df.loc[df["outcome_norm"] == "served"]
          .groupby("month_4w")["total_profit_clp"].sum()
          .reset_index(name="prof_clp")
    )
    monthly_prof["prof_clp_fmt"] = monthly_prof["prof_clp"].map(lambda x: f"{x:,.0f} CLP")
    display(monthly_prof)
    
    # Profit por tipo de dia
    profit_day_df = (
        df.loc[df["outcome_norm"] == "served"]
          .groupby("dia_tipo_norm")["total_profit_clp"].sum()
          .reset_index(name="profit_clp")
    )
    profit_day_df = profit_day_df.rename(columns={"dia_tipo_norm": "dia_tipo"})
    profit_day_df["profit_clp"] = profit_day_df["profit_clp"] * factor_anual
    profit_day_df["costo_cajas_clp"] = profit_day_df["dia_tipo"].map(lambda t: engine.LANE_COST_BY_DAYTYPE_STR.get(str(t), 0.0))
    profit_day_df["profit_clp_degradado"] = profit_day_df["profit_clp"] - profit_day_df["costo_cajas_clp"]
    profit_day_display = profit_day_df.assign(
        profit_clp_teo=np.nan,
        profit_clp_est=lambda d: d["profit_clp"],
        profit_clp_rel_err_pct=np.nan,
        profit_clp_degradado_rel_err_pct=np.nan,
    )[
        [
            "dia_tipo",
            "profit_clp_teo",
            "profit_clp_est",
            "profit_clp_degradado",
            "costo_cajas_clp",
            "profit_clp_rel_err_pct",
            "profit_clp_degradado_rel_err_pct",
        ]
    ]
    profit_day_title = "PROFIT por tipo de dia (estimado)"
    
    # 3) TAC_p (tasa de abandono/balk por perfil, estimada a partir de la muestra)
    status_cols = ["served", "abandoned", "balked"]

    def _summarize_client_status(data: pd.DataFrame) -> pd.DataFrame:
        subset = data[data["outcome_norm"].isin(status_cols)].copy()
        if subset.empty:
            return pd.DataFrame()
        grouped = (
            subset.groupby(["dia_tipo_norm", "profile", "outcome_norm"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in status_cols:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped = grouped.rename(columns={"dia_tipo_norm": "dia_tipo"})
        return grouped

    def _build_wait_day_profile_summary(data: pd.DataFrame) -> pd.DataFrame:
        served = data[data["outcome_norm"] == "served"].copy()
        if served.empty:
            return pd.DataFrame()
        grouped = (
            served.groupby(["dia_tipo_norm", "profile"])["wait_time_s"]
            .mean()
            .reset_index(name="wait_time_s")
        )
        grouped["dia_tipo"] = grouped["dia_tipo_norm"]
        return grouped[["dia_tipo", "profile", "wait_time_s"]]

    def _build_tac_lane_summary(data: pd.DataFrame) -> pd.DataFrame:
        subset = data[data["outcome_norm"].isin(status_cols)].copy()
        if subset.empty:
            return pd.DataFrame()
        subset["lane_type_norm"] = (
            subset["lane_type"]
            .astype(str)
            .str.lower()
            .map(engine.LANE_NAME_NORMALIZATION)
            .fillna(subset["lane_type"])
        )
        grouped = (
            subset.groupby(["lane_type_norm", "profile", "outcome_norm"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in status_cols:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped["total"] = grouped[status_cols].sum(axis=1)
        for col in status_cols:
            pct_col = f"{col}_pct"
            grouped[pct_col] = np.where(
                grouped["total"] > 0,
                grouped.get(col, 0) / grouped["total"] * 100.0,
                np.nan,
            )
        grouped = grouped.rename(columns={"lane_type_norm": "lane_type"})
        cols = ["lane_type", "profile"] + [f"{c}_pct" for c in status_cols]
        return grouped[cols]

    status_df = df[df["outcome_norm"].isin(status_cols)].copy()
    counts = status_df.groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
    for col in status_cols:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts.sum(axis=1)
    base_tac = counts.assign(
        served_pct=(counts.get("served", 0) / counts["total"] * 100).round(2),
        abandoned_pct=(counts.get("abandoned", 0) / counts["total"] * 100).round(2),
        balked_pct=(counts.get("balked", 0) / counts["total"] * 100).round(2),
    ).reset_index()
    for col in ["served", "abandoned", "balked", "total"]:
        if col in base_tac.columns:
            base_tac[col] = (base_tac[col].astype(float) * factor_anual).round(0)
    tac_profile = base_tac.copy()
    tac_display = tac_profile.copy()
    tac_title = "TAC por perfil (estimado)"
    
    # 4) TMC_p (tiempo medio de espera por perfil, solo servidos, estimado)
    base_tmc = (
        df.loc[df["outcome_norm"] == "served"]
          .groupby("profile")["wait_time_s"].mean()
          .reset_index(name="tmc_wait_time_s")
    )
    base_tmc["tmc_fmt"] = base_tmc["tmc_wait_time_s"].map(lambda s: f"{s:.2f} s")
    tmc_df = base_tmc.copy()
    tmc_display = tmc_df.copy()
    tmc_title = "TMC por perfil (estimado)"
    
    # 5) TAC_HV (tasa de abandono alto volumen por perfil, estimada)
    hv = _high_volume_slice(df)
    hv_counts = hv[hv["outcome_norm"].isin(status_cols)].groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
    hv_counts["total"] = hv_counts.sum(axis=1)
    hv_stats = hv_counts.assign(
        tac_hv_pct=(((hv_counts.get("abandoned", 0) + hv_counts.get("balked", 0)) / hv_counts["total"]) * 100).round(2)
    ).reset_index()
    for col in ["abandoned", "balked", "served", "total"]:
        if col in hv_stats.columns:
            hv_stats[col] = (hv_stats[col].astype(float) * factor_anual).round(2)
    hv_title = "TAC alto volumen (estimado)"
    hv_display_df: Optional[pd.DataFrame] = hv_stats.copy()
    hv_formatted_df: Optional[pd.DataFrame] = None
    
    # 5b) Tiempos de servicio por tipo de caja y perfil
    service_time_sim = (
        df.loc[df["outcome_norm"] == "served"]
          .groupby(["lane_type", "profile"])["service_time_s"]
          .mean()
          .reset_index(name="service_time_mean_est")
    )
    service_time_sim["lane_type"] = (
        service_time_sim["lane_type"].astype(str).str.lower().map(engine.LANE_NAME_NORMALIZATION).fillna(service_time_sim["lane_type"])
    )
    service_time_display = service_time_sim.copy()
    service_time_title = "Tiempos de servicio por tipo de caja (estimado)"
    service_time_formatted = service_time_sim.copy()
    service_time_formatted["service_time_mean_teo"] = np.nan
    service_time_formatted["service_time_mean_rel_err_pct"] = np.nan
    service_time_formatted = service_time_formatted[
        ["lane_type", "profile", "service_time_mean_teo", "service_time_mean_est", "service_time_mean_rel_err_pct"]
    ]
    
    # 6) KPIs segmentados por tipo de dia
    df["dia_tipo_norm"] = (
        df["day"]
        .map(lambda d: engine.DAY_TYPE_BY_DAYNUM.get(int(d), "desconocido") if pd.notna(d) else "desconocido")
        .str.lower()
    )
    kpi_day_rows: list[dict[str, float]] = []
    for dia_tipo, slice_df in df.groupby("dia_tipo_norm"):
        status_slice = slice_df[slice_df["outcome_norm"].isin(status_cols)]
        total_status = status_slice.shape[0]
        served_ct = int((status_slice["outcome_norm"] == "served").sum())
        abandoned_ct = int((status_slice["outcome_norm"] == "abandoned").sum())
        balked_ct = int((status_slice["outcome_norm"] == "balked").sum())
        profit_total = float(slice_df.loc[slice_df["outcome_norm"] == "served", "total_profit_clp"].sum())
        wait_mean = float(slice_df.loc[slice_df["outcome_norm"] == "served", "wait_time_s"].mean())
        kpi_day_rows.append(
            {
                "dia_tipo": dia_tipo,
                "total_eventos": total_status,
                "served": served_ct,
                "abandoned": abandoned_ct,
                "balked": balked_ct,
                "served_pct": round((served_ct / total_status * 100) if total_status else 0.0, 2),
                "abandoned_pct": round((abandoned_ct / total_status * 100) if total_status else 0.0, 2),
                "balked_pct": round((balked_ct / total_status * 100) if total_status else 0.0, 2),
                "profit_served_clp": profit_total,
                "wait_time_avg_s": round(wait_mean, 2),
            }
        )
    
    kpi_day_df = pd.DataFrame(kpi_day_rows).sort_values("dia_tipo").reset_index(drop=True)
    if not kpi_day_df.empty:
        for col in ["total_eventos", "served", "abandoned", "balked", "profit_served_clp"]:
            if col in kpi_day_df.columns:
                kpi_day_df[col] = (kpi_day_df[col].astype(float) * factor_anual).round(2)
        kpi_day_df["profit_fmt"] = kpi_day_df["profit_served_clp"].map(lambda x: f"{x:,.0f} CLP")
    kpi_day_display = kpi_day_df.copy()
    kpi_day_title = "KPIs por tipo de dia (estimado)"
    # Comparacion con KPIs teoricos (outputs_teoricos)
    output_root_path_TEO = Path("outputs_teoricos")
    try:
        df_teo = load_customers_year(output_root_path_TEO)
    except FileNotFoundError:
        df_teo = None
        print(f"No se encontraron datos en {output_root_path_TEO} para comparar KPIs teoricos.")
    
    if df_teo is not None:
        df_teo["dia_tipo_norm"] = (
            df_teo["day"]
            .map(lambda d: engine.DAY_TYPE_BY_DAYNUM.get(int(d), "desconocido") if pd.notna(d) else "desconocido")
            .str.lower()
        )
        profit_teo_total = df_teo.loc[df_teo["outcome_norm"] == "served", "total_profit_clp"].sum()
        profit_compare_df.loc[0, "case_base"] = profit_teo_total
        
        # Calcular profit teórico degradado (profit_teo - costos)
        profit_teo_degradado = profit_teo_total - lane_cost_annual
        
        if profit_teo_total:
            profit_compare_df.loc[0, "error_crudo_pct"] = (
                (prof_clp_estimado_anual - profit_teo_total) / profit_teo_total * 100.0
            )
            # Error degradado debe compararse contra profit_teo_degradado, NO contra profit_teo_total
            if profit_teo_degradado != 0:
                profit_compare_df.loc[0, "error_degradado_pct"] = (
                    (profit_degradado_anual - profit_teo_degradado) / profit_teo_degradado * 100.0
                )
            else:
                profit_compare_df.loc[0, "error_degradado_pct"] = np.nan
        profit_day_teo = (
            df_teo.loc[df_teo["outcome_norm"] == "served"]
              .groupby("dia_tipo_norm")["total_profit_clp"]
              .sum()
              .reset_index(name="profit_clp")
        )
        profit_day_teo = profit_day_teo.rename(columns={"dia_tipo_norm": "dia_tipo"})
        profit_day_cmp = _compare_kpi_tables(
            profit_day_df,
            profit_day_teo,
            key_cols=["dia_tipo"],
            numeric_cols=["profit_clp"],
            titulo="Profit por tipo de dia (Estimado vs Teorico)",
        )
        if profit_day_cmp is not None:
            profit_day_cmp = profit_day_cmp.merge(
                profit_day_df[["dia_tipo", "costo_cajas_clp", "profit_clp_degradado"]],
                on="dia_tipo",
                how="left",
            )
            # Calcular profit degradado teórico (profit_teo - costos)
            profit_day_cmp["profit_clp_degradado_teo"] = profit_day_cmp["profit_clp_teo"] - profit_day_cmp["costo_cajas_clp"]
            
            # Error porcentual del profit degradado (comparado contra degradado_teo, NO contra crudo_teo)
            profit_day_cmp["profit_clp_degradado_rel_err_pct"] = np.where(
                profit_day_cmp["profit_clp_degradado_teo"] != 0,
                (profit_day_cmp["profit_clp_degradado"] - profit_day_cmp["profit_clp_degradado_teo"]) / profit_day_cmp["profit_clp_degradado_teo"] * 100.0,
                np.nan,
            )
            profit_day_display = profit_day_cmp[
                [
                    "dia_tipo",
                    "profit_clp_teo",
                    "profit_clp_est",
                    "profit_clp_degradado_teo",
                    "profit_clp_degradado",
                    "costo_cajas_clp",
                    "profit_clp_rel_err_pct",
                    "profit_clp_degradado_rel_err_pct",
                ]
            ]
            profit_day_title = "PROFIT por tipo de dia (estimado vs teorico)"
        status_teo = df_teo[df_teo["outcome_norm"].isin(status_cols)].copy()
        counts_teo = status_teo.groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
        counts_teo["total"] = counts_teo.sum(axis=1)
        tac_teo = counts_teo.assign(
            served_pct=(counts_teo.get("served", 0) / counts_teo["total"] * 100).round(2),
            abandoned_pct=(counts_teo.get("abandoned", 0) / counts_teo["total"] * 100).round(2),
            balked_pct=(counts_teo.get("balked", 0) / counts_teo["total"] * 100).round(2),
        ).reset_index()
    
        tmc_teo = (
            df_teo.loc[df_teo["outcome_norm"] == "served"]
              .groupby("profile")["wait_time_s"].mean()
              .reset_index(name="tmc_wait_time_s")
        )
        tmc_teo["tmc_fmt"] = tmc_teo["tmc_wait_time_s"].map(lambda s: f"{s:.2f} s")
    
        hv_teo = _high_volume_slice(df_teo)
        if not hv_teo.empty:
            hv_teo_counts = hv_teo[hv_teo["outcome_norm"].isin(status_cols)].groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
            hv_teo_counts["total"] = hv_teo_counts.sum(axis=1)
            hv_teo_stats = hv_teo_counts.assign(
                tac_hv_pct=(((hv_teo_counts.get("abandoned", 0) + hv_teo_counts.get("balked", 0)) / hv_teo_counts["total"]) * 100).round(2)
            ).reset_index()
            hv_compare = (
                hv_stats.rename(columns={"tac_hv_pct": "tac_hv_pct_est"})
                .merge(
                    hv_teo_stats.rename(columns={"tac_hv_pct": "tac_hv_pct_teo"})[["profile", "tac_hv_pct_teo"]],
                    on="profile",
                    how="inner",
                )
            )
            hv_compare["tac_hv_pct_rel_err_pct"] = np.where(
                hv_compare["tac_hv_pct_teo"] != 0,
                (hv_compare["tac_hv_pct_est"] - hv_compare["tac_hv_pct_teo"]) / hv_compare["tac_hv_pct_teo"] * 100.0,
                np.nan,
            )
            hv_title = "TAC alto volumen (estimado vs teorico)"
            hv_display_df = hv_compare
            hv_formatted_df = hv_compare[["profile", "tac_hv_pct_teo", "tac_hv_pct_est", "tac_hv_pct_rel_err_pct"]]
    
        service_time_teo = (
            df_teo.loc[df_teo["outcome_norm"] == "served"]
              .groupby(["lane_type", "profile"])["service_time_s"]
              .mean()
              .reset_index(name="service_time_mean_teo_s")
        )
        service_time_teo["lane_type"] = (
            service_time_teo["lane_type"].astype(str).str.lower().map(engine.LANE_NAME_NORMALIZATION).fillna(service_time_teo["lane_type"])
        )
        if not service_time_teo.empty:
            service_time_est = service_time_sim.rename(columns={"service_time_mean_est": "service_time_mean"})
            service_time_base = service_time_teo.rename(columns={"service_time_mean_teo_s": "service_time_mean"})
            service_time_cmp = _compare_kpi_tables(
                service_time_est,
                service_time_base,
                key_cols=["lane_type", "profile"],
                numeric_cols=["service_time_mean"],
                titulo="Tiempos de servicio por tipo de caja (Estimado vs Teorico)",
            )
            if service_time_cmp is not None:
                service_time_display = service_time_cmp
                service_time_formatted = service_time_cmp[
                    [
                        "lane_type",
                        "profile",
                        "service_time_mean_teo",
                        "service_time_mean_est",
                        "service_time_mean_rel_err_pct",
                    ]
                ]
                service_time_title = "Tiempos de servicio por tipo de caja (estimado vs teorico)"
    
        kpi_day_teo_rows: list[dict[str, float]] = []
        for dia_tipo, slice_df in df_teo.groupby("dia_tipo_norm"):
            status_slice = slice_df[slice_df["outcome_norm"].isin(status_cols)]
            total_status = status_slice.shape[0]
            served_ct = int((status_slice["outcome_norm"] == "served").sum())
            abandoned_ct = int((status_slice["outcome_norm"] == "abandoned").sum())
            balked_ct = int((status_slice["outcome_norm"] == "balked").sum())
            profit_total = float(slice_df.loc[slice_df["outcome_norm"] == "served", "total_profit_clp"].sum())
            wait_mean = float(slice_df.loc[slice_df["outcome_norm"] == "served", "wait_time_s"].mean())
            kpi_day_teo_rows.append(
                {
                    "dia_tipo": dia_tipo,
                    "total_eventos": total_status,
                    "served": served_ct,
                    "abandoned": abandoned_ct,
                    "balked": balked_ct,
                    "served_pct": round((served_ct / total_status * 100) if total_status else 0.0, 2),
                    "abandoned_pct": round((abandoned_ct / total_status * 100) if total_status else 0.0, 2),
                    "balked_pct": round((balked_ct / total_status * 100) if total_status else 0.0, 2),
                    "profit_served_clp": profit_total,
                    "wait_time_avg_s": round(wait_mean, 2),
                }
            )
        kpi_day_teo = pd.DataFrame(kpi_day_teo_rows).sort_values("dia_tipo").reset_index(drop=True)
    
        compare_cols_tac = ["served", "abandoned", "balked", "total", "served_pct", "abandoned_pct", "balked_pct"]
        tac_cmp = _compare_kpi_tables(
            tac_profile,
            tac_teo,
            key_cols=["profile"],
            numeric_cols=compare_cols_tac,
            titulo="TAC por perfil (Estimado vs Teorico)",
        )
        if tac_cmp is not None:
            tac_display = tac_cmp
            tac_title = "TAC por perfil (estimado vs teorico)"
    
        tmc_cmp = _compare_kpi_tables(
            tmc_df,
            tmc_teo,
            key_cols=["profile"],
            numeric_cols=["tmc_wait_time_s"],
            titulo="TMC por perfil (Estimado vs Teorico)",
        )
        if tmc_cmp is not None:
            tmc_display = tmc_cmp
            tmc_title = "TMC por perfil (estimado vs teorico)"
    
        compare_cols_day = [
            "total_eventos",
            "served",
            "abandoned",
            "balked",
            "served_pct",
            "abandoned_pct",
            "balked_pct",
            "profit_served_clp",
            "wait_time_avg_s",
        ]
        kpi_day_cmp = _compare_kpi_tables(
            kpi_day_df,
            kpi_day_teo,
            key_cols=["dia_tipo"],
            numeric_cols=compare_cols_day,
            titulo="KPIs por tipo de dia (Estimado vs Teorico)",
        )
        if kpi_day_cmp is not None:
            kpi_day_display = kpi_day_cmp
            kpi_day_title = "KPIs por tipo de dia (estimado vs teorico)"

        client_counts_df = None
        est_counts = _summarize_client_status(df)
        teo_counts = _summarize_client_status(df_teo)
        if not est_counts.empty and not teo_counts.empty:
            client_counts_df = est_counts.merge(teo_counts, on=["dia_tipo", "profile"], how="inner", suffixes=("_est", "_teo"))
            for col in status_cols:
                est_col = f"{col}_est"
                teo_col = f"{col}_teo"
                err_col = f"{col}_rel_err_pct"
                client_counts_df[est_col] = client_counts_df[est_col].astype(float) * factor_anual
                client_counts_df[err_col] = np.where(
                    client_counts_df[teo_col] != 0,
                    (client_counts_df[est_col] - client_counts_df[teo_col]) / client_counts_df[teo_col] * 100.0,
                    np.nan,
                )
            client_counts_df = client_counts_df[
                [
                    "dia_tipo",
                    "profile",
                    "served_est",
                    "served_teo",
                    "served_rel_err_pct",
                    "abandoned_est",
                    "abandoned_teo",
                    "abandoned_rel_err_pct",
                    "balked_est",
                    "balked_teo",
                    "balked_rel_err_pct",
                ]
            ]
        def _count_days_by_type(source_df: pd.DataFrame) -> dict[str, int]:
            uniq = source_df[["dia_tipo_norm", "week", "day"]].drop_duplicates()
            return uniq.groupby("dia_tipo_norm").size().to_dict()

        def _build_little_summary(source_df: pd.DataFrame, days_map: dict[str, int]) -> pd.DataFrame:
            served = source_df[source_df["outcome_norm"] == "served"].copy()
            if served.empty:
                return pd.DataFrame()
            counts = (
                served.groupby(["dia_tipo_norm", "profile"])
                .size()
                .rename("served_total")
                .reset_index()
            )
            waits = (
                served.groupby(["dia_tipo_norm", "profile"])["wait_time_s"]
                .mean()
                .rename("wait_mean_s")
                .reset_index()
            )
            merged = counts.merge(waits, on=["dia_tipo_norm", "profile"], how="left")
            merged["dia_tipo"] = merged["dia_tipo_norm"]
            merged["days_count"] = merged["dia_tipo"].map(days_map).astype(float)
            merged = merged[merged["days_count"] > 0]
            jornada = engine.CLOSE_S - engine.OPEN_S
            merged["served_per_day"] = merged["served_total"] / merged["days_count"]
            merged["wait_mean_s"] = merged["wait_mean_s"].astype(float)
            merged["lambda_per_s"] = merged["served_per_day"] / float(jornada or 1)
            merged["L_value"] = merged["lambda_per_s"] * merged["wait_mean_s"]
            return merged[["dia_tipo", "profile", "served_per_day", "wait_mean_s", "lambda_per_s", "L_value"]]

        little_summary_df = None
        days_est = _count_days_by_type(df)
        days_teo = _count_days_by_type(df_teo)
        little_est = _build_little_summary(df, days_est)
        little_teo = _build_little_summary(df_teo, days_teo)
        if not little_est.empty and not little_teo.empty:
            little_summary_df = little_est.merge(
                little_teo,
                on=["dia_tipo", "profile"],
                suffixes=("_est", "_teo"),
                how="inner",
            )
            for col in ["served_per_day", "wait_mean_s", "lambda_per_s", "L_value"]:
                est_col = f"{col}_est"
                teo_col = f"{col}_teo"
                err_col = f"{col}_rel_err_pct"
                little_summary_df[err_col] = np.where(
                    little_summary_df[teo_col] != 0,
                    (little_summary_df[est_col] - little_summary_df[teo_col]) / little_summary_df[teo_col] * 100.0,
                    np.nan,
                )
        else:
            little_summary_df = None
    else:
        client_counts_df = None
        little_summary_df = None

    wait_day_profile_df = None
    wait_est = _build_wait_day_profile_summary(df)
    if not wait_est.empty:
        wait_day_profile_df = wait_est.rename(columns={"wait_time_s": "wait_time_s_est"})
        if df_teo is not None:
            wait_teo = _build_wait_day_profile_summary(df_teo)
        else:
            wait_teo = pd.DataFrame()
        if not wait_teo.empty:
            wait_day_profile_df = wait_day_profile_df.merge(
                wait_teo.rename(columns={"wait_time_s": "wait_time_s_teo"}),
                on=["dia_tipo", "profile"],
                how="left",
            )
        else:
            wait_day_profile_df["wait_time_s_teo"] = np.nan
        wait_day_profile_df["wait_time_s_rel_err_pct"] = np.where(
            wait_day_profile_df["wait_time_s_teo"] != 0,
            (wait_day_profile_df["wait_time_s_est"] - wait_day_profile_df["wait_time_s_teo"])
            / wait_day_profile_df["wait_time_s_teo"]
            * 100.0,
            np.nan,
        )

    tac_lane_formatted = None
    lane_est = _build_tac_lane_summary(df)
    if not lane_est.empty:
        tac_lane_formatted = lane_est.rename(
            columns={f"{col}_pct": f"{col}_pct_est" for col in status_cols}
        )
        if df_teo is not None:
            lane_teo = _build_tac_lane_summary(df_teo)
        else:
            lane_teo = pd.DataFrame()
        if not lane_teo.empty:
            tac_lane_formatted = tac_lane_formatted.merge(
                lane_teo.rename(columns={f"{col}_pct": f"{col}_pct_teo" for col in status_cols}),
                on=["lane_type", "profile"],
                how="left",
            )
        else:
            for col in status_cols:
                tac_lane_formatted[f"{col}_pct_teo"] = np.nan
        for col in status_cols:
            est_col = f"{col}_pct_est"
            teo_col = f"{col}_pct_teo"
            err_col = f"{col}_pct_rel_err_pct"
            tac_lane_formatted[err_col] = np.where(
                tac_lane_formatted[teo_col] != 0,
                (tac_lane_formatted[est_col] - tac_lane_formatted[teo_col]) / tac_lane_formatted[teo_col] * 100.0,
                np.nan,
            )
    
    print(f"\n{profit_day_title}")
    display(profit_day_display)
    
    print(f"\n{tac_title}")
    display(tac_display)
    
    print(f"\n{tmc_title}")
    display(tmc_display)
    
    print(f"\n{hv_title}")
    display(hv_display_df if hv_display_df is not None else hv_stats)
    
    print(f"\n{service_time_title}")
    display(service_time_display)
    
    if not kpi_day_display.empty:
        print(f"\n{kpi_day_title}")
        display(kpi_day_display)
    
    tables_to_export = {
        "profit_summary": profit_compare_df,
        "profit_daytype": profit_day_display,
        "monthly_profit": monthly_prof,
        "tac_profile": tac_display,
        "tmc_profile": tmc_display,
        "tac_high_volume": hv_display_df if hv_display_df is not None else hv_stats,
        "kpi_daytype": kpi_day_display,
        "service_time_lane_profile": service_time_display,
        "client_counts": client_counts_df,
        "little_law": little_summary_df,
    }
    export_path = export_kpi_report(resultados_root, tables_to_export)
    if export_path:
        print(f"\nKPIs exportados a {export_path}")
    formatted_path = export_formatted_excel_report(
        export_root=resultados_root,
        profit_df=profit_compare_df,
        profit_day_df=profit_day_display,
        wait_day_profile_df=wait_day_profile_df,
        service_time_df=service_time_formatted,
        tac_lane_df=tac_lane_formatted,
        tac_df=tac_display,
        tmc_df=tmc_display,
        hv_df=hv_formatted_df,
        client_counts_df=client_counts_df,
        little_df=little_summary_df,
        profit_anual_estimado=prof_clp_estimado_anual,
    )
    if formatted_path:
        print(f"Reporte formateado exportado a {formatted_path}")
    
    elapsed = time.perf_counter() - run_start
    print(f"\nTiempo total de ejecucion del simulador: {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")

    return {"tables_path": export_path, "formatted_path": formatted_path}
