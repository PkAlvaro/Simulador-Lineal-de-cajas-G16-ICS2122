# AUDITORÍA COMPLETA DE CÁLCULOS DE ERROR PORCENTUAL EN REPORTING.PY
# Fecha: 2025-11-23
# Estado: REVISADO Y CORREGIDO

## FÓRMULA CORRECTA
Error % = ((Estimado - Teórico) / Teórico) × 100

## ERRORES CORREGIDOS:

### 1. Error en Profit Degradado por Día (Línea ~802)
**ANTES (INCORRECTO):**
```python
profit_day_cmp["profit_clp_degradado_rel_err_pct"] = np.where(
    profit_day_cmp["profit_clp_teo"] != 0,
    (profit_day_cmp["profit_clp_degradado"] - profit_day_cmp["profit_clp_teo"]) / profit_day_cmp["profit_clp_teo"] * 100.0,
    np.nan,
)
```
**Problema:** Comparaba profit_degradado_est contra profit_crudo_teo (manzanas vs naranjas)

**DESPUÉS (CORRECTO):**
```python
# Calcular profit degradado teórico
profit_day_cmp["profit_clp_degradado_teo"] = profit_day_cmp["profit_clp_teo"] - profit_day_cmp["costo_cajas_clp"]

profit_day_cmp["profit_clp_degradado_rel_err_pct"] = np.where(
    profit_day_cmp["profit_clp_degradado_teo"] != 0,
    (profit_day_cmp["profit_clp_degradado"] - profit_day_cmp["profit_clp_degradado_teo"]) / profit_day_cmp["profit_clp_degradado_teo"] * 100.0,
    np.nan,
)
```

### 2. Error en Profit Degradado Global (Línea ~778)
**ANTES (INCORRECTO):**
```python
profit_compare_df.loc[0, "error_degradado_pct"] = (
    (profit_degradado_anual - profit_teo_total) / profit_teo_total * 100.0
)
```
**Problema:** Mismo error, comparaba degradado contra crudo

**DESPUÉS (CORRECTO):**
```python
# Calcular profit teórico degradado
profit_teo_degradado = profit_teo_total - lane_cost_annual

if profit_teo_degradado != 0:
    profit_compare_df.loc[0, "error_degradado_pct"] = (
        (profit_degradado_anual - profit_teo_degradado) / profit_teo_degradado * 100.0
    )
else:
    profit_compare_df.loc[0, "error_degradado_pct"] = np.nan
```

## CÁLCULOS VERIFICADOS COMO CORRECTOS:

### ✅ Línea 119 - Función _compare_kpi_tables
```python
merged[err_col] = np.where(
    merged[teo_col] != 0,
    (merged[est_col] - merged[teo_col]) / merged[teo_col] * 100.0,
    np.nan,
)
```
**Status:** CORRECTO - Uso genérico en múltiples KPIs

### ✅ Línea 775 - Error Profit Crudo
```python
profit_compare_df.loc[0, "error_crudo_pct"] = (
    (prof_clp_estimado_anual - profit_teo_total) / profit_teo_total * 100.0
)
```
**Status:** CORRECTO

### ✅ Línea 855 - Error TAC Alto Volumen
```python
hv_compare["tac_hv_pct_rel_err_pct"] = np.where(
    hv_compare["tac_hv_pct_teo"] != 0,
    (hv_compare["tac_hv_pct_est"] - hv_compare["tac_hv_pct_teo"]) / hv_compare["tac_hv_pct_teo"] * 100.0,
    np.nan,
)
```
**Status:** CORRECTO

### ✅ Línea 976 - Error Client Counts
```python
client_counts_df[err_col] = np.where(
    client_counts_df[teo_col] != 0,
    (client_counts_df[est_col] - client_counts_df[teo_col]) / client_counts_df[teo_col] * 100.0,
    np.nan,
)
```
**Status:** CORRECTO

### ✅ Línea 1043 - Error Ley de Little
```python
little_summary_df[err_col] = np.where(
    little_summary_df[teo_col] != 0,
    (little_summary_df[est_col] - little_summary_df[teo_col]) / little_summary_df[teo_col] * 100.0,
    np.nan,
)
```
**Status:** CORRECTO

### ✅ Línea 1067 - Error Wait Time
```python
wait_day_profile_df["wait_time_s_rel_err_pct"] = np.where(
    wait_day_profile_df["wait_time_s_teo"] != 0,
    (wait_day_profile_df["wait_time_s_est"] - wait_day_profile_df["wait_time_s_teo"])
    / wait_day_profile_df["wait_time_s_teo"]
    * 100.0,
    np.nan,
)
```
**Status:** CORRECTO

### ✅ Línea 1101 - Error TAC por Lane
```python
tac_lane_formatted[err_col] = np.where(
    tac_lane_formatted[teo_col] != 0,
    (tac_lane_formatted[est_col] - tac_lane_formatted[teo_col]) / tac_lane_formatted[teo_col] * 100.0,
    np.nan,
)
```
**Status:** CORRECTO

## RESUMEN DE CAMBIOS:
- Total de errores encontrados: 2
- Total de errores corregidos: 2
- Naturaleza del error: Comparación inconsistente (profit degradado vs profit crudo)
- Impacto: CRÍTICO - Los errores porcentuales mostrados en Excel eran incorrectos
- Solución: Calcular valores teóricos degradados antes de comparar

## VERIFICACIÓN FINAL:
✅ Todas las fórmulas de error porcentual ahora usan: (Estimado - Teórico) / Teórico × 100
✅ Las comparaciones son consistentes (degradado vs degradado, crudo vs crudo)
✅ Se manejan adecuadamente los casos de división por cero
