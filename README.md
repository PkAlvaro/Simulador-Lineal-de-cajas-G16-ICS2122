> **Disclaimer**: Este documento README.md, así como gran parte del código del proyecto, han sido generados con asistencia de Inteligencia Artificial.

# SIMULADOR FINAL P2

Este repositorio contiene el simulador integral de cajas de supermercado (Proyecto P2). El objetivo es emular el flujo completo de clientes (arribos, asignación de cajas, tiempos de servicio/paciencia, balking y métricas operacionales) y compararlo contra la base teórica para realizar calibraciones recurrentes.

## Estructura principal

- `simulator/engine.py`: motor del simulador (arribos PPNH, lógica de colas, tiempos de servicio, balking, reporting).
- `simulator/reporting.py`: generación de KPI y exportes en CSV/XLSX (incluye hojas formateadas con profit, TAC, TMC, tiempos de servicio, tabla de clientes y análisis de Little).
- `simulator/policy_planner.py`: módulo de planificación estratégica que gestiona la optimización secuencial multi-año (2025-2030) y la evaluación de políticas de inversión única (`single_investment`).
- `tools/`: scripts auxiliares para reconstruir parámetros desde los outputs teóricos. Entre otros:
  - `rebuild_arrivals.py`: genera los `.npz` de lambdas por perfil/prioridad/medio de pago/día. Cuenta con la opción `--apply-little` para escalar las tasas de llegada usando la ley de Little.
  - `calc_service_time_multipliers.py`: calcula multiplicadores por `(lane_type, profile)` comparando los tiempos teóricos vs. la predicción del modelo `service_time_model.json`. El resultado se guarda en `service_time/service_time_multipliers.csv` y se aplica automáticamente en el simulador.
  - `run_sensitivity_plan.py`: ejecuta un análisis de sensibilidad variando los límites de ítems para cajas rápidas y *self-checkout* bajo distintos escenarios definidos en un JSON.
- `arrivals_npz/`: lambdas PPNH por perfil (se generan con el script anterior).
- `service_time/`: definiciones del modelo de tiempo de servicio y los multiplicadores.
- `patience/`: distribuciones de paciencia (por perfil, prioridad y medio de pago) que se cargan en cada corrida.
- `optimizador_cajas.py`: optimizador GRASP+SAA+ búsqueda local que usa el simulador para evaluar configuraciones de cajas por tipo de día, calculando el objetivo `profit anual estimado – costo anual`.

## Flujo de calibración

1. **Datos teóricos**: descargar/actualizar `outputs_teoricos/Week-*-Day-*/` (contienen los archivos `customers.csv` y `time_log.csv` reales).
2. **Reconstruir lambdas PPNH**:
   ```bash
   python tools/rebuild_arrivals.py --root outputs_teoricos --output-dir arrivals_npz
   # Opcional: escalar las series usando Little
   python tools/rebuild_arrivals.py --apply-little
   ```
3. **Generar multiplicadores de tiempos de servicio**:
   ```bash
   python tools/calc_service_time_multipliers.py \
       --root outputs_teoricos \
       --model service_time/service_time_model.json \
       --output service_time/service_time_multipliers.csv
   ```
   El simulador leerá automáticamente ese archivo y ajustará los tiempos según `(lane_type, profile)`.
4. **Ejecutar simulación** (menú principal):
   - `1` → simular N semanas y generar KPIs.
   - `2` → simular el año completo (52 semanas).
   - `3` → optimizador de cajas (GRASP+SAA) para cada tipo de día, combinando resultados y descontando costo anual de infraestructura.
   - `4` → planificación secuencial multi-año: optimiza políticas para el horizonte 2026-2030 basándose en proyecciones de demanda (segmentos pesimista, regular, optimista) y re-optimizando año a año o buscando una política única de inversión.

## Métricas generadas

`simulator/reporting.py` exporta dos archivos por corrida:

- `resultados/kpi_resumen_YYYYMMDD_HHMMSS.xlsx`: tablas comparativas básicas (profit total, profit por día, TAC, TMC, TAC HV, tiempos de servicio).
- `resultados/kpi_formato_YYYYMMDD_HHMMSS.xlsx`: versión formateada (hoja principal + pestañas “Clientes” y “Little”). Estas hojas incluyen:
  - Comparación simulación vs. caso base (`served/abandoned/balked` por perfil y tipo de día).
  - Análisis de la ley de Little (`λ`, `W`, `L` teóricos vs. simulados).

## Análisis de Sensibilidad

Para evaluar el impacto de cambiar los límites de productos en cajas Express y Self-Checkout, se utiliza el script dedicado:

```bash
python tools/run_sensitivity_plan.py \
    --scenarios scenarios_limits.json \
    --segments pesimista,regular,optimista \
    --output-dir resultados_sensibilidad
```

Esto generará reportes comparativos en `resultados_sensibilidad/` para cada escenario configurado en el JSON.

## Notas adicionales

- El motor usa los `.npz` de `arrivals_npz/` (no PKL). Asegúrate de regenerarlos cuando existan cambios en la base teórica.
- Los scripts de calibración están pensados para correrse **antes** de la simulación y no dependen de los resultados de esa corrida; así los multiplicadores y tasas permanecen estables.
- El optimizador usa el mismo motor (no hay “simulador paralelo”). Todas las políticas x = (regular, express, priority, self_checkout) se aplican mediante `enforce_lane_constraints` y `update_current_lane_policy`.
