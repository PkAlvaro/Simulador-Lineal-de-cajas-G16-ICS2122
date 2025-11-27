# Resumen del Ajuste de Distribuciones de Paciencia

**Fecha**: 2025-11-24
**Fuente de datos**: `outputs_teoricos` (solo clientes abandonados)
**Total clientes abandonados**: 211,144
**Total combinaciones ajustadas**: 92

## Resumen de Validacion

**Combinaciones validadas**: 79 (con al menos 30 observaciones)

### Metricas de Error en Media

- **Error promedio**: 6.03%
- **Error std**: 26.30%
- **Error mediana**: 14.98%

### Distribucion de Errores

- **< 10% error abs**: 35.4% de las combinaciones
- **< 20% error abs**: 57.0% de las combinaciones
- **< 30% error abs**: 77.2% de las combinaciones

## Mejores Ajustes (Top 5)

| Profile | Day Type | Media Teorica (s) | Media Ajustada (s) | Error (%) |
|---------|----------|-------------------|---------------------|-----------|
| weekly_planner | tipo_2 | 112.95 | 113.37 | 0.37 |
| family_cart | tipo_2 | 96.24 | 95.64 | -0.62 |
| express_basket | tipo_3 | 77.17 | 75.83 | -1.74 |
| weekly_planner | tipo_2 | 119.54 | 122.36 | 2.36 |
| deal_hunter | tipo_2 | 127.19 | 124.05 | -2.47 |

## Peores Ajustes (Top 5)

| Profile | Day Type | Media Teorica (s) | Media Ajustada (s) | Error (%) |
|---------|----------|-------------------|---------------------|-----------|
| deal_hunter | tipo_3 | 70.68 | 137.51 | 94.56 |
| self_checkout_fan | tipo_3 | 122.87 | 224.03 | 82.32 |
| express_basket | tipo_3 | 64.48 | 99.61 | 54.49 |
| express_basket | tipo_1 | 67.46 | 102.38 | 51.77 |
| family_cart | tipo_3 | 103.96 | 155.45 | 49.52 |

## Observaciones

1. **Los 3 peores ajustes son para tipo_3 (Viernes/Sabado)**: Esto sugiere que el comportamiento de paciencia en fin de semana puede ser diferente, posiblemente debido a:
   - Menor volumen de datos de entrenamiento
   - Comportamiento más heterogeneo de clientes en fin de semana

2. **Los mejores ajustes son principalmente tipo_2 (Martes-Jueves)**: Estos dias tienen más datos y comportamiento más homogeneo.

3. **Error general aceptable**: Con un 57% de combinaciones con error < 20%, el ajuste es razonablemente bueno para la mayoria de los casos.

## Archivos Generados

- `data/patience/patience_distribution_profile_priority_payment_day.csv`: Parametros de distribuciones ajustadas
- `data/patience/validation_results.csv`: Resultados completos de la validacion
- `data/patience/validation_errors.png`: Histogramas de errores

## Proximos Pasos

1. **Calibracion fina**: Considerar ajustar manualmente los parametros de las combinaciones con mayor error
2. **Segmentacion adicional**: Evaluar si es necesario segmentar tipo_3 en viernes vs sabado
3. **Prueba en simulacion**: Ejecutar simulaciones con las nuevas paciencias y comparar resultados con teor ico
