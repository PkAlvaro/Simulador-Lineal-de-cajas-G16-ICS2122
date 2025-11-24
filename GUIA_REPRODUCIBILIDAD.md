# GUÍA DE REPRODUCIBILIDAD: CONTROL DE SEMILLAS ALEATORIAS

## PROBLEMA ORIGINAL
Al ejecutar la simulación múltiples veces, los resultados variaban debido a la aleatoriedad inherente del sistema. Esto dificultaba:
- Debugging y análisis de errores
- Comparación between versiones del código
- Validación científica de resultados
- Análisis estadístico riguroso

## SOLUCIÓN IMPLEMENTADA: Semilla Maestra

Se implementó un **sistema centralizado de control de semillas** que permite reproducibilidad completa.

### SEMILLA MAESTRA
```python
_MASTER_SEED = 42  # Por defecto, puedes cambiarla
```

### GENERADORES ALEATORIOS (RNGs) CONTROLADOS

Todos los RNGs ahora derivan de la semilla maestra:

| RNG | Semilla | Uso |
|-----|---------|-----|
| `arrival_rng` | `_MASTER_SEED + 100 + week*7 + day` | Tiempos entre llegadas (Lognormal) y selección de perfiles |
| `RNG_ITEMS` | `_MASTER_SEED + 1` | Cantidad de ítems por cliente |
| `RNG_PROFIT` | `_MASTER_SEED + 2` | Ruido en cálculo de profit |
| `ServiceTimeModel.rng` | `_MASTER_SEED + 3` | Tiempos de servicio (residuos) |
| `PatienceDistribution.rng` | `_MASTER_SEED + 4` | Tiempos de paciencia |

**Nota:** `arrival_rng` varía por día para evitar patrones idénticos entre días, pero sigue siendo reproducible.

## CÓMO USAR

### Opción 1: Usar semilla predeterminada (42)
```python
import simulator.engine as engine

# La semilla 42 se usa automáticamente
stats = engine.simulacion_periodos(num_weeks=2)
```

Ejecutar esto **siempre** dará los mismos resultados.

### Opción 2: Cambiar la semilla maestra
```python
import simulator.engine as engine

# Para análisis A
engine.set_global_seed(100)
stats_A = engine.simulacion_periodos(num_weeks=2)

# Para análisis B
engine.set_global_seed(200)
stats_B = engine.simulacion_periodos(num_weeks=2)
```

- `stats_A` siempre será idéntico cuando uses semilla 100
- `stats_B` siempre será idéntico cuando uses semilla 200
- `stats_A` será diferente a `stats_B`

### Opción 3: Múltiples réplicas (Monte Carlo)
```python
import simulator.engine as engine
import numpy as np

# Ejecutar 10 réplicas independientes
resultados = []
for replica in range(10):
    engine.set_global_seed(42 + replica)  # Semillas diferentes: 42, 43, 44, ...
    stats = engine.simulacion_periodos(num_weeks=2)
    resultados.append(stats)

# Calcular intervalos de confianza, etc.
profits = [sum(s['profit'] for s in run) for run in resultados]
print(f"Profit promedio: {np.mean(profits):,.0f}")
print(f"Std: {np.std(profits):,.0f}")
```

## FLUJO INTERNO

```
Usuario llama set_global_seed(42)
        ↓
_MASTER_SEED = 42
        ↓
        ├─→ RNG_ITEMS inicializado con seed=43
        ├─→ RNG_PROFIT inicializado con seed=44
        ├─→ SERVICE_TIME_MODEL.rng con seed=45
        ├─→ PatienceDistribution.rng con seed=46
        └─→ arrival_rng (por día) = seed + 100 + week*7 + day
```

## VERIFICACIÓN DE REPRODUCIBILIDAD

Para probar que funciona:

```python
import simulator.engine as engine

# Primera ejecución
engine.set_global_seed(123)
stats1 = engine.simulacion_periodos(num_weeks=1)

# Segunda ejecución (misma semilla)
engine.set_global_seed(123)
stats2 = engine.simulacion_periodos(num_weeks=1)

# Verificar igualdad
for i, (s1, s2) in enumerate(zip(stats1, stats2)):
    assert s1['total_clientes'] == s2['total_clientes'], f"Día {i} difiere!"
    
print("✓ Reproducibilidad verificada: resultados idénticos")
```

## RECOMENDACIONES

### Para debugging/desarrollo:
```python
engine.set_global_seed(42)  # Siempre la misma
```

### Para análisis estadístico robusto:
```python
# Ejecutar N réplicas
for i in range(N):
    engine.set_global_seed(1000 + i)
    # ... ejecutar simulación
```

### Para comparar dos versiones del código:
1. Usa la **misma** semilla en ambas versiones
2. Cualquier diferencia en los resultados indica cambios en la lógica (no en la aleatoriedad)

## NOTAS TÉCNICAS

- **Independencia entre RNGs**: Cada RNG tiene su propia secuencia, evitando correlación espuria.
- **SimPy**: SimPy NO usa estos RNGs (usa su propia aleatoriedad interna mínima para eventos).
- **Legacy np.random.X()**: Se estableció `np.random.seed()` como respaldo para código antiguo.

## LIMITACIONES

- Si modificas manualmente algún RNG después de `set_global_seed()`, perderás reproducibilidad.
- Agregar/quitar código que llama a RNGs cambiará los resultados (esperado).
