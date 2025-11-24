import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
DATA_DIR = Path("outputs_teoricos")
SETUP_TIME_ESTIMATE = 15.0  # Asumimos 15s de setup promedio para despejar la tasa

def validate_sco_distribution():
    print("=== VALIDACIÓN ESTADÍSTICA MODELO SELF-CHECKOUT ===")
    
    # 1. Cargar datos de SCO
    print("Cargando datos de Self-Checkout...")
    data = []
    # Leemos una muestra representativa (primeros 50 días) para no saturar memoria
    files = sorted(list(DATA_DIR.glob("Week-*-Day-*/customers.csv")))[:50]
    
    if not files:
        print("Error: No se encontraron archivos en outputs_teoricos/")
        return

    for f in files:
        try:
            df = pd.read_csv(f, usecols=["lane_type", "items", "service_time_s"])
            # Normalizar nombres
            df.columns = [c.strip().lower() for c in df.columns]
            # Filtrar SCO
            sco = df[df["lane_type"].astype(str).str.lower().isin(["self_checkout", "sco", "autocaja"])]
            sco = sco.dropna()
            sco = sco[sco["service_time_s"] > 0]
            sco = sco[sco["items"] > 0]
            data.append(sco)
        except:
            continue
            
    if not data:
        print("No se encontraron datos de Self-Checkout.")
        return

    full_df = pd.concat(data)
    print(f"Datos cargados: {len(full_df)} transacciones SCO.")

    # 2. Calcular Tasa de Procesamiento Observada (Segundos por Ítem)
    # Rate = (TotalTime - Setup) / Items
    # Si (TotalTime - Setup) < 0, asumimos que fue una transacción muy rápida o setup fue menor.
    # Para el ajuste, filtramos valores positivos y razonables.
    
    # Usamos una estimación conservadora: Rate = Time / Items (asumiendo setup diluido o pequeño)
    # O mejor: Rate = (Time - 10) / Items para ser más precisos, filtrando negativos
    
    full_df["calculated_rate"] = (full_df["service_time_s"] - SETUP_TIME_ESTIMATE) / full_df["items"]
    
    # Filtrar ruido (tasas negativas o irreales)
    valid_rates = full_df[
        (full_df["calculated_rate"] > 0.5) &  # Mínimo 0.5s por item
        (full_df["calculated_rate"] < 120.0)  # Máximo 2 min por item (outlier extremo)
    ]["calculated_rate"].values

    print(f"Muestras válidas para ajuste: {len(valid_rates)}")
    print(f"Media Observada: {np.mean(valid_rates):.2f} s/item")
    print(f"Mediana Observada: {np.median(valid_rates):.2f} s/item")

    # 3. Ajuste de Distribuciones
    print("\nProbando distribuciones candidatas...")
    distributions = [
        ("Lognormal", stats.lognorm),
        ("Gamma", stats.gamma),
        ("Weibull", stats.weibull_min),
        ("Exponencial", stats.expon),
        ("Normal", stats.norm)
    ]

    results = []
    
    for name, dist in distributions:
        # Fit
        params = dist.fit(valid_rates, floc=0) # Forzamos loc=0 para tasas (no pueden ser negativas)
        
        # KS Test
        ks_stat, p_value = stats.kstest(valid_rates, dist.name, args=params)
        
        # Log Likelihood
        log_lik = np.sum(dist.logpdf(valid_rates, *params))
        
        results.append({
            "Dist": name,
            "KS Stat": ks_stat,
            "P-Value": p_value,
            "LogLik": log_lik,
            "Params": params
        })

    # Ordenar por mejor ajuste (menor KS)
    results.sort(key=lambda x: x["KS Stat"])
    
    print("\nResultados del Ajuste (Ordenado por KS):")
    print(f"{'Distribución':<15} | {'KS Stat':<10} | {'P-Value':<10} | {'LogLik':<10} | {'Parámetros'}")
    print("-" * 80)
    
    for r in results:
        params_str = ", ".join([f"{p:.2f}" for p in r["Params"]])
        print(f"{r['Dist']:<15} | {r['KS Stat']:.4f}     | {r['P-Value']:.2e}   | {r['LogLik']:.0f}       | {params_str}")

    best = results[0]
    print(f"\n>>> CONCLUSIÓN: La mejor distribución es {best['Dist'].upper()}")
    print(f"    Parámetros sugeridos para engine.py: {best['Params']}")

    # 4. Visualización
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_rates, stat="density", bins=50, color="skyblue", alpha=0.6, label="Datos Observados")
    
    x = np.linspace(0, np.percentile(valid_rates, 99), 1000)
    
    # Plot Best Fit
    dist = dict(distributions)[best["Dist"]]
    pdf = dist.pdf(x, *best["Params"])
    plt.plot(x, pdf, 'r-', lw=3, label=f"Mejor Ajuste: {best['Dist']}")
    
    # Plot Lognormal (si no ganó, para comparar)
    if best["Dist"] != "Lognormal":
        ln_params = [r["Params"] for r in results if r["Dist"] == "Lognormal"][0]
        pdf_ln = stats.lognorm.pdf(x, *ln_params)
        plt.plot(x, pdf_ln, 'g--', lw=2, label="Lognormal (Referencia)")

    plt.title("Distribución de Velocidad de Escaneo (Self-Checkout)\n(Segundos por Ítem)")
    plt.xlabel("Segundos / Ítem")
    plt.ylabel("Densidad")
    plt.legend()
    plt.xlim(0, 60)
    
    out_path = Path("report_visualizations/sco_rate_validation.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"\nGráfico guardado en: {out_path}")

if __name__ == "__main__":
    validate_sco_distribution()
