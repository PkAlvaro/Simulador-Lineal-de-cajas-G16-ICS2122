"""
Valida la calidad del ajuste de distribuciones de paciencia comparando con datos teoricos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import json


def load_theoretical_patience():
    """Carga paciencias de clientes abandonados desde outputs teoricos"""
    root = Path("outputs_teoricos")
    all_data = []
    
    # Mapeo de dia de la semana a tipo de dia
    day_to_type = {
        1: "tipo_1",  # Lunes
        2: "tipo_2",  # Martes
        3: "tipo_2",  # Miercoles
        4: "tipo_2",  # Jueves
        5: "tipo_3",  # Viernes
        6: "tipo_3",  # Sabado
        0: "tipo_1"   # Domingo (7 mod 7 = 0)
    }
    
    for csv_file in root.rglob("customers.csv"):
        df = pd.read_csv(csv_file)
        
        # Inferir day_type del nombre del directorio
        # Formato: Week-X-Day-Y
        dir_name = csv_file.parent.name
        if "Day-" in dir_name:
            day_num = int(dir_name.split("Day-")[1])
            day_of_week = day_num % 7
            day_type = day_to_type[day_of_week]
        else:
            continue
        
        # Solo clientes abandonados
        df_balked = df[df["outcome"] == "abandoned"].copy()
        
        if len(df_balked) == 0:
            continue
            
        # Asignar day_type
        df_balked["day_type"] = day_type
        all_data.append(df_balked[["profile", "priority", "payment_method", "day_type", "patience_s"]])
    
    return pd.concat(all_data, ignore_index=True)


def load_fitted_distributions():
    """Carga parametros de distribuciones ajustadas"""
    df = pd.read_csv("data/patience/patience_distribution_profile_priority_payment_day.csv")
    return df[df["method"] == "param"]


def sample_from_params(dist_name, params_str, n_samples=10000):
    """Genera muestras de una distribucion parametrizada"""
    params = json.loads(params_str)
    
    if dist_name == "genpareto":
        # scipy.stats.genpareto: genpareto(c, loc, scale)
        return stats.genpareto.rvs(c=params[0], loc=params[1], scale=params[2], size=n_samples)
    elif dist_name == "weibull_min":
        # scipy.stats.weibull_min: weibull_min(c, loc, scale)
        return stats.weibull_min.rvs(c=params[0], loc=params[1], scale=params[2], size=n_samples)
    elif dist_name == "gamma":
        # scipy.stats.gamma: gamma(a, loc, scale)
        return stats.gamma.rvs(a=params[0], loc=params[1], scale=params[2], size=n_samples)
    elif dist_name == "lognorm":
        # scipy.stats.lognorm: lognorm(s, loc, scale)
        return stats.lognorm.rvs(s=params[0], loc=params[1], scale=params[2], size=n_samples)
    elif dist_name == "norm":
        # scipy.stats.norm: norm(loc, scale)
        return stats.norm.rvs(loc=params[0], scale=params[1], size=n_samples)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def validate_fit():
    """Valida el ajuste comparando medias y percentiles"""
    print("Cargando datos teoricos...")
    df_theo = load_theoretical_patience()
    
    print("Cargando distribuciones ajustadas...")
    df_fit = load_fitted_distributions()
    
    results = []
    
    for idx, row in df_fit.iterrows():
        # Filtrar datos teoricos para esta combinacion
        mask = (
            (df_theo["profile"] == row["profile"]) &
            (df_theo["priority"] == row["priority"]) &
            (df_theo["payment_method"] == row["payment_method"]) &
            (df_theo["day_type"] == row["day_type"])
        )
        
        theo_data = df_theo[mask]["patience_s"].values
        
        if len(theo_data) < 30:
            continue
        
        # Generar muestras de la distribucion ajustada
        fitted_samples = sample_from_params(row["distribution"], row["params"], n_samples=10000)
        
        # Comparar estadisticas
        theo_mean = np.mean(theo_data)
        fitted_mean = np.mean(fitted_samples)
        
        theo_median = np.median(theo_data)
        fitted_median = np.median(fitted_samples)
        
        theo_p90 = np.percentile(theo_data, 90)
        fitted_p90 = np.percentile(fitted_samples, 90)
        
        results.append({
            "profile": row["profile"],
            "priority": row["priority"],
            "payment_method": row["payment_method"],
            "day_type": row["day_type"],
            "n_theo": len(theo_data),
            "theo_mean": theo_mean,
            "fitted_mean": fitted_mean,
            "mean_error_pct": 100 * (fitted_mean - theo_mean) / theo_mean,
            "theo_median": theo_median,
            "fitted_median": fitted_median,
            "median_error_pct": 100 * (fitted_median - theo_median) / theo_median,
            "theo_p90": theo_p90,
            "fitted_p90": fitted_p90,
            "distribution": row["distribution"]
        })
    
    df_results = pd.DataFrame(results)
    
    # Resumen
    print("\n=== RESUMEN DE VALIDACION ===")
    print(f"Total combinaciones validadas: {len(df_results)}")
    print(f"\nError promedio en media: {df_results['mean_error_pct'].mean():.2f}%")
    print(f"Error promedio en mediana: {df_results['median_error_pct'].mean():.2f}%")
    print(f"\nDistribucion de errores absolutos en media:")
    print(df_results['mean_error_pct'].abs().describe())
    
    # Guardar resultados
    output_file = Path("data/patience/validation_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nResultados completos guardados en: {output_file}")
    
    # Casos con mayor error
    print("\n=== CASOS CON MAYOR ERROR (Top 10) ===")
    top_errors = df_results.nlargest(10, "mean_error_pct", keep="all")[
        ["profile", "priority", "payment_method", "day_type", "theo_mean", "fitted_mean", "mean_error_pct"]
    ]
    print(top_errors.to_string())
    
    # Plot histograma de errores
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filtrar NaN para evitar errores en histograma
    mean_errors = df_results["mean_error_pct"].dropna()
    median_errors = df_results["median_error_pct"].dropna()
    
    axes[0].hist(mean_errors, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Error en Media (%)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribucion de Errores en Media")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(median_errors, bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Error en Mediana (%)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribucion de Errores en Mediana")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("data/patience/validation_errors.png", dpi=150, bbox_inches="tight")
    print("\nGrafico guardado en: data/patience/validation_errors.png")
    
    return df_results


if __name__ == "__main__":
    validate_fit()
