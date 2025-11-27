"""
Compara distribuciones parametricas para los tiempos inter-arrival rescalados (RIATs).
Usa estadigrafos solidos para determinar cual distribucion ajusta mejor.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_ROOT = Path("outputs_teoricos")
NPZ_DIR = Path("data/arrivals_npz")
OUTPUT_DIR = Path("report_visualizations")
SAMPLE_DAYS = 50

def load_riats():
    """Carga y calcula los RIATs de los datos teoricos."""
    print("=== CALCULANDO RIATs DE DATOS TEORICOS ===\n")
    
    riats = []
    lambda_cache = {}
    
    processed_count = 0
    for day_dir in sorted(list(DATA_ROOT.glob("Week-*-Day-*"))):
        if processed_count >= SAMPLE_DAYS:
            break
            
        time_log = day_dir / "time_log.csv"
        if not time_log.exists():
            continue
            
        try:
            df = pd.read_csv(time_log, usecols=["event_type", "timestamp_s"])
            arrivals = df[df["event_type"] == "arrival"].sort_values("timestamp_s")
            
            if arrivals.empty:
                continue
                
            # Determinar tipo de dia
            day_num = int(day_dir.name.split("-")[-1])
            day_type_map = {
                1: "tipo_1", 2: "tipo_1", 3: "tipo_2", 4: "tipo_1", 
                5: "tipo_2", 6: "tipo_2", 7: "tipo_3"
            }
            day_type = day_type_map.get(day_num, "tipo_1")
            
            # Construir lambda total para este dia
            if day_type not in lambda_cache:
                total_lambda = np.zeros(14 * 3600 + 1)
                
                for npz_file in NPZ_DIR.glob("lambda_*.npz"):
                    data = np.load(npz_file)
                    keys = data["keys"]
                    lambdas = data["lambdas"]
                    bin_left = data["bin_left_s"]
                    
                    for i, key in enumerate(keys):
                        parts = str(key).split("|")
                        if parts[0] == day_type:
                            lam_per_sec = lambdas[i] / 60.0
                            
                            for b in range(len(bin_left)):
                                start = int(bin_left[b])
                                end = int(bin_left[b+1]) if b+1 < len(bin_left) else 14*3600
                                val = lam_per_sec[b]
                                total_lambda[start:end] += val
                                
                lambda_cache[day_type] = total_lambda
            
            lam_series = lambda_cache[day_type]
            
            # Calcular RIATs
            times = arrivals["timestamp_s"].values
            times = times[(times >= 0) & (times < len(lam_series))]
            
            if len(times) < 2:
                continue
                
            times = np.sort(times)
            lam_cumsum = np.cumsum(lam_series)
            indices = np.searchsorted(np.arange(len(lam_cumsum)), times)
            cum_values = lam_cumsum[indices.clip(0, len(lam_cumsum)-1)]
            diffs = np.diff(cum_values)
            diffs = diffs[diffs > 1e-6]
            
            riats.extend(diffs)
            processed_count += 1
            
        except Exception as e:
            print(f"Error procesando {day_dir}: {e}")
            continue
    
    riats = np.array(riats)
    print(f"Total Intervalos: {len(riats)}")
    print(f"Media: {np.mean(riats):.4f}")
    print(f"Desv. Std: {np.std(riats):.4f}")
    print(f"Mediana: {np.median(riats):.4f}\n")
    
    return riats


def fit_exponential(data):
    """Ajusta distribucion Exponencial."""
    # Para Exponencial, el parametro es 1/media
    loc, scale = stats.expon.fit(data, floc=0)
    return {'loc': loc, 'scale': scale}


def fit_gamma(data):
    """Ajusta distribucion Gamma."""
    # Gamma(a, scale) donde E[X] = a*scale
    a, loc, scale = stats.gamma.fit(data, floc=0)
    return {'a': a, 'loc': loc, 'scale': scale}


def fit_weibull(data):
    """Ajusta distribucion Weibull."""
    c, loc, scale = stats.weibull_min.fit(data, floc=0)
    return {'c': c, 'loc': loc, 'scale': scale}


def fit_lognormal(data):
    """Ajusta distribucion Lognormal."""
    s, loc, scale = stats.lognorm.fit(data, floc=0)
    return {'s': s, 'loc': loc, 'scale': scale}


def compute_goodness_of_fit(data, dist_name, params):
    """Calcula estadigrafos de bondad de ajuste."""
    
    # Crear distribucion
    if dist_name == 'Exponencial':
        dist = stats.expon(loc=params['loc'], scale=params['scale'])
    elif dist_name == 'Gamma':
        dist = stats.gamma(params['a'], loc=params['loc'], scale=params['scale'])
    elif dist_name == 'Weibull':
        dist = stats.weibull_min(params['c'], loc=params['loc'], scale=params['scale'])
    elif dist_name == 'Lognormal':
        dist = stats.lognorm(params['s'], loc=params['loc'], scale=params['scale'])
    else:
        return {}
    
    # Kolmogorov-Smirnov
    ks_stat, ks_pvalue = stats.kstest(data, dist.cdf)
    
    # Anderson-Darling (solo para algunas distribuciones)
    try:
        if dist_name == 'Exponencial':
            ad_result = stats.anderson(data, dist='expon')
            ad_stat = ad_result.statistic
        else:
            ad_stat = None
    except:
        ad_stat = None
    
    # Log-likelihood y AIC/BIC
    log_likelihood = np.sum(dist.logpdf(data))
    n_params = len(params)
    n_data = len(data)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_data) - 2 * log_likelihood
    
    # Chi-cuadrado (usando bins)
    n_bins = 50
    observed, bin_edges = np.histogram(data, bins=n_bins)
    expected = []
    for i in range(len(bin_edges)-1):
        prob = dist.cdf(bin_edges[i+1]) - dist.cdf(bin_edges[i])
        expected.append(prob * len(data))
    expected = np.array(expected)
    
    # Evitar divisiones por cero
    mask = expected > 5
    if mask.sum() > 0:
        chi2_stat = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        chi2_dof = mask.sum() - n_params - 1
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, chi2_dof) if chi2_dof > 0 else None
    else:
        chi2_stat = None
        chi2_pvalue = None
    
    return {
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'AD_statistic': ad_stat,
        'Chi2_statistic': chi2_stat,
        'Chi2_pvalue': chi2_pvalue,
        'Log_Likelihood': log_likelihood,
        'AIC': aic,
        'BIC': bic
    }


def compare_distributions():
    """Compara todas las distribuciones y genera reporte."""
    
    # Cargar datos
    riats = load_riats()
    
    # Distribuciones a probar
    distributions = {
        'Exponencial': fit_exponential,
        'Gamma': fit_gamma,
        'Weibull': fit_weibull,
        'Lognormal': fit_lognormal
    }
    
    results = []
    fitted_params = {}
    
    print("=== AJUSTANDO DISTRIBUCIONES ===\n")
    
    for dist_name, fit_func in distributions.items():
        print(f"Ajustando {dist_name}...")
        params = fit_func(riats)
        fitted_params[dist_name] = params
        
        gof = compute_goodness_of_fit(riats, dist_name, params)
        
        result = {
            'Distribucion': dist_name,
            **params,
            **gof
        }
        results.append(result)
    
    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    # Ordenar por AIC (menor es mejor)
    df_results = df_results.sort_values('AIC')
    
    print("\n=== RESULTADOS DE BONDAD DE AJUSTE ===\n")
    print(df_results.to_string(index=False))
    
    # Guardar resultados
    OUTPUT_DIR.mkdir(exist_ok=True)
    df_results.to_csv(OUTPUT_DIR / "arrival_distribution_comparison.csv", index=False)
    print(f"\nResultados guardados en {OUTPUT_DIR / 'arrival_distribution_comparison.csv'}")
    
    # Generar graficos
    plot_comparison(riats, fitted_params, df_results)
    
    # Recomendar mejor distribucion
    best_dist = df_results.iloc[0]['Distribucion']
    best_aic = df_results.iloc[0]['AIC']
    best_ks = df_results.iloc[0]['KS_statistic']
    
    print(f"\n=== RECOMENDACION ===")
    print(f"Mejor distribucion (segun AIC): {best_dist}")
    print(f"  AIC: {best_aic:.2f}")
    print(f"  KS statistic: {best_ks:.4f}")
    print(f"  KS p-value: {df_results.iloc[0]['KS_pvalue']:.4f}")
    
    print(f"\nParametros de {best_dist}:")
    for key, value in fitted_params[best_dist].items():
        print(f"  {key}: {value:.6f}")
    
    return df_results, fitted_params


def plot_comparison(data, fitted_params, df_results):
    """Genera graficos comparativos."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x_plot = np.linspace(0, np.percentile(data, 99), 1000)
    
    # Histograma de datos
    for ax in axes:
        ax.hist(data, bins=100, density=True, alpha=0.3, color='gray', label='Datos')
    
    # Plot cada distribucion
    colors = ['red', 'blue', 'green', 'orange']
    
    for idx, (dist_name, params) in enumerate(fitted_params.items()):
        ax = axes[idx]
        
        if dist_name == 'Exponencial':
            pdf = stats.expon.pdf(x_plot, loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Gamma':
            pdf = stats.gamma.pdf(x_plot, params['a'], loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Weibull':
            pdf = stats.weibull_min.pdf(x_plot, params['c'], loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Lognormal':
            pdf = stats.lognorm.pdf(x_plot, params['s'], loc=params['loc'], scale=params['scale'])
        
        ax.plot(x_plot, pdf, color=colors[idx], linewidth=2, label=f'{dist_name} (ajustada)')
        
        # Agregar estadigrafos
        row = df_results[df_results['Distribucion'] == dist_name].iloc[0]
        aic = row['AIC']
        ks = row['KS_statistic']
        
        ax.set_title(f'{dist_name}\nAIC={aic:.1f}, KS={ks:.4f}')
        ax.set_xlabel('RIAT')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arrival_distribution_comparison.png", dpi=200)
    print(f"Grafico guardado en {OUTPUT_DIR / 'arrival_distribution_comparison.png'}")
    
    # Q-Q plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (dist_name, params) in enumerate(fitted_params.items()):
        ax = axes[idx]
        
        if dist_name == 'Exponencial':
            dist = stats.expon(loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Gamma':
            dist = stats.gamma(params['a'], loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Weibull':
            dist = stats.weibull_min(params['c'], loc=params['loc'], scale=params['scale'])
        elif dist_name == 'Lognormal':
            dist = stats.lognorm(params['s'], loc=params['loc'], scale=params['scale'])
        
        # Q-Q plot
        sample_quantiles = np.percentile(data, np.linspace(0.1, 99.9, 100))
        theoretical_quantiles = dist.ppf(np.linspace(0.001, 0.999, 100))
        
        ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
        
        # Linea de referencia
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfecto ajuste')
        
        ax.set_title(f'Q-Q Plot: {dist_name}')
        ax.set_xlabel('Cuantiles Teoricos')
        ax.set_ylabel('Cuantiles Observados')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arrival_qq_plots.png", dpi=200)
    print(f"Q-Q plots guardados en {OUTPUT_DIR / 'arrival_qq_plots.png'}")


if __name__ == "__main__":
    df_results, fitted_params = compare_distributions()
