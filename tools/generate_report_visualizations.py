import os
import sys
import csv
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import pickle
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuración global
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "report_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DATA_CACHE = CACHE_DIR / "visualization_data.pkl"

# Mapeo de días a tipos de día (basado en engine.py)
DAY_TYPE_MAP = {
    1: "Tipo 1",
    2: "Tipo 1",
    3: "Tipo 2",
    4: "Tipo 1",
    5: "Tipo 2",
    6: "Tipo 2",
    7: "Tipo 3",
}

# Mapeo para cargar NPZ (de engine.py)
_DAYTYPE_MAP_NPZ = {
    "Tipo 1": "tipo_1",
    "Tipo 2": "tipo_2",
    "Tipo 3": "tipo_3",
}

def get_day_type(day_num):
    return DAY_TYPE_MAP.get(day_num, "Unknown")

def ensure_subdir(name):
    path = OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_items_summary():
    """Carga el resumen de distribuciones de ítems."""
    path = PROJECT_ROOT / "data" / "items_distribution_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error cargando items_distribution_summary.csv: {e}")
        return pd.DataFrame()

def get_item_fit_info(summary_df, profile, priority, payment, day_type):
    """Busca la información de ajuste para una combinación específica."""
    if summary_df.empty:
        return None
        
    # Intentar coincidencia exacta o 'ALL'
    # Prioridad: Exacta > ALL
    
    # Filtrar por profile (siempre exacto)
    subset = summary_df[summary_df["profile"] == profile]
    if subset.empty:
        return None
        
    # Filtrar por priority
    prio_match = subset[subset["priority"] == priority]
    if prio_match.empty:
        prio_match = subset[subset["priority"] == "ALL"]
    subset = prio_match
    
    # Filtrar por payment
    pay_match = subset[subset["payment_method"] == payment]
    if pay_match.empty:
        pay_match = subset[subset["payment_method"] == "ALL"]
    subset = pay_match
    
    # Filtrar por day_type
    day_match = subset[subset["day_type"] == day_type]
    if day_match.empty:
        day_match = subset[subset["day_type"] == "ALL"]
    subset = day_match
    
    if not subset.empty:
        return subset.iloc[0]
    return None

def parse_array_string(s):
    """Parsea strings de listas como '[1, 2, 3]'."""
    try:
        return ast.literal_eval(s)
    except:
        return []

class MetricCollector:
    def __init__(self):
        # Key: (profile, day_type)
        self.arrivals = defaultdict(list) 
        self.arrival_times = defaultdict(list)
        
        # Key: (profile, priority, payment_method, day_type)
        self.items = defaultdict(list)
        self.patience = defaultdict(list)
        self.profit = defaultdict(list) # Stores (items, profit) tuples
        
        # Key: (lane_type, profile)
        self.service = defaultdict(list) # Stores (items, service_time) tuples

def process_raw_data(root_dir, sample_days=364):
    """Procesa los archivos CSV raw y extrae métricas agregadas."""
    root = Path(root_dir)
    collector = MetricCollector()
    
    day_dirs = sorted(list(root.glob("Week-*-Day-*")))
    if not day_dirs:
        print("No se encontraron directorios de datos.")
        return collector

    print(f"Procesando {min(len(day_dirs), sample_days)} días de datos...")
    
    count = 0
    for day_dir in tqdm(day_dirs):
        if count >= sample_days:
            break
            
        try:
            day_num = int(day_dir.name.split("-")[-1])
            day_type = get_day_type(day_num)
        except:
            day_type = "Unknown"

        csv_path = day_dir / "customers.csv"
        if not csv_path.exists():
            continue
            
        try:
            # Leer solo columnas necesarias para ahorrar memoria
            cols = [
                "profile", "arrival_time_s", "items", 
                "patience_s", "service_time_s", "total_profit_clp",
                "priority", "payment_method", "lane_type"
            ]
            # Verificar headers primero
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
                header = [h.strip().lower() for h in header]
            
            use_cols = [c for c in cols if c in header]
            
            df = pd.read_csv(csv_path, usecols=use_cols)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Limpieza básica
            df = df.dropna(subset=["arrival_time_s", "items"])
            for c in ["arrival_time_s", "items", "patience_s", "service_time_s", "total_profit_clp"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            
            # Procesar Arrivals (Profile + DayType)
            # Necesitamos calcular IATs por día para no mezclar días
            for profile, group in df.groupby("profile"):
                group = group.sort_values("arrival_time_s")
                times = group["arrival_time_s"].values
                if len(times) > 1:
                    iats = np.diff(times)
                    iats = iats[(iats > 0) & (iats < 3600)]
                    collector.arrivals[(profile, day_type)].extend(iats)
                
                collector.arrival_times[(profile, day_type)].extend(times)

            # Procesar Items, Patience, Profit (Full Key)
            # Agrupar para eficiencia
            group_cols = ["profile", "priority", "payment_method"]
            # Asegurar que existan columnas
            valid_group_cols = [c for c in group_cols if c in df.columns]
            
            if len(valid_group_cols) == 3:
                for (prof, prio, pay), group in df.groupby(valid_group_cols):
                    key = (prof, prio, pay, day_type)
                    
                    if "items" in group.columns:
                        collector.items[key].extend(group["items"].dropna().values)
                    
                    if "patience_s" in group.columns:
                        pats = group["patience_s"].dropna().values
                        pats = pats[pats > 0]
                        collector.patience[key].extend(pats)
                        
                    if "total_profit_clp" in group.columns and "items" in group.columns:
                        # Guardar tuplas (items, profit)
                        sub = group[["items", "total_profit_clp"]].dropna()
                        if not sub.empty:
                            collector.profit[key].extend(list(zip(sub["items"].values, sub["total_profit_clp"].values)))

            # Procesar Service Time (LaneType + Profile)
            if "lane_type" in df.columns and "service_time_s" in df.columns:
                for (lane, prof), group in df.groupby(["lane_type", "profile"]):
                    sub = group[["items", "service_time_s"]].dropna()
                    sub = sub[sub["service_time_s"] > 0]
                    if not sub.empty:
                        collector.service[(lane, prof)].extend(list(zip(sub["items"].values, sub["service_time_s"].values)))

            count += 1
        except Exception as e:
            # print(f"Error en {day_dir}: {e}")
            continue
            
    return collector

def plot_inter_arrival_times(iats, profile, day_type):
    """Genera histograma de tiempos entre llegadas con comparación estadística de KDE vs Paramétrica."""
    if len(iats) < 50:
        return

    iats = np.array(iats)
    
    plt.figure()
    # Histograma de datos observados
    sns.histplot(iats, stat="density", bins='auto', color="skyblue", alpha=0.4, label="Datos Observados")
    
    x = np.linspace(0, np.percentile(iats, 99.5), 1000)
    
    # --- 1. Ajuste Paramétrico ---
    distributions = [
        ("Exponencial", stats.expon),
        ("Lognormal", stats.lognorm),
        ("Weibull", stats.weibull_min),
        ("Gamma", stats.gamma)
    ]
    
    best_param_name = None
    best_param_ks = 1.0
    best_param_p = 0.0
    best_param_pdf = None
    best_param_loglik = -np.inf
    
    for name, dist in distributions:
        try:
            if name == "Exponencial":
                params = dist.fit(iats, floc=0)
            else:
                params = dist.fit(iats, floc=0)
                
            # KS Test
            ks_stat, p_val = stats.kstest(iats, dist.name, args=params)
            
            # Log-Likelihood (Sum of logpdf)
            loglik = np.sum(dist.logpdf(iats, *params))
            
            if ks_stat < best_param_ks:
                best_param_ks = ks_stat
                best_param_p = p_val
                best_param_name = name
                best_param_pdf = dist.pdf(x, *params)
                best_param_loglik = loglik
        except:
            continue

    # --- 2. Ajuste KDE (Búsqueda de Kernel) ---
    best_kde_pdf = None
    best_kde_label = "KDE"
    best_kde_loglik = -np.inf
    kde_success = False
    
    try:
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV
        
        # Subsample para KDE grid search si es muy grande para velocidad
        sample_for_grid = iats
        if len(iats) > 2000:
            sample_for_grid = np.random.choice(iats, 2000, replace=False)

        bw_scott = np.std(sample_for_grid) * len(sample_for_grid)**(-0.2)
        bandwidths = np.linspace(bw_scott * 0.5, bw_scott * 2, 5) # Reducido a 5 para velocidad
        
        grid = GridSearchCV(
            KernelDensity(),
            {'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential'], 'bandwidth': bandwidths},
            cv=3
        )
        grid.fit(sample_for_grid.reshape(-1, 1))
        
        kde_model = grid.best_estimator_
        # Recalcular score en full dataset (o subsample grande)
        best_kde_loglik = kde_model.score(sample_for_grid.reshape(-1, 1)) * (len(iats)/len(sample_for_grid))
        
        log_dens = kde_model.score_samples(x.reshape(-1, 1))
        best_kde_pdf = np.exp(log_dens)
        best_kde_label = f"KDE ({kde_model.kernel}, bw={kde_model.bandwidth:.2f})"
        kde_success = True
        
    except ImportError:
        try:
            kde = stats.gaussian_kde(iats)
            best_kde_pdf = kde(x)
            best_kde_label = "KDE (Gaussian - Scipy)"
            best_kde_loglik = np.sum(np.log(kde(iats)))
            kde_success = True
        except:
            pass
    except Exception as e:
        print(f"Error en KDE: {e}")

    # --- 3. Plotting y Decisión ---
    if best_param_name:
        rejected = best_param_p < 0.05
        style = 'r--' if rejected else 'r-'
        label_txt = f"Paramétrica: {best_param_name}\n(p={best_param_p:.2e})"
        plt.plot(x, best_param_pdf, style, lw=2, label=label_txt)
    
    if kde_success:
        plt.plot(x, best_kde_pdf, 'k-.', lw=2.5, label=f"{best_kde_label}")
    
    use_kde = False
    reason = ""
    
    if best_param_name is None:
        use_kde = True
        reason = "No fit paramétrico"
    elif best_param_p < 0.05:
        use_kde = True
        reason = "Paramétrica Rechazada (p<0.05)"
    else:
        use_kde = False
        reason = "Ajuste Paramétrico Aceptable"

    decision_text = f"Recomendación: {'KDE' if use_kde else best_param_name}"
    decision_sub = f"({reason})"
    
    color_box = "black" if use_kde else "darkgreen"
    
    plt.text(0.95, 0.95, f"{decision_text}\n{decision_sub}", transform=plt.gca().transAxes, 
             ha='right', va='top', fontsize=10, fontweight='bold', color=color_box,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    plt.title(f"Inter-Arrival Times Analysis\nProfile: {profile} | Day: {day_type}")
    plt.xlabel("Tiempo entre llegadas (s)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.xlim(0, np.percentile(iats, 99.5))
    
    out_dir = ensure_subdir("arrivals/inter_arrival")
    filename = out_dir / f"arrival_{profile}_{day_type.replace(' ', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_arrival_time_distribution(times, profile, day_type):
    """Genera histograma de la hora de llegada (Observed Arrivals)."""
    if len(times) < 50:
        return

    times = np.array(times)
    # Asumiendo que la simulación parte a las 8:00 AM (0s = 8:00)
    arrival_hours = times / 3600.0 + 8.0
    
    plt.figure()
    sns.histplot(arrival_hours, bins=28, stat="count", color="orange", alpha=0.6, label="Llegadas Observadas")
    
    plt.title(f"Distribución Horaria de Llegadas\nProfile: {profile} | Day: {day_type}")
    plt.xlabel("Hora del Día (8:00 - 22:00)")
    plt.ylabel("Cantidad de Clientes")
    plt.xlim(8, 22)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_dir = ensure_subdir("arrivals/observed_distribution")
    filename = out_dir / f"arrival_time_{profile}_{day_type.replace(' ', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_lambda_process(profile, priority, payment, day_type):
    """Plotea la tasa de llegada lambda(t) desde los archivos NPZ."""
    npz_path = PROJECT_ROOT / "data" / "arrivals_npz" / f"lambda_{profile}.npz"
    if not npz_path.exists():
        return

    try:
        data = np.load(npz_path, allow_pickle=False)
        keys = data["keys"]
        lambdas = data["lambdas"]
        bin_left = data["bin_left_s"]
        
        dt_key = _DAYTYPE_MAP_NPZ.get(day_type, day_type.lower().replace(" ", "_"))
        target_key = f"{dt_key}|{priority}|{payment}"
        
        idx = -1
        for i, k in enumerate(keys):
            if str(k) == target_key:
                idx = i
                break
        
        if idx == -1:
            return

        lambda_series = lambdas[idx]
        
        plt.figure()
        hours = bin_left / 3600.0 + 8.0 
        
        plt.step(hours, lambda_series, where='post', color='purple', lw=2)
        plt.fill_between(hours, lambda_series, step="post", alpha=0.2, color='purple')
        
        plt.title(f"Proceso de Llegada (Lambda)\n{profile} | {priority} | {payment} | {day_type}")
        plt.xlabel("Hora del Día")
        plt.ylabel("Clientes / Hora (Lambda)")
        plt.xlim(8, 22) 
        plt.grid(True, alpha=0.3)
        
        out_dir = ensure_subdir("arrivals/lambda_process")
        filename = out_dir / f"lambda_{profile}_{priority}_{payment}_{day_type.replace(' ', '')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error ploteando lambda para {profile}: {e}")

def plot_items_distribution(items, summary_df, profile, priority, payment, day_type):
    """Genera histograma de ítems."""
    if len(items) < 20:
        return
    
    items = np.array(items)

    plt.figure()
    bins = np.arange(0, items.max() + 2) - 0.5
    plt.hist(items, bins=bins, density=True, color="lightgreen", alpha=0.6, label="Datos Observados", rwidth=0.8)
    
    fit_info = get_item_fit_info(summary_df, profile, priority, payment, day_type)
    
    if fit_info is not None:
        fit_type = fit_info.get("fit_type", "parametric")
        
        if fit_type == "kde":
            try:
                support = parse_array_string(fit_info["kde_support"])
                probs = parse_array_string(fit_info["kde_probs"])
                if support and probs:
                    plt.plot(support, probs, 'm-o', lw=2, ms=4, label='Ajuste KDE (Simulación)')
            except Exception as e:
                print(f"Error ploteando KDE para {profile}: {e}")
                
        elif fit_type == "parametric":
            dist_name = fit_info.get("fit_distribution", "unknown")
            x = np.arange(1, items.max() + 1)
            mu = np.mean(items)
            
            if dist_name == "poisson":
                pmf = stats.poisson.pmf(x, mu)
                plt.plot(x, pmf, 'b--o', lw=1, ms=4, label=f'Poisson ($\\lambda={mu:.1f}$)')
            elif dist_name == "nbinom":
                var = np.var(items)
                if var > mu:
                    p = mu / var
                    n = (mu * p) / (1 - p)
                    pmf = stats.nbinom.pmf(x, n, p)
                    plt.plot(x, pmf, 'r-s', lw=1, ms=4, label=f'Neg. Binomial\n($n={n:.1f}, p={p:.2f}$)')
            else:
                pmf = stats.poisson.pmf(x, mu)
                plt.plot(x, pmf, 'b--o', lw=1, ms=4, label=f'Poisson ($\\lambda={mu:.1f}$)')

    plt.title(f"Items Distribution\n{profile} | {priority} | {payment} | {day_type}")
    plt.xlabel("Cantidad de Ítems")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.xlim(0, np.percentile(items, 99.5))
    
    out_dir = ensure_subdir("items")
    filename = out_dir / f"items_{profile}_{priority}_{payment}_{day_type.replace(' ', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_patience_distribution(patience, profile, priority, payment, day_type):
    """Genera histograma de paciencia."""
    if len(patience) < 20:
        return
    
    patience = np.array(patience)

    plt.figure()
    sns.histplot(patience, stat="density", bins=30, color="salmon", alpha=0.6, label="Datos Observados")
    
    x = np.linspace(0, np.percentile(patience, 99), 1000)
    
    # Weibull
    try:
        params_w = stats.weibull_min.fit(patience, floc=0)
        pdf_w = stats.weibull_min.pdf(x, *params_w)
        plt.plot(x, pdf_w, 'b-', lw=2, label=f'Weibull ($k={params_w[0]:.2f}$)')
    except:
        pass
    
    # Lognormal
    try:
        params_l = stats.lognorm.fit(patience, floc=0)
        pdf_l = stats.lognorm.pdf(x, *params_l)
        plt.plot(x, pdf_l, 'g--', lw=2, label=f'Lognormal ($\\sigma={params_l[0]:.2f}$)')
    except:
        pass
    
    plt.title(f"Patience Distribution\n{profile} | {priority} | {payment} | {day_type}")
    plt.xlabel("Tiempo de Paciencia (s)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.xlim(0, np.percentile(patience, 99))
    
    out_dir = ensure_subdir("patience")
    filename = out_dir / f"patience_{profile}_{priority}_{payment}_{day_type.replace(' ', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_profit_regression(data_tuples, profile, priority, payment, day_type):
    """Genera scatter plot de profit."""
    if len(data_tuples) < 20:
        return

    # Subsample si es muy grande
    if len(data_tuples) > 500:
        indices = np.random.choice(len(data_tuples), 500, replace=False)
        sample = [data_tuples[i] for i in indices]
    else:
        sample = data_tuples
        
    x = np.array([d[0] for d in sample])
    y = np.array([d[1] for d in sample])
    
    plt.figure()
    plt.scatter(x, y, alpha=0.3, s=15, c='green', label='Datos Observados')
    
    if np.sum(x * x) > 0:
        slope = np.sum(x * y) / np.sum(x * x)
    else:
        slope = 0
    
    x_line = np.linspace(0, x.max(), 100)
    y_line = slope * x_line
    plt.plot(x_line, y_line, 'k--', lw=2, label=f'Regresión (y={slope:.1f}x)')
    
    df_sample = pd.DataFrame({'x': x, 'y': y})
    corr = df_sample['x'].corr(df_sample['y'])
    
    plt.title(f"Profit Model (Corr: {corr:.2f})\n{profile} | {priority} | {payment} | {day_type}")
    plt.xlabel("Cantidad de Ítems")
    plt.ylabel("Profit Total (CLP)")
    plt.legend()
    
    upper_lim = np.nanpercentile(y, 99)
    if np.isfinite(upper_lim):
        plt.ylim(0, upper_lim)
    
    out_dir = ensure_subdir("profit")
    filename = out_dir / f"profit_{profile}_{priority}_{payment}_{day_type.replace(' ', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_service_time_regression(data_tuples, lane_type, profile):
    """Genera scatter plot de servicio por lane_type y profile, con modelo estocástico para SCO."""
    if len(data_tuples) < 20:
        return

    if len(data_tuples) > 500:
        indices = np.random.choice(len(data_tuples), 500, replace=False)
        sample = [data_tuples[i] for i in indices]
    else:
        sample = data_tuples
        
    x_obs = np.array([d[0] for d in sample])
    y_obs = np.array([d[1] for d in sample])
    
    plt.figure()
    plt.scatter(x_obs, y_obs, alpha=0.3, s=15, c='gray', label='Datos Observados')
    
    is_sco = str(lane_type).lower() in ["self_checkout", "sco", "autocaja"]
    
    if is_sco:
        x_sim = np.linspace(1, max(x_obs.max(), 20), 50)
        p05, p50, p95 = [], [], []
        rng = np.random.default_rng(42)
        
        for items in x_sim:
            n_sim = 1000
            # Setup: max(5.0, Normal(15.0, 5.0))
            setup = np.maximum(5.0, rng.normal(15.0, 5.0, n_sim))
            # Rate: Lognormal(mean=1.5, sigma=0.6)
            rate = rng.lognormal(1.5, 0.6, n_sim)
            rate = np.clip(rate, 0.5, 60.0)
            times = setup + items * rate
            
            p05.append(np.percentile(times, 5))
            p50.append(np.median(times))
            p95.append(np.percentile(times, 95))
            
        plt.plot(x_sim, p50, 'r-', lw=2, label='Mediana Teórica (Estocástica)')
        plt.fill_between(x_sim, p05, p95, color='red', alpha=0.15, label='Rango Esperado (5%-95%)')
        title_suffix = "(Modelo Estocástico: Setup~N(15,5), Rate~LogN(1.5, 0.6))"
    else:
        df_sample = pd.DataFrame({'items': x_obs, 'service_time_s': y_obs})
        sns.regplot(
            data=df_sample, 
            x="items", 
            y="service_time_s", 
            scatter=False,
            line_kws={'color':'red', 'label': 'Regresión Lineal'}
        )
        title_suffix = f"(Corr: {df_sample['items'].corr(df_sample['service_time_s']):.2f})"
    
    plt.title(f"Service Time Model {title_suffix}\nLane: {lane_type} | Profile: {profile}")
    plt.xlabel("Cantidad de Ítems")
    plt.ylabel("Tiempo de Servicio (s)")
    plt.legend()
    
    upper_lim = np.nanpercentile(y_obs, 99)
    if np.isfinite(upper_lim):
        plt.ylim(0, upper_lim * 1.2)
    
    out_dir = ensure_subdir("service_time")
    filename = out_dir / f"service_{lane_type}_{profile}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    print("=== GENERADOR DE VISUALIZACIONES OPTIMIZADO ===")
    
    data_root = PROJECT_ROOT / "outputs_teoricos"
    if not data_root.exists():
        print(f"Error: No se encuentra {data_root}")
        return
        
    items_summary = load_items_summary()
    
    # Cargar o Procesar Datos
    collector = None
    if VISUALIZATION_DATA_CACHE.exists():
        print(f"Cargando datos procesados desde {VISUALIZATION_DATA_CACHE}...")
        try:
            with open(VISUALIZATION_DATA_CACHE, 'rb') as f:
                collector = pickle.load(f)
            print("Datos cargados correctamente.")
        except Exception as e:
            print(f"Error cargando cache: {e}. Se reprocesarán los datos.")
            
    if collector is None:
        collector = process_raw_data(data_root, sample_days=364)
        print(f"Guardando datos procesados en {VISUALIZATION_DATA_CACHE}...")
        try:
            with open(VISUALIZATION_DATA_CACHE, 'wb') as f:
                pickle.dump(collector, f)
        except Exception as e:
            print(f"No se pudo guardar cache: {e}")

    # Generar Plots
    print("\nGenerando Visualizaciones...")
    
    # 1. Arrivals
    print("-> Arrivals...")
    for (profile, day_type), iats in tqdm(collector.arrivals.items()):
        plot_inter_arrival_times(iats, profile, day_type)
        if (profile, day_type) in collector.arrival_times:
            plot_arrival_time_distribution(collector.arrival_times[(profile, day_type)], profile, day_type)

    # 2. Items, Patience, Profit, Lambda
    print("-> Items, Patience, Profit, Lambda...")
    # Obtener todas las claves únicas
    all_keys = set(collector.items.keys()) | set(collector.patience.keys()) | set(collector.profit.keys())
    
    for key in tqdm(list(all_keys)):
        profile, priority, payment, day_type = key
        
        if key in collector.items:
            plot_items_distribution(collector.items[key], items_summary, profile, priority, payment, day_type)
        
        if key in collector.patience:
            plot_patience_distribution(collector.patience[key], profile, priority, payment, day_type)
            
        if key in collector.profit:
            plot_profit_regression(collector.profit[key], profile, priority, payment, day_type)
            
        plot_lambda_process(profile, priority, payment, day_type)

    # 3. Service Time
    print("-> Service Time...")
    for (lane_type, profile), data in tqdm(collector.service.items()):
        plot_service_time_regression(data, lane_type, profile)
    
    print(f"\n¡Listo! Visualizaciones guardadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
