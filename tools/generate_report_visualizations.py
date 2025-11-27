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

# Cargar modelo de tiempos de servicio para visualizacion
SERVICE_MODEL_PATH = PROJECT_ROOT / "data/service_time/service_time_model.json"
SERVICE_MODEL_DATA = {}
if SERVICE_MODEL_PATH.exists():
    try:
        with open(SERVICE_MODEL_PATH, encoding="utf-8") as f:
            SERVICE_MODEL_DATA = json.load(f)
    except Exception as e:
        print(f"Error cargando modelo de servicio: {e}")

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
    """Genera histograma de tiempos entre llegadas con ajuste Gamma."""
    if len(iats) < 50:
        return

    iats = np.array(iats)
    
    plt.figure()
    # Histograma de datos observados
    sns.histplot(iats, stat="density", bins='auto', color="skyblue", alpha=0.5, label="Datos Observados")
    
    x = np.linspace(0, np.percentile(iats, 99.5), 1000)
    
    # Ajuste Gamma
    try:
        # Fit Gamma distribution
        params = stats.gamma.fit(iats, floc=0)
        shape, loc, scale = params
        
        # KS Test
        ks_stat, p_val = stats.kstest(iats, 'gamma', args=params)
        
        # Plot Gamma fit
        gamma_pdf = stats.gamma.pdf(x, *params)
        plt.plot(x, gamma_pdf, 'r-', lw=2.5, label=f'Gamma (shape={shape:.2f}, scale={scale:.2f})')
        
        # Add statistics text box
        stats_text = f"KS Test: p={p_val:.3f}\n"
        stats_text += f"Media: {np.mean(iats):.1f}s\n"
        stats_text += f"Mediana: {np.median(iats):.1f}s"
        
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 ha='right', va='top', fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
        
    except Exception as e:
        print(f"Error ajustando Gamma para {profile} - {day_type}: {e}")
    
    plt.title(f"Inter-Arrival Times (Gamma Distribution)\nProfile: {profile} | Day: {day_type}")
    plt.xlabel("Tiempo entre llegadas (s)")
    plt.ylabel("Densidad")
    plt.legend(loc='upper right')
    plt.xlim(0, np.percentile(iats, 99.5))
    plt.grid(True, alpha=0.3)
    
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
    """Genera scatter plot de servicio por lane_type y profile, comparando con el modelo configurado."""
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
    
    # Normalizar nombre de lane
    norm_lane = str(lane_type).lower().strip()
    if norm_lane in ["sco", "autocaja"]: norm_lane = "self_checkout"
    
    model_type = SERVICE_MODEL_DATA.get("type", "legacy")
    
    # Variables de estado
    is_stochastic = False
    is_linear = False
    stoch_params = {}
    lin_params = {}
    
    # 1. Detectar tipo de modelo para este lane
    if model_type == "hybrid_multivariate_stochastic":
        lane_model = SERVICE_MODEL_DATA.get("models", {}).get(norm_lane)
        if lane_model:
            method = lane_model.get("model_method")
            if method == "stochastic_rate":
                is_stochastic = True
                stoch_params = lane_model.get("params", {})
                title_suffix = "(Modelo Estocástico Ajustado)"
            else:
                is_linear = True
                lin_params = lane_model
                title_suffix = f"(Modelo Híbrido R2={lane_model.get('stats', {}).get('r2', 0):.2f})"
                
    elif model_type == "segmented_multivariate":
        lane_model = SERVICE_MODEL_DATA.get("models", {}).get(norm_lane)
        if lane_model:
            is_linear = True
            lin_params = lane_model
            title_suffix = f"(Modelo Multivariado R2={lane_model.get('stats', {}).get('r2', 0):.2f})"
            
    elif model_type == "segmented_by_lane":
        lane_model = SERVICE_MODEL_DATA.get("models", {}).get(norm_lane)
        if lane_model:
            is_linear = True
            # Adaptar formato simple a generico
            coeffs = lane_model.get("coeffs", {})
            lin_params = {
                "coeffs": {"Intercept": coeffs.get("intercept"), "items": coeffs.get("slope")}
            }
            title_suffix = f"(Modelo Ajustado R2={lane_model.get('stats', {}).get('r2', 0):.2f})"

    # Fallback legacy para SCO si no hay modelo especifico
    if not is_stochastic and not is_linear and norm_lane == "self_checkout":
        is_stochastic = True
        title_suffix = "(Modelo Estocástico Default)"
        # Params default
        stoch_params = {
            "setup_time": 15.0,
            "rate_dist": {"mu": 1.5, "sigma": 0.6}
        }

    # 2. Graficar según tipo
    if is_linear:
        coeffs = lin_params.get("coeffs", {})
        intercept = float(coeffs.get("Intercept", 0.0))
        slope = float(coeffs.get("items", 0.0))
        
        x_range = np.linspace(max(0, x_obs.min()), x_obs.max(), 100)
        y_model = intercept + slope * x_range
        
        label_txt = f'Modelo Base (Int={intercept:.1f}, Slope={slope:.2f})'
        if "multivariate" in model_type:
            label_txt += " [+Efectos]"
            
        plt.plot(x_range, y_model, 'r-', lw=2.5, label=label_txt)
        
    elif is_stochastic:
        # Extraer parametros
        setup = float(stoch_params.get("setup_time", 15.0))
        dist_params = stoch_params.get("rate_dist", {})
        mu = float(dist_params.get("mu", 1.5))
        sigma = float(dist_params.get("sigma", 0.6))
        
        x_sim = np.linspace(1, max(x_obs.max(), 20), 50)
        p05, p50, p95 = [], [], []
        rng = np.random.default_rng(42)
        
        for items in x_sim:
            n_sim = 1000
            # Setup con ruido para visualizacion realista
            setup_val = np.maximum(5.0, rng.normal(setup, 5.0, n_sim))
            
            rate = rng.lognormal(mu, sigma, n_sim)
            rate = np.clip(rate, 0.1, 120.0)
            times = setup_val + items * rate
            
            p05.append(np.percentile(times, 5))
            p50.append(np.median(times))
            p95.append(np.percentile(times, 95))
            
        plt.plot(x_sim, p50, 'r-', lw=2, label=f'Mediana (Setup={setup:.0f}, Rate~LogN)')
        plt.fill_between(x_sim, p05, p95, color='red', alpha=0.15, label='Rango 5%-95%')

    else:
        # Fallback total
        df_sample = pd.DataFrame({'items': x_obs, 'service_time_s': y_obs})
        if len(df_sample) > 1 and df_sample['items'].std() > 0:
             sns.regplot(
                data=df_sample, 
                x="items", 
                y="service_time_s", 
                scatter=False,
                line_kws={'color':'red', 'label': 'Regresión Lineal (Ad-hoc)'}
            )
             title_suffix = f"(Corr: {df_sample['items'].corr(df_sample['service_time_s']):.2f})"
        else:
             title_suffix = "(Datos Insuficientes)"
    
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
