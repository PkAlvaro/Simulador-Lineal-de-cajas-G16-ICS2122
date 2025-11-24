
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def analyze_attributes(root_dir, sample_days=364):
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: No se encuentra {root}")
        return

    print(f"Analizando atributos en: {root}")
    
    all_data = []
    days_processed = 0
    
    # 1. Recolección de datos
    for day_dir in root.glob("Week-*-Day-*"):
        if days_processed >= sample_days:
            break
        csv_path = day_dir / "customers.csv"
        if not csv_path.exists():
            continue
            
        try:
            df = pd.read_csv(csv_path)
            cols = ["profile", "items", "patience_s"]
            if not all(c in df.columns for c in cols):
                continue
                
            # Limpieza básica
            df["profile"] = df["profile"].astype(str).str.strip().str.lower()
            df["items"] = pd.to_numeric(df["items"], errors="coerce")
            df["patience_s"] = pd.to_numeric(df["patience_s"], errors="coerce")
            df = df.dropna(subset=["items", "patience_s"])
            
            all_data.append(df[cols])
            days_processed += 1
            print(f"Procesado: {day_dir.name}", end="\r")
            
        except Exception:
            continue

    if not all_data:
        print("\nNo se encontraron datos válidos.")
        return

    full_df = pd.concat(all_data)
    profiles = full_df["profile"].unique()
    
    print("\n" + "="*80)
    print(f"ANÁLISIS DE ATRIBUTOS DE CLIENTES (N={len(full_df)} registros)")
    print("="*80)

    # 2. Análisis por Perfil
    for profile in profiles:
        subset = full_df[full_df["profile"] == profile]
        n = len(subset)
        if n < 50:
            continue
            
        print(f"\n>>> PERFIL: {profile.upper()} (N={n})")
        
        # --- ANÁLISIS DE ÍTEMS (Discreto) ---
        items = subset["items"].values
        mean_i = np.mean(items)
        var_i = np.var(items)
        dispersion = var_i / mean_i if mean_i > 0 else 0
        
        print(f"  [ÍTEMS] Media: {mean_i:.2f} | Varianza: {var_i:.2f} | Dispersión: {dispersion:.2f}")
        
        # Ajuste Poisson
        # KS test no es ideal para discreto, pero usaremos una aproximación o Chi2 simple
        # Aquí comparamos log-likelihood o simplemente error cuadrático de frecuencias para simplificar la visualización
        
        print(f"    {'Distribución':<15} | {'Parámetros Estimados'}")
        print(f"    {'-'*50}")
        
        # Poisson: lambda = mean
        print(f"    {'Poisson':<15} | lambda={mean_i:.2f}")
        
        # Binomial Negativa: r, p
        # Mean = r(1-p)/p, Var = r(1-p)/p^2
        if var_i > mean_i:
            p_est = mean_i / var_i
            r_est = (mean_i * p_est) / (1 - p_est)
            print(f"    {'Neg. Binomial':<15} | n={r_est:.2f}, p={p_est:.2f}")
        else:
            print(f"    {'Neg. Binomial':<15} | (No aplica, Varianza < Media)")

        # --- ANÁLISIS DE PACIENCIA (Continuo) ---
        patience = subset["patience_s"].values
        patience = patience[patience > 0] # Filtrar ceros para lognormal/weibull
        
        print(f"\n  [PACIENCIA] Media: {np.mean(patience):.2f} s | Std: {np.std(patience):.2f} s")
        print(f"    {'Distribución':<15} | {'KS Stat':<8} | {'Parámetros'}")
        print(f"    {'-'*60}")
        
        dists = [
            ("Exponencial", stats.expon),
            ("Lognormal", stats.lognorm),
            ("Weibull", stats.weibull_min),
            ("Normal", stats.norm)
        ]
        
        best_dist = None
        best_ks = 1.0
        
        for name, dist in dists:
            try:
                params = dist.fit(patience)
                ks, p = stats.kstest(patience, name.lower().replace("exponencial","expon").replace("normal","norm"), args=params)
                param_str = ", ".join([f"{p:.2f}" for p in params])
                print(f"    {name:<15} | {ks:.4f}   | {param_str}")
                
                if ks < best_ks:
                    best_ks = ks
                    best_dist = name
            except:
                pass
        
        print(f"    >> MEJOR AJUSTE PACIENCIA: {best_dist.upper()}")
        print("-" * 80)

if __name__ == "__main__":
    root = Path("outputs_teoricos")
    if not root.exists():
        root = Path(r"c:\Users\alvar\OneDrive\Escritorio\niniefinal\SIMULADOR FINAL P2\outputs_teoricos")
    analyze_attributes(root)
