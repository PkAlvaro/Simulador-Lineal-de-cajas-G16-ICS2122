
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def analyze_arrivals(root_dir, sample_days=364):
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: No se encuentra el directorio {root}")
        return

    print(f"Analizando datos en: {root}")
    
    # Collect all customer data
    all_arrivals = []
    
    days_processed = 0
    for day_dir in root.glob("Week-*-Day-*"):
        if days_processed >= sample_days:
            break
            
        csv_path = day_dir / "customers.csv"
        if not csv_path.exists():
            continue
            
        try:
            df = pd.read_csv(csv_path)
            if "arrival_time_s" not in df.columns or "profile" not in df.columns:
                continue
                
            # Normalize profile names
            df["profile"] = df["profile"].astype(str).str.strip().str.lower()
            df["arrival_time_s"] = pd.to_numeric(df["arrival_time_s"], errors="coerce")
            df = df.dropna(subset=["arrival_time_s"])
            
            all_arrivals.append(df[["profile", "arrival_time_s"]])
            days_processed += 1
            print(f"Procesado: {day_dir.name}")
            
        except Exception as e:
            print(f"Error leyendo {csv_path}: {e}")

    if not all_arrivals:
        print("No se encontraron datos validos.")
        return

    full_df = pd.concat(all_arrivals)
    
    print("\n" + "="*60)
    print("RESULTADOS DEL ANALISIS DE DISTRIBUCION DE LLEGADAS")
    print("="*60)
    
    profiles = full_df["profile"].unique()
    
    distributions = [
        ("Exponencial", stats.expon),
        ("Lognormal", stats.lognorm),
        ("Weibull", stats.weibull_min),
        ("Gamma", stats.gamma),
        ("Normal", stats.norm)
    ]

    for profile in profiles:
        subset = full_df[full_df["profile"] == profile].sort_values("arrival_time_s")
        
        # Calculate Inter-Arrival Times (IAT)
        # We need to do this per day to avoid huge gaps between days, 
        # but for simplicity in this aggregate view, let's just take the diff
        # and filter out huge gaps (e.g. > 3600s which likely imply day change if we concatenated blindly)
        # BETTER APPROACH: Calculate IAT per day chunk and concatenate IATs
        
        iats = []
        for day_df in all_arrivals:
            p_df = day_df[day_df["profile"] == profile].sort_values("arrival_time_s")
            if len(p_df) < 2:
                continue
            diffs = np.diff(p_df["arrival_time_s"])
            # Filter out zero or negative diffs just in case
            diffs = diffs[diffs > 0]
            iats.extend(diffs)
            
        if len(iats) < 50:
            print(f"\nPerfil: {profile} (Insuficientes datos: {len(iats)})")
            continue
            
        data = np.array(iats)
        
        print(f"\nPerfil: {profile} (N={len(data)})")
        print(f"  Media IAT: {np.mean(data):.4f} s")
        print(f"  Std IAT:   {np.std(data):.4f} s")
        print("-" * 40)
        print(f"  {'Distribucion':<15} | {'KS Stat':<10} | {'P-Value':<10} | {'Parametros'}")
        print("-" * 40)
        
        best_dist = None
        best_p = -1
        best_name = ""
        
        for name, dist in distributions:
            try:
                # Fit distribution
                params = dist.fit(data)
                
                # KS Test
                ks_stat, p_value = stats.kstest(data, name.lower().replace("exponencial", "expon").replace("normal", "norm"), args=params)
                
                # Format params for display
                param_str = ", ".join([f"{p:.2f}" for p in params])
                
                print(f"  {name:<15} | {ks_stat:.4f}     | {p_value:.2e}   | {param_str}")
                
                if p_value > best_p:
                    best_p = p_value
                    best_dist = dist
                    best_name = name
            except Exception:
                pass
        
        print("-" * 40)
        print(f"  >> MEJOR AJUSTE: {best_name.upper()}")

if __name__ == "__main__":
    # Adjust path as needed
    root = Path("outputs_teoricos")
    if not root.exists():
        # Try absolute path if running from different cwd
        root = Path(r"c:\Users\alvar\OneDrive\Escritorio\niniefinal\SIMULADOR FINAL P2\outputs_teoricos")
    
    analyze_arrivals(root)
