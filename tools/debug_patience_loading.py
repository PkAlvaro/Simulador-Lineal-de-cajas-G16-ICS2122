"""
Debug script para verificar que datos teoricos se estan cargando correctamente
"""
import pandas as pd
from pathlib import Path

root = Path("outputs_teoricos")
all_data = []

day_to_type = {
    1: "tipo_1",
    2: "tipo_2",
    3: "tipo_2",
    4: "tipo_2",
    5: "tipo_3",
    6: "tipo_3",
    0: "tipo_1"
}

for csv_file in root.rglob("customers.csv"):
    df = pd.read_csv(csv_file)
    
    dir_name = csv_file.parent.name
    if "Day-" in dir_name:
        day_num = int(dir_name.split("Day-")[1])
        day_of_week = day_num % 7
        day_type = day_to_type[day_of_week]
    else:
        continue
    
    df_balked = df[df["outcome"] == "abandoned"].copy()
    
    if len(df_balked) == 0:
        continue
        
    df_balked["patience_s"] = df_balked["wait_time_s"]
    df_balked["day_type"] = day_type
    all_data.append(df_balked[["profile", "priority", "payment_method", "day_type", "patience_s"]])

df_theo = pd.concat(all_data, ignore_index=True)

print(f"\nTotal observaciones teoricas: {len(df_theo)}")
print(f"\nPrimeras 10 filas:")
print(df_theo.head(10))

# Test: filtrar "regular, no_priority, card, tipo_1"
mask = (
    (df_theo["profile"] ==  "regular") &
    (df_theo["priority"] == "no_priority") &
    (df_theo["payment_method"] == "card") &
    (df_theo["day_type"] == "tipo_1")
)
subset = df_theo[mask]
print(f"\nTest para regular/no_priority/card/tipo_1:")
print(f"  Observaciones: {len(subset)}")
print(f"  Media paciencia: {subset['patience_s'].mean():.2f} s")
print(f"  Mediana paciencia: {subset['patience_s'].median():.2f} s")

# Check unique values
print(f"\nUnique profiles: {sorted(df_theo['profile'].unique())}")
print(f"Unique priorities: {sorted(df_theo['priority'].unique())}")
print(f"Unique payment_methods: {sorted(df_theo['payment_method'].unique())}")
print(f"Unique day_types: {sorted(df_theo['day_type'].unique())}")
