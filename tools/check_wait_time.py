import pandas as pd

df = pd.read_csv("outputs_teoricos/Week-1-Day-1/customers.csv")
df_abandoned = df[df["outcome"] == "abandoned"]

print(f"Abandonados: {len(df_abandoned)}")
print(f"wait_time_s es NaN: {df_abandoned['wait_time_s'].isna().sum()}")
print(f"\nPrimeras 10 filas:")
print(df_abandoned[["customer_id", "outcome", "wait_time_s", "patience_s"]].head(10))
