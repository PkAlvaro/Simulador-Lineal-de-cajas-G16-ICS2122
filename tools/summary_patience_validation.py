import pandas as pd

df = pd.read_csv("data/patience/validation_results.csv")

print("=== RESUMEN FINAL DEL AJUSTE DE PACIENCIAS ===\n")
print(f"Total combinaciones validadas: {len(df)}")
print(f"\nError promedio en media: {df['mean_error_pct'].mean():.2f}%")
print(f"Error std en media: {df['mean_error_pct'].std():.2f}%")
print(f"Error mediana en media: {df['mean_error_pct'].abs().median():.2f}%")

print(f"\n% combinaciones con error abs < 10%: {100 * (df['mean_error_pct'].abs() < 10).sum() / len(df):.1f}%")
print(f"% combinaciones con error abs < 20%: {100 * (df['mean_error_pct'].abs() < 20).sum() / len(df):.1f}%")
print(f"% combinaciones con error abs < 30%: {100 * (df['mean_error_pct'].abs() < 30).sum() / len(df):.1f}%")

print("\n=== MEJORES AJUSTES (Top 5) ===")
df["abs_error"] = df["mean_error_pct"].abs()
best = df.nsmallest(5, "abs_error")
print(best[["profile", "day_type", "theo_mean", "fitted_mean", "mean_error_pct"]].to_string(index=False))

print("\n=== PEORES AJUSTES (Top 5) ===")
worst = df.nlargest(5, "abs_error")
print(worst[["profile", "day_type", "theo_mean", "fitted_mean", "mean_error_pct"]].to_string(index=False))
