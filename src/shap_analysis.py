"""
SHAP — Top-5 Segmentos Críticos em São Luís (2024)
Carrega modelo_final.pkl e analisa os 5 segmentos de maior risco.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import shap

BASE_DIR  = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR   = f"{BASE_DIR}\\csv"
MODEL_DIR = f"{BASE_DIR}\\1projeto"

with open(f"{MODEL_DIR}\\modelo_final.pkl", "rb") as f:
    meta = pickle.load(f)

modelo   = meta["modelo"]
FEATURES = meta["features"]
print(f"Modelo carregado | Features: {FEATURES}")

# São Luís 2024
df = pd.read_csv(f"{CSV_DIR}\\11_training_dataset_final.csv", low_memory=False)
df_sl_2024 = df[(df["cidade"] == "sao_luis") & (df["ano"] == 2024)].copy()
df_sl_2024["risco"] = modelo.predict_proba(df_sl_2024[FEATURES])[:, 1]
top5 = df_sl_2024.nlargest(5, "risco").reset_index(drop=True)

print(f"\nAnalisando {len(df_sl_2024):,} segmentos de São Luís em 2024...")

explainer = shap.TreeExplainer(modelo)
shap_vals = explainer.shap_values(top5[FEATURES])

print("\nTop-5 Segmentos Críticos em São Luís (2024):")
print("=" * 60)

for rank, row in top5.iterrows():
    sv = shap_vals[rank]
    top_feats = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[:4]
    lat = row["lat_mid"]
    lon = row["lon_mid"]
    seg = row.get("segment_id", "?")
    print(f"\n  [{rank+1}] segment_id={seg}")
    print(f"       Risco: {row['risco']:.3f} | ICP: {row.get('icp','?'):.1f}")
    print(f"       Localização: lat={lat:.4f}, lon={lon:.4f}")
    print(f"       flood_zone: {row.get('flood_zone_final',0):.2f} | "
          f"slope: {row.get('slope_pct',0):.1f}%")
    print(f"       Fatores SHAP (impacto no risco):")
    for feat, sv_val in top_feats:
        direcao = "↑ aumenta risco" if sv_val > 0 else "↓ reduz risco"
        print(f"         {feat:<30} SHAP={sv_val:+.4f}  ({direcao})")

# Salvar CSV top5
cols_out = ["segment_id", "cidade", "ano", "lat_mid", "lon_mid",
            "risco", "icp", "flood_zone_final", "highway_code",
            "slope_pct", "chuva_media_anual_mm", "load_proxy_1_10"]
cols_out = [c for c in cols_out if c in top5.columns]
top5[cols_out].to_csv(f"{CSV_DIR}\\12_shap_top5_sao_luis.csv",
                       index=False, encoding="utf-8")
print(f"\n✅ Top-5 salvo: {CSV_DIR}\\12_shap_top5_sao_luis.csv")
