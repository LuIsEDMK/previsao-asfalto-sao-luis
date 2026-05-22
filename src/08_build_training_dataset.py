"""
Tarefa 4 — Dataset Final de Treinamento
Une: road network + elevation + flood + density + rainfall + ICP Y labels.
Saída: csv/11_training_dataset_final.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"

# ── Carregar todos os dados ──────────────────────────────────────────────────
print("Carregando dados...")
road    = pd.read_csv(f"{CSV_DIR}\\01_road_network_osmnx.csv", low_memory=False)
elev    = pd.read_csv(f"{CSV_DIR}\\02_elevation_slope.csv")
flood   = pd.read_csv(f"{CSV_DIR}\\03_flood_zones.csv")
density = pd.read_csv(f"{CSV_DIR}\\04_urban_density.csv")
chuva   = pd.read_csv(f"{CSV_DIR}\\05_rainfall_by_segment.csv")
icp_y   = pd.read_csv(f"{CSV_DIR}\\10_icp_y_labels.csv")

for nome, dframe in [("road",road),("elev",elev),("flood",flood),
                     ("density",density),("chuva",chuva),("icp_y",icp_y)]:
    print(f"  {nome}: {dframe.shape}")

# ── Montar base estática por segmento ────────────────────────────────────────
print("\nMontando base estática por segmento...")
base = road.copy()

# Elevation + slope
base = base.merge(
    elev[["segment_id","elevation_m","slope_pct","flood_risk_slope"]],
    on="segment_id", how="left"
)

# Flood zones
base = base.merge(
    flood[["segment_id","dist_water_m","flood_zone_final"]],
    on="segment_id", how="left"
)

# Density
dens_cols = [c for c in density.columns if "densidade" in c.lower() or "density" in c.lower()]
dens_key  = "segment_id" if "segment_id" in density.columns else None
if dens_key:
    base = base.merge(density[[dens_key] + dens_cols], on=dens_key, how="left")
else:
    # Fallback: merge por cidade
    cidade_dens = density.groupby("cidade")[dens_cols].first().reset_index() if "cidade" in density.columns else None
    if cidade_dens is not None and "cidade" in base.columns:
        base = base.merge(cidade_dens, on="cidade", how="left")

# Rainfall
chuva_feats = ["chuva_media_anual_mm","chuva_media_jan_mm","chuva_media_jul_mm",
               "max_30d_hist_mm","prob_chuva_extrema","n_extremos_por_ano","razao_sazonalidade"]
chuva_feats = [c for c in chuva_feats if c in chuva.columns]
base = base.merge(chuva[["segment_id"] + chuva_feats], on="segment_id", how="left")

print(f"  Base estática: {base.shape}")

# ── Codificar highway_type → highway_code ────────────────────────────────────
HIGHWAY_MAP = {
    "motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
    "tertiary": 5, "residential": 6, "living_street": 7,
    "unclassified": 8, "service": 9, "track": 10, "path": 11,
    "footway": 12, "cycleway": 13, "other": 14,
}

def encode_hw(hw):
    hw = str(hw).lower()
    for k, v in HIGHWAY_MAP.items():
        if k in hw:
            return v
    return 14

if "highway_type" in base.columns:
    base["highway_code"] = base["highway_type"].apply(encode_hw)
elif "highway" in base.columns:
    base["highway_code"] = base["highway"].apply(encode_hw)

# ── Cruzar com ICP Y (10 anos) ────────────────────────────────────────────────
print("Cruzando com ICP Y (10 anos)...")
df_final = icp_y.merge(
    base[["segment_id","cidade","highway_code","length_m","maxspeed_kmh","lanes","oneway",
          "load_proxy_1_10","lat_mid","lon_mid","flood_zone_final","dist_water_m","flood_risk_slope",
          "elevation_m","slope_pct"] + chuva_feats +
         [c for c in dens_cols if c in base.columns]],
    on="segment_id", how="left"
)

df_final["y_source"] = "icp_degradacao"

print(f"  Dataset final: {df_final.shape}")

# ── Preencher NaN numéricos ───────────────────────────────────────────────────
num_cols = df_final.select_dtypes(include="number").columns
df_final[num_cols] = df_final[num_cols].fillna(0)

# ── Selecionar e ordenar features finais ─────────────────────────────────────
FEATURES = [
    "highway_code", "length_m", "maxspeed_kmh", "lanes", "oneway", "load_proxy_1_10",
    "flood_zone_final", "dist_water_m", "flood_risk_slope",
    "elevation_m", "slope_pct",
    "chuva_media_anual_mm", "chuva_media_jan_mm",
]
# Adicionar densidade se disponível
dens_final = [c for c in dens_cols if c in df_final.columns]
FEATURES = FEATURES + dens_final

FEATURES_OK = [f for f in FEATURES if f in df_final.columns]
FEATURES_MISS = [f for f in FEATURES if f not in df_final.columns]

print(f"\nFeatures disponíveis ({len(FEATURES_OK)}): {FEATURES_OK}")
if FEATURES_MISS:
    print(f"Features ausentes: {FEATURES_MISS}")

META_COLS = ["segment_id","cidade","ano","lat_mid","lon_mid","y_icp","y_source","icp",
             "taxa_ajustada","fator_total","mes_pico_os"]
META_OK   = [c for c in META_COLS if c in df_final.columns]

saida_cols = META_OK + FEATURES_OK
df_out = df_final[saida_cols].copy()

# ── Estatísticas finais ───────────────────────────────────────────────────────
print("\nEstatísticas do dataset final:")
print(f"  Shape: {df_out.shape}")
print(f"  Y=1 total: {df_out['y_icp'].sum():,} ({df_out['y_icp'].mean()*100:.1f}%)")
print(f"  Cidades:")
for cidade, grp in df_out.groupby("cidade"):
    n_segs = grp["segment_id"].nunique()
    pct_y  = grp["y_icp"].mean() * 100
    print(f"    {cidade}: {n_segs:,} segmentos, Y=1={pct_y:.1f}%")
print(f"\n  Anos: {sorted(df_out['ano'].unique())}")
print(f"  Features OK: {FEATURES_OK}")

# Verificar NaN remanescentes
nan_check = df_out[FEATURES_OK].isna().sum()
nan_cols  = nan_check[nan_check > 0]
if len(nan_cols) > 0:
    print(f"\n  ⚠  NaN remanescentes: {nan_cols.to_dict()}")
else:
    print("\n  ✅ Zero NaN nas features")

# ── Salvar ───────────────────────────────────────────────────────────────────
saida = f"{CSV_DIR}\\11_training_dataset_final.csv"
df_out.to_csv(saida, index=False, encoding="utf-8")
print(f"\n✅ Dataset final salvo: {saida}")
print(f"   {df_out.shape[0]:,} linhas × {df_out.shape[1]} colunas")
