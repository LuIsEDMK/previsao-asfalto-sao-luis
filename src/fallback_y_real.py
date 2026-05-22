"""
Tarefa 3 — Y por Degradação Física (ICP — Índice de Condição do Pavimento)
Substitui Y sintético quando não há OS reais.
Saída: csv/10_icp_y_labels.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"

# ── Carregar dados base ──────────────────────────────────────────────────────
print("Carregando dados base...")
road    = pd.read_csv(f"{CSV_DIR}\\01_road_network_osmnx.csv", low_memory=False)
elev    = pd.read_csv(f"{CSV_DIR}\\02_elevation_slope.csv")
flood   = pd.read_csv(f"{CSV_DIR}\\03_flood_zones.csv")
chuva   = pd.read_csv(f"{CSV_DIR}\\05_rainfall_by_segment.csv")

print(f"  road: {road.shape} | elev: {elev.shape} | flood: {flood.shape} | chuva: {chuva.shape}")

# Merge base — chave: segment_id
df = road.copy()
df = df.merge(elev[["segment_id","elevation_m","slope_pct","flood_risk_slope"]],
              on="segment_id", how="left")
df = df.merge(flood[["segment_id","flood_zone_final","dist_water_m"]],
              on="segment_id", how="left")
chuva_cols = [c for c in chuva.columns if c.startswith("chuva_")]
df = df.merge(chuva[["segment_id"] + chuva_cols], on="segment_id", how="left")

num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(0)
print(f"  Dataset merged: {df.shape}")

# ── Parâmetros ICP ───────────────────────────────────────────────────────────
# Taxa de degradação anual por tipo de via (pontos de ICP/ano)
TAXA_BASE = {
    "motorway":   2.0,   # rodovias — alta qualidade, manutenção federal
    "trunk":      2.5,
    "primary":    3.5,   # arteriais principais
    "secondary":  4.0,
    "tertiary":   4.5,
    "residential":5.0,   # locais — pior qualidade construtiva
    "living_street":5.5,
    "unclassified":5.0,
    "service":    5.5,
    "track":      6.0,
    "path":       6.5,
    "footway":    4.0,
    "cycleway":   4.0,
    "other":      5.0,
}

def taxa_por_highway(hw):
    hw = str(hw).lower()
    for k in TAXA_BASE:
        if k in hw:
            return TAXA_BASE[k]
    return TAXA_BASE["other"]

# Cidades e estação chuvosa (meses com >120mm média histórica)
CIDADE_RAIN_PEAK = {
    "sao_luis":  {"mes_inicio": 1, "mes_fim": 5, "meses": [1,2,3,4,5]},    # jan-mai
    "fortaleza": {"mes_inicio": 2, "mes_fim": 5, "meses": [2,3,4,5]},      # fev-mai
    "recife":    {"mes_inicio": 5, "mes_fim": 8, "meses": [5,6,7,8]},      # mai-ago
}

def cidade_do_segmento(cidade_str):
    if pd.isna(cidade_str):
        return "sao_luis"
    c = str(cidade_str).lower()
    if "fortaleza" in c or "cear" in c:
        return "fortaleza"
    if "recife" in c or "pernambuco" in c:
        return "recife"
    return "sao_luis"

# ── Calcular fatores ICP ─────────────────────────────────────────────────────
print("\nCalculando fatores ICP...")

# Taxa base por tipo de via
taxa_col = "highway_type" if "highway_type" in df.columns else "highway"
df["taxa_base"] = df[taxa_col].apply(lambda x: taxa_por_highway(str(x)))

# Fator chuva: chuva acima de 300mm/trimestre acelera degradação
chuva_anual = df["chuva_media_anual_mm"] if "chuva_media_anual_mm" in df.columns else pd.Series(1500.0, index=df.index)
chuva_trimestral = chuva_anual / 4  # aproximação trimestral
df["fator_chuva"] = 1.0 + ((chuva_trimestral - 300) / 1000).clip(-0.3, 0.8)

# Fator alagamento (flood_zone_final: 0=seco, 1=alagável)
fz = df["flood_zone_final"].clip(0, 1)
df["fator_alagamento"] = 1.0 + 0.8 * fz  # varia de 1.0 (seco) a 1.8 (alagável)

# Fator declividade: inclinações moderadas (~2-5%) drenam bem (fator < 1)
# inclinações extremas (>10%) acumulam dano mecânico por veículos (fator > 1)
slope = df["slope_pct"].clip(0, 30) if "slope_pct" in df.columns else pd.Series(1.0, index=df.index)
df["fator_declividade"] = np.where(
    slope < 1,   1.30,   # plano = acumula água → mais degradação
    np.where(
        slope < 5,  0.95,   # suave = drenagem boa
        np.where(
            slope < 15, 1.05,  # moderado = stress mecânico
            1.20              # acentuado = erosão + stress
        )
    )
)

# Fator carga (vias principais têm mais tráfego pesado)
load_col = next((c for c in ["load_proxy_1_10","load_proxy"] if c in df.columns), None)
if load_col:
    df["fator_carga"] = 1.0 + 0.04 * (df[load_col] - 5.5).clip(-4, 4)
else:
    df["fator_carga"] = 1.0

# Fator total
df["fator_total"] = (
    df["fator_chuva"] *
    df["fator_alagamento"] *
    df["fator_declividade"] *
    df["fator_carga"]
)

# Taxa ajustada final
df["taxa_ajustada"] = (df["taxa_base"] * df["fator_total"]).clip(0.5, 15.0)

print(f"  Taxa ajustada: min={df['taxa_ajustada'].min():.2f}, "
      f"média={df['taxa_ajustada'].mean():.2f}, max={df['taxa_ajustada'].max():.2f}")
print(f"  Fator total: min={df['fator_total'].min():.2f}, "
      f"média={df['fator_total'].mean():.2f}, max={df['fator_total'].max():.2f}")

# ── Simular ICP Anual 2015–2024 (vetorizado) ─────────────────────────────────
print("\nSimulando ICP 2015–2024 (vetorizado)...")

ANOS = list(range(2015, 2025))
N    = len(df)
cidade_col = next((c for c in ["cidade","city","municipio","municipality"] if c in df.columns), None)

# Mapear cidade para colunas
if cidade_col:
    cidades_arr = df[cidade_col].apply(cidade_do_segmento).values
else:
    cidades_arr = np.full(N, "sao_luis")

# Mês pico de OS por segmento (calculado uma vez)
def mes_os_de_cidade(c):
    meses = CIDADE_RAIN_PEAK.get(c, CIDADE_RAIN_PEAK["sao_luis"])["meses"]
    return min(meses[-1] + 1, 12)

mes_pico_arr = np.array([mes_os_de_cidade(c) for c in cidades_arr])

# Heterogeneidade inicial: ruído ±5 com semente global fixa
rng_global = np.random.default_rng(seed=42)
ruido_inicial = rng_global.uniform(-5, 5, size=N)

# ICP inicial por segmento
icp_arr = np.clip(100.0 + ruido_inicial, 30.0, 100.0)
taxa     = df["taxa_ajustada"].values
fator_t  = df["fator_total"].values
seg_ids  = df["segment_id"].values

# Amplificação por ano (El Niño simplificado — cidades nordestinas)
AMP_ANOS = {2015: 1.15, 2016: 1.15, 2020: 1.15, 2022: 1.15}

frames = []
for ano in ANOS:
    amp = AMP_ANOS.get(ano, 1.0)
    icp_arr = np.maximum(0.0, icp_arr - taxa * amp)
    y_arr   = (icp_arr < 40).astype(np.int8)

    frames.append(pd.DataFrame({
        "segment_id":    seg_ids,
        "ano":           ano,
        "icp":           np.round(icp_arr, 2),
        "y_icp":         y_arr,
        "taxa_ajustada": np.round(taxa, 3),
        "fator_total":   np.round(fator_t, 3),
        "mes_pico_os":   mes_pico_arr,
    }))

icp_df = pd.concat(frames, ignore_index=True)
print(f"  Simulação: {len(icp_df):,} registros ({N:,} segmentos × {len(ANOS)} anos)")

# ── Validação ─────────────────────────────────────────────────────────────────
print("\nValidação do Y_ICP:")
total = len(icp_df)
positivos = icp_df["y_icp"].sum()
pct = positivos / total * 100
print(f"  Y=1: {positivos:,} ({pct:.1f}%)")
print(f"  Y=0: {total - positivos:,} ({100-pct:.1f}%)")

# Distribuição por ano (deve crescer — degradação acumulada)
print("\n  Y=1 por ano (deve crescer):")
por_ano = icp_df.groupby("ano")["y_icp"].mean() * 100
for ano, pct_ano in por_ano.items():
    bar = "█" * int(pct_ano / 2)
    print(f"    {ano}: {pct_ano:5.1f}% {bar}")

# Distribuição por cidade
if cidade_col:
    print("\n  Y=1 por cidade:")
    icp_com_cidade = icp_df.merge(
        df[["segment_id", cidade_col]].drop_duplicates("segment_id"),
        on="segment_id", how="left"
    )
    for cidade_val, grp in icp_com_cidade.groupby(cidade_col):
        pct_c = grp["y_icp"].mean() * 100
        print(f"    {cidade_val}: {pct_c:.1f}%")

# Verificar concentração em segmentos de risco (flood_zone alto)
icp_2024 = icp_df[icp_df["ano"] == 2024].merge(
    df[["segment_id","flood_zone_final"]].drop_duplicates("segment_id"),
    on="segment_id", how="left"
)
corr = icp_2024[["y_icp","flood_zone_final"]].corr().iloc[0, 1]
print(f"\n  Correlação Y_ICP × flood_zone (2024): {corr:.3f}")
if corr > 0.50:
    print("  ⚠  ATENÇÃO: correlação alta — verificar pesos do modelo ICP")
elif corr > 0.20:
    print("  ✅ Correlação moderada — discriminação física presente")
else:
    print("  ✅ Correlação baixa — modelo ICP independente do flood_zone")

# ── Salvar ───────────────────────────────────────────────────────────────────
saida = f"{CSV_DIR}\\10_icp_y_labels.csv"
icp_df.to_csv(saida, index=False, encoding="utf-8")
print(f"\n✅ ICP Y salvo: {saida}")
print(f"   Shape: {icp_df.shape}")
print(f"   Colunas: {list(icp_df.columns)}")

# Resumo estatístico do ICP final (ano 2024)
icp_final = icp_df[icp_df["ano"] == 2024]["icp"]
print(f"\n  ICP final (2024):")
print(f"    Média:   {icp_final.mean():.1f}")
print(f"    Mediana: {icp_final.median():.1f}")
print(f"    <40 (crítico):  {(icp_final < 40).sum():,} segmentos")
print(f"    40-70 (regular):{((icp_final >= 40) & (icp_final < 70)).sum():,} segmentos")
print(f"    >=70 (bom):     {(icp_final >= 70).sum():,} segmentos")
