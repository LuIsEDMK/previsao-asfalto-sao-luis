"""
Tarefa 6 — Previsões São Luís com Análise Econômica
Categorias de risco + custo preventivo vs emergencial em R$.
Saída: csv/13_previsoes_economicas_sao_luis.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

BASE_DIR  = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR   = f"{BASE_DIR}\\csv"
MODEL_DIR = f"{BASE_DIR}\\1projeto"

# ── Carregar modelo ──────────────────────────────────────────────────────────
with open(f"{MODEL_DIR}\\modelo_final.pkl", "rb") as f:
    meta = pickle.load(f)

modelo   = meta["modelo"]
FEATURES = meta["features"]
print(f"Modelo carregado | AUROC OOD = {meta.get('auroc_ood_sao_luis', '?'):.4f}")

# ── Carregar São Luís 2024 ───────────────────────────────────────────────────
df = pd.read_csv(f"{CSV_DIR}\\11_training_dataset_final.csv", low_memory=False)
df_sl = df[(df["cidade"] == "sao_luis") & (df["ano"] == 2024)].copy()
print(f"\nSegmentos São Luís 2024: {len(df_sl):,}")

# ── Predizer risco ───────────────────────────────────────────────────────────
df_sl["risco_prob"] = modelo.predict_proba(df_sl[FEATURES])[:, 1]
df_sl["risco_pred"] = (df_sl["risco_prob"] >= 0.5).astype(int)

# ── Classificar categoria de risco ──────────────────────────────────────────
def categoria(prob):
    if prob >= 0.75:
        return "CRÍTICO"
    elif prob >= 0.50:
        return "ALTO"
    elif prob >= 0.30:
        return "MÉDIO"
    else:
        return "BAIXO"

df_sl["categoria_risco"] = df_sl["risco_prob"].apply(categoria)

# ── Custos por categoria ─────────────────────────────────────────────────────
# Custo por segmento (em R$) — baseado em estimativas DNIT/SEMUSC
CUSTOS = {
    "CRÍTICO": {"preventivo": 15_000, "emergencial": 45_000},
    "ALTO":    {"preventivo":  3_500, "emergencial": 12_000},
    "MÉDIO":   {"preventivo":    800, "emergencial":  3_000},
    "BAIXO":   {"preventivo":    200, "emergencial":    600},
}

df_sl["custo_preventivo_R$"]   = df_sl["categoria_risco"].map(
    lambda c: CUSTOS[c]["preventivo"])
df_sl["custo_emergencial_R$"]  = df_sl["categoria_risco"].map(
    lambda c: CUSTOS[c]["emergencial"])
df_sl["economia_R$"]           = (
    df_sl["custo_emergencial_R$"] - df_sl["custo_preventivo_R$"]
)
df_sl["razao_custo"]           = (
    df_sl["custo_emergencial_R$"] / df_sl["custo_preventivo_R$"]
)

# ── Relatório econômico ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ANÁLISE ECONÔMICA — SÃO LUÍS 2024")
print("=" * 60)

total_segs = len(df_sl)
for cat in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
    grp = df_sl[df_sl["categoria_risco"] == cat]
    n   = len(grp)
    pct = n / total_segs * 100
    custo_prev  = grp["custo_preventivo_R$"].sum()
    custo_emer  = grp["custo_emergencial_R$"].sum()
    economia    = grp["economia_R$"].sum()
    print(f"\n  {cat} — {n:,} segmentos ({pct:.1f}%)")
    print(f"    Custo preventivo total: R$ {custo_prev:>14,.0f}")
    print(f"    Custo emergencial total:R$ {custo_emer:>14,.0f}")
    print(f"    Economia potencial:      R$ {economia:>14,.0f}")

total_prev = df_sl["custo_preventivo_R$"].sum()
total_emer = df_sl["custo_emergencial_R$"].sum()
total_econ = df_sl["economia_R$"].sum()

print(f"\n{'─'*60}")
print(f"  TOTAIS (todos os {total_segs:,} segmentos):")
print(f"    Custo 100% preventivo: R$ {total_prev:>14,.0f}")
print(f"    Custo 100% emergencial:R$ {total_emer:>14,.0f}")
print(f"    Economia total possível:R$ {total_econ:>14,.0f}")
print(f"    Razão médio emer/prev: {total_emer/total_prev:.1f}x")

# Distribuição de probabilidade
print(f"\n  Distribuição de probabilidade de risco:")
for cat in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
    n   = (df_sl["categoria_risco"] == cat).sum()
    pct = n / total_segs * 100
    bar = "█" * int(pct / 2)
    print(f"    {cat:<9} {n:>6,}  ({pct:5.1f}%) {bar}")

# Top 10 segmentos mais críticos
print(f"\n  Top 10 segmentos mais críticos (lat, lon, risco):")
top10 = df_sl.nlargest(10, "risco_prob")[["segment_id","lat_mid","lon_mid",
                                           "risco_prob","categoria_risco",
                                           "custo_preventivo_R$","custo_emergencial_R$"]]
for _, r in top10.iterrows():
    print(f"    {r['segment_id']} | "
          f"lat={r['lat_mid']:.4f}, lon={r['lon_mid']:.4f} | "
          f"risco={r['risco_prob']:.3f} | {r['categoria_risco']} | "
          f"prev=R${r['custo_preventivo_R$']:,.0f}")

# ── Salvar CSV final ─────────────────────────────────────────────────────────
cols_salvar = [
    "segment_id", "cidade", "ano", "lat_mid", "lon_mid",
    "risco_prob", "risco_pred", "categoria_risco",
    "custo_preventivo_R$", "custo_emergencial_R$", "economia_R$", "razao_custo",
    "icp", "flood_zone_final", "highway_code", "slope_pct",
    "chuva_media_anual_mm", "load_proxy_1_10",
]
cols_salvar = [c for c in cols_salvar if c in df_sl.columns]
df_out = df_sl[cols_salvar].copy()

saida = f"{CSV_DIR}\\13_previsoes_economicas_sao_luis.csv"
df_out.to_csv(saida, index=False, encoding="utf-8")
print(f"\n✅ Previsões econômicas salvas: {saida}")
print(f"   {len(df_out):,} segmentos × {len(df_out.columns)} colunas")

# Resumo executivo (para usar no relatório)
print(f"\n  RESUMO EXECUTIVO (para gestores):")
criticos = (df_sl["categoria_risco"] == "CRÍTICO").sum()
altos    = (df_sl["categoria_risco"] == "ALTO").sum()
econ_critico = df_sl[df_sl["categoria_risco"] == "CRÍTICO"]["economia_R$"].sum()
print(f"  • {criticos} trechos em estado CRÍTICO necessitam intervenção imediata")
print(f"  • {altos} trechos em estado ALTO devem ser monitorados mensalmente")
print(f"  • Intervir preventivamente nos trechos CRÍTICOS economizaria")
print(f"    R$ {econ_critico:,.0f} comparado ao custo emergencial")
print(f"  • ROI médio da manutenção preventiva: {total_emer/total_prev:.1f}x o custo preventivo")
