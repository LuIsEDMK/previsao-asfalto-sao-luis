"""
Projeto: Preditor de Falhas em Vias - Sao Luis MA (v2 - Dados Publicos)
Tarefa 3: Elevacao e Declividade via Open-Elevation (NASA SRTM)
Saida:    02_elevation_slope.csv

Estrategia: arredondamento em 2dp (~1.1 km) para reduzir pontos unicos
de ~150k para ~2k — resolucao compativel com SRTM 30m das cidades costeiras
planas (Sao Luis, Fortaleza, Recife: altitude media < 30m).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import math
import requests
import pandas as pd
import numpy as np

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"
ENTRADA  = f"{CSV_DIR}\\01_road_network_osmnx.csv"
SAIDA    = f"{CSV_DIR}\\02_elevation_slope.csv"

# 2 casas decimais = ~1.1 km de resolução (3-4x SRTM 30m)
# reduz pontos únicos de ~150k para ~2k para cidades costeiras planas
PRECISION = 2
BATCH     = 100   # pontos por request
DELAY     = 0.5   # segundos entre requests
TIMEOUT   = 30    # timeout por request

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─── Classificação de declividade ────────────────────────────────

def categorizar_slope(slope_pct):
    if slope_pct < 2:
        return "plano", 3      # agua empoca — alto risco
    elif slope_pct < 5:
        return "suave", 2
    elif slope_pct < 10:
        return "moderado", 1
    else:
        return "acentuado", 0  # escoamento rapido — baixo risco

# ─── API Open-Elevation ───────────────────────────────────────────

def buscar_elevacoes_batch(pontos):
    """
    Busca elevacao de uma lista de (lat, lon) via Open-Elevation API.
    Retorna lista de elevacoes em metros (None se falhar).
    """
    url     = "https://api.open-elevation.com/api/v1/lookup"
    payload = {
        "locations": [
            {"latitude": lat, "longitude": lon}
            for lat, lon in pontos
        ]
    }
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT,
                          headers={"Content-Type": "application/json",
                                   "Accept": "application/json"})
        if r.status_code == 200:
            resultados = r.json().get("results", [])
            return [res.get("elevation") for res in resultados]
    except Exception:
        pass
    return [None] * len(pontos)

# ─── Fallback: Open-Topo-Data (SRTM 30m) ─────────────────────────

def buscar_elevacoes_opentopodata(pontos):
    """Fallback alternativo usando api.opentopodata.org."""
    loc_str = "|".join(f"{lat},{lon}" for lat, lon in pontos[:100])
    url     = f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            results = r.json().get("results", [])
            return [res.get("elevation") for res in results]
    except Exception:
        pass
    return [None] * len(pontos)

# ─── Buscar elevacoes com fallback ───────────────────────────────

def buscar_todos_pontos(pontos_unicos, usar_opentopodata=False):
    """
    Busca elevacao de todos os pontos unicos em batches.
    Retorna dict {(lat,lon): elevation_m}.
    """
    elevacoes = {}
    n         = len(pontos_unicos)
    n_ok      = 0
    n_fail    = 0

    print(f"  Total de pontos unicos ({PRECISION}dp): {n}")
    print(f"  Batches de {BATCH}: {math.ceil(n/BATCH)} requests")
    fn_api = buscar_elevacoes_opentopodata if usar_opentopodata else buscar_elevacoes_batch

    for i in range(0, n, BATCH):
        batch = pontos_unicos[i:i + BATCH]
        nums  = fn_api(batch)

        for (lat, lon), elev in zip(batch, nums):
            if elev is not None:
                elevacoes[(lat, lon)] = round(float(elev), 1)
                n_ok += 1
            else:
                elevacoes[(lat, lon)] = 0.0
                n_fail += 1

        progresso = min(i + BATCH, n)
        print(f"  [{progresso}/{n}] ok={n_ok} fallback={n_fail}")
        if n > BATCH:
            time.sleep(DELAY)

    taxa_ok = n_ok / n * 100 if n > 0 else 0
    print(f"  API respondeu: {taxa_ok:.1f}% dos pontos")
    return elevacoes, (taxa_ok > 50)

# ─── EXECUCAO PRINCIPAL ───────────────────────────────────────────

sep("TAREFA 3 — Elevacao e Declividade (NASA SRTM via Open-Elevation)")

if os.path.exists(SAIDA):
    df_ex = pd.read_csv(SAIDA)
    print(f"  [INFO] Arquivo ja existe com {len(df_ex):,} linhas.")
    if len(df_ex) >= 100:
        print("  Nada a fazer. Delete o arquivo para re-processar.")
        print(df_ex[["cidade","elevation_m","slope_pct","slope_category"]].groupby("cidade").agg(
            n=("elevation_m","count"),
            elev_media=("elevation_m","mean"),
            slope_media=("slope_pct","mean")
        ).round(2).to_string())
        sys.exit(0)

sep("Carregando rede viaria")
df = pd.read_csv(ENTRADA)
print(f"  Segmentos carregados: {len(df):,}")
print(f"  Cidades: {df['cidade'].value_counts().to_dict()}")

# Coletar pontos unicos com precisao reduzida
sep(f"Coletando pontos unicos ({PRECISION} casas decimais)")
pontos_set = set()
for _, row in df.iterrows():
    for pref in ["mid", "start", "end"]:
        lat = row.get(f"lat_{pref}")
        lon = row.get(f"lon_{pref}")
        if pd.notna(lat) and pd.notna(lon):
            pontos_set.add((round(float(lat), PRECISION),
                            round(float(lon), PRECISION)))

pontos_lista = sorted(pontos_set)
print(f"  Pontos unicos: {len(pontos_lista):,}")
print(f"  (vs ~150k com 4dp — reducao de {150000//max(len(pontos_lista),1)}x)")

# Testar API principal
sep("Buscando elevacoes")
print("  Teste da API Open-Elevation com 1 ponto...")
teste = buscar_elevacoes_batch([(-2.53, -44.30)])
api_principal_ok = (teste[0] is not None)

if api_principal_ok:
    print(f"  [OK] Open-Elevation respondeu: {teste[0]}m")
    elevacoes, api_ok = buscar_todos_pontos(pontos_lista, usar_opentopodata=False)
    fonte = "open-elevation"
else:
    print("  [AV] Open-Elevation nao respondeu — tentando Open-Topo-Data...")
    teste2 = buscar_elevacoes_opentopodata([(-2.53, -44.30)])
    if teste2[0] is not None:
        print(f"  [OK] Open-Topo-Data respondeu: {teste2[0]}m")
        elevacoes, api_ok = buscar_todos_pontos(pontos_lista, usar_opentopodata=True)
        fonte = "opentopodata-srtm30m"
    else:
        print("  [AV] Ambas as APIs indisponiveis — usando fallback zero")
        elevacoes = {p: 0.0 for p in pontos_lista}
        api_ok    = False
        fonte     = "fallback_zero"

# ─── Calcular slope por segmento ─────────────────────────────────
sep("Calculando declividade por segmento")

registros = []
for _, row in df.iterrows():
    def get_elev(pref):
        lat = round(float(row.get(f"lat_{pref}", 0)), PRECISION)
        lon = round(float(row.get(f"lon_{pref}", 0)), PRECISION)
        return elevacoes.get((lat, lon), 0.0)

    elev_mid   = get_elev("mid")
    elev_start = get_elev("start")
    elev_end   = get_elev("end")
    length_m   = float(row.get("length_m", 0)) or 1.0

    dh        = abs(elev_end - elev_start)
    slope_pct = round(dh / length_m * 100, 2)

    cat, risco = categorizar_slope(slope_pct)

    registros.append({
        "segment_id":        row["segment_id"],
        "cidade":            row["cidade"],
        "elevation_m":       elev_mid,
        "elevation_start_m": elev_start,
        "elevation_end_m":   elev_end,
        "slope_pct":         slope_pct,
        "slope_category":    cat,
        "flood_risk_slope":  risco,
        "elevation_source":  fonte,
    })

df_out = pd.DataFrame(registros)
df_out.to_csv(SAIDA, index=False, encoding="utf-8")

sep("RESUMO FINAL — TAREFA 3")
print(f"  Elevacao fonte: {fonte}")
print(f"  Precisao: {PRECISION} casas decimais (~{10**(2-PRECISION)*111:.0f}km grid)")
print(f"  Segmentos processados: {len(df_out):,}")
print()
print("  Por cidade:")
print(df_out.groupby("cidade")[["elevation_m","slope_pct","flood_risk_slope"]].mean().round(2).to_string())
print()
print("  Distribuicao slope_category:")
print(df_out["slope_category"].value_counts().to_string())
print()
print(f"  [OK] {SAIDA}")
print(f"  TAREFA 3 CONCLUIDA")
