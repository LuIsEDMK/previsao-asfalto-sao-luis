"""
Projeto: Preditor de Falhas em Vias - Sao Luis MA (v2 - Dados Publicos)
Tarefa 6: Chuva por Segmento via Open-Meteo (grade 11km)
Saida:    05_rainfall_by_segment.csv

Estrategia:
- Arredondar lat_mid/lon_mid para 1 decimal (grade ~11km Open-Meteo)
- Baixar serie diaria 2015-2024 para cada ponto unico (~20-30 pontos)
- Calcular estatisticas de longo prazo por ponto de grade
- Atribuir ao segmento mais proximo (por celula de grade)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time, warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"
CACHE_DIR= f"{BASE_DIR}\\cache_osmnx"
ENTRADA  = f"{CSV_DIR}\\01_road_network_osmnx.csv"
SAIDA    = f"{CSV_DIR}\\05_rainfall_by_segment.csv"

DATA_INI = "2015-01-01"
DATA_FIM = "2024-12-31"
TIMEOUT  = 60

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─── Open-Meteo Archive API ───────────────────────────────────────

def baixar_serie_diaria(lat, lon):
    """
    Baixa precipitacao diaria (mm) de 2015-2024 via Open-Meteo Archive.
    Retorna Series indexada por data, ou None se falhar.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  DATA_INI,
        "end_date":    DATA_FIM,
        "daily":       "precipitation_sum",
        "timezone":    "America/Fortaleza",
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            datas  = data["daily"]["time"]
            precip = data["daily"]["precipitation_sum"]
            serie = pd.Series(
                [float(v) if v is not None else 0.0 for v in precip],
                index=pd.to_datetime(datas),
                name="precip_mm"
            )
            return serie.clip(lower=0)
    except Exception as e:
        print(f"  [AV] Open-Meteo ({lat},{lon}): {e}")
    return None

def calcular_stats(serie):
    """
    Calcula estatisticas de longo prazo de chuva a partir de serie diaria.
    Retorna dict com features para o modelo.
    """
    if serie is None or len(serie) == 0:
        return {
            "chuva_media_anual_mm":   0.0,
            "chuva_media_jan_mm":     0.0,
            "chuva_media_jul_mm":     0.0,
            "max_30d_hist_mm":        0.0,
            "prob_chuva_extrema":     0.0,
            "n_extremos_por_ano":     0.0,
            "razao_sazonalidade":     1.0,
            "fonte_chuva":            "fallback_zero",
        }

    # Media anual
    anual = serie.resample("YE").sum()
    chuva_media_anual = round(float(anual.mean()), 1)

    # Media por mes
    mensal_medio = serie.groupby(serie.index.month).mean() * 30.44
    jan = round(float(mensal_medio.get(1, 0)), 1)
    jul = round(float(mensal_medio.get(7, 0)), 1)
    razao = round(jan / max(jul, 1), 2)

    # Maximo acumulado em 30 dias (janela rolante)
    max_30d = round(float(serie.rolling(30, min_periods=1).sum().max()), 1)

    # Eventos extremos: dias com > 50mm
    extremos = (serie > 50).astype(int)
    n_anos = len(serie) / 365.25
    n_ext_ano = round(float(extremos.sum()) / max(n_anos, 1), 2)
    prob_ext_90d = round(min(1.0, 1 - (1 - extremos.mean()) ** 90), 3)

    return {
        "chuva_media_anual_mm": chuva_media_anual,
        "chuva_media_jan_mm":   jan,
        "chuva_media_jul_mm":   jul,
        "max_30d_hist_mm":      max_30d,
        "prob_chuva_extrema":   prob_ext_90d,
        "n_extremos_por_ano":   n_ext_ano,
        "razao_sazonalidade":   razao,
        "fonte_chuva":          "open-meteo",
    }

# ─── EXECUCAO PRINCIPAL ───────────────────────────────────────────

sep("TAREFA 6 — Chuva por Segmento (Open-Meteo grade 11km)")

if os.path.exists(SAIDA):
    df_ex = pd.read_csv(SAIDA)
    print(f"  [INFO] Arquivo ja existe com {len(df_ex):,} linhas. Nada a fazer.")
    print(df_ex.groupby("cidade")[["chuva_media_anual_mm","n_extremos_por_ano","razao_sazonalidade"]].mean().round(2).to_string())
    sys.exit(0)

sep("Carregando rede viaria")
df = pd.read_csv(ENTRADA)
print(f"  Segmentos: {len(df):,}")

# Coletar pontos de grade unicos (1dp = ~11km, resolucao Open-Meteo)
sep("Identificando pontos de grade unicos (1 decimal = ~11km)")
df["lat_grid"] = df["lat_mid"].round(1)
df["lon_grid"] = df["lon_mid"].round(1)

grade_unicos = df[["lat_grid","lon_grid"]].drop_duplicates()
print(f"  Pontos de grade: {len(grade_unicos)}")
print(grade_unicos.to_string(index=False))

# Baixar serie para cada ponto de grade
sep("Baixando series diarias Open-Meteo (2015-2024)")
cache_stats = {}  # (lat,lon) -> dict de stats

for _, row in grade_unicos.iterrows():
    lat = float(row["lat_grid"])
    lon = float(row["lon_grid"])
    chave = (lat, lon)

    cache_path = f"{CACHE_DIR}/chuva_{lat}_{lon}.parquet"
    if os.path.exists(cache_path):
        try:
            serie = pd.read_parquet(cache_path)["precip_mm"]
            print(f"  [CACHE] ({lat},{lon}) — {len(serie)} dias")
            cache_stats[chave] = calcular_stats(serie)
            continue
        except Exception:
            pass

    print(f"  [API] ({lat},{lon})...", end=" ", flush=True)
    serie = baixar_serie_diaria(lat, lon)
    if serie is not None:
        print(f"{len(serie)} dias | anual={serie.resample('YE').sum().mean():.0f}mm")
        pd.DataFrame({"precip_mm": serie}).to_parquet(cache_path)
        cache_stats[chave] = calcular_stats(serie)
    else:
        print("FALHOU — usando fallback")
        cache_stats[chave] = calcular_stats(None)

    time.sleep(0.3)

# Atribuir stats de chuva a cada segmento
sep("Atribuindo estatisticas de chuva aos segmentos")

registros = []
for _, row in df.iterrows():
    chave = (float(row["lat_grid"]), float(row["lon_grid"]))
    stats = cache_stats.get(chave, calcular_stats(None))

    registros.append({
        "segment_id": row["segment_id"],
        "cidade":     row["cidade"],
        "lat_grid":   chave[0],
        "lon_grid":   chave[1],
        **stats,
    })

df_out = pd.DataFrame(registros)
df_out.to_csv(SAIDA, index=False, encoding="utf-8")

sep("RESUMO FINAL — TAREFA 6")
print(f"  Total: {len(df_out):,} segmentos")
print()
print("  Por cidade:")
print(df_out.groupby("cidade")[[
    "chuva_media_anual_mm",
    "chuva_media_jan_mm",
    "chuva_media_jul_mm",
    "n_extremos_por_ano",
    "razao_sazonalidade"
]].mean().round(2).to_string())
print()
print(f"  [OK] {SAIDA}")
print(f"  TAREFA 6 CONCLUIDA")
