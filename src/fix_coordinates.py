"""
Projeto: Preditor de Falhas em Vias - São Luís / MA
Script:  fix_coordinates.py
Tarefa:  Substituir coordenadas estimadas por coordenadas reais
         via geocodificação OpenStreetMap (Nominatim / geopy)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import shutil
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

CSV_DIR = r"w:\projetos vscode\projetos para prefeitura\csv"
COD_DIR = r"w:\projetos vscode\projetos para prefeitura\1projeto\codigos"

ARQUIVO_SEG    = f"{CSV_DIR}/01_road_segments.csv"
ARQUIVO_BACKUP = f"{CSV_DIR}/01_road_segments_backup.csv"
ARQUIVO_ANA    = f"{COD_DIR}/08_analytical_dataset.csv"

# Limites válidos para São Luís / MA
LAT_MIN, LAT_MAX = -2.70, -2.35
LON_MIN, LON_MAX = -44.45, -44.10

# ═══════════════════════════════════════════════════════════
# PASSO 1 — Backup antes de qualquer alteração
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 1 — Backup do arquivo original")
print("="*60)

shutil.copy2(ARQUIVO_SEG, ARQUIVO_BACKUP)
print(f"  [OK] Backup salvo em: {ARQUIVO_BACKUP}")

# ═══════════════════════════════════════════════════════════
# PASSO 2 — Carregar segmentos
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 2 — Geocodificação via Nominatim (OSM)")
print("="*60)
print("  Aguarde: 1 segundo de delay obrigatório entre chamadas\n")

seg = pd.read_csv(ARQUIVO_SEG, encoding="utf-8")
total = len(seg)

# Inicializar geocoder
geolocator = Nominatim(user_agent="sao_luis_road_project", timeout=10)

# Colunas de resultado
seg["lat_mid"]        = seg["lat_start"]  # padrão: manter original
seg["lon_mid"]        = seg["lon_start"]
seg["geocode_status"] = "falhou_manter_original"
seg["geocode_source"] = "manual_estimado"

n_ok   = 0
n_fail = 0

for idx, row in seg.iterrows():
    numero = idx + 1
    nome_rua = str(row["street_name"]).strip()
    bairro   = str(row["bairro"]).strip()
    query    = f"{nome_rua}, {bairro}, São Luís, Maranhão, Brasil"

    location = None
    try:
        location = geolocator.geocode(query)
    except GeocoderTimedOut:
        print(f"  [{numero:02d}/{total}] {nome_rua}, {bairro} → TIMEOUT ⚠")
    except GeocoderServiceError as e:
        print(f"  [{numero:02d}/{total}] {nome_rua}, {bairro} → ERRO SERVICO: {e} ⚠")

    if location:
        lat = round(location.latitude,  6)
        lon = round(location.longitude, 6)

        seg.at[idx, "lat_mid"]        = lat
        seg.at[idx, "lon_mid"]        = lon
        seg.at[idx, "lat_start"]      = round(lat - 0.0009, 6)
        seg.at[idx, "lat_end"]        = round(lat + 0.0009, 6)
        seg.at[idx, "lon_start"]      = round(lon - 0.0009, 6)
        seg.at[idx, "lon_end"]        = round(lon + 0.0009, 6)
        seg.at[idx, "geocode_status"] = "ok"
        seg.at[idx, "geocode_source"] = "nominatim_osm"
        n_ok += 1
        print(f"  [{numero:02d}/{total}] {nome_rua}, {bairro}"
              f" → lat={lat:.4f}, lon={lon:.4f} OK")
    else:
        n_fail += 1
        print(f"  [{numero:02d}/{total}] {nome_rua}, {bairro} → FALHOU (mantendo original) AV")

    # Delay obrigatório para respeitar rate limit do OSM
    time.sleep(1)

# ═══════════════════════════════════════════════════════════
# PASSO 3 — Validar coordenadas
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 3 — Validação de coordenadas")
print("="*60)

# Calcular lat_mid e lon_mid para linhas que ainda não têm
# (segmentos que falharam mantêm os valores originais calculados)
mask_fail = seg["geocode_status"] == "falhou_manter_original"
if mask_fail.any():
    seg.loc[mask_fail, "lat_mid"] = (
        (seg.loc[mask_fail, "lat_start"] + seg.loc[mask_fail, "lat_end"]) / 2
    ).round(6)
    seg.loc[mask_fail, "lon_mid"] = (
        (seg.loc[mask_fail, "lon_start"] + seg.loc[mask_fail, "lon_end"]) / 2
    ).round(6)

alertas = 0
for _, row in seg.iterrows():
    lat = row["lat_mid"]
    lon = row["lon_mid"]
    fora_lat = not (LAT_MIN <= lat <= LAT_MAX)
    fora_lon = not (LON_MIN <= lon <= LON_MAX)
    if fora_lat or fora_lon:
        alertas += 1
        print(f"  [ALERTA] {row['segment_id']} | {row['street_name']}"
              f" → lat={lat:.5f}, lon={lon:.5f} FORA DOS LIMITES DE SAO LUIS!")
    else:
        print(f"  [OK] {row['segment_id']} | lat={lat:.5f}, lon={lon:.5f}"
              f" ({row['geocode_status']})")

if alertas == 0:
    print(f"\n  Todas as {total} coordenadas dentro dos limites de São Luís.")
else:
    print(f"\n  ATENCAO: {alertas} segmento(s) com coordenadas suspeitas!")

# ═══════════════════════════════════════════════════════════
# PASSO 4 — Salvar 01_road_segments.csv atualizado
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 4 — Salvando arquivos atualizados")
print("="*60)

seg.to_csv(ARQUIVO_SEG, index=False, encoding="utf-8")
print(f"  [OK] 01_road_segments.csv atualizado ({total} linhas)")

# ═══════════════════════════════════════════════════════════
# PASSO 5 — Atualizar 08_analytical_dataset.csv
# ═══════════════════════════════════════════════════════════
ana = pd.read_csv(ARQUIVO_ANA, encoding="utf-8")

# Remover colunas de coordenada antigas se existirem
coord_cols = ["lat_start", "lon_start", "lat_end", "lon_end",
              "lat_mid", "lon_mid", "geocode_status", "geocode_source"]
for c in coord_cols:
    if c in ana.columns:
        ana = ana.drop(columns=[c])

# Merge com as novas coordenadas
novas_coords = seg[["segment_id"] + coord_cols].copy()
ana = ana.merge(novas_coords, on="segment_id", how="left")

ana.to_csv(ARQUIVO_ANA, index=False, encoding="utf-8")
print(f"  [OK] 08_analytical_dataset.csv atualizado com novas coordenadas")
print(f"       Colunas adicionadas: {coord_cols}")

# ═══════════════════════════════════════════════════════════
# RESUMO FINAL
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RESUMO FINAL")
print("="*60)
print(f"  Geocodificados com sucesso : {n_ok}/{total}")
print(f"  Mantidos (falha)           : {n_fail}/{total}")
print(f"  Alertas de range           : {alertas}")
print(f"\n  Arquivos gerados:")
print(f"    • {ARQUIVO_BACKUP}  (backup original)")
print(f"    • {ARQUIVO_SEG}     (coordenadas corrigidas)")
print(f"    • {ARQUIVO_ANA}     (analytical dataset atualizado)")
print()

# Tabela final com todas as coordenadas
print("  Coordenadas finais por segmento:")
print(f"  {'ID':<12} {'lat_mid':>10} {'lon_mid':>11} {'status':<30} {'rua'}")
print("  " + "-"*85)
for _, r in seg.iterrows():
    print(f"  {r['segment_id']:<12} {r['lat_mid']:>10.5f} {r['lon_mid']:>11.5f}"
          f"  {r['geocode_status']:<28}  {r['street_name']}")
