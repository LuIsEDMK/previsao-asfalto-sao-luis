"""
Projeto: Preditor de Falhas em Vias - Sao Luis MA (v2 - Dados Publicos)
Tarefa 4: Zonas de Alagamento via CEMADEN / OSMnx
Saida:    03_flood_zones.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time, requests, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import osmnx as ox

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"
CACHE_DIR= f"{BASE_DIR}\\cache_osmnx"
ENTRADA  = f"{CSV_DIR}\\01_road_network_osmnx.csv"
SAIDA    = f"{CSV_DIR}\\03_flood_zones.csv"

ox.settings.use_cache    = True
ox.settings.cache_folder = CACHE_DIR
ox.settings.log_console  = False

CIDADES_NOME = {
    "sao_luis":  "São Luís, Maranhão, Brasil",
    "fortaleza": "Fortaleza, Ceará, Brasil",
    "recife":    "Recife, Pernambuco, Brasil",
}

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─── Método 4A: CEMADEN ───────────────────────────────────────────

def tentar_cemaden(cidade_id):
    """Tenta obter dados de risco CEMADEN para a cidade."""
    if cidade_id != "sao_luis":
        return None
    url = ("http://sjc.salvar.cemaden.gov.br/resources/graficos/"
           "municipio/getRiscoMunicipio.php?uf=MA&municipio=São Luís")
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 100:
            print(f"  [CEMADEN] Dados obtidos: {len(r.content)} bytes")
            return r.json() if r.headers.get("content-type","").startswith("application/json") else None
    except Exception as e:
        print(f"  [CEMADEN] Indisponível: {e}")
    return None

# ─── Método 4B: Feições de água via OSMnx ────────────────────────

def baixar_agua_osm(cidade_id, cidade_nome):
    """Baixa corpos d'água do OpenStreetMap para a cidade."""
    cache_file = f"{CACHE_DIR}/water_{cidade_id}.gpkg"

    if os.path.exists(cache_file):
        print(f"  [CACHE] Carregando agua de {cache_file}")
        try:
            return gpd.read_file(cache_file)
        except Exception:
            pass

    print(f"  [OSM] Baixando corpos d agua para {cidade_id}...")
    gdfs = []

    # Rios, lagos, zonas úmidas
    for tags in [
        {"natural": ["water", "wetland"]},
        {"waterway": ["river", "stream", "canal", "drain"]},
        {"landuse": "basin"},
    ]:
        try:
            gdf = ox.features_from_place(cidade_nome, tags=tags)
            if not gdf.empty:
                gdfs.append(gdf[["geometry"]].copy())
                print(f"  {list(tags.keys())[0]}: {len(gdf)} feições")
            time.sleep(1)
        except Exception as e:
            print(f"  [AV] {tags}: {e}")
            continue

    if not gdfs:
        print(f"  [AV] Nenhum corpo d agua encontrado para {cidade_id}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf_agua = pd.concat(gdfs, ignore_index=True)
    gdf_agua = gdf_agua[gdf_agua.geometry.notna()].copy()
    gdf_agua.crs = "EPSG:4326"
    gdf_agua.to_file(cache_file, driver="GPKG")
    print(f"  [OK] {len(gdf_agua)} feições de agua salvas")
    return gdf_agua

# ─── Calcular distância ao corpo d'água mais próximo ─────────────

def calcular_dist_agua(lats, lons, gdf_agua):
    """
    Para uma lista de pontos (lat, lon), retorna a distância mínima
    ao corpo d'água mais próximo em metros.
    Usa projeção UTM para distâncias métricas.
    """
    if gdf_agua.empty:
        return np.full(len(lats), 9999.0)

    # Criar GeoDataFrame dos pontos de segmentos
    geometrias = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    gdf_pts    = gpd.GeoDataFrame({"geometry": geometrias}, crs="EPSG:4326")

    # Reprojetar para UTM (zona 23S para NE do Brasil)
    try:
        gdf_pts_utm  = gdf_pts.to_crs("EPSG:31983")
        gdf_agua_utm = gdf_agua.to_crs("EPSG:31983")
    except Exception:
        gdf_pts_utm  = gdf_pts.to_crs("+proj=utm +zone=23 +south +datum=WGS84")
        gdf_agua_utm = gdf_agua.to_crs("+proj=utm +zone=23 +south +datum=WGS84")

    # Unir todos os corpos d'água em um único objeto para cálculo rápido
    agua_unida = gdf_agua_utm.geometry.union_all()
    if agua_unida is None or agua_unida.is_empty:
        return np.full(len(lats), 9999.0)

    distancias = gdf_pts_utm.geometry.distance(agua_unida).values
    return np.round(distancias, 1)

# ─── Classificar flood_zone a partir da distância ────────────────

def dist_para_flood_zone(dist_m):
    if dist_m < 100:
        return 1.0
    elif dist_m < 300:
        return 0.5
    else:
        return 0.0

# ─── EXECUÇÃO PRINCIPAL ───────────────────────────────────────────

sep("TAREFA 4 — Zonas de Alagamento (CEMADEN / OSMnx)")

if os.path.exists(SAIDA):
    df_ex = pd.read_csv(SAIDA)
    print(f"  [INFO] Arquivo ja existe com {len(df_ex):,} linhas. Nada a fazer.")
    print(df_ex.groupby("cidade")[["dist_water_m","flood_zone_final"]].mean().round(2).to_string())
    sys.exit(0)

sep("Carregando rede viaria")
df = pd.read_csv(ENTRADA)
print(f"  Segmentos: {len(df):,}")

registros_finais = []

for cidade_id, cidade_nome in CIDADES_NOME.items():
    sep(f"Processando {cidade_id}")
    df_cid = df[df["cidade"] == cidade_id].copy()
    print(f"  {len(df_cid):,} segmentos")

    # Tentar CEMADEN (só São Luís)
    cemaden_data = tentar_cemaden(cidade_id)
    flood_cemaden = 0.0  # padrão: sem dado CEMADEN

    # Baixar água OSM
    gdf_agua = baixar_agua_osm(cidade_id, cidade_nome)

    # Calcular distâncias
    print(f"  Calculando distancias ao corpo d agua mais proximo...")
    t0 = time.time()
    distancias = calcular_dist_agua(
        df_cid["lat_mid"].values,
        df_cid["lon_mid"].values,
        gdf_agua
    )
    elapsed = round(time.time() - t0, 1)
    print(f"  Calculado em {elapsed}s")
    print(f"  Distancia media: {distancias.mean():.0f}m | min: {distancias.min():.0f}m")

    for i, (_, row) in enumerate(df_cid.iterrows()):
        dist = float(distancias[i]) if i < len(distancias) else 9999.0
        fz_osm   = dist_para_flood_zone(dist)
        fz_final = max(flood_cemaden, fz_osm)

        registros_finais.append({
            "segment_id":     row["segment_id"],
            "cidade":         cidade_id,
            "dist_water_m":   dist,
            "flood_zone_osm": fz_osm,
            "flood_zone_final": fz_final,
            "flood_source":   "cemaden+osm" if cemaden_data else "osm_only",
        })

    n_alto = sum(1 for d in distancias if d < 100)
    n_med  = sum(1 for d in distancias if 100 <= d < 300)
    print(f"  Alto risco (<100m):  {n_alto:,} ({n_alto/len(distancias)*100:.1f}%)")
    print(f"  Med risco (100-300m):{n_med:,} ({n_med/len(distancias)*100:.1f}%)")

sep("Salvando")
df_out = pd.DataFrame(registros_finais)
df_out.to_csv(SAIDA, index=False, encoding="utf-8")

print(f"  [OK] {SAIDA}")
print(f"  Total: {len(df_out):,} segmentos")
print()
print("  Por cidade:")
print(df_out.groupby("cidade")[["dist_water_m","flood_zone_final"]].mean().round(2).to_string())
print()
print("  TAREFA 4 CONCLUIDA")
