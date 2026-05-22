"""
Projeto: Preditor de Falhas em Vias - Sao Luis MA (v2 - Dados Publicos)
Tarefa 2: Baixar rede viaria real via OSMnx
Saida:    01_road_network_osmnx.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd

import osmnx as ox

# ─── Diretórios ───────────────────────────────────────────────────
BASE_DIR  = r"w:\projetos vscode\projetos para prefeitura"
COD_DIR   = f"{BASE_DIR}\\1projeto\\codigos"
CSV_DIR   = f"{BASE_DIR}\\csv"
CACHE_DIR = f"{BASE_DIR}\\cache_osmnx"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CSV_DIR,   exist_ok=True)

# Configurar cache do osmnx
ox.settings.use_cache    = True
ox.settings.cache_folder = CACHE_DIR
ox.settings.log_console  = False

SAIDA = f"{CSV_DIR}\\01_road_network_osmnx.csv"

# ─── Cidades ──────────────────────────────────────────────────────
CIDADES = {
    "sao_luis":  "São Luís, Maranhão, Brasil",
    "fortaleza": "Fortaleza, Ceará, Brasil",
    "recife":    "Recife, Pernambuco, Brasil",
}

# ─── Tabelas de imputação por tipo de via ─────────────────────────
SPEED_IMPUT = {
    "motorway": 80, "motorway_link": 60, "trunk": 70, "trunk_link": 60,
    "primary": 60, "primary_link": 50, "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30, "residential": 30,
    "living_street": 20, "unclassified": 30, "road": 30,
}
LANES_IMPUT = {
    "motorway": 3, "motorway_link": 2, "trunk": 3, "trunk_link": 2,
    "primary": 2, "primary_link": 1, "secondary": 2, "secondary_link": 1,
    "tertiary": 1, "tertiary_link": 1, "residential": 1,
    "living_street": 1, "unclassified": 1, "road": 1,
}

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─── Funções de limpeza ───────────────────────────────────────────

def normalizar_highway(val):
    """Retorna o tipo principal de via (primeiro item se for lista)."""
    if isinstance(val, list):
        val = val[0] if val else "unclassified"
    s = str(val).strip().lower()
    # Mapear variantes para tipo canônico
    for base in ["motorway", "trunk", "primary", "secondary",
                 "tertiary", "residential", "living_street"]:
        if s.startswith(base):
            return s.split("_link")[0] if "_link" in s else s
    return "unclassified"

def limpar_speed(val, highway):
    """Converte maxspeed para km/h inteiro."""
    if val is None or val != val:   # None ou NaN
        return SPEED_IMPUT.get(highway, 30)
    if isinstance(val, list):
        val = val[0]
    s = str(val).strip()
    try:
        # Remover unidades e pegar primeiro valor de lista separada por ';'
        s = s.split(";")[0].split(",")[0].strip()
        v = float(s.replace(" mph", "").replace("mph", "")
                   .replace(" km/h", "").replace("km/h", ""))
        if "mph" in str(val).lower():
            v = v * 1.60934
        return max(10, min(150, round(v)))
    except (ValueError, AttributeError):
        return SPEED_IMPUT.get(highway, 30)

def limpar_lanes(val, highway):
    """Converte lanes para int."""
    if val is None or val != val:
        return LANES_IMPUT.get(highway, 1)
    if isinstance(val, list):
        val = val[0]
    try:
        return max(1, min(8, int(str(val).split(";")[0].strip())))
    except (ValueError, AttributeError):
        return LANES_IMPUT.get(highway, 1)

def limpar_nome(val):
    """Extrai string de nome da via (pode ser lista)."""
    if val is None or val != val:
        return ""
    if isinstance(val, list):
        val = val[0] if val else ""
    return str(val).strip() if val else ""

def calcular_load_proxy(highway, lanes, speed):
    """
    Proxy de carga de tráfego (1-10) baseado em tipo de via, faixas e velocidade.
    """
    hw_score = {
        "motorway": 5, "trunk": 5, "primary": 4,
        "secondary": 3, "tertiary": 2,
        "residential": 1, "living_street": 1, "unclassified": 1,
    }.get(highway, 1)

    lane_score = 3 if lanes >= 4 else (2 if lanes == 3 else (1 if lanes == 2 else 0))
    speed_score = 2 if speed >= 60 else (1 if speed >= 40 else 0)

    return max(1, min(10, hw_score + lane_score + speed_score))

# ─── Processar edges de uma cidade ────────────────────────────────

def processar_edges(edges_gdf, cidade_id):
    """Extrai features de cada aresta do grafo viário."""
    registros = []
    erros = 0

    for idx, edge in edges_gdf.iterrows():
        try:
            geom = edge.get("geometry")
            if geom is None or geom.is_empty:
                continue

            # Coordenadas do segmento
            centroid  = geom.centroid
            lat_mid   = round(centroid.y, 6)
            lon_mid   = round(centroid.x, 6)
            coords    = list(geom.coords)
            lat_start = round(coords[0][1], 6)
            lon_start = round(coords[0][0], 6)
            lat_end   = round(coords[-1][1], 6)
            lon_end   = round(coords[-1][0], 6)

            # osmid único: combinar nó-origem + nó-destino + chave de aresta
            if isinstance(idx, tuple) and len(idx) >= 2:
                osmid = f"{idx[0]}-{idx[1]}-{idx[2] if len(idx) > 2 else 0}"
            elif "u" in edge.index and "v" in edge.index:
                osmid = f"{edge['u']}-{edge['v']}-{edge.get('key', 0)}"
            else:
                osmid = str(idx)

            # Features de tipo de via
            hw_raw  = edge.get("highway", "unclassified")
            highway = normalizar_highway(hw_raw)
            speed   = limpar_speed(edge.get("maxspeed"), highway)
            lanes   = limpar_lanes(edge.get("lanes"), highway)
            oneway  = int(bool(edge.get("oneway", False)))
            length  = round(float(edge.get("length", 0)), 1)
            nome    = limpar_nome(edge.get("name", ""))

            registros.append({
                "segment_id":      f"OSM-{osmid}",
                "cidade":          cidade_id,
                "street_name":     nome,
                "highway_type":    highway,
                "length_m":        length,
                "maxspeed_kmh":    speed,
                "lanes":           lanes,
                "oneway":          oneway,
                "lat_mid":         lat_mid,
                "lon_mid":         lon_mid,
                "lat_start":       lat_start,
                "lon_start":       lon_start,
                "lat_end":         lat_end,
                "lon_end":         lon_end,
                "load_proxy_1_10": calcular_load_proxy(highway, lanes, speed),
            })
        except Exception:
            erros += 1
            continue

    if erros > 0:
        print(f"  [AV] {erros} segmentos ignorados por erro de geometria")
    return pd.DataFrame(registros)

# ─── Download por cidade ───────────────────────────────────────────

def baixar_cidade(cidade_id, cidade_nome):
    """Baixa rede viária de uma cidade com cache local."""
    cache_gpkg = f"{CACHE_DIR}/network_{cidade_id}.gpkg"

    if os.path.exists(cache_gpkg):
        print(f"  [CACHE] Carregando de {cache_gpkg}")
        try:
            edges = gpd.read_file(cache_gpkg)
            print(f"  [OK] {len(edges)} arestas carregadas do cache")
            return edges
        except Exception as e:
            print(f"  [AV] Cache corrompido ({e}), re-baixando...")

    print(f"  [DOWNLOAD] Baixando via OSMnx (pode levar alguns minutos)...")
    t0 = time.time()

    G = ox.graph_from_place(cidade_nome, network_type="drive",
                             simplify=True, retain_all=False)
    _, edges = ox.graph_to_gdfs(G)
    elapsed = round(time.time() - t0, 1)
    print(f"  [OK] Download em {elapsed}s — {len(edges)} arestas")

    print(f"  [SAVE] Salvando cache: {cache_gpkg}")
    edges.to_file(cache_gpkg, driver="GPKG")

    return edges

# ─── EXECUÇÃO PRINCIPAL ───────────────────────────────────────────

sep("TAREFA 2 — Rede Viaria Real via OSMnx")
print(f"  osmnx {ox.__version__}")
print(f"  Cache: {CACHE_DIR}")
print(f"  Saida: {SAIDA}")

# Verificar se output já existe
if os.path.exists(SAIDA):
    print(f"\n  [INFO] Arquivo de saida ja existe. Carregando para verificar...")
    df_existente = pd.read_csv(SAIDA)
    print(f"  Linhas existentes: {len(df_existente):,}")
    print(f"  Cidades: {df_existente['cidade'].value_counts().to_dict()}")
    print(f"\n  Para re-baixar, delete {SAIDA} e rode novamente.")
    # Continuar para garantir que está completo
    cidades_existentes = set(df_existente["cidade"].unique())
    cidades_faltando = set(CIDADES.keys()) - cidades_existentes
    if not cidades_faltando:
        print(f"  Todas as cidades presentes. Nada a fazer.")
        # Imprimir resumo e sair
        sep("RESUMO")
        for c in CIDADES:
            n = (df_existente["cidade"] == c).sum()
            print(f"  {c:12s}: {n:,} segmentos")
        print(f"  {'TOTAL':12s}: {len(df_existente):,} segmentos")
        sys.exit(0)
    else:
        print(f"  Cidades faltando: {cidades_faltando}. Baixando...")
        todos_dfs = [df_existente]
else:
    todos_dfs = []
    cidades_faltando = set(CIDADES.keys())

# Baixar cidades que faltam
for cidade_id, cidade_nome in CIDADES.items():
    if cidade_id not in cidades_faltando:
        continue

    sep(f"{cidade_id.upper()}: {cidade_nome}")

    try:
        edges = baixar_cidade(cidade_id, cidade_nome)
        df_cidade = processar_edges(edges, cidade_id)
        print(f"  Processados: {len(df_cidade):,} segmentos")

        # Amostra de tipos de via
        hw_counts = df_cidade["highway_type"].value_counts().head(5)
        print(f"  Tipos de via mais comuns:\n{hw_counts.to_string()}")

        todos_dfs.append(df_cidade)

    except Exception as e:
        print(f"  [ERRO] {cidade_id}: {e}")
        print(f"  Pulando {cidade_id} e continuando...")
        continue

# ─── Consolidar e salvar ──────────────────────────────────────────

if not todos_dfs:
    print("\n[ERRO FATAL] Nenhuma cidade baixada com sucesso.")
    sys.exit(1)

sep("Consolidando e salvando")

df_final = pd.concat(todos_dfs, ignore_index=True)

# Remover duplicatas (osmid pode repetir entre reintentos)
n_antes = len(df_final)
df_final = df_final.drop_duplicates(subset=["segment_id", "cidade"]).reset_index(drop=True)
n_dedup  = n_antes - len(df_final)
if n_dedup > 0:
    print(f"  {n_dedup} duplicatas removidas")

df_final.to_csv(SAIDA, index=False, encoding="utf-8")
print(f"  [OK] {SAIDA}")
print(f"  Colunas: {list(df_final.columns)}")

sep("RESUMO FINAL — TAREFA 2")
for cidade_id in CIDADES:
    n = (df_final["cidade"] == cidade_id).sum()
    print(f"  {cidade_id:12s}: {n:,} segmentos")
print(f"  {'TOTAL':12s}: {len(df_final):,} segmentos")
print(f"\n  Distribuicao por tipo de via:")
print(df_final["highway_type"].value_counts().head(8).to_string())
print(f"\n  load_proxy_1_10 media por cidade:")
print(df_final.groupby("cidade")["load_proxy_1_10"].mean().round(2).to_string())
print(f"\n  TAREFA 2 CONCLUIDA")
