"""
Projeto: Preditor de Falhas em Vias - São Luís / MA
Etapa:   Coleta, Limpeza e Transformação de Dados
Autor:   Data Science - Prefeitura de São Luís
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8")

CSV_DIR = r"w:\projetos vscode\projetos para prefeitura\csv"
OUT_DIR = r"w:\projetos vscode\projetos para prefeitura\1projeto\codigos"
DATA_DIR = CSV_DIR + "\\"

# UTILIDADES

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)

def info(msg):   print(f"  ✔  {msg}")
def warn(msg):   print(f"  ⚠  {msg}")
def issue(msg):  print(f"  ✘  {msg}")

def quality_report(df, name):
    """Gera relatório de qualidade de dados."""
    print(f"\n  [{name}]  {df.shape[0]} linhas × {df.shape[1]} colunas")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        info("Sem valores nulos")
    else:
        for col, n in missing.items():
            pct = n / len(df) * 100
            warn(f"Nulos em '{col}': {n} ({pct:.1f}%)")
    dups = df.duplicated().sum()
    if dups:
        issue(f"{dups} linhas duplicadas encontradas")
    else:
        info("Sem duplicatas")

# ═══════════════════════════════════════════════════════════
# PASSO 1 — CARREGAR DADOS BRUTOS
# ═══════════════════════════════════════════════════════════

segments   = pd.read_csv(f"{CSV_DIR}/01_road_segments.csv")
repairs    = pd.read_csv(f"{CSV_DIR}/02_repair_history.csv")
rainfall   = pd.read_csv(f"{CSV_DIR}/03_rainfall_data.csv")
traffic    = pd.read_csv(f"{CSV_DIR}/04_traffic_load.csv")
complaints = pd.read_csv(f"{CSV_DIR}/05_citizen_complaints.csv")

# ═══════════════════════════════════════════════════════════
# PASSO 2 — DIAGNÓSTICO DE QUALIDADE
# ═══════════════════════════════════════════════════════════

quality_report(segments,   "01_road_segments")
quality_report(repairs,    "02_repair_history")
quality_report(rainfall,   "03_rainfall_data")
quality_report(traffic,    "04_traffic_load")
quality_report(complaints, "05_citizen_complaints")

# ═══════════════════════════════════════════════════════════
# PASSO 3 — LIMPEZA: SEGMENTOS
# ═══════════════════════════════════════════════════════════

seg = segments.copy()

# Padronizar strings
for col in ["street_name", "bairro", "zone", "pavement_type", "road_class"]:
    seg[col] = seg[col].str.strip().str.lower()

# Corrigir tipo de pavimento (normalizar variações de escrita)
pav_map = {
    "asfalto": "asfalto",
    "asphalt": "asfalto",
    "paralelepipedo": "paralelepipedo",
    "paralelepípedo": "paralelepipedo",
    "terra": "terra",
    "concreto": "concreto",
}
seg["pavement_type"] = seg["pavement_type"].map(pav_map).fillna("desconhecido")

# Converter booleanos (sim/nao → 1/0)
for col in ["truck_route", "bus_route", "flood_zone"]:
    seg[col] = seg[col].str.lower().map({"sim": 1, "nao": 0, "yes": 1, "no": 0})

# Validar coordenadas (São Luís: lat ~-2.4 a -2.6 / lon ~-44.1 a -44.4)
lat_ok = seg["lat_start"].between(-2.7, -2.3) & seg["lat_end"].between(-2.7, -2.3)
lon_ok = seg["lon_start"].between(-44.5, -44.0) & seg["lon_end"].between(-44.5, -44.0)
invalid_coords = seg[~(lat_ok & lon_ok)]
if len(invalid_coords):
    issue(f"{len(invalid_coords)} segmentos com coordenadas fora de São Luís — verificar!")
else:
    info("Todas as coordenadas dentro dos limites de São Luís")

# Calcular idade da pavimentação
current_year = datetime.now().year
seg["pavement_age_years"] = current_year - seg["pavement_year"]
seg["years_since_overlay"] = seg["last_overlay_year"].apply(
    lambda y: current_year - y if pd.notna(y) else np.nan
)

# Codificar tipo de pavimento (para o modelo)
pav_code = {"asfalto": 1, "paralelepipedo": 2, "terra": 3, "concreto": 4, "desconhecido": 0}
seg["pavement_type_code"] = seg["pavement_type"].map(pav_code)

# ═══════════════════════════════════════════════════════════
# PASSO 3B — LIMPEZA: HISTÓRICO DE REPAROS
# ═══════════════════════════════════════════════════════════

rep = repairs.copy()

# Converter datas
rep["order_date"]      = pd.to_datetime(rep["order_date"],      errors="coerce")
rep["completion_date"] = pd.to_datetime(rep["completion_date"], errors="coerce")

# Detectar datas inválidas
invalid_dates = rep["order_date"].isna().sum()
if invalid_dates:
    issue(f"{invalid_dates} ordens com data inválida — removendo")
    rep = rep.dropna(subset=["order_date"])

# Calcular duração real do reparo
rep["repair_duration_days"] = (rep["completion_date"] - rep["order_date"]).dt.days

# Detectar durações absurdas (negativas ou > 60 dias)
outliers = rep[(rep["repair_duration_days"] < 0) | (rep["repair_duration_days"] > 60)]
if len(outliers):
    warn(f"{len(outliers)} reparos com duração suspeita (negativa ou >60 dias)")
    rep.loc[rep["repair_duration_days"] < 0, "repair_duration_days"] = np.nan

# Remover duplicatas exatas
before = len(rep)
rep = rep.drop_duplicates(subset=["segment_id", "order_date", "repair_type"])
removed = before - len(rep)
if removed:
    warn(f"{removed} reparos duplicados removidos")
else:
    info("Sem duplicatas em reparos")

# Padronizar tipo de reparo
rep["repair_type"] = rep["repair_type"].str.strip().str.lower()
rep["failure_type"] = rep["failure_type"].str.strip().str.lower()
rep["priority"]    = rep["priority"].str.strip().str.lower()

# Extrair ano/mês para joins temporais
rep["repair_year"]  = rep["order_date"].dt.year
rep["repair_month"] = rep["order_date"].dt.month

# Custo nulo → imputar com mediana do tipo de reparo
median_cost = rep.groupby("repair_type")["repair_cost_brl"].median()
rep["repair_cost_brl"] = rep.apply(
    lambda row: median_cost.get(row["repair_type"], rep["repair_cost_brl"].median())
    if pd.isna(row["repair_cost_brl"]) else row["repair_cost_brl"],
    axis=1
)

# ═══════════════════════════════════════════════════════════
# PASSO 3C — LIMPEZA: DADOS DE CHUVA
# ═══════════════════════════════════════════════════════════

rain = rainfall.copy()

# Validar valores de chuva (São Luís: 0–600mm/mês é razoável)
rain_outliers = rain[rain["rainfall_mm"] > 600]
if len(rain_outliers):
    warn(f"{len(rain_outliers)} meses com chuva >600mm — verificar sensor")

rain_negative = rain[rain["rainfall_mm"] < 0]
if len(rain_negative):
    issue(f"{len(rain_negative)} registros com chuva negativa — corrigindo para 0")
    rain.loc[rain["rainfall_mm"] < 0, "rainfall_mm"] = 0

# Criar coluna de data para ordenação e cálculos
rain["date"] = pd.to_datetime(rain[["year", "month"]].assign(day=1))
rain = rain.sort_values("date").reset_index(drop=True)

# Calcular precipitações acumuladas (janelas temporais)
rain["rainfall_30d"]  = rain["rainfall_mm"]
rain["rainfall_90d"]  = rain["rainfall_mm"].rolling(3,  min_periods=1).sum()
rain["rainfall_180d"] = rain["rainfall_mm"].rolling(6,  min_periods=1).sum()
rain["rainfall_365d"] = rain["rainfall_mm"].rolling(12, min_periods=1).sum()

# Identificar meses críticos (top quartil de chuva)
q75 = rain["rainfall_mm"].quantile(0.75)
rain["is_critical_month"] = (rain["rainfall_mm"] >= q75).astype(int)

# Preencher meses faltantes com interpolação linear
full_dates = pd.date_range(rain["date"].min(), rain["date"].max(), freq="MS")
rain = rain.set_index("date").reindex(full_dates)
numeric_cols = ["rainfall_mm", "rainy_days", "max_daily_mm",
                "extreme_events_50mm", "rainfall_30d",
                "rainfall_90d", "rainfall_180d", "rainfall_365d"]
rain[numeric_cols] = rain[numeric_cols].interpolate(method="linear")
rain = rain.reset_index().rename(columns={"index": "date"})
rain["year"]  = rain["date"].dt.year
rain["month"] = rain["date"].dt.month

# ═══════════════════════════════════════════════════════════
# PASSO 3D — LIMPEZA: TRÁFEGO
# ═══════════════════════════════════════════════════════════

traf = traffic.copy()

# Validar percentual de veículos pesados (0–100%)
invalid_pct = traf[~traf["heavy_vehicles_pct"].between(0, 100)]
if len(invalid_pct):
    issue(f"{len(invalid_pct)} registros com % de pesados inválido")

# Preencher load_index nulo com estimativa baseada em volume + pesados
mask = traf["load_index_1_10"].isna()
if mask.sum():
    traf.loc[mask, "load_index_1_10"] = (
        (traf.loc[mask, "avg_daily_vehicles"] / 2000 +
         traf.loc[mask, "heavy_vehicles_pct"] / 5)
        .clip(1, 10)
        .round()
    )
    warn(f"{mask.sum()} load_index imputados por fórmula")

# Para segmentos sem dado de tráfego, usar mediana por road_class
traf_latest = traf.sort_values("measurement_year").groupby("segment_id").last().reset_index()

# Merge com road_class para imputação
traf_latest = traf_latest.merge(
    seg[["segment_id", "road_class"]], on="segment_id", how="left"
)
median_load_by_class = traf_latest.groupby("road_class")["load_index_1_10"].median()

# ═══════════════════════════════════════════════════════════
# PASSO 3E — LIMPEZA: RECLAMAÇÕES
# ═══════════════════════════════════════════════════════════

comp = complaints.copy()

comp["report_date"]      = pd.to_datetime(comp["report_date"],      errors="coerce")
comp["resolution_date"]  = pd.to_datetime(comp["resolution_date"],  errors="coerce")

# Remover duplicatas (mesmo segmento, mesma data, mesmo tipo)
before = len(comp)
comp = comp.drop_duplicates(subset=["segment_id", "report_date", "complaint_type"])
removed = before - len(comp)
if removed:
    warn(f"{removed} reclamações duplicadas removidas (mesmo segmento/dia/tipo)")

# Padronizar severidade
sev_map = {"leve": 1, "moderado": 2, "grave": 3, "muito_grave": 4}
comp["severity_code"] = comp["severity_reported"].str.lower().map(sev_map).fillna(2)

# Preencher days_to_resolve calculando quando possível
mask = comp["days_to_resolve"].isna() & comp["resolution_date"].notna()
comp.loc[mask, "days_to_resolve"] = (
    (comp.loc[mask, "resolution_date"] - comp.loc[mask, "report_date"]).dt.days
)

comp["report_year"]  = comp["report_date"].dt.year
comp["report_month"] = comp["report_date"].dt.month

# ═══════════════════════════════════════════════════════════
# PASSO 4 — ENGENHARIA DE FEATURES
# ═══════════════════════════════════════════════════════════

# ── 4A: Agregações de reparo por segmento ──────────────────
rep_agg = rep.groupby("segment_id").agg(
    total_repairs          = ("repair_id",         "count"),
    total_repair_cost      = ("repair_cost_brl",   "sum"),
    avg_repair_cost        = ("repair_cost_brl",   "mean"),
    last_repair_year       = ("repair_year",        "max"),
    last_repair_month      = ("repair_month",       lambda x: rep.loc[x.index, "repair_month"].iloc[-1]),
    emergency_repairs      = ("priority",           lambda x: (x == "urgente").sum()),
    buraco_count           = ("failure_type",       lambda x: (x == "buraco").sum()),
    afundamento_count      = ("failure_type",       lambda x: (x == "afundamento").sum()),
).reset_index()

rep_agg["years_since_last_repair"] = current_year - rep_agg["last_repair_year"]
rep_agg["recurrence_rate"]         = rep_agg["total_repairs"] / (
    (current_year - 2019) + 1
)  # reparos por ano

# ── 4B: Agregações de reclamações por segmento ─────────────
comp_agg = comp.groupby("segment_id").agg(
    total_complaints       = ("complaint_id",   "count"),
    avg_severity           = ("severity_code",  "mean"),
    muito_grave_count      = ("severity_code",  lambda x: (x == 4).sum()),
    unique_years           = ("report_year",    "nunique"),
).reset_index()

# ── 4C: Features de chuva sazonais ─────────────────────────
rain_season = rain.copy()
rain_season["is_rainy_season"] = rain_season["month"].between(1, 6).astype(int)

avg_rainy    = rain_season[rain_season["is_rainy_season"] == 1]["rainfall_mm"].mean()
avg_dry      = rain_season[rain_season["is_rainy_season"] == 0]["rainfall_mm"].mean()
rain_ratio   = avg_rainy / avg_dry if avg_dry > 0 else 0

# ═══════════════════════════════════════════════════════════
# PASSO 5 — MONTAR DATASET FINAL
# ═══════════════════════════════════════════════════════════

# Base: todos os segmentos
final = seg[[
    "segment_id", "street_name", "bairro", "zone",
    "pavement_type", "pavement_type_code", "pavement_age_years",
    "years_since_overlay", "road_class",
    "flood_zone", "truck_route", "bus_route"
]].copy()

# Merge: tráfego
final = final.merge(
    traf_latest[["segment_id", "load_index_1_10", "avg_daily_vehicles",
                 "heavy_vehicles_pct", "traffic_class"]],
    on="segment_id", how="left"
)

# Imputar load_index faltante pela mediana da road_class
final["load_index_1_10"] = final.apply(
    lambda row: median_load_by_class.get(row["road_class"], 5)
    if pd.isna(row["load_index_1_10"]) else row["load_index_1_10"],
    axis=1
)

# Merge: histórico de reparos
final = final.merge(rep_agg, on="segment_id", how="left")
final["total_repairs"]          = final["total_repairs"].fillna(0).astype(int)
final["total_repair_cost"]      = final["total_repair_cost"].fillna(0)
final["emergency_repairs"]      = final["emergency_repairs"].fillna(0).astype(int)
final["buraco_count"]           = final["buraco_count"].fillna(0).astype(int)
final["afundamento_count"]      = final["afundamento_count"].fillna(0).astype(int)
final["recurrence_rate"]        = final["recurrence_rate"].fillna(0)
final["years_since_last_repair"]= final["years_since_last_repair"].fillna(
    final["pavement_age_years"]  # se nunca reparado, usar idade da pavimentação
)

# Merge: reclamações
final = final.merge(comp_agg, on="segment_id", how="left")
final["total_complaints"]  = final["total_complaints"].fillna(0).astype(int)
final["avg_severity"]      = final["avg_severity"].fillna(0)
final["muito_grave_count"] = final["muito_grave_count"].fillna(0).astype(int)

# Adicionar chuva mais recente disponível
latest_rain = rain.sort_values("date").iloc[-1]
final["current_rainfall_30d"]  = latest_rain["rainfall_30d"]
final["current_rainfall_90d"]  = latest_rain["rainfall_90d"]
final["current_rainfall_180d"] = latest_rain["rainfall_180d"]
final["current_month"]         = latest_rain["month"]
final["is_rainy_season"]       = int(latest_rain["month"] in range(1, 7))

# ── Score de Risco Composto (heurístico, pré-modelo) ───────
# Pesos calibrados para o contexto de São Luís:
# Chuva recente 30% + Idade pavimento 20% + Tráfego 20%
# + Histórico 15% + Reclamações 10% + Baixada 5%
def compute_risk_score(row):
    rain_score    = min(row["current_rainfall_30d"] / 500, 1.0) * 30
    age_score     = min(row["pavement_age_years"] / 30, 1.0) * 20
    traffic_score = min(row["load_index_1_10"] / 10, 1.0) * 20
    history_score = min(row["total_repairs"] / 5, 1.0) * 15
    complaint_score = min(row["total_complaints"] / 5, 1.0) * 10
    flood_score   = row["flood_zone"] * 5
    return round(rain_score + age_score + traffic_score +
                 history_score + complaint_score + flood_score, 1)

final["risk_score"] = final.apply(compute_risk_score, axis=1)

# Categoria de risco
def risk_category(score):
    if score >= 80: return "critico"
    if score >= 60: return "alto"
    if score >= 40: return "medio"
    return "baixo"

final["risk_category"] = final["risk_score"].apply(risk_category)
final["priority_rank"] = final["risk_score"].rank(ascending=False, method="min").fillna(0).astype(int)
final = final.sort_values("risk_score", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════
# PASSO 6 — SALVAR OUTPUTS
# ═══════════════════════════════════════════════════════════

# Dataset analítico completo
final.to_csv(f"{OUT_DIR}/08_analytical_dataset.csv", index=False)

# Top 10 segmentos de risco (para o dashboard)
top10 = final.head(10)[[
    "priority_rank", "segment_id", "street_name", "bairro",
    "risk_score", "risk_category",
    "pavement_age_years", "load_index_1_10", "flood_zone",
    "total_repairs", "total_complaints",
    "current_rainfall_30d"
]]
top10.to_csv(f"{OUT_DIR}/09_top_risk_segments.csv", index=False)

# Relatório de qualidade de dados
quality_log = []
for df, name in [(seg, "segments"), (rep, "repairs"), (rain, "rainfall"),
                 (traf, "traffic"), (comp, "complaints")]:
    for col in df.columns:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            quality_log.append({
                "file": name,
                "column": col,
                "null_count": n_null,
                "null_pct": round(n_null / len(df) * 100, 1)
            })

pd.DataFrame(quality_log).to_csv(f"{OUT_DIR}/10_data_quality_log.csv", index=False)

# Sumário de custos por segmento
cost_summary = rep.groupby("segment_id").agg(
    n_repairs          = ("repair_id",       "count"),
    total_cost_brl     = ("repair_cost_brl", "sum"),
    avg_cost_brl       = ("repair_cost_brl", "mean"),
    max_cost_brl       = ("repair_cost_brl", "max"),
    n_emergency        = ("priority",        lambda x: (x == "urgente").sum()),
).reset_index().merge(seg[["segment_id", "street_name", "bairro"]], on="segment_id")
cost_summary.to_csv(f"{OUT_DIR}/11_cost_summary.csv", index=False)

# ═══════════════════════════════════════════════════════════
# RESUMO EXECUTIVO
# ═══════════════════════════════════════════════════════════

total_spent = rep["repair_cost_brl"].sum()
avg_reactive = rep[rep["priority"] == "urgente"]["repair_cost_brl"].mean()
avg_preventive = rep[rep["priority"].isin(["media", "baixa"])]["repair_cost_brl"].mean()
savings_potential = avg_reactive - avg_preventive if avg_preventive else 0

print(f"""
  Custo total histórico de reparos:   R$ {total_spent:,.2f}
  Custo médio reparo EMERGENCIAL:     R$ {avg_reactive:,.2f}
  Custo médio reparo PREVENTIVO:      R$ {avg_preventive:,.2f}
  Potencial de economia por reparo:   R$ {savings_potential:,.2f}

  Segmento mais problemático:
    {final.iloc[0]['street_name']} ({final.iloc[0]['bairro']})
    Score de risco: {final.iloc[0]['risk_score']} — {final.iloc[0]['risk_category'].upper()}
    Reparos históricos: {final.iloc[0]['total_repairs']}
    Reclamações registradas: {final.iloc[0]['total_complaints']}

  Próximos passos:
    → Carregar 08_analytical_dataset.csv no modelo XGBoost
    → Validar top-10 segmentos com equipes de campo
    → Integrar alerta de chuva do CEMADEN via API
""")

print("  Pipeline concluído com sucesso.\n")

# ═══════════════════════════════════════════════════════════
# TAREFA 2A — validate_csvs()
# Roda checagens automáticas em todos os CSVs do projeto e
# imprime relatório de qualidade resumido.
# ═══════════════════════════════════════════════════════════

def validate_csvs():
    """Valida todos os CSVs do projeto e imprime relatório de qualidade."""

    # Ranges válidos por arquivo/coluna  (min, max)
    RANGES = {
        "01_road_segments":     {"pavement_year":  (1950, 2024),
                                  "length_m":       (50,   2000)},
        "02_repair_history":    {"repair_cost_brl": (100,  500_000)},
        "03_rainfall_data":     {"rainfall_mm":     (0,    700),
                                  "rainy_days":      (0,    31)},
        "04_traffic_load":      {"load_index_1_10":  (1,   10),
                                  "heavy_vehicles_pct": (0, 100)},
        "06_ml_training_dataset": {"LABEL_failed_next_30d": (0, 1),
                                   "LABEL_failed_next_90d": (0, 1)},
        "07_risk_scores":       {"risk_score_0_100": (0, 100)},
    }

    # Tipos obrigatórios por coluna (amostra)
    DTYPES = {
        "segment_id": "object",
        "pavement_year": "float64",
        "rainfall_mm": "float64",
        "risk_score_0_100": "float64",
    }

    arquivos = {
        "01_road_segments":       f"{CSV_DIR}/01_road_segments.csv",
        "02_repair_history":      f"{CSV_DIR}/02_repair_history.csv",
        "03_rainfall_data":       f"{CSV_DIR}/03_rainfall_data.csv",
        "04_traffic_load":        f"{CSV_DIR}/04_traffic_load.csv",
        "05_citizen_complaints":  f"{CSV_DIR}/05_citizen_complaints.csv",
        "06_ml_training_dataset": f"{CSV_DIR}/06_ml_training_dataset.csv",
        "07_risk_scores":         f"{CSV_DIR}/07_risk_scores.csv",
        "08_analytical_dataset":  f"{OUT_DIR}/08_analytical_dataset.csv",
    }

    erros_criticos  = 0
    avisos          = 0
    relatorio       = []

    print("\n" + "="*65)
    print("  RELATORIO DE QUALIDADE DE DADOS — validate_csvs()")
    print("="*65)

    for nome, caminho in arquivos.items():
        try:
            df = pd.read_csv(caminho)
        except FileNotFoundError:
            print(f"\n  [ERRO] {nome}: arquivo nao encontrado em {caminho}")
            erros_criticos += 1
            continue

        erros_arq = 0
        avisos_arq = 0
        linhas = []

        # 1. Nulos
        nulos = df.isnull().sum()
        nulos = nulos[nulos > 0]
        criticos_nulo = {"pavement_year", "segment_id", "rainfall_mm",
                          "risk_score_0_100", "LABEL_failed_next_90d",
                          "load_index_1_10", "pavement_age_years", "risk_score"}
        for col, n in nulos.items():
            pct = n / len(df) * 100
            if col in criticos_nulo:
                linhas.append(f"    [CRITICO] '{col}': {n} nulos ({pct:.1f}%)")
                erros_arq += 1
            else:
                linhas.append(f"    [AV] '{col}': {n} nulos ({pct:.1f}%) — nao critico")
                avisos_arq += 1

        # 2. Duplicatas
        dups = df.duplicated().sum()
        if dups > 0:
            linhas.append(f"    [AV] {dups} linhas duplicadas")
            avisos_arq += 1

        # 3. Ranges válidos
        if nome in RANGES:
            for col, (vmin, vmax) in RANGES[nome].items():
                if col in df.columns:
                    fora = df[(df[col] < vmin) | (df[col] > vmax)][col].dropna()
                    if len(fora) > 0:
                        linhas.append(f"    [CRITICO] '{col}': {len(fora)} valores fora "
                                       f"de [{vmin}, {vmax}]  ex: {fora.values[:3]}")
                        erros_arq += 1

        # 4. Tipos
        for col, tipo_esp in DTYPES.items():
            if col in df.columns:
                tipo_real = str(df[col].dtype)
                if tipo_esp == "object" and df[col].dtype != object:
                    linhas.append(f"    [AV] '{col}' esperado string, encontrado {tipo_real}")
                    avisos_arq += 1

        # Resumo do arquivo
        status = "[OK]" if erros_arq == 0 else "[FALHA]"
        print(f"\n  {status} {nome}  ({df.shape[0]} lin x {df.shape[1]} col)")
        for l in linhas:
            print(l)
        if not linhas:
            print("    Sem problemas detectados.")

        erros_criticos += erros_arq
        avisos         += avisos_arq
        relatorio.append({"arquivo": nome, "linhas": df.shape[0], "colunas": df.shape[1],
                           "erros_criticos": erros_arq, "avisos": avisos_arq,
                           "nulos_totais": int(nulos.sum()),
                           "duplicatas": int(dups)})

    print("\n" + "="*65)
    print(f"  RESUMO FINAL: {erros_criticos} erros criticos | {avisos} avisos")
    if erros_criticos == 0:
        print("  Todos os CSVs passaram nas checagens criticas.")
    else:
        print("  ATENCAO: corrija os erros criticos antes de treinar o modelo.")
    print("="*65 + "\n")

    return pd.DataFrame(relatorio)

# ═══════════════════════════════════════════════════════════
# TAREFA 2B — project_features(years_ahead=3)
# Projeta features dos segmentos para 2025, 2026 e 2027.
# Salva em 12_feature_projections_2025_2027.csv
# ═══════════════════════════════════════════════════════════

def project_features(years_ahead: int = 3) -> pd.DataFrame:
    """
    Projeta features de cada segmento para os próximos years_ahead anos.

    Retorna DataFrame com (15 segmentos × 12 meses × years_ahead) linhas
    e salva em 12_feature_projections_2025_2027.csv.
    """
    ANO_BASE = 2024

    # Carregar datasets corrigidos
    ana  = pd.read_csv(f"{OUT_DIR}/08_analytical_dataset.csv")
    rain = pd.read_csv(f"{CSV_DIR}/03_rainfall_data.csv")

    # Média histórica de chuva por mês (rolling já calculado no dataset de treino)
    rain["rainfall_30d"]  = rain["rainfall_mm"]
    rain["rainfall_90d"]  = rain["rainfall_mm"].rolling(3,  min_periods=1).sum()
    rain["rainfall_180d"] = rain["rainfall_mm"].rolling(6,  min_periods=1).sum()

    media_mensal = rain.groupby("month").agg(
        rainfall_30d_hist  = ("rainfall_30d",  "mean"),
        rainfall_90d_hist  = ("rainfall_90d",  "mean"),
        rainfall_180d_hist = ("rainfall_180d", "mean"),
        extreme_events_hist= ("extreme_events_50mm", "mean"),
    ).reset_index()

    linhas = []
    anos_proj = range(ANO_BASE + 1, ANO_BASE + 1 + years_ahead)

    for _, seg_row in ana.iterrows():
        for delta_ano, ano in enumerate(anos_proj, start=1):
            for mes in range(1, 13):
                rain_mes = media_mensal[media_mensal["month"] == mes].iloc[0]

                # Grupo A — variáveis que evoluem +1/ano
                pav_age_proj    = seg_row["pavement_age_years"] + delta_ano
                rep_years_proj  = seg_row["years_since_last_repair"] + delta_ano

                # Grupo B — tráfego: +2,5% ao ano
                load_proj = round(
                    seg_row["load_index_1_10"] * (1.025 ** delta_ano))
                load_proj = int(min(max(load_proj, 1), 10))

                # Constantes
                flood_zone          = seg_row["flood_zone"]
                pavement_type_code  = seg_row["pavement_type_code"]
                truck_route         = seg_row["truck_route"]

                linhas.append({
                    "segment_id":            seg_row["segment_id"],
                    "street_name":           seg_row["street_name"],
                    "bairro":                seg_row["bairro"],
                    "ano":                   ano,
                    "mes":                   mes,
                    # --- features projetadas ---
                    "pavement_age_proj":     pav_age_proj,
                    "years_since_last_repair_proj": rep_years_proj,
                    "load_index_proj":       load_proj,
                    "flood_zone":            flood_zone,
                    "pavement_type_code":    pavement_type_code,
                    "truck_route":           truck_route,
                    # --- chuva histórica projetada ---
                    "rainfall_30d_proj":     round(rain_mes["rainfall_30d_hist"], 1),
                    "rainfall_90d_proj":     round(rain_mes["rainfall_90d_hist"], 1),
                    "rainfall_180d_proj":    round(rain_mes["rainfall_180d_hist"], 1),
                    "extreme_events_proj":   round(rain_mes["extreme_events_hist"], 1),
                    # --- features auxiliares do base ---
                    "total_complaints":      seg_row["total_complaints"],
                    "total_repairs":         seg_row["total_repairs"],
                    "emergency_repairs":     seg_row["emergency_repairs"],
                    "pavement_type":         seg_row["pavement_type"],
                    "risk_score_2024":       seg_row["risk_score"],
                })

    proj_df = pd.DataFrame(linhas)
    saida   = f"{OUT_DIR}/12_feature_projections_2025_2027.csv"
    proj_df.to_csv(saida, index=False, encoding="utf-8")

    n_seg = ana.shape[0]
    print(f"\n  [OK] 12_feature_projections_2025_2027.csv gerado")
    print(f"       {n_seg} segmentos x 12 meses x {years_ahead} anos = {len(proj_df)} linhas")
    print(f"       Salvo em: {saida}")
    return proj_df


# ═══════════════════════════════════════════════════════════
# EXECUÇÃO DAS FUNÇÕES DA TAREFA 2
# ═══════════════════════════════════════════════════════════

section("TAREFA 2A — Validação automática de todos os CSVs")
relatorio_qualidade = validate_csvs()

section("TAREFA 2B — Projeção de features 2025-2027")
projecoes = project_features(years_ahead=3)
print(projecoes[projecoes["mes"] == 1][
    ["segment_id", "ano", "mes", "pavement_age_proj",
     "years_since_last_repair_proj", "load_index_proj", "rainfall_90d_proj"]
].to_string(index=False))
print(f"\n  Tarefa 2 concluída.\n")
