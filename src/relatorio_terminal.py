"""
Tarefa 9 — Relatório Terminal Final (ASCII Box)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd

BASE_DIR  = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR   = f"{BASE_DIR}\\csv"
MODEL_DIR = f"{BASE_DIR}\\1projeto"

with open(f"{MODEL_DIR}\\modelo_final.pkl", "rb") as f:
    meta = pickle.load(f)

prev = pd.read_csv(f"{CSV_DIR}\\13_previsoes_economicas_sao_luis.csv")
total_s  = len(prev)
criticos = (prev["categoria_risco"] == "CRÍTICO").sum()
altos    = (prev["categoria_risco"] == "ALTO").sum()
medios   = (prev["categoria_risco"] == "MÉDIO").sum()
baixos   = (prev["categoria_risco"] == "BAIXO").sum()

col_prev = "custo_preventivo_R$"
col_emer = "custo_emergencial_R$"
col_econ = "economia_R$"

total_prev = prev[col_prev].sum()
total_emer = prev[col_emer].sum()
total_econ = prev[col_econ].sum()
econ_crit  = prev[prev["categoria_risco"] == "CRÍTICO"][col_econ].sum()

auroc_wf  = meta.get("auroc_walk_forward_mean", 0)
auroc_std = meta.get("auroc_walk_forward_std", 0)
auroc_sl  = meta.get("auroc_ood_sao_luis", 0)

W = 72

def div_h():  return "+" + "=" * W + "+"
def div_s():  return "+" + "-" * W + "+"
def box(txt): return "| " + txt.ljust(W - 4) + " |"

print()
print(div_h())
print(box("  PREDITOR DE FALHAS EM VIAS URBANAS  --  Sao Luis / MA"))
print(box("  Manutencao Preditiva com ML e Dados 100% Publicos"))
print(div_h())
print(box(""))
print(box("  STATUS DAS TAREFAS"))
print(box(""))
print(box("  [T1] fetch_sao_paulo_os.py    -- 5 fontes SP testadas -> FALLBACK"))
print(box("  [T2] (nao executada -- sem OS reais de SP)"))
print(box("  [T3] fallback_y_real.py       -- Modelo ICP Degradacao Fisica OK"))
print(box("  [T4] build_final_dataset.py   -- 2.069.500 linhas x 27 colunas OK"))
print(box("  [T5] train_final_model.py     -- XGBoost + Walk-Forward + SHAP OK"))
print(box("  [T6] predict_sao_luis_final.py-- Previsoes + Analise Economica OK"))
print(box("  [T7] build_dashboard_final.py -- Dashboard HTML 2.1 MB OK"))
print(box("  [T8] generate_final_docs.py   -- README + Relatorio + Decisoes OK"))
print(box("  [T9] relatorio_terminal.py    -- Este relatorio"))
print(div_s())
print(box(""))
print(box("  DATASET"))
print(box(""))
print(box("  Cidades:   Fortaleza (CE)  +  Recife (PE)  +  Sao Luis (MA)"))
print(box("  Segmentos: 206.950 (OSMnx / OpenStreetMap)"))
print(box("  Periodo:   2015 - 2024 (10 anos)"))
print(box("  Total:     2.069.500 linhas x 27 colunas"))
print(box("  Features:  15 (via + alagamento + topografia + chuva + densidade)"))
print(box("  Y:         ICP Degradacao Fisica -- Y=1 se ICP < 40 (critico)"))
print(div_s())
print(box(""))
print(box("  MODELO  (XGBoost, n=200, depth=4, lr=0.05, scale_pos_weight=auto)"))
print(box(""))
print(box(f"  AUROC Walk-Forward (Fortaleza+Recife):   {auroc_wf:.4f} +/- {auroc_std:.4f}"))
print(box(f"  AUROC OOD Sao Luis (cidade inedita):     {auroc_sl:.4f}"))
print(box(f"  Delta WF -> OOD:                          {auroc_sl-auroc_wf:+.4f}"))
print(box(""))
print(box("  Feature Importance (Top 5):"))
print(box("    1. flood_zone_final     89.1%  [zona de alagamento]"))
print(box("    2. flood_risk_slope      2.8%  [risco por declividade]"))
print(box("    3. slope_pct             2.4%  [declividade]"))
print(box("    4. highway_code          2.1%  [tipo de via]"))
print(box("    5. lanes                 1.0%  [numero de faixas]"))
print(div_s())
print(box(""))
print(box("  PREVISOES SAO LUIS 2024"))
print(box(""))
n_criticos_bar = int(criticos / total_s * 30)
n_altos_bar    = int(altos    / total_s * 30)
n_medios_bar   = int(medios   / total_s * 30)
n_baixos_bar   = int(baixos   / total_s * 30)
print(box(f"  CRITICO  {criticos:>6,} ({criticos/total_s*100:5.1f}%) [{'#'*n_criticos_bar}]"))
print(box(f"  ALTO     {altos:>6,} ({altos/total_s*100:5.1f}%) [{'#'*n_altos_bar}]"))
print(box(f"  MEDIO    {medios:>6,} ({medios/total_s*100:5.1f}%) [{'#'*n_medios_bar}]"))
print(box(f"  BAIXO    {baixos:>6,} ({baixos/total_s*100:5.1f}%) [{'#'*n_baixos_bar}]"))
print(div_s())
print(box(""))
print(box("  ANALISE ECONOMICA  (estimativas DNIT/SEMUSC em R$)"))
print(box(""))
print(box(f"  Custo 100% preventivo:    R$ {total_prev/1e6:>8.1f} milhoes"))
print(box(f"  Custo 100% emergencial:   R$ {total_emer/1e6:>8.1f} milhoes"))
print(box(f"  ECONOMIA POTENCIAL:       R$ {total_econ/1e6:>8.0f} MILHOES"))
print(box(f"  ROI manutencao preventi:  {total_emer/total_prev:>5.1f}x o investimento"))
print(box(""))
print(box(f"  Intervir nos {criticos:,} trechos CRITICOS economizaria"))
print(box(f"  R$ {econ_crit/1e6:.0f} milhoes em relacao a resposta emergencial."))
print(div_s())
print(box(""))
print(box("  ARQUIVOS GERADOS"))
print(box(""))
print(box("  1projeto/modelo_final.pkl                  XGBoost + metadados"))
print(box("  1projeto/dashboard_final.html  (2.1 MB)   Mapa interativo Leaflet"))
print(box("  1projeto/README.md             (5.3 KB)   Documentacao tecnica"))
print(box("  1projeto/relatorio_final_executivo.txt     Linguagem de gestor"))
print(box("  1projeto/decisoes_tecnicas.txt (6.4 KB)   Registro de decisoes DT"))
print(box("  csv/10_icp_y_labels.csv                    Y por ICP (2.07M linhas)"))
print(box("  csv/11_training_dataset_final.csv          Dataset de treino"))
print(box("  csv/12_metricas_final.csv                  Metricas Walk-Forward"))
print(box("  csv/12_shap_top5_sao_luis.csv              SHAP top-5 segmentos"))
print(box("  csv/13_previsoes_economicas_sao_luis.csv   60.933 segs + custos R$"))
print(div_s())
print(box(""))
print(box("  FONTES DE DADOS  (100% publicas, zero dados da prefeitura)"))
print(box(""))
print(box("  [OK] Rede viaria:     OpenStreetMap via OSMnx 2.1"))
print(box("  [OK] Elevacao:        NASA SRTM via Open-Elevation API (2dp grid)"))
print(box("  [OK] Alagamento:      OSMnx water features (proxy geometrico)"))
print(box("  [OK] Densidade urb.:  IBGE Censo 2022 (hardcoded por municipio)"))
print(box("  [OK] Pluviosidade:    ERA5 via Open-Meteo Archive API"))
print(box("  [--] OS reais SP:     5 fontes testadas -- todas inacessiveis"))
print(box("         CKAN dados.sp | slugs diretos | Base dos Dados"))
print(box("         GitHub prefeitura-sp | URLs diretas -- FALLBACK ativado"))
print(div_h())
print()
print("  Projeto concluido com sucesso.")
print(f"  AUROC OOD Sao Luis = {auroc_sl:.4f}  |  "
      f"Economia potencial = R$ {total_econ/1e6:.0f}M")
print()
