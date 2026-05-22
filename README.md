# Preditor de Risco de Falhas em Vias Urbanas
### São Luís, Maranhão — Dados 100% Públicos

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Dados-100%25_Públicos-success)
![AUROC](https://img.shields.io/badge/AUROC_OOD-0.6796-orange)
![Segments](https://img.shields.io/badge/Segmentos-60.933-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)

![Demo](assets/gif_D_antes_depois.gif)

---

## Resultados

| Métrica | Valor |
|---|---|
| AUROC Out-of-Domain (São Luís) | **0.6796** |
| AUROC Walk-Forward (Fortaleza + Recife) | **0.9680 ± 0.0330** |
| Segmentos mapeados | **60.933** |
| Segmentos em estado crítico (2024) | **6.031 (9,9%)** |
| Cidades de treino | Fortaleza (CE) + Recife (PE) |
| Período histórico | 2015–2024 (10 anos) |
| Dados da prefeitura utilizados | **Zero** |
| Economia potencial (preventivo vs emergencial) | **R$ 347M** |

> **Nota sobre os AUROCs:** O AUROC Walk-Forward (0,9680) é alto porque treino e
> teste ocorrem nas **mesmas cidades** — mede extrapolação temporal, não geográfica.
> O AUROC OOD **0,6796** é a performance real: São Luís nunca vista durante o treino.

---

## Fontes de Dados (100% Públicas)

| Dado | Fonte | Status | Volume |
|---|---|---|---|
| Rede viária | OpenStreetMap via OSMnx 2.1 | OK | 206.950 segmentos |
| Elevação / Declividade | NASA SRTM via Open-Elevation API | OK | 2dp grid (~1,1 km) |
| Zonas de alagamento | OSMnx water features (proxy geométrico) | OK | Por segmento |
| Densidade urbana | IBGE Censo 2022 | OK | Por município |
| Pluviosidade | ERA5 via Open-Meteo Archive API | OK | 120 meses (2015-2024) |
| OS reais (São Paulo) | 5 fontes tentadas (CKAN, GitHub, URLs diretas) | FALLBACK | Inacessíveis |

---

## Arquitetura do Pipeline

```
OpenStreetMap (OSMnx)      --+
NASA SRTM (Open-Elevation) --+
OSMnx water features       --+--> Features geoespaciais (206.950 segmentos)
IBGE Censo 2022            --+
ERA5 (Open-Meteo)          --+
                              |
                              v
ICP Degradação Física ------> Y fisicamente motivado (Y=1 se ICP < 40)
  taxa_anual x fator_chuva x fator_alagamento x fator_declividade x fator_carga
                              |
                              v
XGBoost Classifier ---------> Walk-Forward 5 folds (temporal)
  n=200, depth=4, lr=0.05     Train: Fortaleza + Recife | OOD: São Luís
                              |
                              v
Previsões + R$ -------------> Dashboard Leaflet.js + Análise econômica
```

---

## Features do Modelo (15 features)

| Grupo | Features | Importância |
|---|---|---|
| Alagamento | flood_zone_final, dist_water_m, flood_risk_slope | **89,1%** |
| Topografia | elevation_m, slope_pct | 4,5% |
| Via | highway_code, length_m, maxspeed_kmh, lanes, oneway, load_proxy_1_10 | 3,8% |
| Pluviosidade | chuva_media_anual_mm, chuva_media_jan_mm | 1,0% |
| Densidade | densidade_hab_km2, urban_density_score | ~0% |

`flood_zone_final` domina com 89,1%. Correlação Y x flood_zone = 0,459 (moderada — sem data leakage, confirmado por ablation study).

---

## Walk-Forward Validation

| Fold | Treino | Teste | AUROC |
|---|---|---|---|
| 1 | 2015-2019 | 2020 | 0.9845 |
| 2 | 2015-2020 | 2021 | 0.9291 |
| 3 | 2015-2021 | 2022 | 0.9996 |
| 4 | 2015-2022 | 2023 | 0.9995 |
| 5 | 2015-2023 | 2024 | 0.9271 |
| **OOD** | **2015-2024** | **São Luís** | **0.6796** |

---

## Estrutura do Repositório

```
1projeto/
├── data/
│   ├── raw/               <- Dados brutos (03_rainfall_data.csv -- Open-Meteo)
│   ├── processed/         <- CSVs processados (< 50MB -- gerados pelo pipeline)
│   └── simulated/         <- CSVs simulados -- NAO usar em análises (ver LEIA-ME.txt)
├── src/                   <- Scripts da pipeline (01_ a 12_ + utilitários)
├── eda/                   <- 7 EDAs com dados reais
├── model/                 <- modelo_final.pkl + metricas_final.csv
├── reports/               <- Relatórios e decisões técnicas (DT-01 a DT-08)
├── dashboard/             <- dashboard_final.html (Leaflet.js)
├── assets/                <- GIFs LinkedIn (gif_A ... gif_D)
├── notebooks/             <- Projeto_Predictivo.ipynb
├── templates/             <- template_os_semusc.xlsx (importação SEMUSC)
├── requirements.txt
└── README.md
```

---

## Como Reproduzir

```bash
git clone https://github.com/[usuario]/preditor-vias-sao-luis
cd preditor-vias-sao-luis
pip install -r requirements.txt

python src/01_fetch_road_network.py       # OSMnx -- ~5 min
python src/02_fetch_elevation.py          # NASA SRTM -- ~30 seg
python src/03_fetch_flood_zones.py        # OSMnx water features
python src/04_fetch_ibge_density.py       # IBGE Censo 2022
python src/05_fetch_rainfall_segments.py  # Open-Meteo ERA5
python src/fallback_y_real.py             # Modelo ICP (Y)
python src/08_build_training_dataset.py   # Merge 206.950 x 10 anos
python src/09_train_final_model.py        # XGBoost Walk-Forward
python src/10_predict_sao_luis.py         # Previsões + análise R$
python src/11_build_dashboard.py          # Dashboard HTML

# Arquivos > 50MB são gerados localmente e não estão no repositório.
```

---

## EDAs (Dados Reais)

| EDA | Descrição |
|---|---|
| [eda_01_rede_viaria.png](eda/eda_01_rede_viaria.png) | Rede viária de São Luís -- 60.933 segmentos (OSMnx) |
| [eda_02_chuva_real.png](eda/eda_02_chuva_real.png) | Precipitação ERA5 2015-2024 (120 meses) |
| [eda_03_elevacao_declividade.png](eda/eda_03_elevacao_declividade.png) | Elevação e declividade -- NASA SRTM |
| [eda_04_risco_por_bairro.png](eda/eda_04_risco_por_bairro.png) | Risco por zona geográfica (grid 0,03 graus) |
| [eda_05_feature_importance.png](eda/eda_05_feature_importance.png) | Importância de features -- XGBoost |
| [eda_06_auroc_walkforward.png](eda/eda_06_auroc_walkforward.png) | Walk-Forward vs OOD São Luís |
| [eda_07_previsoes_sao_luis.png](eda/eda_07_previsoes_sao_luis.png) | Mapa de risco -- São Luís 2024 |

---

## Análise Econômica

| Categoria | Segmentos | Custo Preventivo | Custo Emergencial | Economia |
|---|---|---|---|---|
| CRÍTICO | 6.031 (9,9%) | R$ 90,5M | R$ 271,4M | **R$ 180,9M** |
| ALTO | 17.166 (28,2%) | R$ 60,1M | R$ 206,0M | **R$ 145,9M** |
| MÉDIO | 2.832 (4,6%) | R$ 2,3M | R$ 8,5M | R$ 6,2M |
| BAIXO | 34.904 (57,3%) | R$ 7,0M | R$ 20,9M | R$ 14,0M |
| **TOTAL** | **60.933** | **R$ 159,8M** | **R$ 506,8M** | **R$ 347,0M** |

Estimativas baseadas em tabelas DNIT/SEMUSC. ROI da manutencao preventiva: 3,2x.

---

## Limitações e Honestidade

1. **Y sintético** -- A variável-alvo é gerada pelo Modelo ICP de Degradação Física, não por ordens de serviço reais. O modelo aprende padrões de degradação física, não de demanda real de reparo.

2. **OS reais não obtidas** -- 5 fontes abertas de SP foram tentadas (todas inacessíveis). Portal Recife retornou dados de zoneamento incompatíveis.

3. **Gap WF -> OOD** -- AUROC WF = 0,9680 vs OOD = 0,6796. Delta = -0,29. Esperado para validação geográfica cross-city.

4. **flood_zone domina (89,1%)** -- O modelo tem poder discriminativo concentrado nessa feature. Em zonas sem variação de alagamento, a discriminação é limitada.

5. **Custos em R$** -- Estimativas por categoria. Valores reais a validar com SEMUSC/DNIT.

---

## Próximos Passos

- OS reais da SEMUSC -> retreinar com Y real (templates/template_os_semusc.xlsx)
- Dados CEMADEN para zonas de alagamento reais (substitui proxy OSMnx)
- Expandir para outras cidades do Maranhão
- API REST para consulta de risco por coordenada

---

## Decisões Técnicas

Ver reports/decisoes_tecnicas.txt -- 8 decisões documentadas (DT-01 a DT-08).

---

## Tecnologias

Python 3.14 · XGBoost 3.2 · OSMnx 2.1 · pandas 3.0 · numpy 2.4 · SHAP 0.51 · scikit-learn 1.8 · Leaflet.js · Open-Meteo API · NASA SRTM · IBGE Censo 2022

---

Treino: Fortaleza (CE) + Recife (PE) -- Validação OOD: São Luís (MA) -- Zero dados da prefeitura.
