# 🛣️ Previsão de Falhas Viárias (Asfalto) - São Luís

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Modelo-XGBoost-orange)

## 📌 Sobre o Projeto

Este projeto consiste no desenvolvimento do ciclo de vida completo de um **Modelo Preditivo baseado em Machine Learning (XGBoost)** para antecipar falhas na infraestrutura viária (buracos e ruptura de asfalto) no município de São Luís, Maranhão.

Historicamente, a Secretaria de Obras atua de forma reativa. O objetivo deste sistema é converter a abordagem de manutenção de **reativa para preditiva**, permitindo prever falhas com até **90 dias de antecedência** e proporcionando uma redução projetada superior a **58% nos custos emergenciais** de zeladoria urbana.

---

## 📊 Principais Resultados

* **Performance do Modelo:** 
  * **Recall:** 83,3% (Identifica corretamente mais de 8 em cada 10 buracos antes de ocorrerem).
  * **ROC-AUC:** 0.778.
* **Impacto de Negócio (ROI):** Redução simulada de gastos com manutenção de R$ 72.000,00 (reativo/emergencial) para R$ 29.900,00 (preventivo). Uma economia de **58,5%**.
* **Principais Preditores:**
  1. Tempo desde o último reparo (38.9%)
  2. Reclamações acumuladas nos últimos 30 dias na ouvidoria (22.5%)
  3. Idade da rua (8.2%)
  4. Chuva acumulada em 90 dias (7.8%)

---

## 🏗️ Estrutura de Dados (Data Wrangling)

A base analítica foi consolidada a partir de 5 fontes:
1. **Segmentos Viários:** Idade do pavimento, tipo (asfalto, paralelepípedo, terra) e classe.
2. **Histórico de Manutenções:** Tipo e custo de consertos anteriores.
3. **Eventos Climáticos:** Precipitação (chuvas) acumulada em 30, 90 e 180 dias.
4. **Tráfego:** Índice de carga de veículos pesados.
5. **Ouvidoria (156):** Denúncias e reclamações de cidadãos.

---

## 📁 Estrutura do Repositório

* `Projeto_Predictivo.ipynb` - Notebook completo consolidando toda a análise e modelagem.
* `codigos/` - Scripts Python modulares:
  * `pipeline_limpeza.py`: Tratamento e preparação de dados.
  * `eda_analysis.py`: Análise Exploratória de Dados (EDA).
  * `modelagem.py`: Treinamento e avaliação do modelo XGBoost.
  * `avaliacao_negocio.py`: Cálculo do ROI e impacto financeiro.
  * `export_model.py`: Exportação do modelo para produção usando `joblib`.
  * `fix_map_coords.py`: Correção e manipulação de dados geoespaciais.
* `eda_outputs/` - Gráficos e resultados gerados durante a Análise Exploratória.
* `ml_outputs/` - Gráficos de performance do modelo (Curva ROC, Feature Importance, etc).
* `relatorio_tecnico.tex` - Relatório técnico acadêmico detalhado do projeto em formato LaTeX.

---

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual (`venv` ou `conda`).

Instale as dependências necessárias (geralmente `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`):

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### Ordem de Execução dos Scripts

Para reproduzir o projeto a partir dos dados brutos:

1. Acesse o diretório `codigos/`
2. Execute a limpeza: `python pipeline_limpeza.py`
3. Gere as análises: `python eda_analysis.py`
4. Treine o modelo: `python modelagem.py`
5. Avalie os custos: `python avaliacao_negocio.py`
6. Exporte o modelo final: `python export_model.py`

Ou, alternativamente, consulte o notebook `Projeto_Predictivo.ipynb` para uma visualização interativa do processo ponta-a-ponta.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualização:** Matplotlib, Seaborn
* **Implantação (Planejada/Mencionada):** Streamlit (Dashboard interativo), Joblib (Serialização de Modelos)

---

**Autor:** Prefeitura de São Luís - MA | Equipe de Inteligência de Dados
