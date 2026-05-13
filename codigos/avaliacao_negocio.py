"""
Projeto: Preditor de Falhas em Vias - São Luís / MA
Etapa:   Avaliação e Interpretação (Business Evaluation)
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# 1. CARREGAR DADOS E RETREINAR O MODELO PARA OBTER PREDIÇÕES
BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "ml_outputs")
os.makedirs(OUT, exist_ok=True)

data_path = os.path.join(PARENT, "06_ml_training_dataset.csv")
df = pd.read_csv(data_path)

df['years_since_last_repair'] = df['years_since_last_repair'].replace('NA', np.nan)
df['years_since_last_repair'] = pd.to_numeric(df['years_since_last_repair'])
df['years_since_last_repair'].fillna(df['road_age_years'], inplace=True)

drop_cols = ['segment_id', 'year', 'month', 'notes', 'LABEL_failed_next_30d', 'LABEL_failed_next_90d', 'repair_cost_total_brl', 'cumulative_repairs']
X = df.drop(columns=drop_cols, errors='ignore')
y = df['LABEL_failed_next_90d']

# Salvamos os índices para podermos referenciar depois os custos
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df.index, test_size=0.3, stratify=y, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Melhor modelo
best_xgb = XGBClassifier(
    learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
best_xgb.fit(X_train_scaled, y_train)

y_pred = best_xgb.predict(X_test_scaled)
y_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

# 2. MÉTRICAS TÉCNICAS (PERFORMANCE)
print("="*60)
print("  AVALIAÇÃO TÉCNICA (PERFORMANCE DO MODELO)")
print("="*60)
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"  AUC (Área sob a Curva ROC): {auc:.4f}")
print(f"  F1-Score (Harmônica entre Precisão e Recall): {f1:.4f}")
print(f"  Acurácia Global: {acc:.4f}\n")
print(f"  Matriz de Confusão:")
print(f"    Verdadeiros Negativos (Acertou que não falha): {cm[0][0]}")
print(f"    Falsos Positivos (Alarme falso):               {cm[0][1]}")
print(f"    Falsos Negativos (Deixou passar o buraco):     {cm[1][0]}")
print(f"    Verdadeiros Positivos (Detectou o buraco):     {cm[1][1]}")

# 3. RESULTADOS DE NEGÓCIO (BUSINESS INTERPRETATION)
print("\n" + "="*60)
print("  AVALIAÇÃO DE NEGÓCIOS (BUSINESS IMPACT)")
print("="*60)

# Custo médio derivado do relatório executivo e dicionário de dados
# Reparo preventivo médio: ~ R$ 3.500
# Reparo emergencial (falha não prevista): ~ R$ 12.000
# Custo de vistoriar um alarme falso (inspeção): ~ R$ 200

COST_PREVENTIVE = 3500
COST_EMERGENCY = 12000
COST_INSPECTION_FALSE_ALARM = 200

# Cálculo de Custos do Cenário ATUAL (100% Reativo nas vias que falharam)
# Vias que realmente falharam na amostra de teste
total_falhas_reais = np.sum(y_test == 1)
custo_cenario_atual = total_falhas_reais * COST_EMERGENCY

# Cálculo de Custos com o MODELO PREDITIVO
# Verdadeiros Positivos (Detectou preventivamente -> Custo Preventivo)
custo_vp = cm[1][1] * COST_PREVENTIVE
# Falsos Negativos (O modelo errou e não detectou -> Custo Emergencial)
custo_fn = cm[1][0] * COST_EMERGENCY
# Falsos Positivos (Modelo previu falha, vistoria foi feita, mas estava bom -> Custo de Inspeção)
custo_fp = cm[0][1] * COST_INSPECTION_FALSE_ALARM

custo_cenario_ia = custo_vp + custo_fn + custo_fp
economia_reais = custo_cenario_atual - custo_cenario_ia
economia_pct = (economia_reais / custo_cenario_atual) * 100 if custo_cenario_atual > 0 else 0

print(f"  Simulação para a amostra de teste ({len(y_test)} segmentos, onde {total_falhas_reais} falhariam):\n")
print(f"  Cenário ATUAL (Reativo): R$ {custo_cenario_atual:,.2f}")
print(f"  Cenário com IA (Preditivo): R$ {custo_cenario_ia:,.2f}")
print(f"  -> ECONOMIA ESTIMADA: R$ {economia_reais:,.2f} ({economia_pct:.1f}% de redução de custos)")

# Projetando para toda a cidade de São Luís (~ 2000 trechos críticos)
# Vamos escalar o resultado da amostra (que tinha 12 segmentos)
escala_cidade = 2000 / len(y_test)
economia_cidade_projetada = economia_reais * escala_cidade

print(f"\n  Projeção anual para a Prefeitura (aplicando a 2000 vias críticas):")
print(f"  -> ECONOMIA PROJETADA: R$ {economia_cidade_projetada:,.2f} por ano.")

# Gerando gráfico de ROI (Return on Investment)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Custo Atual (Reativo)', 'Custo c/ IA (Preditivo)']
valores = [custo_cenario_atual, custo_cenario_ia]
cores = ['#e74c3c', '#2ecc71']

bars = ax.bar(labels, valores, color=cores, width=0.5)
ax.set_ylabel('Custo em R$ (Milhares)')
ax.set_title('Simulação de Custo em Manutenção (Amostra de Teste)')

for b in bars:
    altura = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., altura + 1000,
            f'R$ {altura:,.0f}', ha='center', fontweight='bold')

# Adiciona linha de economia
if economia_reais > 0:
    ax.annotate(f'Economia:\nR$ {economia_reais:,.0f}', 
                xy=(1, custo_cenario_ia), 
                xytext=(0.5, custo_cenario_atual),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_business_impact.png"))
plt.close()
print("\n  Gráfico de impacto de negócio salvo.")
