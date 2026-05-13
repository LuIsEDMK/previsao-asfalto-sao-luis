"""
Projeto: Preditor de Falhas em Vias - São Luís / MA
Etapa:   Modelagem e Machine Learning (Modelling)
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# 1. CARREGAR DADOS
BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "ml_outputs")
os.makedirs(OUT, exist_ok=True)

print("="*60)
print("  TREINAMENTO E AVALIAÇÃO DE MODELOS DE MACHINE LEARNING")
print("="*60)

data_path = os.path.join(PARENT, "06_ml_training_dataset.csv")
df = pd.read_csv(data_path)

# 2. PRÉ-PROCESSAMENTO
print("\n▶ 1. PRÉ-PROCESSAMENTO DOS DADOS")

# Tratar valores 'NA' na coluna years_since_last_repair
df['years_since_last_repair'] = df['years_since_last_repair'].replace('NA', np.nan)
df['years_since_last_repair'] = pd.to_numeric(df['years_since_last_repair'])

# Preencher nan em years_since_last_repair com a idade do pavimento
df['years_since_last_repair'].fillna(df['road_age_years'], inplace=True)

# Remover colunas que não são features numéricas ou são labels vazamentos
drop_cols = ['segment_id', 'year', 'month', 'notes', 'LABEL_failed_next_30d', 'LABEL_failed_next_90d', 'repair_cost_total_brl', 'cumulative_repairs']
X = df.drop(columns=drop_cols, errors='ignore')

# Target principal: falha nos próximos 90 dias (mais dados para modelo de manutenção preventiva)
y = df['LABEL_failed_next_90d']

print(f"  Shape de X (features): {X.shape}")
print(f"  Shape de y (target): {y.shape}")
print(f"  Proporção das classes no target: \n{y.value_counts(normalize=True)*100}")

# Divisão Treino e Teste (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(f"  Tamanho do Treino: {len(X_train)} amostras")
print(f"  Tamanho do Teste: {len(X_test)} amostras")

# Escalonamento e imputação (para os poucos nulos que podem sobrar)
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# 3. SELEÇÃO E TREINAMENTO DOS MODELOS BÁSICOS (BASELINE)
print("\n▶ 2. TREINAMENTO DE MODELOS E VALIDAÇÃO CRUZADA (BASELINE)")

models = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    results.append({
        "Model": name,
        "ROC-AUC CV Mean": cv_scores.mean(),
        "ROC-AUC CV Std": cv_scores.std()
    })
    print(f"  {name:22s} -> ROC-AUC Médio (CV=5): {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values(by='ROC-AUC CV Mean', ascending=False)

# O melhor modelo é escolhido baseado no AUC
best_model_name = results_df.iloc[0]['Model']
print(f"\n  >> Melhor modelo na validação cruzada: {best_model_name}")

# 4. TUNING DE HIPERPARÂMETROS DO XGBOOST (ASSUMINDO QUE SEJA O MELHOR)
print("\n▶ 3. OTIMIZAÇÃO (TUNING) DO MELHOR MODELO (XGBoost)")

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Em datasets muito pequenos (como nosso sample), o GridSearch pode não variar muito. 
# cv=3 para evitar fold muito pequeno
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
grid_search.fit(X_train_scaled, y_train)

best_xgb = grid_search.best_estimator_
print(f"  Melhores hiperparâmetros encontrados:")
for k, v in grid_search.best_params_.items():
    print(f"    {k}: {v}")

# 5. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
print("\n▶ 4. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")

# Predições
y_pred = best_xgb.predict(X_test_scaled)
y_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

# Métricas
auc_test = roc_auc_score(y_test, y_prob)
print(f"  ROC-AUC no Teste: {auc_test:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred))

# Gerar Matriz de Confusão e Curva ROC em imagem
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão (Teste)')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='red', label=f'ROC curve (AUC = {auc_test:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[1].set_xlabel('Taxa Falsos Positivos')
axes[1].set_ylabel('Taxa Verdadeiros Positivos')
axes[1].set_title('Curva ROC')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_avaliacao_modelo.png"))
plt.close()

# 6. IMPORTÂNCIA DAS VARIÁVEIS (FEATURE IMPORTANCE)
print("\n▶ 5. IMPORTÂNCIA DAS VARIÁVEIS PREDITORAS")

importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (XGBoost)")
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
plt.xlabel("Importância Relativa")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_feature_importance.png"))
plt.close()

for i in indices[:5]:
    print(f"  {features[i]:25s}: {importances[i]*100:.1f}%")

print(f"\n{'='*60}")
print(f"  Modelagem concluída! Gráficos salvos em: {OUT}")
print(f"{'='*60}")
