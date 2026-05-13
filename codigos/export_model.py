"""
Script para treinar o modelo final em TODOS os dados e exportá-lo 
junto com o scaler para ser usado no Dashboard de Produção.
"""
import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "deploy")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PARENT, "06_ml_training_dataset.csv"))

# Tratar NA
df['years_since_last_repair'] = df['years_since_last_repair'].replace('NA', np.nan)
df['years_since_last_repair'] = pd.to_numeric(df['years_since_last_repair'])
df['years_since_last_repair'].fillna(df['road_age_years'], inplace=True)

drop_cols = ['segment_id', 'year', 'month', 'notes', 'LABEL_failed_next_30d', 'LABEL_failed_next_90d', 'repair_cost_total_brl', 'cumulative_repairs']
X = df.drop(columns=drop_cols, errors='ignore')
y = df['LABEL_failed_next_90d']

# Escalonamento
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# Treinamento com os melhores hiperparâmetros
xgb = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8,
                    use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_scaled, y)

# Salvar Modelo, Scaler e Imputer, e as features esperadas
joblib.dump(xgb, os.path.join(OUT_DIR, "xgb_model.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(OUT_DIR, "imputer.pkl"))
joblib.dump(list(X.columns), os.path.join(OUT_DIR, "features.pkl"))

print(f"Modelo e pré-processadores salvos em: {OUT_DIR}")
