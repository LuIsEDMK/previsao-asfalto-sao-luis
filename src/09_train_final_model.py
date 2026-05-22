"""
Tarefa 5 — Treino Final com Walk-Forward + SHAP
Walk-Forward: treina em Fortaleza+Recife, valida em anos futuros.
OOD: valida em São Luís (cidade nunca vista durante treino).
Saída: modelo_final.pkl, csv/12_metricas_final.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, warnings, pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

BASE_DIR   = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR    = f"{BASE_DIR}\\csv"
MODEL_DIR  = f"{BASE_DIR}\\1projeto"

# ── Carregar dataset ─────────────────────────────────────────────────────────
print("Carregando dataset final...")
df = pd.read_csv(f"{CSV_DIR}\\11_training_dataset_final.csv", low_memory=False)
print(f"  Shape: {df.shape}")

# Features numéricas apenas (exclui texto como fonte_densidade)
FEATURES_NUM = [
    "highway_code", "length_m", "maxspeed_kmh", "lanes", "oneway", "load_proxy_1_10",
    "flood_zone_final", "dist_water_m", "flood_risk_slope",
    "elevation_m", "slope_pct",
    "chuva_media_anual_mm", "chuva_media_jan_mm",
    "densidade_hab_km2", "urban_density_score",
]
FEATURES = [f for f in FEATURES_NUM if f in df.columns]
print(f"  Features: {FEATURES}")

TARGET = "y_icp"

# ── Walk-Forward Validation ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  WALK-FORWARD VALIDATION (Fortaleza + Recife)")
print("="*60)

# Só cidades de treino
df_tr_cidades = df[df["cidade"].isin(["fortaleza", "recife"])].copy()
df_sl         = df[df["cidade"] == "sao_luis"].copy()

FOLDS = [
    (list(range(2015, 2020)), [2020]),
    (list(range(2015, 2021)), [2021]),
    (list(range(2015, 2022)), [2022]),
    (list(range(2015, 2023)), [2023]),
    (list(range(2015, 2024)), [2024]),
]

resultados_wf = []
modelos_fold  = []

for i, (anos_train, anos_test) in enumerate(FOLDS, 1):
    X_train = df_tr_cidades[df_tr_cidades["ano"].isin(anos_train)][FEATURES]
    y_train = df_tr_cidades[df_tr_cidades["ano"].isin(anos_train)][TARGET]
    X_test  = df_tr_cidades[df_tr_cidades["ano"].isin(anos_test)][FEATURES]
    y_test  = df_tr_cidades[df_tr_cidades["ano"].isin(anos_test)][TARGET]

    spw = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    modelo = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)],
               verbose=False)

    prob_test = modelo.predict_proba(X_test)[:, 1]
    auroc     = roc_auc_score(y_test, prob_test) if y_test.nunique() > 1 else float("nan")

    print(f"  Fold {i} | Train anos {anos_train[0]}-{anos_train[-1]} → "
          f"Test {anos_test[0]} | AUROC={auroc:.4f} | "
          f"train_n={len(y_train):,} | test_n={len(y_test):,}")

    resultados_wf.append({"fold": i, "anos_treino": f"{anos_train[0]}-{anos_train[-1]}",
                           "ano_teste": anos_test[0], "auroc_wf": auroc,
                           "n_train": len(y_train), "n_test": len(y_test)})
    modelos_fold.append(modelo)

auroc_wf_vals = [r["auroc_wf"] for r in resultados_wf]
print(f"\n  AUROC Walk-Forward: {np.mean(auroc_wf_vals):.4f} ± {np.std(auroc_wf_vals):.4f}")

# ── Modelo Final — treino em TODOS os anos das 2 cidades ────────────────────
print("\n" + "="*60)
print("  MODELO FINAL (Fortaleza + Recife, todos os anos)")
print("="*60)

X_all  = df_tr_cidades[FEATURES]
y_all  = df_tr_cidades[TARGET]
spw_all = max(1.0, (y_all == 0).sum() / max((y_all == 1).sum(), 1))

modelo_final = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=spw_all,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)
modelo_final.fit(X_all, y_all, verbose=False)
print(f"  Treinado em {len(y_all):,} amostras (spw={spw_all:.2f})")

# ── OOD Validation — São Luís ────────────────────────────────────────────────
print("\n" + "="*60)
print("  OOD VALIDATION — São Luís (cidade nunca vista durante treino)")
print("="*60)

X_sl = df_sl[FEATURES]
y_sl = df_sl[TARGET]
prob_sl = modelo_final.predict_proba(X_sl)[:, 1]
auroc_sl = roc_auc_score(y_sl, prob_sl) if y_sl.nunique() > 1 else float("nan")

pred_sl_bin = (prob_sl >= 0.5).astype(int)
prec_sl  = precision_score(y_sl, pred_sl_bin, zero_division=0)
rec_sl   = recall_score(y_sl, pred_sl_bin, zero_division=0)
f1_sl    = f1_score(y_sl, pred_sl_bin, zero_division=0)

print(f"  AUROC OOD São Luís: {auroc_sl:.4f}")
print(f"  Precision: {prec_sl:.4f} | Recall: {rec_sl:.4f} | F1: {f1_sl:.4f}")
print(f"  N amostras: {len(y_sl):,} | Y=1: {y_sl.sum():,} ({y_sl.mean()*100:.1f}%)")

# ── Feature Importance ───────────────────────────────────────────────────────
print("\n  Feature Importance (modelo final):")
fi = pd.Series(modelo_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<30} {imp*100:5.1f}% {bar}")

# ── SHAP — Top-5 Segmentos Críticos em São Luís ───────────────────────────────
print("\n" + "="*60)
print("  SHAP — Top-5 Segmentos Críticos em São Luís (2024)")
print("="*60)

try:
    import shap

    # Pegar segmentos de São Luís em 2024 com maior risco
    df_sl_2024 = df_sl[df_sl["ano"] == 2024].copy()
    df_sl_2024["risco"] = modelo_final.predict_proba(df_sl_2024[FEATURES])[:, 1]
    top5 = df_sl_2024.nlargest(5, "risco")

    explainer = shap.TreeExplainer(modelo_final)
    shap_vals = explainer.shap_values(top5[FEATURES])

    print("\n  Segmentos mais críticos e seus fatores explicativos:")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        idx_in_top5 = rank - 1
        sv = shap_vals[idx_in_top5]
        top_feats = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[:3]

        lat = row.get("lat_mid", "?")
        lon = row.get("lon_mid", "?")
        seg_id = row.get("segment_id", "?")
        risco  = row["risco"]
        icp_val = row.get("icp", "?")

        print(f"\n  [{rank}] seg_id={seg_id} | risco={risco:.3f} | ICP={icp_val}")
        print(f"       lat={lat:.4f}, lon={lon:.4f}" if isinstance(lat, float) else f"       lat={lat}, lon={lon}")
        print(f"       Fatores principais:")
        for feat, sv_val in top_feats:
            direcao = "↑risco" if sv_val > 0 else "↓risco"
            print(f"         {feat}: SHAP={sv_val:+.4f} ({direcao})")

    # Salvar SHAP top5
    shap_df = top5[["segment_id","cidade","ano","lat_mid","lon_mid","risco","icp"] +
                   [c for c in FEATURES if c in top5.columns]].copy()
    shap_df.to_csv(f"{CSV_DIR}\\12_shap_top5_sao_luis.csv", index=False, encoding="utf-8")
    print(f"\n  ✅ SHAP top5 salvo: {CSV_DIR}\\12_shap_top5_sao_luis.csv")

except ImportError:
    print("  ⚠  SHAP não instalado — pulando análise SHAP")
    print("     Para instalar: pip install shap")
except Exception as e:
    print(f"  ⚠  Erro no SHAP: {e}")

# ── Métricas consolidadas ────────────────────────────────────────────────────
metricas = pd.DataFrame(resultados_wf)
metricas_ood = pd.DataFrame([{
    "fold": "OOD_SL", "anos_treino": "2015-2024", "ano_teste": "SL_all",
    "auroc_wf": auroc_sl, "n_train": len(y_all), "n_test": len(y_sl)
}])
metricas_all = pd.concat([metricas, metricas_ood], ignore_index=True)
metricas_all.to_csv(f"{CSV_DIR}\\12_metricas_final.csv", index=False, encoding="utf-8")
print(f"\n✅ Métricas salvas: {CSV_DIR}\\12_metricas_final.csv")

# ── Salvar modelo ────────────────────────────────────────────────────────────
meta = {
    "features": FEATURES,
    "target": TARGET,
    "auroc_walk_forward_mean": float(np.mean(auroc_wf_vals)),
    "auroc_walk_forward_std":  float(np.std(auroc_wf_vals)),
    "auroc_ood_sao_luis":      float(auroc_sl),
    "precision_sl":            float(prec_sl),
    "recall_sl":               float(rec_sl),
    "f1_sl":                   float(f1_sl),
    "n_train":                 int(len(y_all)),
    "y_source":                "icp_degradacao",
    "versao":                  "final_v3",
    "modelo":                  modelo_final,
}

pkl_path = f"{MODEL_DIR}\\modelo_final.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(meta, f)

print(f"✅ Modelo salvo: {pkl_path}")
print(f"\n{'='*60}")
print(f"  RESUMO FINAL DO MODELO")
print(f"{'='*60}")
print(f"  AUROC Walk-Forward: {np.mean(auroc_wf_vals):.4f} ± {np.std(auroc_wf_vals):.4f}")
print(f"  AUROC OOD São Luís: {auroc_sl:.4f}")
print(f"  Δ WF→OOD:           {auroc_sl - np.mean(auroc_wf_vals):+.4f}")
print(f"  Features:           {len(FEATURES)}")
print(f"  Y=source:           ICP Degradação Física")
