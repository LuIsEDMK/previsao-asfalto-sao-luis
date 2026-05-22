"""
Projeto: Preditor de Falhas em Vias - Sao Luis MA (v2 - Dados Publicos)
Tarefa 8: Ordens de Servico Publicas (Fortaleza / Recife / SP)
Saida:    07_public_repair_orders.csv

Tenta portais em ordem:
  8A) Fortaleza: dados.fortaleza.ce.gov.br (CKAN)
  8B) Recife:    dados.recife.pe.gov.br (CKAN)
  8C) Sao Paulo: dados.prefeitura.sp.gov.br (CKAN) — como referencia
  8D) Fallback:  tabela sintetica anotada com fonte
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time, warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"
SAIDA    = f"{CSV_DIR}\\07_public_repair_orders.csv"

TIMEOUT  = 20
SEMENTE  = 42

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ─── 8A: Fortaleza CKAN ──────────────────────────────────────────

def tentar_fortaleza():
    """Busca datasets de manutencao viaria em dados.fortaleza.ce.gov.br."""
    base = "https://dados.fortaleza.ce.gov.br/api/3/action"
    termos = ["pavimento", "tapa buraco", "manutencao via", "obras viarias"]

    for termo in termos:
        try:
            r = requests.get(f"{base}/package_search",
                             params={"q": termo, "rows": 5},
                             timeout=TIMEOUT)
            if r.status_code == 200:
                res = r.json()
                count = res.get("result", {}).get("count", 0)
                if count > 0:
                    print(f"  [FORTALEZA] '{termo}': {count} datasets")
                    # Tentar baixar primeiro recurso CSV
                    pkgs = res["result"]["results"]
                    for pkg in pkgs:
                        for res_item in pkg.get("resources", []):
                            if res_item.get("format", "").upper() in ["CSV", "XLSX"]:
                                url_csv = res_item.get("url", "")
                                try:
                                    df = pd.read_csv(url_csv, encoding="utf-8",
                                                     nrows=500, on_bad_lines="skip")
                                    print(f"  [OK] {len(df)} linhas de '{pkg['name']}'")
                                    return df, "fortaleza_opendata"
                                except Exception:
                                    pass
        except Exception as e:
            print(f"  [FORTALEZA] Erro em '{termo}': {e}")
        time.sleep(0.5)

    print("  [FORTALEZA] Nenhum dataset acessivel")
    return None, None

# ─── 8B: Recife CKAN ─────────────────────────────────────────────

def tentar_recife():
    """Busca datasets de manutencao viaria em dados.recife.pe.gov.br."""
    base = "https://dados.recife.pe.gov.br/api/3/action"
    termos = ["pavimento", "tapa-buraco", "conservacao vias", "manutencao"]

    for termo in termos:
        try:
            r = requests.get(f"{base}/package_search",
                             params={"q": termo, "rows": 5},
                             timeout=TIMEOUT)
            if r.status_code == 200:
                res = r.json()
                count = res.get("result", {}).get("count", 0)
                if count > 0:
                    print(f"  [RECIFE] '{termo}': {count} datasets")
                    pkgs = res["result"]["results"]
                    for pkg in pkgs:
                        for res_item in pkg.get("resources", []):
                            if res_item.get("format", "").upper() in ["CSV", "XLSX"]:
                                url_csv = res_item.get("url", "")
                                try:
                                    df = pd.read_csv(url_csv, encoding="utf-8",
                                                     nrows=500, on_bad_lines="skip")
                                    print(f"  [OK] {len(df)} linhas de '{pkg['name']}'")
                                    return df, "recife_opendata"
                                except Exception:
                                    pass
        except Exception as e:
            print(f"  [RECIFE] Erro em '{termo}': {e}")
        time.sleep(0.5)

    print("  [RECIFE] Nenhum dataset acessivel")
    return None, None

# ─── 8C: Sao Paulo CKAN ──────────────────────────────────────────

def tentar_sao_paulo():
    """Busca datasets de tapa-buraco / pavimento em dados.prefeitura.sp.gov.br."""
    base = "https://dados.prefeitura.sp.gov.br/api/3/action"
    termos = ["tapa buraco", "pavimento", "recape"]

    for termo in termos:
        try:
            r = requests.get(f"{base}/package_search",
                             params={"q": termo, "rows": 3},
                             timeout=TIMEOUT)
            if r.status_code == 200:
                res = r.json()
                count = res.get("result", {}).get("count", 0)
                if count > 0:
                    print(f"  [SP] '{termo}': {count} datasets")
                    pkgs = res["result"]["results"]
                    for pkg in pkgs:
                        for res_item in pkg.get("resources", []):
                            if res_item.get("format", "").upper() == "CSV":
                                url_csv = res_item.get("url", "")
                                try:
                                    df = pd.read_csv(url_csv, encoding="latin-1",
                                                     nrows=500, on_bad_lines="skip")
                                    print(f"  [OK] {len(df)} linhas de '{pkg['name']}'")
                                    return df, "saopaulo_opendata"
                                except Exception:
                                    pass
        except Exception as e:
            print(f"  [SP] Erro em '{termo}': {e}")
        time.sleep(0.5)

    print("  [SP] Nenhum dataset acessivel")
    return None, None

# ─── 8D: Fallback sintetico ───────────────────────────────────────

def gerar_os_sintetica(n=2000):
    """
    Gera tabela sintetica de Ordens de Servico para referencia do modelo.
    Representa a estrutura de dados que viria de portais reais.
    """
    rng = np.random.default_rng(SEMENTE)

    cidades = (
        ["fortaleza"] * (n // 2)
        + ["recife"] * (n // 4)
        + ["sao_paulo"] * (n // 4)
    )
    rng.shuffle(cidades)

    anos = rng.integers(2015, 2025, size=n)
    meses = rng.integers(1, 13, size=n)

    tipos = ["tapa-buraco", "recapeamento", "remendo", "conservacao", "drenagem"]
    pesos = [0.45, 0.20, 0.20, 0.10, 0.05]
    tipo_os = rng.choice(tipos, size=n, p=pesos)

    prioridades = ["urgente", "alta", "media", "baixa"]
    prioridade  = rng.choice(prioridades, size=n, p=[0.15, 0.30, 0.40, 0.15])

    custo_base = {"tapa-buraco": 800, "recapeamento": 15000,
                  "remendo": 1200, "conservacao": 3000, "drenagem": 8000}
    custos = [custo_base[t] * rng.uniform(0.7, 1.5) for t in tipo_os]

    # Status de conclusao (simulado)
    concluido = rng.choice([0, 1], size=n, p=[0.12, 0.88])

    return pd.DataFrame({
        "os_id":          [f"OS-{i+1:05d}" for i in range(n)],
        "cidade":         cidades,
        "ano":            anos,
        "mes":            meses,
        "tipo_servico":   tipo_os,
        "prioridade":     prioridade,
        "custo_estimado": np.round(custos, 2),
        "concluido":      concluido,
        "fonte":          "sintetico_referencia",
    })

# ─── EXECUCAO PRINCIPAL ───────────────────────────────────────────

sep("TAREFA 8 — Ordens de Servico Publicas")

if os.path.exists(SAIDA):
    df_ex = pd.read_csv(SAIDA)
    print(f"  [INFO] Arquivo ja existe com {len(df_ex):,} linhas. Nada a fazer.")
    print(df_ex.groupby(["cidade","tipo_servico"]).size().unstack(fill_value=0).to_string())
    sys.exit(0)

sep("Tentando portais de dados abertos")
df_os = None
fonte  = None

print("  [8A] Fortaleza...")
df_os, fonte = tentar_fortaleza()

if df_os is None:
    print("\n  [8B] Recife...")
    df_os, fonte = tentar_recife()

if df_os is None:
    print("\n  [8C] Sao Paulo...")
    df_os, fonte = tentar_sao_paulo()

if df_os is None:
    print("\n  Nenhum portal acessivel. Usando fallback sintetico (8D).")
    df_os = gerar_os_sintetica(n=2000)
    fonte  = "sintetico_referencia"

sep("Padronizando e salvando")
# Garantir colunas minimas presentes
for col in ["cidade","ano","mes","tipo_servico","prioridade","fonte"]:
    if col not in df_os.columns:
        df_os[col] = "desconhecido"

if "fonte" not in df_os.columns or df_os["fonte"].isna().all():
    df_os["fonte"] = fonte

df_os.to_csv(SAIDA, index=False, encoding="utf-8")

sep("RESUMO FINAL — TAREFA 8")
print(f"  Fonte OS: {fonte}")
print(f"  Total registros: {len(df_os):,}")
print()
print("  Por cidade:")
print(df_os.groupby("cidade").size().to_string())
print()
if "tipo_servico" in df_os.columns:
    print("  Por tipo de servico:")
    print(df_os["tipo_servico"].value_counts().to_string())
print()
print(f"  [OK] {SAIDA}")
print(f"  TAREFA 8 CONCLUIDA")
