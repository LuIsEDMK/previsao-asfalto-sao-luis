"""
Tarefa 1 — Coletar OS Reais de São Paulo
Tenta 5 fontes em ordem; documenta cada tentativa.
Saída: sp_repair_orders_raw.csv (se OK) ou sp_os_status.txt="FALLBACK"
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time, warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd

BASE_DIR = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR  = f"{BASE_DIR}\\csv"
STATUS_FILE = f"{CSV_DIR}\\sp_os_status.txt"
SAIDA_RAW   = f"{CSV_DIR}\\sp_repair_orders_raw.csv"

TIMEOUT = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-project"}

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

def inspecionar(df, fonte):
    """Verifica se df tem o mínimo necessário para o projeto."""
    col_lower = " ".join(df.columns.str.lower())
    tem_data  = any(p in col_lower for p in ["data","date","dt_","abertura","inicio","ano"])
    tem_local = any(p in col_lower for p in ["lat","lon","longitude","latitude",
                                              "endereco","endereço","logradouro","bairro"])
    tem_sp    = df.shape[0] > 10  # qualquer dado substancial
    print(f"    Shape: {df.shape}")
    print(f"    Colunas: {list(df.columns[:8])}")
    print(f"    tem_data={tem_data} | tem_local={tem_local} | tem_sp={tem_sp}")
    if tem_data and tem_local and tem_sp:
        print(f"    ✅ Dataset utilizável")
        return True
    if not tem_local:
        print(f"    ❌ Sem localização — não geocodificável")
    if not tem_data:
        print(f"    ❌ Sem data — impossível criar Y temporal")
    return False

resultados = []

# ═══════════════════════════════════════════
# TENTATIVA 1 — CKAN dados.prefeitura.sp.gov.br
# ═══════════════════════════════════════════
sep("TENTATIVA 1 — CKAN dados.prefeitura.sp.gov.br")
ok1 = False
try:
    r = requests.get(
        "https://dados.prefeitura.sp.gov.br/api/3/action/package_search",
        params={"q": "tapa buraco pavimento manutencao", "rows": 20},
        timeout=TIMEOUT, headers=HEADERS
    )
    print(f"  Status: {r.status_code} | Conteúdo: {len(r.content)} bytes")
    if r.status_code == 200:
        data = r.json()
        count = data.get("result", {}).get("count", 0)
        print(f"  Datasets encontrados: {count}")
        if count > 0:
            pkgs = data["result"]["results"]
            for pkg in pkgs[:5]:
                print(f"    → {pkg.get('name','?')}")
                for res in pkg.get("resources", []):
                    fmt = res.get("format", "").upper()
                    if fmt in ["CSV", "JSON", "XLSX"]:
                        url_csv = res.get("url", "")
                        print(f"      Baixando {fmt}: {url_csv[:80]}...")
                        try:
                            df_raw = pd.read_csv(url_csv, encoding="utf-8",
                                                 nrows=500, on_bad_lines="skip",
                                                 timeout=TIMEOUT)
                            if inspecionar(df_raw, "CKAN SP"):
                                df_raw.to_csv(SAIDA_RAW, index=False, encoding="utf-8")
                                ok1 = True
                                break
                        except Exception as e:
                            print(f"      Falha ao baixar: {e}")
                if ok1:
                    break
        else:
            print("  Nenhum dataset retornado")
    else:
        print(f"  HTTP {r.status_code}")
except Exception as e:
    print(f"  Erro: {e}")

resultados.append(f"[TENTATIVA 1] CKAN SP → {'✅ OK' if ok1 else '❌ Sem dados utilizáveis'}")

# ═══════════════════════════════════════════
# TENTATIVA 2 — Dataset específico Tapa Buraco SP
# ═══════════════════════════════════════════
sep("TENTATIVA 2 — Datasets Específicos de Pavimentação")
ok2 = False
if not ok1:
    slugs = [
        "servicos-de-tapa-buraco",
        "manutencao-de-vias-publicas",
        "ordens-de-servico-de-zeladoria",
        "pavimentacao",
        "conservacao-de-vias",
    ]
    base = "https://dados.prefeitura.sp.gov.br/api/3/action/package_show"
    for slug in slugs:
        try:
            r = requests.get(base, params={"id": slug}, timeout=TIMEOUT, headers=HEADERS)
            print(f"  [{slug}] status={r.status_code}", end=" ")
            if r.status_code == 200:
                pkg = r.json().get("result", {})
                print(f"→ {pkg.get('title', '?')}")
                for res in pkg.get("resources", [])[:3]:
                    if res.get("format", "").upper() in ["CSV", "JSON"]:
                        try:
                            df_raw = pd.read_csv(res["url"], nrows=300,
                                                 encoding="utf-8", on_bad_lines="skip")
                            if inspecionar(df_raw, slug):
                                df_raw.to_csv(SAIDA_RAW, index=False, encoding="utf-8")
                                ok2 = True
                                break
                        except Exception:
                            pass
            else:
                print()
            if ok2:
                break
        except Exception as e:
            print(f"  [{slug}] erro: {e}")

resultados.append(f"[TENTATIVA 2] Datasets SP → {'✅ OK' if ok2 else '❌ Não acessíveis'}")

# ═══════════════════════════════════════════
# TENTATIVA 3 — Base dos Dados (basedosdados.org)
# ═══════════════════════════════════════════
sep("TENTATIVA 3 — Base dos Dados (basedosdados.org)")
ok3 = False
if not ok1 and not ok2:
    try:
        r = requests.get(
            "https://basedosdados.org/api/3/action/package_search",
            params={"q": "sao paulo pavimentacao buraco tapa", "rows": 10},
            timeout=TIMEOUT, headers=HEADERS
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            count = data.get("result", {}).get("count", 0)
            print(f"  Datasets encontrados: {count}")
            if count > 0:
                for pkg in data["result"]["results"][:5]:
                    print(f"    → {pkg.get('name','?')} | {pkg.get('title','?')}")
    except Exception as e:
        print(f"  Erro: {e}")

resultados.append(f"[TENTATIVA 3] Base dos Dados → {'✅ OK' if ok3 else '❌ Sem dados úteis'}")

# ═══════════════════════════════════════════
# TENTATIVA 4 — SP156 (central de atendimento)
# ═══════════════════════════════════════════
sep("TENTATIVA 4 — SP156 e GitHub prefeitura-sp")
ok4 = False
if not ok1 and not ok2 and not ok3:
    # GitHub API
    try:
        r = requests.get(
            "https://api.github.com/orgs/prefeitura-sp/repos",
            params={"per_page": 30}, timeout=TIMEOUT, headers=HEADERS
        )
        print(f"  GitHub prefeitura-sp: status={r.status_code}")
        if r.status_code == 200:
            repos = r.json()
            print(f"  Repositórios: {len(repos)}")
            for repo in repos:
                if any(p in repo["name"].lower() for p in
                       ["dado","pavi","buraco","manut","servico"]):
                    print(f"    → {repo['name']} | {repo['html_url']}")
    except Exception as e:
        print(f"  GitHub erro: {e}")

resultados.append(f"[TENTATIVA 4] SP156/GitHub → {'✅ OK' if ok4 else '❌ Sem CSV acessível'}")

# ═══════════════════════════════════════════
# TENTATIVA 5 — URLs diretas conhecidas
# ═══════════════════════════════════════════
sep("TENTATIVA 5 — URLs Diretas")
ok5 = False
if not ok1 and not ok2 and not ok3 and not ok4:
    urls = [
        "https://dados.prefeitura.sp.gov.br/dataset/b9d77ea5-4ad1-4b56-9a9a-e8e6e8f2e2f6/resource//download/tapaburaco.csv",
        "https://raw.githubusercontent.com/prefeitura-sp/dados-abertos/main/pavimentacao.csv",
        "https://raw.githubusercontent.com/prefeitura-sp/dados-abertos/main/tapa_buraco.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
            print(f"  {url[:70]}... status={r.status_code}")
            if r.status_code == 200 and len(r.content) > 5000:
                from io import StringIO
                df_raw = pd.read_csv(StringIO(r.text), nrows=300,
                                     encoding="utf-8", on_bad_lines="skip")
                if inspecionar(df_raw, "URL direta"):
                    df_raw.to_csv(SAIDA_RAW, index=False, encoding="utf-8")
                    ok5 = True
                    break
        except Exception as e:
            print(f"  Erro: {e}")

resultados.append(f"[TENTATIVA 5] URLs diretas → {'✅ OK' if ok5 else '❌ Não acessíveis'}")

# ═══════════════════════════════════════════
# RESUMO E STATUS FINAL
# ═══════════════════════════════════════════
sep("RESUMO DAS TENTATIVAS")
sucesso = any([ok1, ok2, ok3, ok4, ok5])
for r_txt in resultados:
    print(f"  {r_txt}")

if sucesso:
    print(f"\n  ✅ OS de SP obtidas → {SAIDA_RAW}")
    status = "OK"
else:
    print(f"\n  ⚠  Zero OS reais obtidas de SP.")
    print(f"  Ativando modo alternativo — Modelo de Degradação Física (Tarefa 3).")
    status = "FALLBACK"

with open(STATUS_FILE, "w", encoding="utf-8") as f:
    f.write(status)
print(f"\n  Status gravado: {STATUS_FILE} = {status}")
