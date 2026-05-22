"""
Projeto: Preditor de Falhas em Vias - Sao Luis / MA
Script:  fetch_inmet_real.py
Tarefa:  Substituir dados de chuva simulados por dados reais
         Metodos: INMET API -> BDMEP -> Open-Meteo (fallback gratuito)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import io
import shutil
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime

CSV_DIR = r"w:\projetos vscode\projetos para prefeitura\csv"
COD_DIR = r"w:\projetos vscode\projetos para prefeitura\1projeto\codigos"

ARQUIVO_CHUVA  = f"{CSV_DIR}/03_rainfall_data.csv"
ARQUIVO_BACKUP = f"{CSV_DIR}/03_rainfall_data_SIMULADO_backup.csv"
ARQUIVO_PROC   = f"{CSV_DIR}/03_rainfall_data_processed.csv"
ARQUIVO_META   = f"{CSV_DIR}/03_rainfall_metadata.txt"
ARQUIVO_ML     = f"{CSV_DIR}/06_ml_training_dataset.csv"
ARQUIVO_ANA    = f"{COD_DIR}/08_analytical_dataset.csv"

ANO_INI = 2015
ANO_FIM = 2024

MESES_PT = {
    1: "janeiro", 2: "fevereiro", 3: "marco", 4: "abril",
    5: "maio", 6: "junho", 7: "julho", 8: "agosto",
    9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
}

def sep(t=""):
    if t:
        print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ===================================================================
# METODO 1 — API INMET oficial
# ===================================================================

def metodo1_inmet_api():
    sep("METODO 1 — API INMET")
    print("  [METODO 1] Tentando API INMET...")

    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

    # Endpoint a) — dados por periodo
    url_a = "https://apitempo.inmet.gov.br/estacao/2015-01-01/2024-12-31/A203"
    try:
        r = requests.get(url_a, headers=headers, timeout=30)
        print(f"  [METODO 1] Status endpoint a): {r.status_code}")
        if r.status_code == 200:
            dados = r.json()
            if isinstance(dados, list) and len(dados) > 10:
                df = _parsear_json_inmet(dados, "INMET_API")
                if df is not None and len(df) >= 10:
                    print(f"  [METODO 1] OK — {len(df)} meses obtidos")
                    return df, "INMET_API", 1
    except Exception as e:
        print(f"  [METODO 1] Erro endpoint a): {e}")

    # Endpoint b) — mensal direto
    url_b = "https://apitempo.inmet.gov.br/estacao/mensal/A203"
    try:
        r = requests.get(url_b, headers=headers, timeout=30)
        print(f"  [METODO 1] Status endpoint b): {r.status_code}")
        if r.status_code == 200:
            dados = r.json()
            if isinstance(dados, list) and len(dados) > 10:
                df = _parsear_json_inmet(dados, "INMET_API")
                if df is not None and len(df) >= 10:
                    print(f"  [METODO 1] OK — {len(df)} meses obtidos")
                    return df, "INMET_API", 1
    except Exception as e:
        print(f"  [METODO 1] Erro endpoint b): {e}")

    print("  [METODO 1] FALHOU — passando para Metodo 2")
    return None, None, None


def _parsear_json_inmet(dados, fonte):
    """Agrega registros horarios/diarios da API INMET em mensais."""
    registros = []
    for d in dados:
        dt_str = (d.get("DT_MEDICAO") or d.get("data")
                  or d.get("DT_INICIO") or d.get("datetime") or "")
        if not dt_str:
            continue
        try:
            dt = pd.to_datetime(str(dt_str)[:10])
        except Exception:
            continue
        ano, mes = dt.year, dt.month
        if not (ANO_INI <= ano <= ANO_FIM):
            continue

        chuva_raw = (d.get("CHUVA") or d.get("PRECIPITACAO_TOTAL")
                     or d.get("precipitacao_total") or d.get("VL_MEDICAO") or 0)
        try:
            chuva = float(str(chuva_raw).replace(",", "."))
            if chuva < 0:
                chuva = 0.0
        except Exception:
            chuva = 0.0

        registros.append({"year": ano, "month": mes, "day": dt.day,
                           "daily_mm": chuva})

    if len(registros) < 30:
        return None

    df_d = pd.DataFrame(registros)
    df_m = df_d.groupby(["year", "month"]).agg(
        rainfall_mm=("daily_mm", "sum"),
        rainy_days=("daily_mm", lambda x: (x > 0.1).sum()),
        max_daily_mm=("daily_mm", "max"),
        extreme_events_50mm=("daily_mm", lambda x: (x >= 50).sum()),
    ).reset_index()
    df_m["data_source"] = fonte
    return df_m


# ===================================================================
# METODO 2 — BDMEP download direto
# ===================================================================

def metodo2_bdmep():
    sep("METODO 2 — Download BDMEP")
    print("  [METODO 2] Tentando download BDMEP...")

    urls = [
        "https://bdmep.inmet.gov.br/dados/automaticas/A203.zip",
        "https://bdmep.inmet.gov.br/dados/automaticas/A203.CSV",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            tamanho = len(r.content) / 1024
            print(f"  [METODO 2] Status: {r.status_code} | Tamanho: {tamanho:.1f} KB")
            if r.status_code == 200 and tamanho > 5:
                conteudo = None
                # Tentar descompactar zip
                try:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    csvs = [n for n in z.namelist()
                            if n.lower().endswith(".csv") or n.lower().endswith(".txt")]
                    if csvs:
                        with z.open(csvs[0]) as fz:
                            conteudo = fz.read().decode("latin-1", errors="replace")
                        print(f"  [METODO 2] ZIP extraido: {csvs[0]}")
                except Exception:
                    conteudo = r.content.decode("latin-1", errors="replace")

                if conteudo:
                    df = _parsear_csv_bdmep(conteudo)
                    if df is not None and len(df) >= 10:
                        print(f"  [METODO 2] OK — {len(df)} meses obtidos")
                        return df, "INMET_BDMEP", 2
        except Exception as e:
            print(f"  [METODO 2] Erro ({url}): {e}")

    # Verificar download manual do usuario
    manual = os.path.join(CSV_DIR, "inmet_manual_download.csv")
    if os.path.exists(manual):
        print(f"  [METODO 2] Arquivo manual encontrado: {manual}")
        try:
            with open(manual, "r", encoding="latin-1", errors="replace") as f:
                conteudo = f.read()
            df = _parsear_csv_bdmep(conteudo)
            if df is not None and len(df) >= 10:
                print(f"  [METODO 2] OK (manual) — {len(df)} meses obtidos")
                return df, "INMET_BDMEP", 2
        except Exception as e:
            print(f"  [METODO 2] Erro ao ler arquivo manual: {e}")

    print("  [METODO 2] FALHOU — passando para Metodo 3")
    return None, None, None


def _parsear_csv_bdmep(conteudo):
    """Parseia CSV BDMEP (formato horario com cabecalho INMET) e agrega mensalmente."""
    linhas = conteudo.splitlines()

    # Localizar linha de cabecalho de dados
    header_idx = None
    for i, linha in enumerate(linhas):
        baixo = linha.lower()
        if ("data" in baixo or "date" in baixo) and ("hora" in baixo or "hour" in baixo or ";" in linha):
            header_idx = i
            break

    if header_idx is None:
        # Tentar sem cabecalho de metadados
        for i, linha in enumerate(linhas):
            if ";" in linha and len(linha.split(";")) >= 3:
                header_idx = i
                break

    if header_idx is None:
        print("  [METODO 2] Cabecalho nao encontrado")
        return None

    try:
        texto = "\n".join(linhas[header_idx:])
        df_raw = pd.read_csv(
            io.StringIO(texto), sep=";", decimal=",",
            encoding="utf-8", on_bad_lines="skip", low_memory=False,
        )
    except Exception as e:
        print(f"  [METODO 2] Erro ao parsear CSV: {e}")
        return None

    # Identificar colunas de data e precipitacao
    col_data, col_chuva = None, None
    for c in df_raw.columns:
        cl = c.lower()
        if ("data" in cl or "date" in cl) and col_data is None:
            col_data = c
        if ("precipita" in cl or "chuva" in cl) and col_chuva is None:
            col_chuva = c

    if not col_data or not col_chuva:
        print(f"  [METODO 2] Colunas nao identificadas. Disponiveis: {list(df_raw.columns[:8])}")
        return None

    df_raw["_dt"] = pd.to_datetime(df_raw[col_data].astype(str).str[:10],
                                   errors="coerce")
    df_raw = df_raw.dropna(subset=["_dt"])
    df_raw["year"]  = df_raw["_dt"].dt.year
    df_raw["month"] = df_raw["_dt"].dt.month
    df_raw["day"]   = df_raw["_dt"].dt.day
    df_raw["_mm"] = pd.to_numeric(
        df_raw[col_chuva].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0).clip(lower=0)

    # Agregar diario
    df_d = df_raw.groupby(["year", "month", "day"])["_mm"].sum().reset_index()
    df_d.rename(columns={"_mm": "daily_mm"}, inplace=True)

    # Agregar mensal
    df_m = df_d.groupby(["year", "month"]).agg(
        rainfall_mm=("daily_mm", "sum"),
        rainy_days=("daily_mm", lambda x: (x > 0.1).sum()),
        max_daily_mm=("daily_mm", "max"),
        extreme_events_50mm=("daily_mm", lambda x: (x >= 50).sum()),
    ).reset_index()
    df_m = df_m[(df_m["year"] >= ANO_INI) & (df_m["year"] <= ANO_FIM)]
    df_m["data_source"] = "INMET_BDMEP"
    return df_m if len(df_m) >= 10 else None


# ===================================================================
# METODO 3 — Open-Meteo (fallback gratuito, sem API key)
# ===================================================================

def metodo3_open_meteo():
    sep("METODO 3 — Open-Meteo API")
    print("  [METODO 3] Tentando Open-Meteo...")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=-2.5297&longitude=-44.3028"
        f"&start_date={ANO_INI}-01-01&end_date={ANO_FIM}-12-31"
        "&daily=precipitation_sum&timezone=America/Fortaleza"
    )
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        print(f"  [METODO 3] Status: {r.status_code}")
        if r.status_code != 200:
            print("  [METODO 3] FALHOU")
            return None, None, None

        dados = r.json()
        datas  = dados["daily"]["time"]
        precip = dados["daily"]["precipitation_sum"]
        print(f"  [METODO 3] Dados recebidos: {len(datas)} dias")

        df_d = pd.DataFrame({
            "date":     pd.to_datetime(datas),
            "daily_mm": pd.Series(precip, dtype=float).fillna(0).clip(lower=0),
        })
        df_d["year"]  = df_d["date"].dt.year
        df_d["month"] = df_d["date"].dt.month

        df_m = df_d.groupby(["year", "month"]).agg(
            rainfall_mm=("daily_mm", "sum"),
            rainy_days=("daily_mm", lambda x: (x > 0.1).sum()),
            max_daily_mm=("daily_mm", "max"),
            extreme_events_50mm=("daily_mm", lambda x: (x >= 50).sum()),
        ).reset_index()
        df_m = df_m[(df_m["year"] >= ANO_INI) & (df_m["year"] <= ANO_FIM)]
        df_m["data_source"] = "open-meteo"

        print(f"  [METODO 3] OK - {len(df_m)} meses agregados")
        return df_m, "open-meteo", 3

    except Exception as e:
        print(f"  [METODO 3] Erro: {e}")
        return None, None, None


# ===================================================================
# INSTRUCOES DE DOWNLOAD MANUAL
# ===================================================================

def imprimir_instrucoes_manuais():
    print("""
  ======================================
  DOWNLOAD MANUAL - INMET BDMEP
  ======================================
  Todos os 3 metodos automaticos falharam.
  Siga estas instrucoes para download manual:

  1. Acesse: https://bdmep.inmet.gov.br/
  2. Clique em "Dados Meteorologicos"
  3. Selecione: Tipo = Automatica
  4. Estacao = A203 - SAO LUIS - MA
  5. Periodo = 01/01/2015 a 31/12/2024
  6. Variavel = Precipitacao Total (mm)
  7. Clique em "Gerar CSV"
  8. Salve como: inmet_manual_download.csv
     na pasta: w:\\projetos vscode\\projetos para prefeitura\\csv\\
  9. Rode novamente: py -X utf8 fetch_inmet_real.py
     O script detectara o arquivo manual automaticamente.
  ======================================
""")


# ===================================================================
# FORMATAR PARA O PADRAO DO PROJETO
# ===================================================================

def formatar_dataframe(df_mensal, fonte):
    """Formata o DataFrame bruto para o padrao exato do projeto."""
    estacao_id   = "A203" if "INMET" in fonte else "open-meteo"
    estacao_nome = "Sao Luis INMET" if "INMET" in fonte else "Open-Meteo Sao Luis"

    df = df_mensal.copy()
    df["year"]          = df["year"].astype(int)
    df["month"]         = df["month"].astype(int)
    df["month_name"]    = df["month"].map(MESES_PT)
    df["station_id"]    = estacao_id
    df["station_name"]  = estacao_nome
    df["rainfall_mm"]   = df["rainfall_mm"].round(1)
    df["rainy_days"]    = df["rainy_days"].astype(int)
    df["max_daily_mm"]  = df["max_daily_mm"].round(1)
    df["extreme_events_50mm"] = df["extreme_events_50mm"].astype(int)
    df["data_source"]   = fonte
    df["notes"]         = ""

    # Filtrar e ordenar
    df = df[(df["year"] >= ANO_INI) & (df["year"] <= ANO_FIM)]
    df = df[df["rainfall_mm"].notna() & (df["rainfall_mm"] >= 0)]
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    colunas = [
        "year", "month", "month_name", "station_id", "station_name",
        "rainfall_mm", "rainy_days", "max_daily_mm", "extreme_events_50mm",
        "data_source", "notes",
    ]
    return df[colunas]


# ===================================================================
# VALIDACAO CLIMATICA
# ===================================================================

def validar_dados(df):
    """Valida plausibilidade climatica para Sao Luis MA."""
    sep("PASSO 4 - VALIDACAO DOS DADOS REAIS")

    # Cobertura temporal
    ano_min  = df["year"].min()
    mes_min  = df[df["year"] == ano_min]["month"].min()
    ano_max  = df["year"].max()
    mes_max  = df[df["year"] == ano_max]["month"].max()
    n_meses  = len(df)
    print(f"\n  Cobertura: {ano_min}-{mes_min:02d} a {ano_max}-{mes_max:02d} ({n_meses} meses)")

    cobertura_ok = (ano_min <= 2019 and ano_max >= 2024)
    if not cobertura_ok:
        print("  [AV] Cobertura menor que o esperado (2019-2024)")
    else:
        print("  [OK] Cobertura temporal adequada (2019-2024)")

    # Validacao de ranges
    alertas = []
    for _, row in df.iterrows():
        mes = row["month"]
        mm  = row["rainfall_mm"]
        if mm > 700:
            alertas.append(f"  [ALERTA] {row['year']}-{mes:02d}: {mm}mm > 700mm (extremo)")
        if mm < 0:
            alertas.append(f"  [ALERTA] {row['year']}-{mes:02d}: {mm}mm negativo!")
    for a in alertas:
        print(a)

    # Estatisticas resumidas
    media_jan_jun = df[df["month"].isin(range(1, 7))]["rainfall_mm"].mean()
    media_jul_dez = df[df["month"].isin(range(7, 13))]["rainfall_mm"].mean()
    mes_mais_chuvoso = (
        df.groupby("month")["rainfall_mm"].mean()
        .idxmax()
    )
    mes_mais_seco = (
        df.groupby("month")["rainfall_mm"].mean()
        .idxmin()
    )
    total_extremos = df["extreme_events_50mm"].sum()
    anos_disponiveis = df["year"].nunique()

    print(f"\n  Media mensal jan-jun : {media_jan_jun:.1f}mm")
    print(f"  Media mensal jul-dez : {media_jul_dez:.1f}mm")
    print(f"  Ratio chuvoso/seco   : {media_jan_jun/media_jul_dez:.1f}x")
    print(f"  Mes mais chuvoso     : {MESES_PT.get(mes_mais_chuvoso, mes_mais_chuvoso)}")
    print(f"  Mes mais seco        : {MESES_PT.get(mes_mais_seco, mes_mais_seco)}")
    print(f"  Eventos extremos/ano : {total_extremos/max(anos_disponiveis, 1):.1f} dias/ano >= 50mm")
    print(f"  Total anual medio    : {df.groupby('year')['rainfall_mm'].sum().mean():.0f}mm")

    # Validacoes booleanas
    ok_chuvosa = media_jan_jun > 200
    ok_seca    = media_jul_dez < 200
    ok_pos     = (df["rainfall_mm"] >= 0).all()
    ok_max     = (df["rainfall_mm"] <= 700).all()

    print(f"\n  VALIDACAO CLIMATICA SAO LUIS:")
    print(f"  {'[OK]' if ok_chuvosa else '[ALERTA]'}  Media jan-jun > 200mm  ({media_jan_jun:.0f}mm)")
    print(f"  {'[OK]' if ok_seca else '[ALERTA]'}  Media jul-dez < 200mm  ({media_jul_dez:.0f}mm)")
    print(f"  {'[OK]' if ok_pos else '[ERRO]'}  Sem valores negativos")
    print(f"  {'[OK]' if ok_max else '[ALERTA]'}  Sem meses acima de 700mm")

    return ok_chuvosa, ok_seca, ok_pos, ok_max


# ===================================================================
# COMPARACAO COM DADOS SIMULADOS
# ===================================================================

def comparar_com_simulados(df_real, df_sim):
    """Imprime tabela comparativa real x simulado para anos em comum."""
    sep("Comparacao Real x Simulado")

    anos_comuns = sorted(set(df_real["year"].unique()) & set(df_sim["year"].unique()))
    if not anos_comuns:
        print("  Nenhum ano em comum para comparacao.")
        return

    print(f"\n  {'Mes':<12} {'Simulado':>10}  {'Real':>10}  {'Diferenca':>12}")
    print("  " + "-"*48)

    for ano in anos_comuns[:3]:  # mostrar ate 3 anos
        for mes in range(1, 13):
            r_real = df_real[(df_real["year"] == ano) & (df_real["month"] == mes)]
            r_sim  = df_sim[(df_sim["year"] == ano) & (df_sim["month"] == mes)]
            if r_real.empty or r_sim.empty:
                continue
            mm_real = r_real["rainfall_mm"].values[0]
            mm_sim  = r_sim["rainfall_mm"].values[0]
            diff    = mm_real - mm_sim
            sinal   = "+" if diff >= 0 else ""
            nome_mes = MESES_PT.get(mes, str(mes))
            print(f"  {nome_mes[:3]}/{ano}     {mm_sim:>9.1f}mm  {mm_real:>9.1f}mm  {sinal}{diff:>9.1f}mm")


# ===================================================================
# CALCULAR JANELAS ROLLING (igual pipeline_limpeza.py PASSO 3C)
# ===================================================================

def calcular_rolling(df):
    """
    Calcula precipitacoes acumuladas com janelas temporais.
    Replica o PASSO 3C do pipeline_limpeza.py.
    """
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values("date").reset_index(drop=True)

    df["rainfall_30d"]  = df["rainfall_mm"]
    df["rainfall_90d"]  = df["rainfall_mm"].rolling(3,  min_periods=1).sum().round(1)
    df["rainfall_180d"] = df["rainfall_mm"].rolling(6,  min_periods=1).sum().round(1)
    df["rainfall_365d"] = df["rainfall_mm"].rolling(12, min_periods=1).sum().round(1)

    # Eventos extremos acumulados 90d (soma dos ultimos 3 meses)
    df["extreme_events_90d"] = (
        df["extreme_events_50mm"].rolling(3, min_periods=1).sum().astype(int)
    )

    return df


# ===================================================================
# ATUALIZAR 06_ml_training_dataset.csv
# ===================================================================

def atualizar_ml_dataset(df_proc):
    """
    Substitui colunas de chuva no 06_ml_training_dataset.csv
    com os valores das janelas rolling calculadas dos dados reais.
    """
    sep("PASSO 5c - Atualizar 06_ml_training_dataset.csv")

    ml = pd.read_csv(ARQUIVO_ML)
    print(f"  ML dataset carregado: {ml.shape[0]} linhas x {ml.shape[1]} colunas")

    # Tabela de lookup: (year, month) -> valores rolling
    lookup = df_proc[["year", "month", "rainfall_30d", "rainfall_90d",
                       "rainfall_180d", "extreme_events_90d"]].copy()
    lookup.rename(columns={
        "rainfall_30d":  "rainfall_30d_mm_new",
        "rainfall_90d":  "rainfall_90d_mm_new",
        "rainfall_180d": "rainfall_180d_mm_new",
        "extreme_events_90d": "extreme_events_90d_new",
    }, inplace=True)

    ml = ml.merge(lookup, on=["year", "month"], how="left")

    cols_substituir = {
        "rainfall_30d_mm":    "rainfall_30d_mm_new",
        "rainfall_90d_mm":    "rainfall_90d_mm_new",
        "rainfall_180d_mm":   "rainfall_180d_mm_new",
        "extreme_events_90d": "extreme_events_90d_new",
    }

    for col_orig, col_new in cols_substituir.items():
        if col_orig in ml.columns and col_new in ml.columns:
            n_nulos = ml[col_new].isna().sum()
            if n_nulos > 0:
                print(f"  [AV] {col_orig}: {n_nulos} linhas sem correspondencia "
                      f"(mantendo valor original)")
                ml[col_orig] = ml[col_new].fillna(ml[col_orig])
            else:
                ml[col_orig] = ml[col_new]
            ml.drop(columns=[col_new], inplace=True)

    ml.to_csv(ARQUIVO_ML, index=False, encoding="utf-8")
    print(f"  [OK] 06_ml_training_dataset.csv atualizado ({ml.shape[0]} linhas)")
    return ml


# ===================================================================
# ATUALIZAR 08_analytical_dataset.csv
# ===================================================================

def atualizar_analytical(df_proc):
    """
    Atualiza current_rainfall_30d/90d/180d no 08_analytical_dataset.csv
    usando o ultimo mes disponivel nos dados reais.
    """
    sep("PASSO 5d - Atualizar 08_analytical_dataset.csv")

    ana = pd.read_csv(ARQUIVO_ANA)
    print(f"  Analytical dataset carregado: {ana.shape[0]} linhas x {ana.shape[1]} colunas")

    # Usar o mes mais recente disponivel
    ultimo = df_proc.sort_values("date").iloc[-1]
    ano_ref  = int(ultimo["year"])
    mes_ref  = int(ultimo["month"])
    r30d     = float(ultimo["rainfall_30d"])
    r90d     = float(ultimo["rainfall_90d"])
    r180d    = float(ultimo["rainfall_180d"])

    print(f"  Referencia: {ano_ref}-{mes_ref:02d} | "
          f"30d={r30d:.1f}mm | 90d={r90d:.1f}mm | 180d={r180d:.1f}mm")

    if "current_rainfall_30d" in ana.columns:
        ana["current_rainfall_30d"]  = r30d
        ana["current_rainfall_90d"]  = r90d
        ana["current_rainfall_180d"] = r180d
        print("  [OK] Colunas current_rainfall_* atualizadas")
    else:
        print("  [AV] Colunas current_rainfall_* nao encontradas no analytical — ignorando")

    ana.to_csv(ARQUIVO_ANA, index=False, encoding="utf-8")
    print(f"  [OK] 08_analytical_dataset.csv atualizado")


# ===================================================================
# ARQUIVO DE METADADOS
# ===================================================================

def salvar_metadados(fonte, metodo_num, n_meses, ok_chuvosa, ok_seca, ok_pos, ok_max):
    hoje = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conteudo = f"""Data de download  : {hoje}
Fonte             : {fonte}
Metodo usado      : {metodo_num}
Estacao           : A203 Sao Luis MA
Periodo           : {ANO_INI}-01-01 a {ANO_FIM}-12-31
Total de registros: {n_meses} meses

VALIDACAO CLIMATICA:
  Media jan-jun > 200mm : {'OK' if ok_chuvosa else 'ALERTA'}
  Media jul-dez < 200mm : {'OK' if ok_seca else 'ALERTA'}
  Sem valores negativos : {'OK' if ok_pos else 'ERRO'}
  Sem meses > 700mm     : {'OK' if ok_max else 'ALERTA'}

OBSERVACOES:
  Dados obtidos automaticamente via metodo {metodo_num}.
  Para validacao oficial, confrontar com BDMEP INMET (bdmep.inmet.gov.br).
"""
    with open(ARQUIVO_META, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"  [OK] 03_rainfall_metadata.txt salvo")


# ===================================================================
# PROGRAMA PRINCIPAL
# ===================================================================

if __name__ == "__main__":
    sep("INICIO — Coleta de Dados Reais de Chuva")
    print(f"  Estacao alvo : A203 - Sao Luis MA")
    print(f"  Periodo      : {ANO_INI} a {ANO_FIM}")
    print(f"  Script       : fetch_inmet_real.py")

    # Carregar dados simulados para comparacao posterior
    df_simulado = pd.read_csv(ARQUIVO_CHUVA)

    # ── Tentar os 3 metodos em ordem ──────────────────────────
    df_bruto, fonte, metodo_num = metodo1_inmet_api()

    if df_bruto is None:
        df_bruto, fonte, metodo_num = metodo2_bdmep()

    if df_bruto is None:
        df_bruto, fonte, metodo_num = metodo3_open_meteo()

    if df_bruto is None:
        print("\n  [ERRO] Todos os 3 metodos falharam.")
        imprimir_instrucoes_manuais()
        sys.exit(1)

    # ── Formatar para o padrao do projeto ─────────────────────
    sep("PASSO 3 - Formatar para padrao do projeto")
    df_final = formatar_dataframe(df_bruto, fonte)
    print(f"  Linhas formatadas: {len(df_final)}")
    print(f"  Anos cobertos: {sorted(df_final['year'].unique())}")

    # ── Validar dados ─────────────────────────────────────────
    ok_chuvosa, ok_seca, ok_pos, ok_max = validar_dados(df_final)

    # ── Comparar com simulados ────────────────────────────────
    comparar_com_simulados(df_final, df_simulado)

    # ── Backup e salvar novo CSV ──────────────────────────────
    sep("PASSO 5a - Backup e substituicao do CSV")
    shutil.copy2(ARQUIVO_CHUVA, ARQUIVO_BACKUP)
    print(f"  [OK] Backup salvo: {ARQUIVO_BACKUP}")

    df_final.to_csv(ARQUIVO_CHUVA, index=False, encoding="utf-8")
    print(f"  [OK] 03_rainfall_data.csv substituido por dados reais ({len(df_final)} linhas)")

    # ── Calcular rolling e salvar processado ──────────────────
    sep("PASSO 5b - Calcular janelas rolling")
    df_proc = calcular_rolling(df_final)
    df_proc.to_csv(ARQUIVO_PROC, index=False, encoding="utf-8")
    print(f"  [OK] 03_rainfall_data_processed.csv salvo ({len(df_proc)} linhas)")
    print(f"  Colunas: {list(df_proc.columns)}")

    # ── Metadados ─────────────────────────────────────────────
    salvar_metadados(fonte, metodo_num, len(df_final),
                     ok_chuvosa, ok_seca, ok_pos, ok_max)

    # ── Atualizar datasets dependentes ────────────────────────
    atualizar_ml_dataset(df_proc)
    atualizar_analytical(df_proc)

    # ── RELATORIO FINAL ───────────────────────────────────────
    sep("RELATORIO FINAL")
    print(f"""
  Fonte utilizada       : {fonte}
  Metodo               : {metodo_num}
  Periodo obtido       : {df_final['year'].min()}-{df_final['month'].min():02d} a \
{df_final['year'].max()}-{df_final['month'].max():02d}
  Total de meses       : {len(df_final)}

  VALIDACAO CLIMATICA SAO LUIS:
  {'[OK]' if ok_chuvosa else '[ALERTA]'}  Medias jan-jun acima de 200mm
  {'[OK]' if ok_seca else '[ALERTA]'}  Medias jul-dez abaixo de 200mm
  {'[OK]' if ok_pos else '[ERRO]'}  Sem valores negativos
  {'[OK]' if ok_max else '[ALERTA]'}  Sem meses acima de 700mm

  ARQUIVOS ATUALIZADOS:
  [OK] {ARQUIVO_BACKUP}
  [OK] {ARQUIVO_CHUVA}
  [OK] {ARQUIVO_META}
  [OK] {ARQUIVO_PROC}
  [OK] {ARQUIVO_ML}
  [OK] {ARQUIVO_ANA}

  PROXIMO PASSO:
  Os dados de chuva agora sao reais.
  Restam para coletar dados reais:
  -> 02_repair_history.csv  (ordens de servico SEMUSC)
  -> 04_traffic_load.csv    (contagem de trafego)
  -> 05_citizen_complaints.csv (ouvidoria)
""")
