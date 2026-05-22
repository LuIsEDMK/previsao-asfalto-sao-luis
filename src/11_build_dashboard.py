"""
Tarefa 7 — Dashboard Final com Análise Econômica e Transparência
Gera dashboard_final.html com mapa de risco + painel econômico.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, warnings, pickle, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

BASE_DIR  = r"w:\projetos vscode\projetos para prefeitura"
CSV_DIR   = f"{BASE_DIR}\\csv"
MODEL_DIR = f"{BASE_DIR}\\1projeto"

# ── Carregar dados ───────────────────────────────────────────────────────────
print("Carregando dados...")
previsoes = pd.read_csv(f"{CSV_DIR}\\13_previsoes_economicas_sao_luis.csv")
metricas  = pd.read_csv(f"{CSV_DIR}\\12_metricas_final.csv")

with open(f"{MODEL_DIR}\\modelo_final.pkl", "rb") as f:
    meta = pickle.load(f)

FEATURES = meta["features"]
auroc_wf = meta.get("auroc_walk_forward_mean", 0)
auroc_sl = meta.get("auroc_ood_sao_luis", 0)

print(f"  Segmentos: {len(previsoes):,} | AUROC OOD: {auroc_sl:.4f}")

# ── Estatísticas econômicas ──────────────────────────────────────────────────
total_prev  = previsoes["custo_preventivo_R$"].sum()
total_emer  = previsoes["custo_emergencial_R$"].sum()
total_econ  = previsoes["economia_R$"].sum()

criticos = (previsoes["categoria_risco"] == "CRÍTICO").sum()
altos    = (previsoes["categoria_risco"] == "ALTO").sum()
medios   = (previsoes["categoria_risco"] == "MÉDIO").sum()
baixos   = (previsoes["categoria_risco"] == "BAIXO").sum()
total_s  = len(previsoes)

# ── Gerar GeoJSON dos segmentos (sample para performance) ────────────────────
print("Gerando GeoJSON...")
SAMPLE = min(8000, len(previsoes))

# Garantir representação de todos os níveis de risco
criticos_df = previsoes[previsoes["categoria_risco"] == "CRÍTICO"]
outros_df   = previsoes[previsoes["categoria_risco"] != "CRÍTICO"]

n_criticos = min(len(criticos_df), 3000)
n_outros   = min(len(outros_df), SAMPLE - n_criticos)

sample = pd.concat([
    criticos_df.sample(n=n_criticos, random_state=42),
    outros_df.sample(n=n_outros, random_state=42),
]).dropna(subset=["lat_mid", "lon_mid"])

COR_RISCO = {
    "CRÍTICO": "#d62728",
    "ALTO":    "#ff7f0e",
    "MÉDIO":   "#bcbd22",
    "BAIXO":   "#2ca02c",
}

features_parts = []
for _, row in sample.iterrows():
    lat  = float(row["lat_mid"])
    lon  = float(row["lon_mid"])
    cat  = str(row["categoria_risco"])
    risco = float(row["risco_prob"])
    cor  = COR_RISCO.get(cat, "#888888")
    seg  = str(row.get("segment_id", "?"))
    prev_r = int(row.get("custo_preventivo_R$", 0))
    emer_r = int(row.get("custo_emergencial_R$", 0))
    icp_v  = float(row.get("icp", 0))
    flood  = float(row.get("flood_zone_final", 0))

    popup = (f"<b>{cat}</b><br>"
             f"Risco: {risco:.1%}<br>"
             f"ICP: {icp_v:.0f}/100<br>"
             f"Flood zone: {flood:.2f}<br>"
             f"Preventivo: R${prev_r:,}<br>"
             f"Emergencial: R${emer_r:,}")
    popup = popup.replace('"', "'")

    feat = (
        '{"type":"Feature","properties":{"categoria":"' + cat + '"'
        + ',"risco":' + str(round(risco, 4))
        + ',"cor":"' + cor + '"'
        + ',"popup":"' + popup + '"}'
        + ',"geometry":{"type":"Point","coordinates":['
        + str(round(lon, 6)) + ',' + str(round(lat, 6)) + ']}}'
    )
    features_parts.append(feat)

geojson_str = '{"type":"FeatureCollection","features":[' + ",".join(features_parts) + ']}'
print(f"  GeoJSON: {len(features_parts):,} segmentos amostrados")

# ── Métricas Walk-Forward para tabela ────────────────────────────────────────
metricas_wf = metricas[metricas["fold"] != "OOD_SL"]
wf_rows = ""
for _, r in metricas_wf.iterrows():
    wf_rows += (
        f"<tr><td>Fold {r['fold']}</td>"
        f"<td>{r['anos_treino']}</td>"
        f"<td>{int(r['ano_teste'])}</td>"
        f"<td>{r['auroc_wf']:.4f}</td></tr>"
    )

# ── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Preditor de Falhas em Vias — São Luís MA</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#0d1117; color:#e6edf3; }}
  .header {{ background:#161b22; border-bottom:1px solid #30363d; padding:16px 24px; display:flex; align-items:center; gap:16px; }}
  .header h1 {{ font-size:1.4rem; font-weight:700; color:#58a6ff; }}
  .header .sub {{ font-size:0.85rem; color:#8b949e; }}
  .badge {{ background:#238636; color:#fff; font-size:0.75rem; padding:3px 10px; border-radius:12px; margin-left:auto; }}
  .container {{ display:grid; grid-template-columns:340px 1fr; height:calc(100vh - 60px); }}
  .sidebar {{ background:#161b22; border-right:1px solid #30363d; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:16px; }}
  #map {{ height:100%; }}
  .card {{ background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:14px; }}
  .card h3 {{ font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:#8b949e; margin-bottom:12px; }}
  .metric-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .metric {{ background:#161b22; border-radius:6px; padding:10px; text-align:center; }}
  .metric .val {{ font-size:1.5rem; font-weight:700; color:#58a6ff; }}
  .metric .lbl {{ font-size:0.72rem; color:#8b949e; margin-top:2px; }}
  .risk-bar {{ display:flex; flex-direction:column; gap:6px; }}
  .risk-row {{ display:flex; align-items:center; gap:8px; }}
  .risk-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
  .risk-label {{ font-size:0.82rem; width:70px; }}
  .risk-fill-bg {{ flex:1; background:#30363d; border-radius:4px; height:8px; overflow:hidden; }}
  .risk-fill {{ height:100%; border-radius:4px; transition:width 0.5s; }}
  .risk-count {{ font-size:0.78rem; color:#8b949e; min-width:50px; text-align:right; }}
  .econ-grid {{ display:grid; grid-template-columns:1fr; gap:6px; }}
  .econ-row {{ display:flex; justify-content:space-between; font-size:0.82rem; padding:6px 8px; border-radius:4px; background:#161b22; }}
  .econ-val {{ font-weight:600; color:#3fb950; }}
  .econ-economia {{ color:#f78166; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
  th {{ text-align:left; color:#8b949e; padding:4px 8px; border-bottom:1px solid #30363d; }}
  td {{ padding:4px 8px; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#21262d; }}
  .legend {{ position:absolute; bottom:20px; right:10px; z-index:1000; background:rgba(22,27,34,0.95);
             border:1px solid #30363d; border-radius:8px; padding:10px 14px; font-size:0.78rem; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; margin:4px 0; }}
  .legend-dot {{ width:10px; height:10px; border-radius:50%; }}
  .auroc-badge {{ display:inline-block; background:#1f6feb; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }}
  .source-tag {{ font-size:0.72rem; color:#8b949e; font-style:italic; }}
  .transparency {{ background:#161822; border:1px solid #388bfd44; border-radius:6px; padding:10px; font-size:0.75rem; line-height:1.6; }}
  .transparency b {{ color:#58a6ff; }}
  .section-title {{ font-size:1rem; font-weight:600; color:#e6edf3; border-bottom:1px solid #30363d; padding-bottom:8px; margin-bottom:10px; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Preditor de Falhas em Vias Urbanas</h1>
    <div class="sub">São Luís — MA | Modelo ICP + XGBoost | Dados 100% públicos</div>
  </div>
  <div class="badge">v3 — Análise Econômica</div>
</div>

<div class="container">
<div class="sidebar">

  <!-- MÉTRICAS DO MODELO -->
  <div class="card">
    <h3>Performance do Modelo</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="val">{auroc_wf:.3f}</div>
        <div class="lbl">AUROC Walk-Forward<br>Fortaleza + Recife</div>
      </div>
      <div class="metric">
        <div class="val">{auroc_sl:.3f}</div>
        <div class="lbl">AUROC OOD<br>São Luís (inédito)</div>
      </div>
      <div class="metric">
        <div class="val">{total_s:,}</div>
        <div class="lbl">Segmentos<br>analisados</div>
      </div>
      <div class="metric">
        <div class="val">15</div>
        <div class="lbl">Features<br>utilizadas</div>
      </div>
    </div>
  </div>

  <!-- DISTRIBUIÇÃO DE RISCO -->
  <div class="card">
    <h3>Distribuição de Risco</h3>
    <div class="risk-bar">
      <div class="risk-row">
        <div class="risk-dot" style="background:#d62728"></div>
        <div class="risk-label">CRÍTICO</div>
        <div class="risk-fill-bg"><div class="risk-fill" style="width:{criticos/total_s*100:.1f}%;background:#d62728"></div></div>
        <div class="risk-count">{criticos:,} ({criticos/total_s*100:.0f}%)</div>
      </div>
      <div class="risk-row">
        <div class="risk-dot" style="background:#ff7f0e"></div>
        <div class="risk-label">ALTO</div>
        <div class="risk-fill-bg"><div class="risk-fill" style="width:{altos/total_s*100:.1f}%;background:#ff7f0e"></div></div>
        <div class="risk-count">{altos:,} ({altos/total_s*100:.0f}%)</div>
      </div>
      <div class="risk-row">
        <div class="risk-dot" style="background:#bcbd22"></div>
        <div class="risk-label">MÉDIO</div>
        <div class="risk-fill-bg"><div class="risk-fill" style="width:{medios/total_s*100:.1f}%;background:#bcbd22"></div></div>
        <div class="risk-count">{medios:,} ({medios/total_s*100:.0f}%)</div>
      </div>
      <div class="risk-row">
        <div class="risk-dot" style="background:#2ca02c"></div>
        <div class="risk-label">BAIXO</div>
        <div class="risk-fill-bg"><div class="risk-fill" style="width:{baixos/total_s*100:.1f}%;background:#2ca02c"></div></div>
        <div class="risk-count">{baixos:,} ({baixos/total_s*100:.0f}%)</div>
      </div>
    </div>
  </div>

  <!-- ANÁLISE ECONÔMICA -->
  <div class="card">
    <h3>Análise Econômica (R$)</h3>
    <div class="econ-grid">
      <div class="econ-row">
        <span>Custo preventivo total</span>
        <span class="econ-val">R$ {total_prev/1e6:.1f}M</span>
      </div>
      <div class="econ-row">
        <span>Custo emergencial total</span>
        <span style="font-weight:600;color:#f78166">R$ {total_emer/1e6:.1f}M</span>
      </div>
      <div class="econ-row" style="background:#1a2a1a;border:1px solid #238636">
        <span><b>Economia potencial</b></span>
        <span style="font-weight:700;color:#3fb950;font-size:1rem">R$ {total_econ/1e6:.0f}M</span>
      </div>
      <div class="econ-row">
        <span>ROI manutenção preventiva</span>
        <span class="econ-val">{total_emer/total_prev:.1f}x</span>
      </div>
    </div>
    <div style="margin-top:10px;font-size:0.75rem;color:#8b949e">
      ⚠ Intervir nos {criticos:,} trechos CRÍTICOS evita R$ {previsoes[previsoes['categoria_risco']=='CRÍTICO']['economia_R$'].sum()/1e6:.0f}M em emergências.
    </div>
  </div>

  <!-- WALK-FORWARD -->
  <div class="card">
    <h3>Walk-Forward Validation</h3>
    <table>
      <tr><th>Fold</th><th>Treino</th><th>Teste</th><th>AUROC</th></tr>
      {wf_rows}
      <tr style="background:#1f2937">
        <td colspan="3"><b>OOD — São Luís</b></td>
        <td><b>{auroc_sl:.4f}</b></td>
      </tr>
    </table>
  </div>

  <!-- TRANSPARÊNCIA -->
  <div class="card">
    <h3>Painel de Transparência</h3>
    <div class="transparency">
      <b>Y (variável-alvo):</b> Modelo ICP de Degradação Física<br>
      — Taxa de degradação por tipo de via<br>
      — Fatores: chuva, zona de alagamento, declividade, carga<br>
      — Y=1 quando ICP &lt; 40 (estado crítico)<br>
      <br>
      <b>Fontes de dados:</b><br>
      • Rede viária: OSMnx (OpenStreetMap)<br>
      • Elevação/declividade: NASA SRTM via Open-Elevation<br>
      • Zonas de alagamento: OSMnx water proximity<br>
      • Densidade urbana: IBGE Censo 2022<br>
      • Pluviosidade: ERA5 via Open-Meteo Archive API<br>
      <br>
      <b>OS reais de SP:</b> 5 fontes tentadas (CKAN, GitHub, URLs diretas) — todas inacessíveis. Modo ICP ativado.<br>
      <br>
      <b>Modelo:</b> XGBoost (n=200, depth=4, lr=0.05)<br>
      <b>Validação:</b> Walk-Forward 5 folds temporal<br>
      <b>OOD:</b> São Luís nunca vista durante treino<br>
    </div>
  </div>

</div><!-- sidebar -->

<!-- MAPA -->
<div style="position:relative">
  <div id="map"></div>
  <div class="legend">
    <div style="font-weight:600;margin-bottom:6px;font-size:0.8rem">Categoria de Risco</div>
    <div class="legend-item"><div class="legend-dot" style="background:#d62728"></div> CRÍTICO (≥75%)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff7f0e"></div> ALTO (50–75%)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#bcbd22"></div> MÉDIO (30–50%)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#2ca02c"></div> BAIXO (&lt;30%)</div>
  </div>
</div>
</div><!-- container -->

<script>
var map = L.map('map').setView([-2.53, -44.30], 12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &amp; CartoDB',
  subdomains: 'abcd', maxZoom: 19
}}).addTo(map);

var geojsonData = {geojson_str};

L.geoJSON(geojsonData, {{
  pointToLayer: function(feat, latlng) {{
    return L.circleMarker(latlng, {{
      radius: feat.properties.risco > 0.75 ? 6 : feat.properties.risco > 0.50 ? 5 : 4,
      fillColor: feat.properties.cor,
      color: '#000',
      weight: 0.3,
      opacity: 0.8,
      fillOpacity: 0.8
    }});
  }},
  onEachFeature: function(feat, layer) {{
    layer.bindPopup(feat.properties.popup);
  }}
}}).addTo(map);
</script>
</body>
</html>"""

# Salvar
saida = f"{MODEL_DIR}\\dashboard_final.html"
with open(saida, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(saida) / 1024
print(f"\n✅ Dashboard final salvo: {saida}")
print(f"   Tamanho: {size_kb:.0f} KB")
print(f"   Segmentos no mapa: {len(features_parts):,}")
