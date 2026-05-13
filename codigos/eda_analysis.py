"""
Projeto: Preditor de Falhas em Vias - São Luís / MA
Etapa:   Análise Exploratória de Dados (EDA)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

# Configuração visual
plt.rcParams.update({'figure.figsize': (12, 7), 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 150})
sns.set_style("whitegrid")
COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "eda_outputs")
os.makedirs(OUT, exist_ok=True)

# ── Carregar dados ──
segments   = pd.read_csv(os.path.join(PARENT, "01_road_segments.csv"))
repairs    = pd.read_csv(os.path.join(PARENT, "02_repair_history.csv"))
rainfall   = pd.read_csv(os.path.join(PARENT, "03_rainfall_data.csv"))
traffic    = pd.read_csv(os.path.join(PARENT, "04_traffic_load.csv"))
complaints = pd.read_csv(os.path.join(PARENT, "05_citizen_complaints.csv"))
ml_data    = pd.read_csv(os.path.join(PARENT, "06_ml_training_dataset.csv"))
risk       = pd.read_csv(os.path.join(PARENT, "07_risk_scores.csv"))
analytical = pd.read_csv(os.path.join(BASE, "08_analytical_dataset.csv"))

repairs["order_date"] = pd.to_datetime(repairs["order_date"])
repairs["completion_date"] = pd.to_datetime(repairs["completion_date"])
complaints["report_date"] = pd.to_datetime(complaints["report_date"])

print("="*60)
print("  ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("  Preditor de Falhas em Vias — São Luís / MA")
print("="*60)

# ═══════════════════════════════════════════════════════════
# 1. ESTATÍSTICAS DESCRITIVAS
# ═══════════════════════════════════════════════════════════
print("\n▶ 1. ESTATÍSTICAS DESCRITIVAS\n")

desc = analytical[['pavement_age_years','load_index_1_10','avg_daily_vehicles',
    'heavy_vehicles_pct','total_repairs','total_repair_cost','total_complaints',
    'risk_score']].describe().round(2)
print(desc.to_string())
desc.to_csv(os.path.join(OUT, "01_descritivas.csv"))

# ═══════════════════════════════════════════════════════════
# 2. DISTRIBUIÇÃO DA VARIÁVEL ALVO (risk_score / risk_category)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma do risk_score
axes[0].hist(analytical['risk_score'].dropna(), bins=10, color='#3498db',
    edgecolor='white', alpha=0.85)
axes[0].axvline(analytical['risk_score'].mean(), color='#e74c3c', ls='--',
    lw=2, label=f"Média: {analytical['risk_score'].mean():.1f}")
axes[0].set_xlabel('Risk Score'); axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição do Score de Risco'); axes[0].legend()

# Contagem por categoria
cat_order = ['baixo','medio','alto']
cat_counts = analytical['risk_category'].value_counts().reindex(cat_order).fillna(0)
bars = axes[1].bar(cat_counts.index, cat_counts.values,
    color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='white')
for b in bars:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
        int(b.get_height()), ha='center', fontweight='bold')
axes[1].set_title('Segmentos por Categoria de Risco')
axes[1].set_ylabel('Quantidade')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_distribuicao_risco.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: distribuição de risco salvo")

# ═══════════════════════════════════════════════════════════
# 3. ANÁLISE POR ZONA E BAIRRO
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

zone_risk = analytical.groupby('zone')['risk_score'].mean().sort_values(ascending=False)
axes[0].barh(zone_risk.index, zone_risk.values, color=COLORS[:len(zone_risk)])
axes[0].set_xlabel('Score de Risco Médio'); axes[0].set_title('Risco Médio por Zona')
for i, v in enumerate(zone_risk.values):
    axes[0].text(v+0.5, i, f'{v:.1f}', va='center', fontweight='bold')

bairro_risk = analytical.groupby('bairro')['risk_score'].mean().sort_values(ascending=False).head(8)
axes[1].barh(bairro_risk.index, bairro_risk.values, color=sns.color_palette("YlOrRd_r", len(bairro_risk)))
axes[1].set_xlabel('Score de Risco Médio'); axes[1].set_title('Top 8 Bairros por Risco')
for i, v in enumerate(bairro_risk.values):
    axes[1].text(v+0.5, i, f'{v:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_risco_zona_bairro.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: risco por zona e bairro salvo")

# ═══════════════════════════════════════════════════════════
# 4. TIPO DE PAVIMENTO vs RISCO
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pav_counts = analytical['pavement_type'].value_counts()
axes[0].pie(pav_counts, labels=pav_counts.index, autopct='%1.0f%%',
    colors=COLORS, startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Distribuição por Tipo de Pavimento')

pav_data = []
pav_labels = []
for pt in analytical['pavement_type'].unique():
    subset = analytical[analytical['pavement_type']==pt]['risk_score'].dropna()
    if len(subset) > 0:
        pav_data.append(subset.values)
        pav_labels.append(pt)
bp = axes[1].boxplot(pav_data, labels=pav_labels,
    patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_title('Score de Risco por Tipo de Pavimento')
axes[1].set_ylabel('Risk Score')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_pavimento_risco.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: pavimento vs risco salvo")

# ═══════════════════════════════════════════════════════════
# 5. CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS
# ═══════════════════════════════════════════════════════════
num_cols = ['pavement_age_years','load_index_1_10','avg_daily_vehicles',
    'heavy_vehicles_pct','total_repairs','total_repair_cost',
    'total_complaints','avg_severity','flood_zone','risk_score']
corr = analytical[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
    cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Matriz de Correlação — Variáveis Preditoras vs Risk Score')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_correlacao.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: matriz de correlação salvo")

corr_target = corr['risk_score'].drop('risk_score').sort_values(ascending=False)
print("\n  Correlações com risk_score:")
for col, val in corr_target.items():
    print(f"    {col:30s} r = {val:+.3f}")

# ═══════════════════════════════════════════════════════════
# 6. ANÁLISE TEMPORAL — CHUVAS E REPAROS
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

rainfall['date'] = pd.to_datetime(rainfall[['year','month']].assign(day=1))
axes[0].fill_between(rainfall['date'], rainfall['rainfall_mm'], alpha=0.3, color='#3498db')
axes[0].plot(rainfall['date'], rainfall['rainfall_mm'], color='#2980b9', lw=2)
axes[0].axhline(rainfall['rainfall_mm'].quantile(0.75), color='#e74c3c', ls='--',
    label=f"P75 = {rainfall['rainfall_mm'].quantile(0.75):.0f}mm")
axes[0].set_ylabel('Precipitação (mm)'); axes[0].set_title('Precipitação Mensal — São Luís')
axes[0].legend()

repairs_monthly = repairs.groupby(repairs['order_date'].dt.to_period('M')).size()
repairs_monthly.index = repairs_monthly.index.to_timestamp()
axes[1].bar(repairs_monthly.index, repairs_monthly.values, width=25, color='#e74c3c', alpha=0.7)
axes[1].set_ylabel('Nº de Reparos'); axes[1].set_title('Reparos por Mês')
axes[1].set_xlabel('Data')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_temporal_chuva_reparos.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: temporal chuvas vs reparos salvo")

# ═══════════════════════════════════════════════════════════
# 7. SAZONALIDADE
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
monthly_avg = rainfall.groupby('month')['rainfall_mm'].mean()
repairs['repair_month'] = repairs['order_date'].dt.month
repair_monthly_avg = repairs.groupby('repair_month').size()

ax2 = ax.twinx()
ax.bar(monthly_avg.index, monthly_avg.values, color='#3498db', alpha=0.5, label='Chuva média (mm)')
ax2.plot(repair_monthly_avg.index, repair_monthly_avg.values, 'o-', color='#e74c3c',
    lw=2.5, markersize=8, label='Nº reparos')
ax.set_xlabel('Mês'); ax.set_ylabel('Precipitação (mm)', color='#3498db')
ax2.set_ylabel('Nº de Reparos', color='#e74c3c')
ax.set_title('Sazonalidade: Chuva vs Reparos por Mês')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_sazonalidade.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: sazonalidade salvo")

# ═══════════════════════════════════════════════════════════
# 8. SCATTER PLOTS — RELAÇÕES COM VARIÁVEL ALVO
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
scatter_vars = [
    ('pavement_age_years', 'Idade do Pavimento (anos)'),
    ('total_repairs', 'Total de Reparos Históricos'),
    ('total_complaints', 'Total de Reclamações'),
    ('load_index_1_10', 'Índice de Carga (1-10)')
]
for ax, (col, label) in zip(axes.flat, scatter_vars):
    data = analytical.dropna(subset=[col, 'risk_score'])
    ax.scatter(data[col], data['risk_score'], c=data['risk_score'],
        cmap='RdYlGn_r', s=100, edgecolors='white', linewidth=0.5, zorder=5)
    if len(data) > 2:
        z = np.polyfit(data[col], data['risk_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[col].min(), data[col].max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#e74c3c', lw=2, alpha=0.7)
        r, pval = stats.pearsonr(data[col], data['risk_score'])
        ax.text(0.05, 0.95, f'r={r:.3f}\np={pval:.3f}', transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel(label); ax.set_ylabel('Risk Score')
    ax.set_title(f'{label} vs Risk Score')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_scatter_preditores.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: scatter plots preditores salvo")

# ═══════════════════════════════════════════════════════════
# 9. ANÁLISE DE CUSTOS
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cost_seg = analytical[analytical['total_repair_cost']>0].sort_values('total_repair_cost', ascending=True)
axes[0].barh(cost_seg['street_name']+' ('+cost_seg['bairro']+')',
    cost_seg['total_repair_cost'], color=sns.color_palette("YlOrRd", len(cost_seg)))
axes[0].set_xlabel('Custo Total (R$)'); axes[0].set_title('Custo Acumulado de Reparos por Segmento')
for i, v in enumerate(cost_seg['total_repair_cost']):
    axes[0].text(v+100, i, f'R${v:,.0f}', va='center', fontsize=9)

repair_types = repairs['repair_type'].value_counts()
axes[1].pie(repair_types, labels=repair_types.index, autopct='%1.0f%%',
    colors=COLORS, startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Distribuição por Tipo de Reparo')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_custos.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: análise de custos salvo")

# ═══════════════════════════════════════════════════════════
# 10. FLOOD ZONE vs RISCO
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

flood_groups = analytical.groupby('flood_zone')['risk_score'].apply(list)
flood_data = [flood_groups.get(0, []), flood_groups.get(1, [])]
bp = axes[0].boxplot(flood_data, labels=['Fora de Baixada','Zona de Alagamento'],
    patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#2ecc71'); bp['boxes'][1].set_facecolor('#e74c3c')
for b in bp['boxes']: b.set_alpha(0.7)
axes[0].set_ylabel('Risk Score'); axes[0].set_title('Risco: Zona de Alagamento vs Normal')

# Teste estatístico
if len(flood_data[0])>1 and len(flood_data[1])>1:
    t_stat, p_val = stats.mannwhitneyu(flood_data[0], flood_data[1], alternative='two-sided')
    axes[0].text(0.5, 0.95, f'Mann-Whitney p={p_val:.4f}', transform=axes[0].transAxes,
        ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Reclamações por canal
channels = complaints['channel'].value_counts()
axes[1].bar(channels.index, channels.values, color=COLORS[:len(channels)], edgecolor='white')
axes[1].set_title('Reclamações por Canal'); axes[1].set_ylabel('Quantidade')
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_flood_reclamacoes.png"), bbox_inches='tight')
plt.close()
print("  ✔ Gráfico: flood zone e reclamações salvo")

# ═══════════════════════════════════════════════════════════
# 11. MAPA DE SÃO LUÍS (Folium)
# ═══════════════════════════════════════════════════════════
print("\n▶ Gerando mapa interativo de São Luís...")
try:
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(location=[-2.53, -44.30], zoom_start=13,
        tiles='CartoDB positron', control_scale=True)

    # Cores por categoria de risco
    risk_colors = {'critico': 'red', 'alto': 'orange', 'medio': 'blue', 'baixo': 'green'}

    # Adicionar segmentos como linhas no mapa
    for _, row in segments.iterrows():
        seg_id = row['segment_id']
        risk_row = risk[risk['segment_id'] == seg_id]
        if len(risk_row) > 0:
            cat = risk_row.iloc[0]['risk_category']
            score = risk_row.iloc[0]['risk_score_0_100']
            action = risk_row.iloc[0]['recommended_action']
        else:
            cat, score, action = 'baixo', 0, 'N/A'

        color = risk_colors.get(cat, 'gray')
        weight = 8 if cat in ('critico','alto') else 4

        popup_html = f"""
        <div style="font-family:Arial;width:250px">
        <h4 style="margin:0;color:{color}">{row['street_name']}</h4>
        <b>Bairro:</b> {row['bairro']}<br>
        <b>Zona:</b> {row['zone']}<br>
        <b>Pavimento:</b> {row['pavement_type']}<br>
        <b>Score:</b> {score} ({cat.upper()})<br>
        <b>Ação:</b> {action.replace('_',' ')}<br>
        <b>Alagamento:</b> {row['flood_zone']}
        </div>"""

        folium.PolyLine(
            [[row['lat_start'], row['lon_start']], [row['lat_end'], row['lon_end']]],
            color=color, weight=weight, opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['street_name']} — {cat.upper()} ({score})"
        ).add_to(m)

    # Adicionar marcadores de reclamações
    complaint_cluster = MarkerCluster(name="Reclamações Cidadãs").add_to(m)
    for _, row in complaints.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            sev = row.get('severity_reported', 'N/A')
            icon_color = 'red' if sev == 'muito_grave' else 'orange' if sev == 'grave' else 'blue'
            folium.Marker(
                [row['lat'], row['lon']],
                popup=f"{row['complaint_type']} — {sev}<br>{row['report_date']}",
                icon=folium.Icon(color=icon_color, icon='exclamation-sign', prefix='glyphicon')
            ).add_to(complaint_cluster)

    # Legenda
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
        background:white;padding:15px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);
        font-family:Arial;font-size:13px">
    <b>Risco Viário — São Luís</b><br>
    <i style="background:red;width:18px;height:4px;display:inline-block;margin-right:5px"></i> Crítico<br>
    <i style="background:orange;width:18px;height:4px;display:inline-block;margin-right:5px"></i> Alto<br>
    <i style="background:blue;width:18px;height:4px;display:inline-block;margin-right:5px"></i> Médio<br>
    <i style="background:green;width:18px;height:4px;display:inline-block;margin-right:5px"></i> Baixo<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    map_path = os.path.join(OUT, "11_mapa_sao_luis.html")
    m.save(map_path)
    print(f"  ✔ Mapa interativo salvo: {map_path}")

except ImportError:
    print("  ⚠ Folium não instalado — mapa não gerado")

# ═══════════════════════════════════════════════════════════
# 12. TESTES DE HIPÓTESES
# ═══════════════════════════════════════════════════════════
print("\n▶ 12. TESTES DE HIPÓTESES\n")

# H1: Segmentos em zona de alagamento têm mais reparos
flood_yes = analytical[analytical['flood_zone']==1]['total_repairs']
flood_no = analytical[analytical['flood_zone']==0]['total_repairs']
if len(flood_yes)>0 and len(flood_no)>0:
    u_stat, p = stats.mannwhitneyu(flood_yes, flood_no, alternative='greater')
    print(f"  H1: Zona alagamento → mais reparos")
    print(f"      Média alagamento: {flood_yes.mean():.1f} vs Fora: {flood_no.mean():.1f}")
    print(f"      Mann-Whitney U={u_stat:.1f}, p={p:.4f} {'✔ SIGNIFICATIVO' if p<0.05 else '✘ Não significativo'}")

# H2: Pavimento mais velho → maior risco
valid = analytical.dropna(subset=['pavement_age_years','risk_score'])
if len(valid)>2:
    r, p = stats.pearsonr(valid['pavement_age_years'], valid['risk_score'])
    print(f"\n  H2: Idade pavimento → maior risco")
    print(f"      Pearson r={r:.3f}, p={p:.4f} {'✔ SIGNIFICATIVO' if p<0.05 else '✘ Não significativo'}")

# H3: Chuva concentrada no 1º semestre (sazonalidade)
sem1 = rainfall[rainfall['month'].between(1,6)]['rainfall_mm']
sem2 = rainfall[rainfall['month'].between(7,12)]['rainfall_mm']
t_stat, p = stats.ttest_ind(sem1, sem2)
print(f"\n  H3: 1º semestre mais chuvoso")
print(f"      Média 1º sem: {sem1.mean():.0f}mm vs 2º sem: {sem2.mean():.0f}mm")
print(f"      t={t_stat:.2f}, p={p:.6f} {'✔ SIGNIFICATIVO' if p<0.05 else '✘ Não significativo'}")

# H4: Reclamações graves concentradas em segmentos de alto risco
high = analytical[analytical['risk_category'].isin(['alto','critico'])]['total_complaints']
low = analytical[analytical['risk_category'].isin(['baixo','medio'])]['total_complaints']
if len(high)>0 and len(low)>0:
    u_stat, p = stats.mannwhitneyu(high, low, alternative='greater')
    print(f"\n  H4: Alto risco → mais reclamações")
    print(f"      Média alto/crítico: {high.mean():.1f} vs baixo/médio: {low.mean():.1f}")
    print(f"      Mann-Whitney U={u_stat:.1f}, p={p:.4f} {'✔ SIGNIFICATIVO' if p<0.05 else '✘ Não significativo'}")

# ═══════════════════════════════════════════════════════════
# 13. RESUMO DE ANOMALIAS
# ═══════════════════════════════════════════════════════════
print("\n▶ 13. ANOMALIAS DETECTADAS\n")

# Segmentos com custo desproporcional
if analytical['total_repair_cost'].sum() > 0:
    top_cost = analytical.nlargest(3, 'total_repair_cost')[['street_name','bairro','total_repair_cost','total_repairs']]
    print("  Segmentos com maior custo acumulado:")
    for _, r in top_cost.iterrows():
        print(f"    → {r['street_name']} ({r['bairro']}): R${r['total_repair_cost']:,.0f} em {r['total_repairs']} reparos")

# Segmento com recorrência alta
recurrent = analytical[analytical['recurrence_rate']>=1.0]
if len(recurrent)>0:
    print(f"\n  Segmentos com recorrência ≥1 reparo/ano:")
    for _, r in recurrent.iterrows():
        print(f"    → {r['street_name']} ({r['bairro']}): {r['recurrence_rate']:.2f} reparos/ano")

print(f"\n{'='*60}")
print(f"  EDA concluída! {len(os.listdir(OUT))} arquivos salvos em: {OUT}")
print(f"{'='*60}")
