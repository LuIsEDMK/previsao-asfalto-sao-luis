import nbformat as nbf
import os

BASE = r"w:\projetos vscode\projetos para prefeitura\1projeto\codigos"

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("# Modelo Preditivo de Falhas Viárias - São Luís / MA\nEste notebook consolida todo o fluxo de trabalho de Data Science: desde a limpeza dos dados até o treinamento do modelo XGBoost."))

files = [
    ("1. Extração e Limpeza de Dados (Data Wrangling)", "pipeline_limpeza.py"),
    ("2. Análise Exploratória (EDA)", "eda_analysis.py"),
    ("3. Modelagem Preditiva (Machine Learning)", "modelagem.py"),
    ("4. Avaliação de Negócio (ROI)", "avaliacao_negocio.py")
]

for title, filename in files:
    nb.cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
    
    path = os.path.join(BASE, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Removemos a maioria dos prints puros para economizar espaço visual no Github
            lines = []
            for line in content.split("\n"):
                if not line.strip().startswith("print(") and not line.strip().startswith("import "):
                    lines.append(line)
            
            clean_code = "import pandas as pd\nimport numpy as np\n" + "\n".join(lines).strip()
            nb.cells.append(nbf.v4.new_code_cell(clean_code))

with open(os.path.join(os.path.dirname(BASE), "Projeto_Predictivo.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook gerado!")
