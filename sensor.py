# =============================================================
# SOFT SENSOR - SELEÇÃO DE VARIÁVEIS + MLP
# Reservatório de Trasona | TCC Engenharia
# Inclui: gráfico de paridade e observado x predito (Cenários I e II)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===================
# LEITURA DOS DADOS
# ===================


import pandas as pd
import numpy as np

dados = pd.read_excel("dados/dados.xlsx")
dados.columns = dados.columns.str.strip()

desc = dados.describe().T[['mean', 'std', 'min', 'max']]
desc.columns = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
desc = desc.round(3)

print(desc.to_string())

# ===================
# VARIÁVEL ALVO
# ===================

y = dados["Microcystis aeruginosa (mm3/l)"]
y_log = np.log1p(y)

# ===================
# VARIÁVEIS
# ===================

variaveis_fq = [
    'Chlorophyll (µg/l)',
    'Water temperature (ºC)',
    'Ambient temperature (ºC)',
    'Secchi depth (m)',
    'Turbidity (NTU)',
    'Total phosphorus (mg P/l)',
    'Phosphate concentration (mg PO43-/l)',
    'Total nitrogen concentration  (mg N/l)',
    'Nitrate concentration  (mg NO3-/l)',
    'Nitrite concentration  (mg NO2-/l)',
    'Ammonium concentration  (mg/l)',
    'Dissolved oxygen concentration (mg O2/l)',
    'Conductivity (µS/cm)',
    'Alkalinity (mgCaCO3/l )',
    'Calcium concentration  (mg/l)',
    'pH values'
]

variaveis_micro = [
    'Cyanotoxin (μg/l)',
    'Woronichinia naegeliana (mm3/l)',
    'Other cyanobacteria species (mm3/l)',
    'Diatoms (mm3/l)',
    'Chrysophytes (mm3/l)',
    'Chlorophytes (mm3/l)',
    'Other phytoplankton species (mm3/l)'
]

# ===================
# FUNÇÃO MODELO
# Retorna métricas + vetores predito/observado (teste e CV completo)
# ===================

def rodar_modelo(X, y_log, nome):

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(12,),
            activation='tanh',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=3000,
            random_state=42
        ))
    ])

    # --- Validação cruzada ---
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y_log,
        cv=kfold,
        scoring='r2'
    )

    # --- Treinamento final ---
    pipeline.fit(X_treino, y_treino)

    y_pred_log = pipeline.predict(X_teste)
    y_pred     = np.expm1(y_pred_log)
    y_real     = np.expm1(y_teste)

    r2   = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)

    print(f"\n=== {nome} ===")
    print(f"R² CV (média):    {scores.mean():.4f}")
    print(f"R² CV (desvio):   {scores.std():.4f}")
    print(f"R² Teste:         {r2:.4f}")
    print(f"RMSE:             {rmse:.4f}")
    print(f"MAE:              {mae:.4f}")

    # --- Predição sobre todo o conjunto via CV (para gráfico temporal) ---
    # Usa cross_val_predict para obter predição de cada amostra
    # quando ela estava no fold de validação
    from sklearn.model_selection import cross_val_predict

    y_pred_cv_log = cross_val_predict(
        pipeline, X, y_log,
        cv=kfold
    )
    y_pred_cv = np.expm1(y_pred_cv_log)
    y_full    = np.expm1(y_log)

    return {
        'nome':       nome,
        'r2cv':       scores.mean(),
        'r2cv_std':   scores.std(),
        'r2_teste':   r2,
        'rmse':       rmse,
        'mae':        mae,
        # conjunto de teste
        'y_real':     y_real.values,
        'y_pred':     y_pred,
        # conjunto completo via CV
        'y_full':     y_full.values,
        'y_pred_cv':  y_pred_cv,
    }

# =============================================================
# CENÁRIO 1 — FÍSICO-QUÍMICO
# =============================================================

X_fq   = dados[variaveis_fq]
res_c1 = rodar_modelo(X_fq, y_log, "CENÁRIO 1 — Físico-Químico")

# =============================================================
# CENÁRIO 2 — FQ + MICROBIOLÓGICO (seleção por correlação)
# =============================================================

print("\n=== ANÁLISE DE CORRELAÇÃO ===")

dados_modelo = dados[
    variaveis_fq + variaveis_micro + ["Microcystis aeruginosa (mm3/l)"]
]

correlacao = dados_modelo.corr()
cor_target = correlacao["Microcystis aeruginosa (mm3/l)"].sort_values(ascending=False)
print(cor_target)

threshold = 0.2
variaveis_relevantes = cor_target[abs(cor_target) > threshold].index.tolist()
variaveis_relevantes.remove("Microcystis aeruginosa (mm3/l)")

print(f"\nVariáveis selecionadas (|r| > {threshold}):")
for v in variaveis_relevantes:
    print(f"  {v}")

X_sel  = dados[variaveis_relevantes]
res_c2 = rodar_modelo(X_sel, y_log, "CENÁRIO 2 — FQ + Microbiológico Selecionado")

# =============================================================
# COMPARAÇÃO FINAL — TABELA
# =============================================================

print("\n=== COMPARAÇÃO FINAL ===")
print(f"{'Modelo':<35} {'R² CV':>8} {'R² Teste':>10} {'RMSE':>8} {'MAE':>8}")
for r in [res_c1, res_c2]:
    print(f"{r['nome']:<35} {r['r2cv']:>8.4f} {r['r2_teste']:>10.4f} "
          f"{r['rmse']:>8.4f} {r['mae']:>8.4f}")

# =============================================================
# GRÁFICOS
# =============================================================


# =============================================================
# CONFIGURAÇÃO DE ESTILO
# =============================================================
plt.style.use('seaborn-v0_8-whitegrid')

cores = {
    "c1": "#1f77b4",   # azul
    "c2": "#d62728",   # vermelho
    "obs": "#2c3e50"   # observado
}

def salvar_figura(fig, nome):
    fig.savefig(nome, dpi=400, bbox_inches='tight')
    print(f"Figura salva: {nome}")


# =============================================================
# FUNÇÕES DE GRÁFICO
# =============================================================

def grafico_paridade(res, cenario, cor, nome_arquivo):
    fig, ax = plt.subplots(figsize=(7, 6))

    lim_min = min(res['y_real'].min(), res['y_pred'].min()) * 0.95
    lim_max = max(res['y_real'].max(), res['y_pred'].max()) * 1.05

    ax.scatter(res['y_real'], res['y_pred'],
               color=cor, alpha=0.75,
               edgecolors='white', linewidths=0.5, s=70)

    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            linestyle='--', linewidth=1.5, color='black',
            label='Ideal (y = x)')

    ax.set_title(
        f"{cenario} — Gráfico de Paridade\n"
        f"R² = {res['r2_teste']:.4f} | RMSE = {res['rmse']:.4f}",
        pad=15
    )

    ax.set_xlabel("Valores Observados (mm³/L)")
    ax.set_ylabel("Valores Preditos (mm³/L)")

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    salvar_figura(fig, nome_arquivo)
    plt.close(fig)


def grafico_temporal(res, cenario, cor, nome_arquivo):
    fig, ax = plt.subplots(figsize=(9, 5))

    obs_idx = np.arange(len(res['y_full']))

    ax.scatter(obs_idx, res['y_full'],
               color=cores["obs"], s=30,
               label='Observado', zorder=3)

    ax.scatter(obs_idx, res['y_pred_cv'],
               color=cor, marker='+', s=50,
               label='Predito (CV)', zorder=2)

    ax.set_title(
        f"{cenario} — Observado × Predito\n"
        f"R² CV = {res['r2cv']:.4f} | Desvio = {res['r2cv_std']:.4f}",
        pad=15
    )

    ax.set_xlabel("Número da observação")
    ax.set_ylabel("Microcystis aeruginosa (mm³/L)")

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    salvar_figura(fig, nome_arquivo)
    plt.close(fig)


# =============================================================
# GERAÇÃO DOS GRÁFICOS
# =============================================================

grafico_paridade(res_c1, "Cenário I", cores["c1"], "c1_paridade.png")
grafico_temporal(res_c1, "Cenário I", cores["c1"], "c1_temporal.png")

grafico_paridade(res_c2, "Cenário II", cores["c2"], "c2_paridade.png")
grafico_temporal(res_c2, "Cenário II", cores["c2"], "c2_temporal.png")

print("\nGráficos salvos em: graficos_sensor_virtual.png")