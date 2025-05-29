import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

TABELAS = {
    'CartPole': 'CartPole/resultados/tabela_comparativa.csv',
    'Acrobot': 'Acrobot/resultados/tabela_comparativa.csv',
    'LunarLander': 'LunarLander/resultados/tabela_comparativa_lunarlander.csv',
}

RECOMPENSA_IDEAL = {
    'CartPole': 475,
    'Acrobot': -100,
    'LunarLander': 200,
}

ALGOS = ['DQN', 'A2C', 'PPO']
def extrair_recompensas_finais(path, ambiente):
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        ultima = df.iloc[-1]
        recompensas = {}
        for algo in ALGOS:
            valor = ultima[f'{algo} (recompensa)']
            media = float(valor.split('±')[0].strip())
            recompensas[algo] = media
        return recompensas
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {str(e)}")
        return None

dados = {}
for ambiente, path in TABELAS.items():
    recompensas = extrair_recompensas_finais(path, ambiente)
    if recompensas is not None:
        dados[ambiente] = recompensas

if not dados:
    print("Nenhum dado foi carregado com sucesso!")
    exit()

fig, axes = plt.subplots(1, len(dados), figsize=(6*len(dados), 8))
fig.suptitle('Comparação de Algoritmos por Ambiente', fontsize=16, y=1.05)

cores = {
    'DQN': '#FF9999',  
    'A2C': '#66B2FF',  
    'PPO': '#99FF99'   
}

for idx, (ambiente, recompensas) in enumerate(dados.items()):
    ax = axes[idx]
    
    x = np.arange(len(ALGOS))
    valores = [recompensas[algo] for algo in ALGOS]
    barras = ax.bar(x, valores, color=[cores[algo] for algo in ALGOS])
    
    ax.axhline(y=RECOMPENSA_IDEAL[ambiente], color='red', linestyle='--', 
               label='Recompensa Ideal', alpha=0.7)
    
    if ambiente == 'Acrobot':
        ax.set_ylim(-550, 0) 
    else:
        ax.set_ylim(0, 550)   
    
   
    for bar in barras:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    ax.set_title(f'{ambiente}', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS, rotation=45)
    ax.set_ylabel('Recompensa Média Final')
    
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend()

plt.tight_layout()

plt.savefig('comparacao_algoritmos_ambientes.png', 
            bbox_inches='tight', 
            dpi=300,
            facecolor='white',
            edgecolor='none')
plt.show() 
