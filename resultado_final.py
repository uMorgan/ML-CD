import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurar estilo do matplotlib
plt.style.use('default')

# Caminhos dos arquivos de resultados
TABELAS = {
    'CartPole': 'CartPole/resultados/tabela_comparativa.csv',
    'Acrobot': 'Acrobot/resultados/tabela_comparativa.csv',
    'LunarLander': 'LunarLander/resultados/tabela_comparativa_lunarlander.csv',
}

# Recompensa ideal para cada ambiente
RECOMPENSA_IDEAL = {
    'CartPole': 475,
    'Acrobot': -100,
    'LunarLander': 200,
}

# Função para extrair a recompensa média final de cada algoritmo
ALGOS = ['DQN', 'A2C', 'PPO']
def extrair_recompensas_finais(path, ambiente):
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        # Pega a última linha (maior checkpoint)
        ultima = df.iloc[-1]
        recompensas = {}
        for algo in ALGOS:
            valor = ultima[f'{algo} (recompensa)']
            # Remove o desvio padrão e converte para float
            media = float(valor.split('±')[0].strip())
            recompensas[algo] = media
        return recompensas
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {str(e)}")
        return None

# Coletar dados
dados = {}
for ambiente, path in TABELAS.items():
    recompensas = extrair_recompensas_finais(path, ambiente)
    if recompensas is not None:
        dados[ambiente] = recompensas

if not dados:
    print("Nenhum dado foi carregado com sucesso!")
    exit()

# Criar figura com subplots para cada ambiente
fig, axes = plt.subplots(1, len(dados), figsize=(6*len(dados), 8))
fig.suptitle('Comparação de Algoritmos por Ambiente', fontsize=16, y=1.05)

# Cores para cada algoritmo
cores = {
    'DQN': '#FF9999',  # Rosa suave
    'A2C': '#66B2FF',  # Azul suave
    'PPO': '#99FF99'   # Verde suave
}

# Plotar gráfico para cada ambiente
for idx, (ambiente, recompensas) in enumerate(dados.items()):
    ax = axes[idx]
    
    # Plotar barras para cada algoritmo
    x = np.arange(len(ALGOS))
    valores = [recompensas[algo] for algo in ALGOS]
    barras = ax.bar(x, valores, color=[cores[algo] for algo in ALGOS])
    
    # Adicionar linha de recompensa ideal
    ax.axhline(y=RECOMPENSA_IDEAL[ambiente], color='red', linestyle='--', 
               label='Recompensa Ideal', alpha=0.7)
    
    # Personalizar eixo y
    if ambiente == 'Acrobot':
        ax.set_ylim(-550, 0)  # Ajuste para Acrobot
    else:
        ax.set_ylim(0, 550)   # Ajuste para CartPole e LunarLander
    
    # Adicionar valores nas barras
    for bar in barras:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    # Personalizar título e labels
    ax.set_title(f'{ambiente}', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS, rotation=45)
    ax.set_ylabel('Recompensa Média Final')
    
    # Adicionar grid suave
    ax.grid(True, alpha=0.3)
    
    # Adicionar legenda
    if idx == 0:
        ax.legend()

# Ajustar layout
plt.tight_layout()

# Salvar figura com alta qualidade
plt.savefig('comparacao_algoritmos_ambientes.png', 
            bbox_inches='tight', 
            dpi=300,
            facecolor='white',
            edgecolor='none')
plt.show() 