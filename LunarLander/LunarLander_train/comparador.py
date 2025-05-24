import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
import time

MODELS = {
    "DQN": "dqn_lunarlander.zip",
    "PPO": "ppo_lunarlander.zip",
    "A2C": "a2c_lunarlander.zip"
}

def avaliar_modelo(model_class, path, env_name="LunarLander-v3", n_episodes=100):
    env = gym.make(env_name)
    model = model_class.load(path)
    
    start_time = time.time()
    rewards, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes, return_episode_rewards=True)
    end_time = time.time()
    
    env.close()
    return np.array(rewards), end_time - start_time

def calcular_metricas(rewards, nome, tempo_execucao):
    media = np.mean(rewards)
    std = np.std(rewards)
    cv = std / media if media != 0 else float('inf')
    tempo_medio = tempo_execucao / len(rewards)

    convergencia = None
    window = 10
    for i in range(len(rewards) - window):
        if np.mean(rewards[i:i+window]) >= 200:
            convergencia = i + window
            break

    print(f"\n🔎 {nome} - Avaliação com {len(rewards)} episódios:")
    print(f"  🎯 Recompensa média: {media:.2f}")
    print(f"  📉 Desvio padrão: {std:.2f}")
    print(f"  📊 Coeficiente de variação: {cv:.2f}")
    print(f"  🕒 Tempo total de avaliação: {tempo_execucao:.2f} s")
    print(f"  ⏱️ Tempo médio por episódio: {tempo_medio:.3f} s")
    print(f"  ⏱️ Episódios até convergência (estimado): {convergencia if convergencia else 'Não convergiu'}")

    return {
        "Algoritmo": nome,
        "Recompensa Média": round(media, 2),
        "Desvio Padrão": round(std, 2),
        "Recompensa Mínima": round(np.min(rewards), 2),
        "Recompensa Máxima": round(np.max(rewards), 2),
        "Coef. de Variação": round(cv, 2),
        "Tempo Total (s)": round(tempo_execucao, 2),
        "Tempo Médio por Episódio (s)": round(tempo_medio, 3),
        "Episódios até Convergência": convergencia if convergencia else "N/A",
        "Recompensas": rewards  
    }

if __name__ == "__main__":
    resultados = []

    for nome, caminho in MODELS.items():
        if nome == "DQN":
            classe = DQN
        elif nome == "PPO":
            classe = PPO
        elif nome == "A2C":
            classe = A2C

        recompensas, tempo = avaliar_modelo(classe, caminho)
        metricas = calcular_metricas(recompensas, nome, tempo)
        resultados.append(metricas)

    nomes = [r["Algoritmo"] for r in resultados]
    medias = [r["Recompensa Média"] for r in resultados]
    desvios = [r["Desvio Padrão"] for r in resultados]
    tempos = [r["Tempo Total (s)"] for r in resultados]
    cvs = [r["Coef. de Variação"] for r in resultados]

    plt.figure(figsize=(10, 6))
    plt.bar(nomes, medias, yerr=desvios, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
    plt.title("Recompensa Média com Desvio Padrão")
    plt.ylabel("Recompensa")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(nomes, tempos, color='orange')
    plt.title("Tempo Total de Avaliação (s)")
    plt.ylabel("Segundos")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot([r["Recompensas"] for r in resultados], labels=nomes, patch_artist=True)
    plt.title("Distribuição das Recompensas")
    plt.ylabel("Recompensa por Episódio")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
