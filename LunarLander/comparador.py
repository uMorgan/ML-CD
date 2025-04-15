import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

MODELS = {
    "DQN": "dqn_lunarlander.zip",
    "PPO": "ppo_lunarlander.zip",
    "A2C": "a2c_lunarlander.zip"
}

def avaliar_modelo(model_class, path, env_name="LunarLander-v3", n_episodes=100):
    env = gym.make(env_name)
    model = model_class.load(path)
    rewards, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes, return_episode_rewards=True)
    env.close()
    return np.array(rewards)

def calcular_metricas(rewards, nome):
    media = np.mean(rewards)
    std = np.std(rewards)
    cv = std / media if media != 0 else float('inf')

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
    print(f"  ⏱️ Episódios até convergência (estimado): {convergencia if convergencia else 'Não convergiu'}")

    return {
        "Algoritmo": nome,
        "Recompensa Média": round(media, 2),
        "Desvio Padrão": round(std, 2),
        "Coef. de Variação": round(cv, 2),
        "Episódios até Convergência": convergencia if convergencia else "N/A"
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

        recompensas = avaliar_modelo(classe, caminho)
        metricas = calcular_metricas(recompensas, nome)
        resultados.append(metricas)

    print("\n📋 Comparação Final:")
    for r in resultados:
        print(r)
