import os
import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordEpisodeStatistics

ENV_ID = "CartPole-v1"
N_EPISODES = 10
CHECKPOINT_STEPS = list(range(50_000, 500_001, 50_000))
MODELS = {
    "DQN": (DQN, "models/dqn_cartpole_checkpoints/dqn_cartpole_{}_steps"),
    "A2C": (A2C, "models/a2c_cartpole_checkpoints/a2c_cartpole_{}_steps"),
    "PPO": (PPO, "models/ppo_cartpole_checkpoints/ppo_cartpole_{}_steps")
}

IDEAL_CONVERGENCE = 475  # Linha de referência para convergência

def evaluate_model(model, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

results = {}
data_rows = []

for algo_name, (algo_class, path_pattern) in MODELS.items():
    print(f"\n🔍 Avaliando {algo_name}...")
    avg_rewards = []
    std_rewards = []
    durations = []

    for step in CHECKPOINT_STEPS:
        path = path_pattern.format(step)
        env = DummyVecEnv([lambda: gym.make(ENV_ID)])

        start_time = time.time()
        model = algo_class.load(path, env=env)
        duration = time.time() - start_time

        avg_reward, std_reward = evaluate_model(model, env, N_EPISODES)
        avg_rewards.append(avg_reward)
        std_rewards.append(std_reward)
        durations.append(duration)

        data_rows.append({
            "Checkpoint": step,
            f"{algo_name}_avg_reward": avg_reward,
            f"{algo_name}_std_reward": std_reward,
            f"{algo_name}_duration": duration
        })

    results[algo_name] = {
        "avg_rewards": avg_rewards,
        "std_rewards": std_rewards,
        "durations": durations
    }

# Criar uma tabela mais limpa e informativa
summary_data = []
for step_idx, step in enumerate(CHECKPOINT_STEPS):
    row = {"Checkpoint (passos)": f"{step/1000:.0f}k"}
    
    # Adicionar dados de cada algoritmo na mesma linha
    for algo_name in MODELS:
        row[f"{algo_name} (recompensa)"] = f"{results[algo_name]['avg_rewards'][step_idx]:.2f} ± {results[algo_name]['std_rewards'][step_idx]:.2f}"
        row[f"{algo_name} (tempo, s)"] = f"{results[algo_name]['durations'][step_idx]:.4f}"
        
        # Verificar convergência
        if results[algo_name]['avg_rewards'][step_idx] >= IDEAL_CONVERGENCE:
            row[f"{algo_name} (convergência)"] = "Sim"
        else:
            row[f"{algo_name} (convergência)"] = "Não"
    
    summary_data.append(row)

# Criar DataFrame melhorado
comparison_df = pd.DataFrame(summary_data)

# Reordenar as colunas para agrupar informações por algoritmo
ordered_columns = ["Checkpoint (passos)"]
for algo_name in MODELS:
    ordered_columns.extend([
        f"{algo_name} (recompensa)", 
        f"{algo_name} (tempo, s)",
        f"{algo_name} (convergência)"
    ])
comparison_df = comparison_df[ordered_columns]

# Mostrar tabela e salvar em CSV
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\n📊 Tabela Comparativa:")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("tabela_comparativa.csv", index=False)

# Gráficos
plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["avg_rewards"], label=algo_name)
plt.axhline(y=IDEAL_CONVERGENCE, color='r', linestyle='--', label='Convergência Ideal')
plt.xlabel("Passos de Treinamento")
plt.ylabel("Recompensa Média")
plt.title("Recompensa Média por Episódio - CartPole-v1")
plt.legend()
plt.grid(True)
plt.savefig("recompensa_media.png")
plt.show()

plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["std_rewards"], label=algo_name)
plt.xlabel("Passos de Treinamento")
plt.ylabel("Desvio Padrão da Recompensa")
plt.title("Estabilidade da Recompensa - CartPole-v1")
plt.legend()
plt.grid(True)
plt.savefig("estabilidade_recompensa.png")
plt.show()

# Gráfico de barras para o tempo de avaliação
plt.figure(figsize=(12, 6))
bar_width = 0.2
bar_positions = np.arange(len(CHECKPOINT_STEPS))
for i, algo_name in enumerate(MODELS):
    offset = (i - 1) * bar_width
    plt.bar(bar_positions + offset, results[algo_name]["durations"], width=bar_width, label=algo_name)
plt.xlabel("Passos de Treinamento")
plt.ylabel("Tempo de Avaliação (s)")
plt.title("Tempo de Avaliação por Checkpoint - CartPole-v1")
plt.xticks(bar_positions, [f"{step/1000}k" for step in CHECKPOINT_STEPS])
plt.legend()
plt.grid(True)
plt.savefig("tempo_avaliacao.png")
plt.show()

# Estimar número de episódios para convergência
print("\n📈 Estimativa de Convergência por Algoritmo:")
for algo_name in MODELS:
    rewards = results[algo_name]["avg_rewards"]
    steps = CHECKPOINT_STEPS
    convergiu = False
    for i, r in enumerate(rewards):
        if r >= IDEAL_CONVERGENCE:
            print(f"{algo_name}: convergência estimada em {steps[i]} passos de treinamento")
            convergiu = True
            break
    if not convergiu:
        print(f"{algo_name}: não convergiu até {steps[-1]} passos")
