import os
import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_ID = "CartPole-v1"
N_EPISODES = 10
EVAL_SEED = 42
CHECKPOINT_STEPS = list(range(100_000, 1_000_001, 100_000))
MODELS = {
    "DQN": (DQN, os.path.join(SCRIPT_DIR, "models/dqn_cartpole_checkpoints/dqn_cartpole_{}_steps")),
    "A2C": (A2C, os.path.join(SCRIPT_DIR, "models/a2c_cartpole_checkpoints/a2c_cartpole_{}_steps")),
    "PPO": (PPO, os.path.join(SCRIPT_DIR, "models/ppo_cartpole_checkpoints/ppo_cartpole_{}_steps"))
}

IDEAL_CONVERGENCE = 475

def make_env():
    env = gym.make(ENV_ID)
    return env

def evaluate_model(model, n_episodes=10):
    env = make_env()
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards), np.std(rewards)

results = {}
data_rows = []

for algo_name, (algo_class, path) in MODELS.items():
    print(f"\n🔍 Avaliando {algo_name}...")
    avg_rewards = []
    std_rewards = []
    durations = []

    for step in CHECKPOINT_STEPS:
        # Criar ambiente para carregar o modelo
        env = DummyVecEnv([make_env])
        
        start_time = time.time()
        model = algo_class.load(path.format(step), env=env)
        duration = time.time() - start_time

        # Avaliar o modelo
        avg_reward, std_reward = evaluate_model(model, N_EPISODES)
        avg_rewards.append(avg_reward)
        std_rewards.append(std_reward)
        durations.append(duration)

        data_rows.append({
            "Checkpoint": step,
            f"{algo_name}_avg_reward": avg_reward,
            f"{algo_name}_std_reward": std_reward,
            f"{algo_name}_duration": duration
        })

        env.close()

    results[algo_name] = {
        "avg_rewards": avg_rewards,
        "std_rewards": std_rewards,
        "durations": durations
    }

summary_data = []
for step_idx, step in enumerate(CHECKPOINT_STEPS):
    row = {"Checkpoint (passos)": f"{step/1000:.0f}k"}
    
    for algo_name in MODELS:
        row[f"{algo_name} (recompensa)"] = f"{results[algo_name]['avg_rewards'][step_idx]:.2f} ± {results[algo_name]['std_rewards'][step_idx]:.2f}"
        row[f"{algo_name} (tempo, s)"] = f"{results[algo_name]['durations'][step_idx]:.4f}"
        
        if results[algo_name]['avg_rewards'][step_idx] >= IDEAL_CONVERGENCE:
            row[f"{algo_name} (convergência)"] = "Sim"
        else:
            row[f"{algo_name} (convergência)"] = "Não"
    
    summary_data.append(row)

comparison_df = pd.DataFrame(summary_data)

ordered_columns = ["Checkpoint (passos)"]
for algo_name in MODELS:
    ordered_columns.extend([
        f"{algo_name} (recompensa)", 
        f"{algo_name} (tempo, s)",
        f"{algo_name} (convergência)"
    ])
comparison_df = comparison_df[ordered_columns]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\n📊 Tabela Comparativa:")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("tabela_comparativa.csv", index=False)

# Estilos personalizados para cada algoritmo
styles = {
    "DQN": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "alpha": 0.8},
    "A2C": {"color": "#2ca02c", "marker": "s", "linestyle": "--", "alpha": 0.8},
    "PPO": {"color": "#9467bd", "marker": "^", "linestyle": "-.", "alpha": 0.8}
}

# Gráfico de Recompensa Média
plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["avg_rewards"], 
             label=algo_name, 
             **styles[algo_name],
             linewidth=2,
             markersize=8)

plt.axhline(y=IDEAL_CONVERGENCE, color='r', linestyle=':', label='Convergência Ideal', linewidth=2)
plt.xlabel("Passos de Treinamento")
plt.ylabel("Recompensa Média")
plt.title("Recompensa Média por Episódio - CartPole-v1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("recompensa_media.png", dpi=300, bbox_inches='tight')
plt.show()

# Gráfico de Estabilidade
plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["std_rewards"], 
             label=algo_name, 
             **styles[algo_name],
             linewidth=2,
             markersize=8)

plt.xlabel("Passos de Treinamento")
plt.ylabel("Desvio Padrão da Recompensa")
plt.title("Estabilidade da Recompensa - CartPole-v1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("estabilidade_recompensa.png", dpi=300, bbox_inches='tight')
plt.show()

# Gráfico de Tempo (usando subplots para melhor visualização)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 3])
fig.suptitle("Tempo de Avaliação por Checkpoint - CartPole-v1", fontsize=14)

# Subplot superior (zoom na discrepância inicial)
for algo_name in MODELS:
    ax1.plot(CHECKPOINT_STEPS, results[algo_name]["durations"], 
             label=algo_name, 
             **styles[algo_name],
             linewidth=2,
             markersize=8)
ax1.set_ylim(0, 0.1)  # Ajuste este valor conforme necessário
ax1.set_ylabel("Tempo (s)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot inferior (visão completa)
for algo_name in MODELS:
    ax2.plot(CHECKPOINT_STEPS, results[algo_name]["durations"], 
             label=algo_name, 
             **styles[algo_name],
             linewidth=2,
             markersize=8)
ax2.set_xlabel("Passos de Treinamento")
ax2.set_ylabel("Tempo (s)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("tempo_avaliacao.png", dpi=300, bbox_inches='tight')
plt.show()

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
