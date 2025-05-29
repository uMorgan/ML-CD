import os
import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "LunarLander-v3"
N_EPISODES = 10
CHECKPOINT_STEPS = list(range(50_000, 500_001, 50_000))
MODELS = {
    "DQN": (DQN, "LunarLander/models/dqn_lunarlander_checkpoints/dqn_lunarlander_{}_steps"),
    "A2C": (A2C, "LunarLander/models/a2c_lunarlander_checkpoints/a2c_lunarlander_{}_steps"),
    "PPO": (PPO, "LunarLander/models/ppo_lunarlander_checkpoints/ppo_lunarlander_{}_steps")
}

IDEAL_CONVERGENCE = 200  # Valor típico para considerar "bom" no LunarLander-v2

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
comparison_df.to_csv("tabela_comparativa_lunarlander.csv", index=False)

plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["avg_rewards"], label=algo_name)
plt.axhline(y=IDEAL_CONVERGENCE, color='r', linestyle='--', label='Convergência Ideal')
plt.xlabel("Passos de Treinamento")
plt.ylabel("Recompensa Média")
plt.title("Recompensa Média por Episódio - LunarLander-v2")
plt.legend()
plt.grid(True)
plt.savefig("recompensa_media_lunarlander.png")
plt.show()

plt.figure(figsize=(12, 6))
for algo_name in MODELS:
    plt.plot(CHECKPOINT_STEPS, results[algo_name]["std_rewards"], label=algo_name)
plt.xlabel("Passos de Treinamento")
plt.ylabel("Desvio Padrão da Recompensa")
plt.title("Estabilidade da Recompensa - LunarLander-v2")
plt.legend()
plt.grid(True)
plt.savefig("estabilidade_recompensa_lunarlander.png")
plt.show()

plt.figure(figsize=(12, 6))
bar_width = 0.2
bar_positions = np.arange(len(CHECKPOINT_STEPS))
for i, algo_name in enumerate(MODELS):
    offset = (i - 1) * bar_width
    plt.bar(bar_positions + offset, results[algo_name]["durations"], width=bar_width, label=algo_name)
plt.xlabel("Passos de Treinamento")
plt.ylabel("Tempo de Avaliação (s)")
plt.title("Tempo de Avaliação por Checkpoint - LunarLander-v2")
plt.xticks(bar_positions, [f"{step/1000}k" for step in CHECKPOINT_STEPS])
plt.legend()
plt.grid(True)
plt.savefig("tempo_avaliacao_lunarlander.png")
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

def avaliar_modelo(modelo_path, algoritmo, num_episodios=100):
    env = gym.make('LunarLander-v2')
    env = DummyVecEnv([lambda: env])
    
    if algoritmo == 'DQN':
        modelo = DQN.load(modelo_path)
    elif algoritmo == 'A2C':
        modelo = A2C.load(modelo_path)
    elif algoritmo == 'PPO':
        modelo = PPO.load(modelo_path)
    else:
        raise ValueError(f"Algoritmo {algoritmo} não suportado")
    
    recompensas = []
    
    for _ in range(num_episodios):
        obs = env.reset()[0]
        recompensa_episodio = 0
        terminado = False
        truncado = False
        
        while not (terminado or truncado):
            acao, _ = modelo.predict(obs, deterministic=True)
            obs, recompensa, terminado, truncado, _ = env.step(acao)
            recompensa_episodio += recompensa
        
        recompensas.append(recompensa_episodio)
    
    env.close()
    
    media = np.mean(recompensas)
    std = np.std(recompensas)
    return f"{media:.2f} ± {std:.2f}"

def gerar_tabela_comparativa():
    os.makedirs('LunarLander/resultados', exist_ok=True)
    
    algoritmos = ['DQN', 'A2C', 'PPO']
    checkpoints = {
        'DQN': 'LunarLander/models/best_model',
        'A2C': 'LunarLander/models/best_model',
        'PPO': 'LunarLander/models/best_model'
    }
    
    resultados = {}
    for algo in algoritmos:
        if os.path.exists(checkpoints[algo]):
            recompensa = avaliar_modelo(checkpoints[algo], algo)
            resultados[algo] = recompensa
        else:
            print(f"Checkpoint não encontrado para {algo}")
            resultados[algo] = "N/A"
    
    df = pd.DataFrame([resultados])
    df.index = ['Recompensa Média']
    
    df.to_csv('LunarLander/resultados/tabela_comparativa_lunarlander.csv')
    print("\nTabela Comparativa:")
    print(df)

if __name__ == "__main__":
    gerar_tabela_comparativa() 