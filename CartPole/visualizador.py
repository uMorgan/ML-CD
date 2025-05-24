import gymnasium as gym
import time
from stable_baselines3 import DQN, A2C, PPO

# === CONFIGURAÇÕES ===
ALGO = "PPO"  # Opções: "DQN", "A2C", "PPO"
ENV_ID = "CartPole-v1"
N_EPISODES = 5

# === Caminho do modelo final ===
CHECKPOINTS = {
    "DQN": "models/dqn_cartpole_checkpoints/dqn_cartpole_500000_steps.zip",
    "A2C": "models/a2c_cartpole_checkpoints/a2c_cartpole_500000_steps.zip",
    "PPO": "models/ppo_cartpole_checkpoints/ppo_cartpole_500000_steps.zip"
}

CHECKPOINT = CHECKPOINTS.get(ALGO)
if CHECKPOINT is None:
    raise ValueError("Algoritmo inválido. Use DQN, A2C ou PPO.")

# === Carregar modelo ===
if ALGO == "DQN":
    model = DQN.load(CHECKPOINT)
elif ALGO == "A2C":
    model = A2C.load(CHECKPOINT)
elif ALGO == "PPO":
    model = PPO.load(CHECKPOINT)

# === Criar ambiente com renderização ===
env = gym.make(ENV_ID, render_mode="human")

# === Rodar episódios ===
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)  # Pequeno delay para melhor visualização

    print(f"🏁 Episódio {ep + 1} finalizado com recompensa total: {total_reward:.2f}")

env.close() 