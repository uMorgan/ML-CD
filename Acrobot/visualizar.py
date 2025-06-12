import gymnasium as gym
import time
from stable_baselines3 import DQN, A2C, PPO

ALGO = "DQN"
ENV_ID = "Acrobot-v1"
N_EPISODES = 5

CHECKPOINTS = {
    "DQN": "dqn_acrobot_models/dqn_acrobot_500000_steps.zip",
    "A2C": "a2c_acrobot_models/a2c_acrobot_500000_steps.zip",
    "PPO": "ppo_acrobot_models/ppo_acrobot_500000_steps.zip"
}

CHECKPOINT = CHECKPOINTS.get(ALGO)
if CHECKPOINT is None:
    raise ValueError("Algoritmo inválido. Use DQN, A2C ou PPO.")

if ALGO == "DQN":
    model = DQN.load(CHECKPOINT)
elif ALGO == "A2C":
    model = A2C.load(CHECKPOINT)
elif ALGO == "PPO":
    model = PPO.load(CHECKPOINT)

env = gym.make(ENV_ID, render_mode="human")

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)

    print(f"🏁 Episódio {ep + 1} finalizado com recompensa total: {total_reward:.2f}")

env.close()
