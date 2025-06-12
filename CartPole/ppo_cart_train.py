import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Criar diretórios para logs e checkpoints
os.makedirs("./logs/ppo_cartpole", exist_ok=True)
os.makedirs("./models/ppo_cartpole_checkpoints", exist_ok=True)

# Criar e configurar o ambiente
env = gym.make("CartPole-v1")

# Configurar o modelo PPO com hiperparâmetros otimizados
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_cartpole",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

# Configurar callback para salvar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/ppo_cartpole_checkpoints/",
    name_prefix="ppo_cartpole"
)

# Treinar o modelo
model.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback,
    progress_bar=True
)

# Salvar o modelo final
model.save("models/ppo_cartpole_final")