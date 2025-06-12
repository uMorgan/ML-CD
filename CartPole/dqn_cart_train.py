import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Criar diretórios para logs e checkpoints
os.makedirs("./logs/dqn_cartpole", exist_ok=True)
os.makedirs("./models/dqn_cartpole_checkpoints", exist_ok=True)

# Criar e configurar o ambiente
env = gym.make("CartPole-v1")

# Configurar o modelo DQN com hiperparâmetros otimizados
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/dqn_cartpole",
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10
)

# Configurar callback para salvar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/dqn_cartpole_checkpoints/",
    name_prefix="dqn_cartpole"
)

# Treinar o modelo
model.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback,
    progress_bar=True
)

# Salvar o modelo final
model.save("models/dqn_cartpole_final")
