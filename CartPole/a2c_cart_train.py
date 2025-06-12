import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Criar diretórios para logs e checkpoints
os.makedirs("./logs/a2c_cartpole", exist_ok=True)
os.makedirs("./models/a2c_cartpole_checkpoints", exist_ok=True)

# Criar e configurar o ambiente
env = gym.make("CartPole-v1")

# Configurar o modelo A2C com hiperparâmetros otimizados
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/a2c_cartpole",
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    rms_prop_eps=1e-5,
    use_rms_prop=True,
    use_sde=False,
    sde_sample_freq=-1,
    normalize_advantage=False
)

# Configurar callback para salvar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/a2c_cartpole_checkpoints/",
    name_prefix="a2c_cartpole"
)

# Treinar o modelo
model.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback,
    progress_bar=True
)

# Salvar o modelo final
model.save("models/a2c_cartpole_final") 