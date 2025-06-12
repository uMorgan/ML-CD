import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Configuração de seeds para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configurações
ENV_ID = "Acrobot-v1"
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 100_000
SAVE_DIR = "models/ppo_acrobot_models"
LOG_DIR = "logs/ppo_acrobot"

# Criar diretórios se não existirem
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Criar e configurar ambiente
env = make_vec_env(ENV_ID, n_envs=1)
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env.seed(SEED)

# Criar ambiente de avaliação
eval_env = make_vec_env(ENV_ID, n_envs=1)
eval_env = VecMonitor(eval_env)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
eval_env.seed(SEED)

# Configurar callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=SAVE_DIR,
    name_prefix="ppo_acrobot"
)

# Criar e treinar modelo
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log=LOG_DIR
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Salvar modelo final
model.save(os.path.join(SAVE_DIR, "ppo_acrobot_final"))
