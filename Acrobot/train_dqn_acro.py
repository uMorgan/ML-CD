import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
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
SAVE_DIR = "models/dqn_acrobot_models"
LOG_DIR = "logs/dqn_acrobot"

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
    name_prefix="dqn_acrobot"
)

# Criar e treinar modelo
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log=LOG_DIR
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Salvar modelo final
model.save(os.path.join(SAVE_DIR, "dqn_acrobot_final"))
