import os
import time
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gym

ENV_ID = "Acrobot-v1"
TOTAL_TIMESTEPS = 500_000
SAVE_FREQ = 50_000
SAVE_DIR = "a2c_acrobot_models"

os.makedirs(SAVE_DIR, exist_ok=True)

env = make_vec_env(ENV_ID, n_envs=1, wrapper_class=Monitor)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/a2c_acrobot")

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=SAVE_DIR,
    name_prefix="a2c_acrobot"
)

start_time = time.time()

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

total_time = time.time() - start_time
model.save(f"{SAVE_DIR}/a2c_acrobot_final")
print(f"\nTempo total de treinamento: {total_time:.2f} segundos")
