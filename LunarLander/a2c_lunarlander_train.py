import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def make_env():
    return gym.make("LunarLander-v2")

env = DummyVecEnv([make_env()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_env.training = False

os.makedirs('models/a2c_lunarlander_checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/a2c_lunarlander_checkpoints',
    log_path='logs',
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='models/a2c_lunarlander_checkpoints',
    name_prefix='lunarlander_model'
)

callbacks = [eval_callback, checkpoint_callback]

model = A2C(
    "MlpPolicy", 
    env,
    learning_rate=0.0003,
    n_steps=20,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="logs"
)

model.learn(
    total_timesteps=1_000_000,
    callback=callbacks,
    progress_bar=True
)

model.save("models/a2c_lunarlander_checkpoints/a2c_lunarlander_final")
env.save("models/a2c_lunarlander_checkpoints/vec_normalize.pkl") 