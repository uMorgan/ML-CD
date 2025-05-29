from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import os

env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

eval_env = gym.make('LunarLander-v2')
eval_env = DummyVecEnv([lambda: eval_env])

os.makedirs('LunarLander/models', exist_ok=True)
os.makedirs('LunarLander/logs', exist_ok=True)

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log='LunarLander/logs')
eval_callback = EvalCallback(eval_env, best_model_save_path='LunarLander/models', log_path='LunarLander/logs', eval_freq=1000, deterministic=True, render=False)
model.learn(total_timesteps=500000, callback=eval_callback) 