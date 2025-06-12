import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def make_env():
    return gym.make("LunarLander-v2")

env = DummyVecEnv([make_env()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_env.training = False

os.makedirs('models/ppo_lunarlander_checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/ppo_lunarlander_checkpoints',
    log_path='logs',
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='models/ppo_lunarlander_checkpoints',
    name_prefix='lunarlander_ppo_model'
)

callbacks = [eval_callback, checkpoint_callback]

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
    tensorboard_log="logs"
)

model.learn(
    total_timesteps=1_000_000,
    callback=callbacks,
    progress_bar=True
)

model.save("models/ppo_lunarlander_checkpoints/ppo_lunarlander_final")
env.save("models/ppo_lunarlander_checkpoints/vec_normalize.pkl") 