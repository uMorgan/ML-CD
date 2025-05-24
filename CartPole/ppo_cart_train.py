import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_cartpole")


checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/ppo_cartpole_checkpoints/",
    name_prefix="ppo_cartpole"
)

model.learn(total_timesteps=500000, callback=checkpoint_callback, progress_bar=True)
model.save("ppo_cartpole_final")