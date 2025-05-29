import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/dqn_cartpole")


checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/dqn_cartpole_checkpoints/",
    name_prefix="dqn_cartpole"
)

model.learn(total_timesteps=500000, callback=checkpoint_callback, progress_bar=True)
model.save("dqn_cartpole_final")
