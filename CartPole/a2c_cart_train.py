import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/a2c_cartpole")


checkpoint_callback = CheckpointCallback(
    save_freq=50000,  
    save_path="./models/a2c_cartpole_checkpoints/",
    name_prefix="a2c_cartpole"
)

model.learn(total_timesteps=500000, callback=checkpoint_callback,progress_bar=True)
model.save("a2c_cartpole_final")