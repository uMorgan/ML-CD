import gym
import time
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")
model = PPO.load("ppo_lunarlander")
obs, _ = env.reset()

for _ in range(5): 
    done = False
    obs, _ = env.reset()
    
    while not done:
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, done, truncated, _ = env.step(action)
        time.sleep(0.05) 

env.close()
