#codigo apenas para testar se a biblioteca está funcionando!
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
