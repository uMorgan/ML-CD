import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")
model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0005, n_steps=5, gamma=0.99, gae_lambda=0.95)
TIMESTEPS = 500_000
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
model.save("a2c_lunarlander")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa Média: {mean_reward} +/- {std_reward}")

env.close()
