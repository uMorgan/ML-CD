import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")

model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, buffer_size=50000, batch_size=64, target_update_interval=500)
TIMESTEPS = 500_000 
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
model.save("dqn_lunarlander")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa Média: {mean_reward} +/- {std_reward}")

env.close()