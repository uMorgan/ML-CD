import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def make_env():
    return gym.make("LunarLander-v2")

env = DummyVecEnv([make_env()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_env.training = False

os.makedirs('models/dqn_lunarlander_checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/dqn_lunarlander_checkpoints',
    log_path='logs',
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='models/dqn_lunarlander_checkpoints',
    name_prefix='dqn_lunarlander'
)

callbacks = [eval_callback, checkpoint_callback]

model = DQN(
    "MlpPolicy", 
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log="logs"
)

model.learn(
    total_timesteps=1_000_000,
    callback=callbacks,
    progress_bar=True
)

model.save("models/dqn_lunarlander_checkpoints/dqn_lunarlander_final")
env.save("models/dqn_lunarlander_checkpoints/vec_normalize.pkl") 