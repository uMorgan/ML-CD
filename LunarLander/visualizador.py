import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
import numpy as np
import matplotlib.pyplot as plt

def visualizar_modelo(modelo_path, algoritmo, num_episodios=5):
    env = gym.make('LunarLander-v2', render_mode='human')
    
    if algoritmo == 'DQN':
        modelo = DQN.load(modelo_path)
    elif algoritmo == 'A2C':
        modelo = A2C.load(modelo_path)
    elif algoritmo == 'PPO':
        modelo = PPO.load(modelo_path)
    else:
        raise ValueError(f"Algoritmo {algoritmo} não suportado")
    
    recompensas = []
    
    for episodio in range(num_episodios):
        obs, _ = env.reset()
        recompensa_episodio = 0
        terminado = False
        truncado = False
        
        while not (terminado or truncado):
            acao, _ = modelo.predict(obs, deterministic=True)
            obs, recompensa, terminado, truncado, _ = env.step(acao)
            recompensa_episodio += recompensa
        
        recompensas.append(recompensa_episodio)
        print(f"Episódio {episodio + 1}: Recompensa = {recompensa_episodio:.2f}")
    
    env.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(recompensas, marker='o')
    plt.title(f'Recompensas por Episódio - {algoritmo}')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualizar_modelo('LunarLander/models/best_model', 'PPO') 