# 🚀 Treinando Agentes no LunarLander-v2 com Aprendizado por Reforço

Este repositório contém três scripts que treinam agentes de IA usando diferentes algoritmos de aprendizado por reforço no ambiente **LunarLander-v2**. Os algoritmos utilizados são:

- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)

Abaixo, você encontrará uma explicação sobre cada um dos métodos, o que cada script faz e como você pode executar e avaliar os modelos.

---

## 📁 Estrutura do Repositório

```
📂 root/
│── dqn_train_lunar.py     # Treina e avalia um modelo DQN
│── ppo_train_lunar.py     # Treina e avalia um modelo PPO
│── a2c_train_lunar.py     # Treina e avalia um modelo A2C
│── dqn_model_view.py      # Visualiza o comportamento do modelo DQN
│── ppo_model_view.py      # Visualiza o comportamento do modelo PPO
│── a2c_model_view.py      # Visualiza o comportamento do modelo A2C
```

---

## 📌 Algoritmos Utilizados

### 🔹 1. DQN (Deep Q-Network)

> **DQN** é um algoritmo de aprendizado por reforço baseado em redes neurais profundas que aproxima a função de valor Q. Ele é eficaz para problemas com ações discretas.

- **📌 Tipo:** Off-policy
- **🛠️ Abordagem:** Valor de Q
- **✅ Vantagens:** Ideal para ambientes com ações discretas.
- **❌ Desvantagens:** Requer ajuste de hiperparâmetros e técnicas como replay de experiência.

#### 🏗️ O que o código faz:
- O script **`dqn_train_lunar.py`** treina um modelo DQN no ambiente **LunarLander-v2** por **100.000 timesteps**.
- O modelo treinado é salvo no arquivo **`dqn_lunarlander`**.
- O código **`dqn_model_view.py`** carrega o modelo treinado e visualiza o desempenho do agente.

---

### 🔹 2. PPO (Proximal Policy Optimization)

> **PPO** é um algoritmo baseado em gradiente de política que melhora a estabilidade do treinamento por meio de atualizações controladas.

- **📌 Tipo:** On-policy
- **🛠️ Abordagem:** Política
- **✅ Vantagens:** Estável e fácil de implementar.
- **❌ Desvantagens:** Requer mais interações com o ambiente do que algoritmos off-policy.

#### 🏗️ O que o código faz:
- O script **`ppo_train_lunar.py`** treina um modelo PPO no **LunarLander-v2** por **100.000 timesteps**.
- O modelo treinado é salvo no arquivo **`ppo_lunarlander`**.
- O código **`ppo_model_view.py`** carrega o modelo treinado e permite visualizar o desempenho do agente.

---

### 🔹 3. A2C (Advantage Actor-Critic)

> **A2C** combina um ator (**policy network**) e um crítico (**value network**) para aprender uma política de ação e estimar o valor do estado.

- **📌 Tipo:** On-policy
- **🛠️ Abordagem:** Actor-Critic
- **✅ Vantagens:** Atualização eficiente da política e bom desempenho em ambientes contínuos.
- **❌ Desvantagens:** Sensível aos hiperparâmetros.

#### 🏗️ O que o código faz:
- O script **`a2c_train_lunar.py`** treina um modelo A2C no **LunarLander-v2** por **100.000 timesteps**.
- O modelo treinado é salvo no arquivo **`a2c_lunarlander`**.
- O código **`a2c_model_view.py`** carrega o modelo treinado e exibe o desempenho do agente.

---

## 📊 Comparação entre DQN, PPO e A2C

| Algoritmo  | Abordagem        | Tipo       | Vantagens  | Desvantagens  |
|------------|-----------------|------------|------------|---------------|
| **DQN**    | Valor de Q       | Off-policy | Ideal para ações discretas | Requer replay de experiência |
| **PPO**    | Política         | On-policy  | Estável e eficiente | Maior custo computacional |
| **A2C**    | Actor-Critic     | On-policy  | Atualização rápida da política | Sensível a hiperparâmetros |

---
## 📈 Avaliação de Desempenho dos Modelos
> Após o treinamento, os modelos são avaliados por meio de três métricas principais:

🎯 Recompensa Média por Episódio: Avalia o desempenho geral do agente.

⏱️ Número de Episódios até a Convergência: Mede o tempo (estimado) para atingir um desempenho estável.

📉 Estabilidade do Desempenho: Calculada pelo desvio padrão e coeficiente de variação das recompensas.

Para isso, utilize o script **`comparador.py`**
---
## ⚙️ Como Rodar os Códigos

### 1️⃣ Instale as dependências:
```bash
pip install pygame
pip install gymnasium 
pip install stable-baselines3[extra]
```

### 2️⃣ Treine o modelo executando o código de treinamento do algoritmo desejado:
```bash
# Para DQN:
python dqn_train_lunar.py

# Para PPO:
python ppo_train_lunar.py

# Para A2C:
python a2c_train_lunar.py
```

### 3️⃣ Visualize o modelo treinado executando o código de visualização:
```bash
# Para DQN:
python dqn_model_view.py

# Para PPO:
python ppo_model_view.py

# Para A2C:
python a2c_model_view.py
```

---

## 🎯 Conclusão

Este repositório oferece uma maneira prática de **treinar, salvar e visualizar** agentes de IA no **LunarLander-v2** usando diferentes algoritmos. Através dos scripts fornecidos, você pode facilmente comparar o desempenho de **DQN, PPO e A2C**, bem como visualizar o comportamento dos agentes treinados em um ambiente gráfico.