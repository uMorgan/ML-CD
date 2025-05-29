# README - LunarLander

Este diretório contém os scripts e resultados para o treinamento e comparação de algoritmos de Aprendizado por Reforço no ambiente **LunarLander-v2**.

## Conteúdo:

- `dqn_lunarlander_train.py`: Script para treinar o agente DQN.
- `a2c_lunarlander_train.py`: Script para treinar o agente A2C.
- `ppo_lunarlander_train.py`: Script para treinar o agente PPO.
- `comparador.py`: Script para comparar o desempenho dos agentes treinados e gerar tabelas e gráficos.
- `visualizador.py`: Script para visualizar o desempenho de um agente treinado em tempo real.
- `models/`: Diretório onde os modelos treinados são salvos.
- `logs/`: Diretório para logs do TensorBoard.
- `resultados/`: Diretório onde são salvos os resultados da comparação (tabela e gráficos).

## Resultados da Comparação:

A comparação dos algoritmos no ambiente LunarLander-v2 gerou os seguintes gráficos:

### Recompensa Média por Episódio

![Recompensa Média por Episódio](resultados/recompensa_media_lunarlander.png)

### Estabilidade da Recompensa

![Estabilidade da Recompensa](resultados/estabilidade_recompensa_lunarlander.png)

### Tempo de Avaliação por Checkpoint

![Tempo de Avaliação por Checkpoint](resultados/tempo_avaliacao_lunarlander.png)

Para gerar esses resultados, execute o script `comparador.py` dentro deste diretório.
