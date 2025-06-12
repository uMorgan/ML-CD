# LunarLander-v2: Pouso de Nave Espacial

## 🎯 Objetivo
O objetivo deste ambiente é pousar uma nave espacial suavemente em uma plataforma, controlando o motor principal e os propulsores laterais.

## 📊 Resultados

### Tabela Comparativa

| Algoritmo | Recompensa Média | Desvio Padrão | Tempo de Avaliação (s) |
|-----------|------------------|---------------|------------------------|
| DQN       | 185.3           | 45.2          | 0.18                  |
| A2C       | 192.7           | 38.6          | 0.15                  |
| PPO       | 198.5           | 32.1          | 0.17                  |

### Convergência
- **DQN**: 700k passos
- **A2C**: Não convergiu até 1000k passos
- **PPO**: 100k passos

## 📈 Análise

### Melhor Algoritmo: PPO
- **Recompensa Média**: 198.5
- **Estabilidade**: Alta (desvio padrão 32.1)
- **Convergência**: Rápida (100k passos)

## 🚀 Como Usar

### 1. Treinamento
```bash
python dqn_lunarlander_train.py
python a2c_lunarlander_train.py
python ppo_lunarlander_train.py
```

### 2. Comparação
```bash
python comparador.py
```

## 📊 Visualização

### Gráficos
- [Recompensa Média](resultados/recompensa_media.png)
- [Estabilidade](resultados/estabilidade_recompensa.png)
- [Tempo de Avaliação](resultados/tempo_avaliacao.png)

### Tabelas
- [Tabela Comparativa](resultados/tabela_comparativa.csv)
- [Tabela de Convergência](resultados/tabela_convergencia.csv)

## 🔙 [Voltar ao README Principal](../README.md)
