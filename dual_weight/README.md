# Dual-Weight PCN (Changepoint Detector)

## Mecanismo

O `DualWeightPCN` mantém duas cópias dos pesos de uma PredictiveCodingNetwork (PCN):
- **Fast Network**: Treinada normalmente a cada amostra usando a regra de aprendizado local (`train_step()`). Adapta-se rapidamente às mudanças na distribuição dos dados.
- **Slow Network**: Nunca é treinada diretamente nos dados. Seus pesos são atualizados gradualmente em direção aos pesos da Fast Network usando média móvel (Polyak averaging): `W_slow = beta_slow * W_fast + (1 - beta_slow) * W_slow`.

### Sinal de Changepoint (Delta Relativo)
Para detectar mudanças de regime (changepoints), comparamos a energia (erro de predição) de ambas as redes sobre a mesma transição de entrada. Essa comparação é feita usando apenas o método `predict()`, que é *read-only* e não altera os pesos:
1. `energia(rede, x, y) = 0.5 * sum((rede.predict(x) - y)^2)`
2. `delta = energia_slow - energia_fast_antes`

Esse `delta` é suavizado através de uma média móvel exponencial (EMA) escalar para reduzir o ruído:
`smoothed_delta = beta_delta * delta + (1 - beta_delta) * smoothed_delta`

Este mecanismo é puramente paramétrico e relativo. Ao comparar duas redes, conseguimos distinguir entre uma "surpresa normal" (exploração intra-tarefa em que ambas as redes erram de forma parecida, mantendo o `delta` baixo) e uma verdadeira mudança de regime (a Fast Network aprende a nova tarefa rápido, enquanto a Slow Network continua ancorada na tarefa antiga, fazendo o `delta` subir significativamente).

## Valores Finais e Resultados dos Testes

Os hiperparâmetros foram ajustados para maximizar a separação entre ruído de exploração e mudanças reais de regime:
- `beta_slow` = 0.02
- `beta_delta` = 0.1
- `threshold` = 0.5
- `weight_lr` = 0.05

### Números Observados
- **Teste A (Resistência a exploração intra-tarefa):** O pico de `smoothed_delta` observado foi de **`0.0014`**, o que é extremamente baixo e seguro (bem abaixo da margem exigida de `0.5 * threshold`, que é `0.25`).
- **Teste B (Detecção de shift real de regime):** O limite (`threshold` = 0.5) foi cruzado em apenas **`5 passos`** após a mudança abrupta de distribuição. O valor de `smoothed_delta` atingiu um pico de **`2.6261`** (muito maior do que a exigência de `2 * threshold`, que é `1.0`) antes de finalmente estabilizar e cair (convergência final em `0.0000` quando a Slow alcança a Fast).
