# Fase 3: Active Inference PCN com Learning Progress

Esta pasta implementa o agente Active Inference operando no `GridWorld` 5x5, agora utilizando **Expected Free Energy (EFE)** completo.

## Substituição do Epsilon-Greedy por EFE Real
Na Fase 2, a exploração era gerada por um epsilon-greedy (que é uma métrica aleatória injetada de fora do modelo). Na Fase 3, nós substituímos isso pela formula completa do Active Inference: `EFE = pragmatic_weight * G - epistemic_weight * LP`.

- **Valor Pragmático (G):** Quão perto o resultado previsto pela PCN está da observação-objetivo (`goal_obs`). Puxa o agente para concluir a tarefa.
- **Valor Epistêmico (LP - Learning Progress):** A taxa de sucesso em reduzir o erro de predição, rastreada por uma *Exponential Moving Average* em duas escalas (fast e slow). Evita o *Dark Room* (o agente se sente estimulado a explorar) mas resiste ao *Noisy TV problem* (ruídos que não diminuem o erro não geram Progresso de Aprendizado, impedindo o agente de ficar "viciado" em aleatoriedades).

*Nota sobre a implementação:* A discretização baseada em grid que gera o `cell_index` no código atual é uma simplificação deliberada de escopo para esta fase. Uma implementação contínua (ex: um VAE/clustering latente) seria mais geral para dados de alta dimensão, mas foge do escopo atual que visa comprovar a matemática da *Expected Free Energy*.

## Hiperparâmetros
- `pragmatic_weight` = 1.0
- `epistemic_weight` = 1.0
Os pesos foram escolhidos para balancear uma boa margem de exploração (LP) garantindo que o agente continue buscando o *goal* primário.

## Como rodar
```bash
python active_inference/run_phase3_experiment.py
pytest active_inference/tests/test_phase3_learning_progress.py
```
