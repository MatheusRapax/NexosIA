# Dream Sampling Probe (Rodada 15 - Gargalo 1)

Esta probe testa a hipótese de que o esgotamento precoce de capacidade da Fase 9 (profundidade 1 de memória) não foi causado por limitação da *Slow Network* em si, mas sim pela política de amostragem dos sonhos.

Na política original, `visit_snapshot` usava contagens cumulativas brutas e os sonhos sorteavam de todas as tarefas de forma igual (unweighted). Nesta probe, aplicamos um EMA (Exponential Moving Average) no acúmulo da importância:

```python
importance_total = gamma_importance * importance_total + snapshot_atual
```

Essa mudança permite proteger as memórias de tarefas mais antigas com mais agressividade (crescimento exponencial do peso de tarefas antigas).

## Resultados

| Model | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-pos-C | Red B-pos-C |
|---|---|---|---|---|---|---|---|
| Sem Sono | 0.0583 | 0.4256 | 0.0494 | 0.2078 | 0.0528 | - | - |
| Com Sono (Orig) | 0.1848 | 0.1437 | 0.3229 | 0.2436 | 0.0258 | -17.23% | 51.19% |
| Com Sono (Acc) | 0.1230 | 0.0988 | 0.4456 | 0.2441 | 0.0593 | -17.47% | -12.25% |

**Conclusão**:
**Achado Negativo Informativo**. A acumulação via EMA de fato sobre-representou a Tarefa A (mais antiga) em relação à B (mais recente), forçando a rede a sonhar mais com A do que na política original. O resultado? A proteção da Tarefa A continuou inexistente (-17.47%, estatisticamente igual ao -17.23% original), e a proteção da Tarefa B **despencou** de +51.19% para -12.25% (perda total de proteção).

Isso refuta a premissa do Gargalo 1 de que o problema era apenas uma falha de "o que está sendo sonhado". Você pode forçar a rede a focar nas memórias antigas, mas se a **capacidade representacional já está saturada**, o replay excessivo de A apenas gera interferência catastrófica tanto com o aprendizado atual (C) quanto com a preservação de B, destruindo tudo. O gargalo real é a capacidade destrutiva da atualização, o que reforça a urgência do Mecanismo F (Poda/Reciclagem) sugerido na Rodada 13.
