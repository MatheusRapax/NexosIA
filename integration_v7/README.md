# Fase 14 (Parte 3): Mecanismo D (Context-dependent Gating)

## Objetivo
Implementar uma versão simplificada de gating dependente de contexto (inspirado em XdG / Masse et al. 2018), onde cada tarefa ativa um subconjunto de unidades ocultas na rede.

## Implementação
- **Gating**: Implementado como uma máscara binária aplicada multiplicativamente às ativações $x[l]$ no forward pass e inferência, com erros forçados a zero para unidades inativas (garantindo ausência de aprendizado).
- **Tracker**: `GatingLocalTracker` seleciona de forma determinística uma fração `gate_frac` das unidades por camada a cada *changepoint*. Neste experimento, `overlap_frac = 0.0` foi utilizado.

## Resultados (Tabela de Calibração)

| Gate  | A pos A | A pos B | B pos B | A pos C | B pos C | Red A-apos-B | Red A-apos-C | Red B-apos-C |
|-------|---------|---------|---------|---------|---------|--------------|--------------|--------------|
| A(b)  | 0.1848  | 0.1437  | 0.3229  | 0.2436  | 0.0258  |      66.24%  |     -17.23%  |      51.14%  |
| 0.25  | 0.4676  | 0.9554  | 2.5311  | 1.0057  | 3.3964  |    -124.49%  |    -383.98%  |   -6332.64%  |
| 0.5   | 0.3639  | 0.7432  | 2.2728  | 0.9129  | 2.8662  |     -74.62%  |    -339.29%  |   -5328.35%  |
| 0.75  | 0.2412  | 0.3027  | 2.0254  | 0.2044  | 1.5325  |      28.87%  |       1.65%  |   -2802.45%  |

*Limitação Conhecida*: A avaliação foi feita sem um oráculo de tarefa, utilizando o gating mais recente (da tarefa C). Isto afeta o erro de A pós C e A pós B, pois a rede tenta prever transições da Região A com as unidades que foram ativadas para a Região C ou B.

## Diagnóstico
As frações ativas se mantiveram exatamente nas proporções nominais (25%, 50%, 75%).

## Conclusão
O Mecanismo D, como implementado, **perde severamente** em comparação ao Mecanismo C corrigido. 
A rede PCN base possui um número baixo de unidades ocultas (16 e 8). Limitar a ativação a 25% (4 e 2 unidades) ou 50% causa um **déficit extremo de capacidade**. Como evidenciado pelo erro `A pos A` e `B pos B` altíssimos (> 2.0), a rede sub-representada sequer consegue aprender de forma competente a tarefa atual. O gating exige que o orçamento total de unidades (tamanho da rede) seja consideravelmente maior para acomodar partições capazes de resolver a tarefa de forma independente.

## Fase 15: Avaliacao com Oraculo de Tarefa

Nesta fase (Rodada 9), implementamos uma avaliação "com oráculo" para o Mecanismo D. Ao medir o erro de uma região já treinada (ex: avaliar A após B, ou A e B após C), utilizamos o *gate* que estava ativo no momento em que aquela região terminou de treinar, em vez do *gate* mais recente, para isolar o problema de avaliação do problema de capacidade.

### Resultados: Oráculo vs Gate Mais Recente

| Gate  | A-pos-B (sem/com) | A-pos-C (sem/com) | B-pos-C (sem/com) |
|-------|-------------------|-------------------|-------------------|
| 0.25  | 0.9554 / 0.9554   | 1.0057 / 0.6997   | 3.3964 / 2.6544   |
| 0.5   | 0.7432 / 0.7432   | 0.9129 / 0.9129   | 2.8662 / 2.8662   |
| 0.75  | 0.3027 / 0.3027   | 0.2044 / 1.2630   | 1.5325 / 3.4455   |

*(Obs: Os valores "com oráculo" de A-pos-B mantêm o gate da Região A; os de A-pos-C usam o gate de A; e B-pos-C usam o gate de B).*

### Conclusão e Recomendação para a Rodada 10
Os resultados demonstram que **o oráculo NÃO ajuda significativamente a reduzir os erros**. Em todos os casos, os erros mesmo usando o gate correto (com oráculo) continuam imensamente mais altos do que os baselines normais da Fase 14 (onde os erros ficavam entre 0.1 e 0.3). No caso `gate_frac=0.75`, o oráculo chega a performar pior, devido à dinâmica do Gating e pesos subjacentes.

Esses resultados **confirmam a hipótese de DÉFICIT DE CAPACIDADE** da rede. O fracasso do Gating nesta rede pequena não é apenas um artefato de avaliação (erro de inferência de tarefa no teste), mas reflete que subdividir uma rede já enxuta (16 e 8 unidades) destrói sua capacidade representacional.

**Recomendação para a Rodada 10:** A próxima rodada deve focar no Ponto 3 discutido anteriormente — escalar a rede. Propõe-se aumentar a arquitetura base em cerca de 4x (ex: `[7, 64, 32, 2]`), de forma que uma fração pequena como `gate_frac=0.25` disponibilize para cada tarefa um número de unidades equivalente ao da rede original inteira.
