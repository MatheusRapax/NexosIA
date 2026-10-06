# Debate — Rodada 5: Normalizando a Penalidade do Mecanismo B

> Quinta rodada de debate, de escopo estreito: a Fase 10 implementou e calibrou o Mecanismo B (EWC Local Online), resolvendo a profundidade 1 (A-após-C: -17.23% → +52.22%), mas em TODOS os valores de `lam` testados (0.01 a 1.0) o MSE de B-pós-B piorou de 0.3229 (só Mecanismo A) para entre 0.41 e 0.90 — a plasticidade para aprender a tarefa seguinte colapsa. Objetivo: desenhar uma normalização de `lam` que resolva esse trade-off, ou determinar com rigor que um `lam` escalar não pode resolvê-lo. Ver [ARTIGO.md](ARTIGO.md) §11-§12 para as duas fases (11 e 12) que testaram experimentalmente as normalizações aqui debatidas — nenhuma resolveu o trade-off, o que viria a motivar a Rodada 7.

## Hipótese inicial e sua correção

Claude Code abriu perguntando se a distribuição de `F_total` entre pesos (pós Região A) seria concentrada ou quase uniforme, com a hipótese de que seria uniforme — já que o colapso aparecia mesmo em `lam=0.01`. O Antigravity apontou o furo: colapso em `lam` baixo não implica uniformidade da distribuição — pode ser um problema de ESCALA ABSOLUTA de `F_total`, independente de sua forma, e os dois eixos (forma/concentração vs. escala absoluta) são ortogonais. Era necessário medir ambos, não inferir um a partir do outro.

## A fórmula saturante e o problema da constante `c`

Claude Code propôs uma fórmula saturante, `lam · (F_total/(F_total+c)) · (w - w_anchor)`, que resolve escala (a penalidade nunca excede `lam`) e seletividade (emerge da própria razão, sem top-k arbitrário) ao mesmo tempo — mas deixou em aberto como estimar `c` sem uma rodada de medição empírica separada. O Antigravity propôs eliminar essa necessidade: `c = median(F_total)`, calculada DINAMICAMENTE a partir da própria distribuição a cada changepoint, autocalibrante em vez de uma constante chutada.

## O furo da mediana em redes pequenas e esparsas

Claude Code aceitou a direção (autocalibração a partir dos próprios dados) mas identificou um furo concreto na escolha específica da mediana: numa rede pequena e esparsa como `[7,16,8,2]`, é bem plausível que mais da metade dos pesos de uma camada tenham `F_total` exatamente zero (caminhos de ativação pouco usados durante o treino de uma região). Se isso acontecer, a mediana por camada é zero, e a fórmula degenera de duas formas simultâneas: pesos com `F_total=0` ficam `0/(0+0)` = NaN, e qualquer peso com `F_total>0` satura IMEDIATAMENTE em 1 — destruindo a gradação que a fórmula deveria dar, na pior forma possível (NaN em produção).

A correção proposta reusou um padrão já validado no próprio projeto — o `epsilon=1.0` como piso mínimo na amostragem priorizada do Mecanismo A (Fase 8): trocar `c = median(F_total)` por `c_layer = mean(F_total da camada) + epsilon_c` (epsilon_c pequeno, ex. 1e-6), calculada por camada (dimensionalidades distintas) e uma única vez por changepoint. A média é robusta à esparsidade (não zera só porque metade dos pesos são zero — a cauda de pesos importantes já a desloca para cima), e o piso `epsilon_c` garante que a fórmula nunca produz NaN mesmo no caso degenerado de `F_total` inteiramente zero (nesse caso, a penalidade cai corretamente para zero — sem Fisher acumulada, sem penalidade).

O Antigravity concordou integralmente com a correção, confirmando que a mediana arriscava divisão por zero ou saturação imediata nesta escala de rede, e que média+epsilon resolve normalização e esparsidade na mesma mudança.

## Decisão — convergido

Mecanismo B, versão corrigida (substituindo a penalidade constante da Fase 10):

- Fórmula: `w -= lam · (F_total[w] / (F_total[w] + c_layer)) · (w - w_anchor[w])`, aplicada por camada.
- `c_layer = mean(F_total da camada) + epsilon_c`, recalculado uma vez por `on_changepoint`, usando o `F_total` já atualizado naquele changepoint (após decaimento + incorporação da Rodada 4).
- `lam` continua precisando de calibração própria — agora como teto superior bem definido da penalidade, não mais multiplicador de uma `F_total` de escala desconhecida — repetindo uma varredura pequena na implementação, já que o significado de `lam` mudou com a normalização.

Esta síntese tornou-se a especificação da Fase 11 — que, ao ser implementada com `c_layer = mean(F_total)` (não `max`, como a Rodada 6 revisaria depois ao ver o resultado), produziu um retrocesso em vez de uma correção (ver ARTIGO.md §11), motivando a Rodada 6.
